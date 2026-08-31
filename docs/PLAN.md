# AI Chessathon — Engine Design & Build Plan

Living document. Owns the "how the agent thinks" detail so it survives context resets.
Last grounded against the live docs on **2026-08-31**.

---

## 0. Ground truth (fetched from the live site, overrides the repo docs)

The canonical docs are `https://aichessathon.com/docs/agent-contract.md` and `/docs/rules.md`.
They differ from this repo's `AGENTS.md`/`docs/IDEAS.md` in ways that change strategy. **Where
they disagree, the live site wins.** Re-fetch before every upload; the docs change.

| Fact | Live rules (authoritative) | Repo docs say | Impact |
|---|---|---|---|
| Learned model | **Not required** — "a classical search is a full entry" | "required to materially drive selection" | Classical engine is a complete entry. Net is optional upside. |
| Dependencies | **No `requirements.txt`.** Only `torch` (CPU), `numpy`, `python-chess`, `onnxruntime`, `numba` importable | mentions requirements.txt | Do not rely on any other package. `numba` is available. |
| Size cap | **≤ 50 MB unzipped** | 200 MB | Model must be small (a few MB NNUE, not a big net). |
| Start positions | **Curated, near-level, not always standard start** | implies standard start | Conventional opening book keyed to startpos is near-worthless. |
| Weights | Must be **self-trained**; `.onnx/.safetensors/.pt` allowed | same | Training *data* may be engine-annotated; what ships must be ours. |
| Banned | Stockfish, Lc0, Maia, **any wrapper** — retroactive DQ | same | Never import or embed one. Judges review source + finalist walkthrough. |

**Hard contract (unchanged):**
- `agent.py` at zip root. Platform does `import agent` and calls `get_move(fen, time_left_ms) -> str`.
- UCI out (`"e2e4"`, `"e7e8q"`). Colour = side to move in the FEN.
- Process lives for one game; **module state persists across our own moves, not across games.**
- **60 s import budget** before the clock starts — do all loading there.
- **120 s + 0.5 s/move, per side**, wall time. Flag = instant loss.
- 1 core, 2 GB RAM, 256 MB `/tmp` (read-write, wiped per game), no network, no GPU.
- Loss triggers: illegal move, malformed output, move payload > 4 KB, crash, OOM, flag.
- 128 concurrent processes share the core; background threads allowed (pondering).
- 300 plies → material adjudication draw. stdout/stderr captured to 8 KB in validation.

**Schedule:** 13-round Swiss on **locked builds, Sep 11**. ~11 days. 6 uploads/team/day, latest
valid plays. Tie-breaks: points, Buchholz, head-to-head, earlier submission.

---

## 1. Strategy & priorities

Because a classical engine is a full entry and we are in **Python on one core**, node counts are
small (low thousands/move, not millions). That inverts the usual C-engine trade: **evaluation
quality and move ordering buy more than raw depth.** Our edge comes, in priority order:

1. **A correct, fast alpha-beta search that never flags and never crashes.** Finishing every game
   cleanly is worth more than any single feature. Most self-inflicted losses are flags and edge-case
   crashes (no legal moves, promotion, en passant).
2. **Move ordering.** Alpha-beta only pays off when good moves come first. This is the single
   biggest strength multiplier and it is cheap.
3. **Quiescence search.** Without it, eval is read mid-exchange and is simply wrong.
4. **A tapered handcrafted evaluation** (material + piece-square + a few terms). This alone beats
   both baselines and is a legitimate competitive eval.
5. **Transposition table + iterative deepening**, which compound with ordering.
6. **(Stretch) a small self-trained NNUE-style eval** swapped in behind the same seam, only if it
   measurably beats the handcrafted eval at equal time. Optional, not on the critical path.

Non-goals early: a conventional opening book (start positions are curated), multi-threading (one
core), deep search (Python is slow — spend the nodes wisely instead).

---

## 2. Module layout

Everything ships in `agent.py` (plus a `weights/` dir if we do the net). Internally, keep it
sectioned so the eval is a swappable seam. Single file, but organized:

```
agent.py
├── constants           piece values, PST tables, search params, mate scores
├── Evaluator (seam)    evaluate(board) -> centipawns from side-to-move POV
│     ├── handcrafted   tapered material + PST + terms      (phase A, ships first)
│     └── nnue          onnxruntime session, batched         (phase B, optional)
├── TranspositionTable  dict keyed on zobrist, survives across moves within a game
├── Search
│     ├── get_move()          entry: budget, iterative deepening, return best
│     ├── _search()           negamax + alpha-beta (fail-soft)
│     ├── _quiesce()          captures/checks to quiet, stand-pat + delta pruning
│     ├── _order_moves()      TT move, MVV-LVA, killers, history
│     └── time/abort plumbing
└── module-level state   TT, killer/history tables, seen-positions set, warm caches
```

`get_move` is thin: parse FEN, set a deadline, run iterative deepening, return the best move from
the last *completed* depth. All the intelligence is in `_search`/`_quiesce`/`_order_moves`.

---

## 3. Board & move representation

- Use `python-chess` (`chess.Board`) — it is preinstalled and correct. Do **not** name any file
  `chess.py`/`types.py`/`random.py`; the zip is first on `sys.path` and would shadow the real one.
- The bottleneck is `board.legal_moves` generation and `push`/`pop`. Minimize both:
  - Generate the move list once per node and reuse it.
  - Prefer `board.push`/`board.pop` (incremental) over constructing new `Board` objects.
- TT key: `chess.polyglot.zobrist_hash(board)` (public, 64-bit, includes side/castling/ep) — or the
  faster internal `board._transposition_key()`. Pick one and be consistent. Zobrist is cleaner.
- `numba` note: it accelerates numpy/scalar code, **not** `python-chess` objects. It only helps if
  we hand it a numpy feature vector (e.g. an NNUE accumulator). Do not expect it to speed up
  move-gen without a full bitboard reimplementation — that's a large, deprioritized effort.

---

## 4. The search — detailed logic

### 4.1 Constants and scores

```python
MATE      = 1_000_000          # score for mate at the node
MATE_IN_MAX = MATE - 1000      # scores above this are "mate in N", encode distance in the gap
INF       = 2_000_000
MAX_PLY   = 128
```

Mate scores are stored **relative to the searching node** and adjusted for ply so the engine
prefers faster mates and delays being mated:
- When returning a mate: `MATE - ply` (mate the opponent) / `-(MATE - ply)` (we are mated).
- On TT store: add `ply` back out; on TT probe: subtract `ply` back in. (Section 4.5.)

### 4.2 Iterative deepening driver (`get_move`)

The driver is the time-management and safety layer. It searches depth 1, 2, 3, … keeping the best
move from each **completed** pass. When time runs out mid-pass, it discards that pass's partial
result and returns the last completed one. This is the discipline that prevents flags.

```python
def get_move(fen, time_left_ms):
    board = chess.Board(fen)

    # 0. Trivial / forced cases — no search needed.
    legal = list(board.legal_moves)
    if not legal:                       # should not happen (referee ends game first), but be safe
        return "0000"                   # null; referee will have already decided
    if len(legal) == 1:
        return legal[0].uci()

    # 1. Budget this move from the clock we were HANDED, not a constant.
    deadline = _compute_deadline(board, time_left_ms)   # monotonic time to stop by (§4.6)

    # 2. Remember this position for repetition awareness across moves (§4.7).
    _seen_positions.append(zobrist(board))

    # 3. Iterative deepening.
    best_move = legal[0]                # always have a legal fallback
    best_score = -INF
    try:
        for depth in range(1, MAX_PLY):
            score, move = _search_root(board, depth, deadline)
            if move is not None:        # completed this depth
                best_move, best_score = move, score
            if _mate_found(best_score): # no point searching deeper into a forced mate
                break
            # optional: if we've used > ~50% of the move budget, another full depth
            # is unlikely to finish — stop early and bank the clock.
            if _out_of_time_for_next_depth(deadline):
                break
    except _TimeUp:
        pass                            # ran out mid-depth; best_move is last completed depth

    return best_move.uci()
```

Root is searched with move ordering seeded by the previous iteration's best move (below), which is
where iterative deepening pays for itself: it hands each depth an almost-optimal ordering for free.

### 4.3 Root search (aspiration optional)

```python
def _search_root(board, depth, deadline):
    alpha, beta = -INF, INF
    best_move, best_score = None, -INF
    for move in _order_moves(board, depth, ply=0, tt_move=_tt_best(board)):
        board.push(move)
        score = -_search(board, depth - 1, -beta, -alpha, ply=1, deadline=deadline)
        board.pop()
        if score > best_score:
            best_score, best_move = score, move
        alpha = max(alpha, score)
    # store for next iteration's ordering + next move's TT
    _tt_store(board, depth, best_score, EXACT, best_move)
    return best_score, best_move
```

Aspiration windows (search a narrow window around the previous score, re-search wider on fail) are
a later optimisation — only add once the plain version is measured and stable.

### 4.4 Negamax with fail-soft alpha-beta (`_search`)

```python
def _search(board, depth, alpha, beta, ply, deadline):
    _check_time(deadline)                       # raises _TimeUp every N nodes (§4.6)

    alpha_orig = alpha

    # --- Transposition probe ---
    tt = _tt_probe(board)
    if tt is not None and tt.depth >= depth:
        v = _from_tt_score(tt.score, ply)
        if   tt.flag == EXACT:        return v
        elif tt.flag == LOWERBOUND:   alpha = max(alpha, v)
        elif tt.flag == UPPERBOUND:   beta  = min(beta, v)
        if alpha >= beta:             return v

    # --- Terminal / draw detection ---
    if board.is_checkmate():          return -(MATE - ply)     # side to move is mated
    if board.is_stalemate() or board.is_insufficient_material() \
       or board.is_repetition(2) or board.can_claim_fifty_moves():
        return 0
    # (Use a 2-fold check inside search so we treat imminent repetition as a draw; the
    #  referee claims 3-fold externally. Tune which fold count to search-draw on.)

    # --- Leaf: drop into quiescence, never evaluate a noisy position ---
    if depth <= 0:
        return _quiesce(board, alpha, beta, ply, deadline)

    # --- (Optional) null-move pruning: skip a move; if we're still winning, prune. ---
    #   Guard with: not in check, non-pawn material present, beta not a mate score.
    #   Reduced-depth search of the opponent; if >= beta, return beta. Verify carefully;
    #   this is where zugzwang bugs hide. Add only after the plain search is solid.

    best_score = -INF
    best_move = None
    tt_move = tt.move if tt is not None else None

    moved = False
    for move in _order_moves(board, depth, ply, tt_move):
        moved = True
        board.push(move)
        # (Optional) Late Move Reductions: search late, quiet moves at reduced depth,
        #   re-search full depth if they surprise us by beating alpha.
        score = -_search(board, depth - 1, -beta, -alpha, ply + 1, deadline)
        board.pop()

        if score > best_score:
            best_score, best_move = score, move
        if score > alpha:
            alpha = score
        if alpha >= beta:                       # beta cutoff (fail-high)
            _register_killer(move, ply)         # move ordering feedback (§4.8)
            _bump_history(move, depth)
            break

    if not moved:                               # no legal moves = mate or stalemate
        return -(MATE - ply) if board.is_check() else 0

    # --- Store result with the right bound flag ---
    flag = (UPPERBOUND if best_score <= alpha_orig       # never rose above original alpha
            else LOWERBOUND if best_score >= beta        # caused a cutoff
            else EXACT)
    _tt_store(board, depth, _to_tt_score(best_score, ply), flag, best_move)
    return best_score
```

**Fail-soft** means we return `best_score` (which may be outside `[alpha, beta]`), giving the TT
tighter bounds. The `alpha_orig` comparison is what makes the UPPER/LOWER/EXACT classification
correct — get this wrong and the TT poisons the search.

### 4.5 Quiescence search (`_quiesce`)

At depth 0 we do **not** evaluate directly; a position mid-capture is misread. Quiescence extends
only forcing moves (captures, and optionally checks) until the position is quiet.

```python
def _quiesce(board, alpha, beta, ply, deadline):
    _check_time(deadline)

    if board.is_checkmate():   return -(MATE - ply)
    if board.is_stalemate() or board.is_insufficient_material():  return 0

    stand_pat = evaluate(board)             # side-to-move POV, centipawns
    if stand_pat >= beta:                   # already too good; opponent won't allow this
        return stand_pat                    # fail-soft
    if stand_pat > alpha:
        alpha = stand_pat

    # Only captures (and promotions); order by MVV-LVA. Optionally include checks near the top.
    for move in _order_captures(board):
        # Delta pruning: if even capturing the target + a margin can't raise alpha, skip.
        if stand_pat + _capture_gain(board, move) + DELTA_MARGIN < alpha and not move.promotion:
            continue
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, ply + 1, deadline)
        board.pop()
        if score >= beta:   return score
        if score > alpha:   alpha = score
    return alpha
```

When in check, quiescence should search **all** evasions (not just captures) or it will hallucinate
safety — handle the in-check case by falling back to a full move list at that node.

### 4.6 Time management & the abort contract

This is the part that most often loses games. Two independent guards:

1. **Per-move budget** (`_compute_deadline`): spend from the clock we were handed.
   ```python
   def _compute_deadline(board, time_left_ms):
       moves_left = _estimate_moves_left(board)     # e.g. max(20, 60 - fullmove_number)
       # Reserve a safety margin: the referee measures wall time and adds only 500 ms grace.
       budget_ms = min(time_left_ms * 0.5,          # never spend >half the remaining clock
                       time_left_ms / moves_left + INCREMENT_MS * 0.8)
       budget_ms = max(budget_ms, MIN_MOVE_MS)      # always think a little
       safety_ms = 150                              # leave the wire/serialisation margin
       return time.monotonic() + (budget_ms - safety_ms) / 1000.0
   ```
   Tune `moves_left` and the fractions empirically in the arena. When very low on time, fall back
   to a depth-1 or ordering-only move instantly.

2. **In-search deadline check** (`_check_time`): budgeting is not enough — one deep iteration can
   overshoot. Check the wall clock **inside** the node loop, not only between depths:
   ```python
   _nodes = 0
   def _check_time(deadline):
       global _nodes
       _nodes += 1
       if _nodes & 2047 == 0 and time.monotonic() >= deadline:
           raise _TimeUp
   ```
   `_TimeUp` unwinds to `get_move`, which returns the **last completed depth's** best move.

**Abort discipline (critical):** a partially-searched depth is *untrustworthy* and must be
discarded — except the root's first move, which is the previous depth's best and is therefore safe
to keep if it already completed. Simplest correct rule: only accept a depth's result if
`_search_root` returned normally for that depth. That is what the `try/except _TimeUp` around the
ID loop guarantees.

### 4.7 Draw / repetition awareness across moves

The process persists within a game, so we keep the positions we've been asked about. Two uses:
- **When winning**, avoid walking into a threefold the referee would claim — vary the shuffle.
- **When losing or equal**, steer toward repetition/50-move to salvage a draw.

```python
_seen_positions: list[int] = []   # module-level; zobrist keys of positions we've moved from
```
Inside eval or move selection, penalise/reward moves that head into a position already in
`_seen_positions` according to the sign of the current evaluation.

### 4.8 Move ordering (`_order_moves`) — the strength multiplier

Order at each node, best-guess first, so alpha-beta prunes early:

1. **TT move** (the best move stored for this position) — first, always.
2. **Winning/equal captures** by **MVV-LVA** (Most Valuable Victim − Least Valuable Attacker):
   `score = 10 * value[victim] - value[attacker]`. Use `board.is_capture`, `piece_type_at`, and
   handle en passant (`board.is_en_passant`).
3. **Promotions** (queen promotions high).
4. **Killer moves**: two quiet moves per ply that caused a beta cutoff at the same ply — try them
   before other quiets.
5. **History heuristic**: quiet moves scored by `history[piece][to_square]`, incremented by
   `depth*depth` on cutoffs. Orders the remaining quiets.
6. Everything else.

Killers and history are module-level tables reset (or aged) each `get_move`. Cheap, and they turn
the search from "searches everything" into "searches the right 3–4 moves first," which is most of
the effective depth in Python.

---

## 5. Evaluation

`evaluate(board) -> int` returns centipawns from the **side-to-move** perspective. This is the seam
the net later plugs into unchanged.

### 5.1 Phase A — handcrafted tapered eval (ships first, is competitive)

- **Material**: pawn 100, knight 320, bishop 330, rook 500, queen 900.
- **Piece-square tables**: per-piece 64-entry tables, mirrored for black. Separate **middlegame**
  and **endgame** tables, blended by a **phase** scalar (count remaining non-pawn material; 1.0 =
  full board, 0.0 = bare kings). This "tapered eval" makes the king march to the centre in endgames
  and stay tucked in the middlegame — a large, cheap gain.
  ```python
  eval = (mg_score * phase + eg_score * (1 - phase))
  ```
- **Cheap positional terms**, each measured before keeping:
  - Bishop pair bonus.
  - Doubled/isolated/passed pawns.
  - Rook on open/semi-open file.
  - King safety: pawn shield in the middlegame; king activity in the endgame (folded into PST via
    the taper).
  - Mobility (legal-move count difference) — but it's expensive (needs move-gen for both sides);
    include only if it pays for its cost in the arena.
- **Precompute** PST lookups and any per-piece tables at import time. Keep `evaluate` allocation-free
  and branch-light; it runs at every leaf and dominates runtime.

### 5.2 Phase B — learned NNUE-style eval (optional upside, behind the same seam)

Only pursue if Phase A is solid and we have arena evidence a net beats it at equal wall-time.

- **Architecture**: small, HalfKP-style or a plain MLP over a sparse board-feature vector
  (piece-on-square one-hot, ~768 inputs) → small hidden layers → scalar centipawn. Keep it tiny so
  we can evaluate **thousands** of leaves/move on one core; a great net you can run 50×/move loses
  to a fast net you run 3000×.
- **Incremental accumulator** (the "NN" in NNUE): maintain the first-layer sum incrementally on
  push/pop instead of recomputing — this is what makes it affordable. `numba` can JIT this numpy
  hot loop.
- **Runtime**: export to **ONNX**, run with **onnxruntime** (faster startup than torch, competitive
  on one core). `sess.set_...` single-threaded. **Batch** the leaves of a search pass into one
  inference call rather than one-at-a-time.
- **Size**: must fit the 50 MB unzipped cap with everything else — a few MB of weights is plenty.
- **Legality**: weights must be **self-trained**. Training data may be engine-annotated (allowed);
  the ban is only on shipping a third-party engine.

### 5.3 The seam

```python
class Evaluator(Protocol):
    def __call__(self, board: chess.Board) -> int: ...

evaluate: Evaluator = handcrafted_eval    # swap to nnue_eval once it wins in the arena
```
Search never knows which eval it called. We A/B them by flipping one binding and running the arena.

---

## 6. Opening / early game

Conventional opening books are **near-worthless here** — start positions are curated and near-level,
not the standard start, so a startpos-keyed book rarely matches. Instead:
- Rely on search + eval from move one. The tapered eval already avoids the gross early blunders a
  book is meant to prevent.
- Optionally, a **tiny** hand-curated table of principled responses to *common* curated openings, if
  we discover the platform reuses a known set (unknown today — verify by observing games).
- Bank clock early: if the position is quiet and the eval is confident, don't over-think move 1.

---

## 7. Performance engineering (one core, Python)

- **Prune, don't grind.** Every 10% better ordering is worth more than micro-optimising a node.
- Minimise `push`/`pop` and `legal_moves` calls; reuse generated lists.
- Cache `evaluate` results in the TT alongside search scores when useful.
- `numba` only where it sees numpy/scalars — realistically the NNUE accumulator, not move-gen.
- Keep `evaluate` and `_order_moves` allocation-free; they are the hot path.
- Never spawn threads during move calculation (single-threaded is fastest on one core). Pondering on
  a background thread during the opponent's turn is *allowed* and a possible late-stage lever, but it
  complicates state and time accounting — defer.

---

## 8. Training pipeline (only if we build the net)

Runs on our machines, before upload; nothing trains at runtime.

1. **Data**: public game databases (Lichess dumps) and/or self-play against our earlier versions.
   Positions may be labelled by an existing engine's evaluation (allowed — the ban is on *shipping*
   an engine). Store `(fen, target_cp or game_result)`.
2. **Target**: either engine eval (regression) or game outcome (win/draw/loss classification), or a
   blend. Regression to centipawns is simplest to slot behind the seam.
3. **Model**: small MLP/NNUE in PyTorch. Train, validate on held-out positions, watch for
   overfitting to the labeller.
4. **Export**: PyTorch → ONNX (opset compatible with the installed `onnxruntime`). Verify parity
   between torch and onnxruntime outputs on a fixed test set.
5. **Ship**: `weights/model.onnx` in the zip (+ update `package.py` includes). Confirm total < 50 MB.
6. **Prove it**: it only ships if it beats Phase A in a few-hundred-game arena at real time control.

---

## 9. Testing & measurement methodology

- **Correctness first.** Play a few hundred games vs `baselines/random` to smoke out crashes on rare
  paths: no legal moves, promotions, en passant, checkmate detection, insufficient material.
- **`make gate`** must stay green: ruff, mypy strict, and two clean games. CI runs it on every push.
- **Measure with the arena, not vibes.** Two games tell you nothing; a 3% change needs hundreds of
  games. Always **alternate colours**, **fix the opponent**, and keep the **previous version** around
  as an opponent — "better than my last one" is the only comparison that counts.
  ```
  uv run python -m harness.arena --opponent baselines/minimax --games 300
  uv run python -m harness.arena --opponent ../prev-version   --games 400
  ```
- **Never flag in testing = never flag in production.** Run some games at the *real* 120 s + 0.5 s
  control (`make play`) to validate time management, not just the fast arena control.
- Keep a scoreboard in this file (below) of each version vs the previous and vs baselines.

---

## 10. Packaging & submission

- `make zip` builds `submission.zip` with `agent.py` at the root (+ `weights/` if present). The
  packager currently lists `requirements.txt` in its includes — harmless (the platform ignores it),
  but do not depend on it. If we ship weights, confirm they're included and the unzip is < 50 MB.
- **Never** import anything outside `torch/numpy/python-chess/onnxruntime/numba` — it crashes the
  agent on the platform even though it runs locally.
- No native binaries in the zip. No obfuscation — a judge reads the source.
- Validate on upload; the dashboard validation log is the authority, not the local harness. Upload
  early to see a real validation log well before the Sep 11 lock.

---

## 11. Milestones (target: locked build by Sep 11)

Each milestone is measurable and must beat the previous version in the arena before it lands.

- **M0 — Scaffold & safety (day 1).** Replace random-mover with negamax α-β + iterative deepening +
  in-search clock check + robust time budget. Material-only eval. Goal: **never flags, never
  crashes**, beats `greedy` decisively. This is the whole safety spine.
- **M1 — Ordering + quiescence + TT (days 2–3).** MVV-LVA, killers, history, TT keyed on zobrist
  surviving across moves, quiescence at leaves. Goal: **beats `minimax`** convincingly and the
  effective depth jumps.
- **M2 — Tapered handcrafted eval (days 3–5).** Middlegame/endgame PST + phase blend + a few proven
  positional terms. Goal: clear arena win vs M1. **This is a strong, complete, legal entry.**
- **M3 — Hardening (days 5–6).** Edge-case fuzzing vs random over hundreds of games, real-time-control
  games, tune time management margins, aspiration windows / null-move / LMR *only if* each proves out
  in the arena. Ship & watch the validation log.
- **M4 — (Stretch) NNUE eval (days 6–10).** Train, export to ONNX, batch leaves, A/B vs M2. Ship only
  if it wins at equal wall-time. If it doesn't beat M2 in time, **M2 is the final submission** — a
  classical entry is fully legal.
- **Lock (Sep 11).** Freeze the best-measured version. Keep the runner-up as a fallback upload.

---

## 12. Risks & footguns

- **Flagging** — the #1 self-inflicted loss. Mitigated by the two-guard time model (§4.6) and the
  abort-and-return-last-completed-depth discipline.
- **TT bound bugs** — wrong UPPER/LOWER/EXACT flags or unadjusted mate scores silently corrupt the
  search. Unit-test the flag logic; verify mate scores prefer shorter mates.
- **Quiescence explosions / in-check handling** — searching all evasions when in check; delta pruning
  to bound capture chains.
- **Importing a non-preinstalled package** — runs locally, crashes on platform. Keep imports to the
  five allowed.
- **Shadowing stdlib/`chess`** — never name a file after a module we import.
- **Size cap (50 MB)** — check unzipped size every time we add weights.
- **Stale assumptions** — the live docs change; re-fetch `agent-contract.md` and `rules.md` before
  each upload and update §0.
- **Opening book over-investment** — start positions are curated; don't sink time into a startpos book.

---

## Scoreboard (fill in as we go)

| Version | vs prev | vs random | vs greedy | vs minimax | Notes |
|---|---|---|---|---|---|
| random (start) | — | — | 10% | ~0% | shipped default |
| M0–M2 classical | — | 97.5% (+19 =1 −0) | 100% (+20 =0 −0) | **100% (+200 =0 −0)** | minimax: 200 games @ 3 s+0.5 s (real increment), 100 as each colour, all checkmates, no flags. random/greedy: 20 games @ 3 s+0.1 s. |
