# Plan: classical engine + ML opening selection

**Decision (Panth + Tristan):** ship the proven **classical alpha-beta engine** as the base for
the whole game, and use **ML only for opening move selection** — picking our move in the first few
plies based on our and the opponent's moves so far. No NNUE eval in the search.

Status: planning only. No branch cut yet.

---

## 1. Why this split

- The NNUE eval currently **loses 20-0** to the classical engine at fast time control (ONNX call per
  node → far shallower search). The classical engine is the strong, reliable workhorse.
- Openings are a *narrow, well-bounded* problem: a small model over the first ~8-12 moves, run a
  handful of times per game, near-instant. Low risk, banks clock time for the middlegame.
- **Rules-clean.** The contract says a *"database of engine moves or evaluations shipped for lookup
  at runtime is an engine, not training data"* — but *"training on positions an existing engine
  labelled is allowed"*, and a trained model is weights, not a lookup DB. So we train a small policy
  net on **human games** and ship the weights. (Static opening books are also explicitly permitted
  within the 50 MB cap; the net is the version Panth wants and it generalizes better — see §7.)

## 2. Base engine

- New branch off **`main`** — it is already the classical alpha-beta engine (iterative-deepening
  negamax, TT, MVV-LVA / killer / history ordering, quiescence, tapered material + PST). No revert
  needed; the NNUE only exists on `ml-eval`.
- Eval stays in `evaluate()`. The opening layer lives in `get_move()`, *before* the search, and does
  not touch eval.

## 3. The ML opening component

**Recommended: a small position policy net.**
- **Input:** board encoding at the current opening position — 768-dim piece-square planes + side to
  move + castling rights. (Reuse the board-encoding code from the NNUE work.)
- **Output:** distribution over moves (from-square x to-square = 4096, plus promotions), **masked to
  legal moves**.
- **Use:** argmax, or sample with a low temperature for variety across games.
- "Based on ours and the opponent's first few moves" is captured implicitly — the position *is* the
  result of those moves, and keying on position handles transpositions for free.

**Alternative considered:** a sequence model over the UCI move list (small GRU/transformer). More
directly "move-sequence aware," but needs tokenization + legality plumbing and buys little over the
position net. Keep as a fallback, not the first build.

## 4. Training data (human games only — provenance matters)

- Source: **Lichess open database** PGN (or a master-games DB). Filter to strong games (rating
  threshold) and standard time controls.
- Extract `(position, move_played, result)` for plies `< N` (start N ~= 16 plies / move 8).
- **Weight targets by outcome** (win for the side to move) and/or player rating, so the net learns
  moves that *score* well, not merely popular ones.
- Aggregate per position → a human move distribution, which we distill into the net. Document that
  it is derived purely from human games (no engine labels), so it is defensible to a judge.

## 5. Pipeline (in `training/`)

Two interchangeable P1 data sources, both emitting the same schema
(`{epd: [{uci, count, score}]}`, `score` = summed result for the side to move):

1a. `training/build_opening_data.py` — parse a Lichess monthly PGN dump. Streams from stdin
    (`--pgn -`) so a `zstd -dc … |` pipe never writes the ~200 GB decompressed file to disk.
1b. `training/crawl_explorer.py` — **primary source (chosen for bandwidth).** Walk the opening
    tree via the Lichess Opening Explorer API; a few MB of traffic instead of a 29 GB download.
    Result-weighted and rating-bucketed. `--self-test` checks the logic offline.

2. `training/train_opening.py` — train the policy net: a small MLP over a 773-dim position
   encoding, cross-entropy over the *full* move space (from*64+to) to the human distribution, so
   inference can argmax over legal moves. Consumes either source unchanged.
3. Save weights as `model/opening.npz` (not ONNX): the agent runs inference with a few numpy
   matmuls, so its only runtime dep is `numpy` (no `onnxruntime`), and it stays judge-readable.
   ~5 MB, well under the 50 MB cap. `encode()` is shared verbatim between trainer and agent.

## 6. Integration in `agent.py`

At import (inside the 60 s budget): build classical tables **and** load the opening net.

In `get_move`:
1. **In-opening gate:** `plies_played < N` (e.g. N ~= 12-16). Otherwise → classical search.
2. Run the policy net on the current position → distribution → **mask to legal moves** → pick top
   (or low-temp sample).
3. **Guards before returning a book move:**
   - **Confidence threshold** — if the net's top move is low-probability (out-of-distribution
     position), fall through to the search instead.
   - **Tactical veto** — a cheap 1-2 ply check so we never play an opening move that hangs material
     outright (protects against offbeat opponent openings leading the net astray).
   - **Legality** — always verify the move is legal in the actual position; an illegal move is an
     instant loss.
4. Else → existing classical alpha-beta search.

Payoff: opening moves are near-instant → **clock banked** for the middlegame (real Elo, especially
vs the old agent that burns time from move 1).

## 7. Why a net over a static book

Opponents are other people's bots and may play offbeat lines (`1.h4`, weird move orders). A curated
main-line book *whiffs* the moment it is out of book; a policy net **degrades gracefully** on unseen
positions and covers transpositions. That generalization is the main argument for ML here.

## 8. Evaluation

- Arena harness: `classical + opening` vs `classical alone`, vs `baselines/our-old-agent`, vs
  `greedy` / `minimax`. Track score **and average time banked**.
- TUI (`play_against_model.py`) to eyeball opening choices by hand.

## 9. Phasing

- **P0** — branch off `main`; confirm classical engine beats baselines (sanity).
- **P1** — opening data pipeline from a PGN corpus.
- **P2** — train policy net, export ONNX.
- **P3** — integrate behind the ply-gate + confidence + tactical + legality guards.
- **P4** — arena eval; tune `N`, confidence threshold, sampling temperature.
- **P5** — package, check < 50 MB, `make gate`, submit.

## 10. Open questions

- `N` (how deep the net drives) and the confidence threshold — tune empirically in P4.
- Training-data volume and training time on available hardware.
- Divide of labour: Panth trains the model; opening data pipeline + `agent.py` integration can
  proceed in parallel against a stub net.
