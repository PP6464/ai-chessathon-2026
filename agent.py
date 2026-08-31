"""The submission entrypoint. The platform imports this file and calls get_move.

A classical engine: iterative-deepening negamax with alpha-beta, a transposition table,
MVV-LVA / killer / history move ordering, quiescence at the leaves, and a tapered
material + piece-square evaluation. No learned model is required by the rules; this is a
complete entry. The evaluation is isolated in `evaluate` so a trained net can replace it later.

Import time runs once per game inside a 60 second budget, before the clock starts. The tables
below are built here, not inside get_move. Module state (the transposition table, killer and
history tables) survives between our moves within a game, never to the next game.
"""

import time
from collections.abc import Hashable

import chess

# --- Scores -----------------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
BISHOP_PAIR = 30
MATE = 1_000_000
MATE_IN_MAX = MATE - 1000  # scores beyond this encode "mate in N"; the gap holds the distance
INF = 2_000_000
MAX_PLY = 64

# Transposition-table entry flags.
EXACT, LOWER, UPPER = 0, 1, 2

# --- Time management --------------------------------------------------------------------------

INCREMENT_MS = 500.0
SAFETY_MS = 120.0  # wall-time margin left for serialisation; the referee grants only 500 ms grace
CLOCK_CHECK_MASK = 2047  # test the wall clock once every this-many-plus-one nodes

# --- Piece-square tables ----------------------------------------------------------------------
# Michniewski's "Simplified Evaluation" tables, written rank 8 first from White's view. Only the
# king differs between phases; the rest reuse one table for both. They are re-indexed by square
# and colour at import so `evaluate` is a flat lookup.

_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]  # fmt: skip
_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]  # fmt: skip
_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]  # fmt: skip
_ROOK = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]  # fmt: skip
_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]  # fmt: skip
_KING_MG = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]  # fmt: skip
_KING_EG = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]  # fmt: skip

_RAW_MG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _BISHOP,
    chess.ROOK: _ROOK,
    chess.QUEEN: _QUEEN,
    chess.KING: _KING_MG,
}
_RAW_EG: dict[chess.PieceType, list[int]] = {**_RAW_MG, chess.KING: _KING_EG}

# Phase weights: 24 at the opening, 0 with only kings and pawns. Used to blend mid/endgame tables.
PHASE_WEIGHT: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
PHASE_TOTAL = 24


def _by_colour(raw: list[int]) -> dict[chess.Color, list[int]]:
    """A rank-8-first White table, re-indexed by square for each colour."""
    return {
        chess.WHITE: [raw[chess.square_mirror(sq)] for sq in range(64)],
        chess.BLACK: [raw[sq] for sq in range(64)],
    }


MG_PST: dict[chess.PieceType, dict[chess.Color, list[int]]] = {
    pt: _by_colour(raw) for pt, raw in _RAW_MG.items()
}
EG_PST: dict[chess.PieceType, dict[chess.Color, list[int]]] = {
    pt: _by_colour(raw) for pt, raw in _RAW_EG.items()
}

# --- Evaluation -------------------------------------------------------------------------------


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns, from the side-to-move's point of view.

    Tapered material + piece-square tables plus a bishop-pair bonus. This is the one seam a
    trained network would replace; search never depends on how the number is produced.
    """
    mg = 0
    eg = 0
    phase = 0
    white_bishops = 0
    black_bishops = 0
    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        value = PIECE_VALUE[pt]
        if piece.color:  # White
            mg += value + MG_PST[pt][chess.WHITE][sq]
            eg += value + EG_PST[pt][chess.WHITE][sq]
            white_bishops += pt == chess.BISHOP
        else:
            mg -= value + MG_PST[pt][chess.BLACK][sq]
            eg -= value + EG_PST[pt][chess.BLACK][sq]
            black_bishops += pt == chess.BISHOP
        phase += PHASE_WEIGHT[pt]

    if white_bishops >= 2:
        mg += BISHOP_PAIR
        eg += BISHOP_PAIR
    if black_bishops >= 2:
        mg -= BISHOP_PAIR
        eg -= BISHOP_PAIR

    phase = min(phase, PHASE_TOTAL)
    score = (mg * phase + eg * (PHASE_TOTAL - phase)) // PHASE_TOTAL  # White's perspective
    return score if board.turn else -score


# --- Search state (module-level, survives across our moves within a game) ---------------------

# key (board._transposition_key()) -> (depth, flag, score, move)
_tt: dict[Hashable, tuple[int, int, int, chess.Move | None]] = {}
_killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
_history: dict[tuple[chess.Color, chess.Square, chess.Square], int] = {}
_nodes = 0
_deadline = 0.0


class _TimeUp(Exception):
    """Raised inside the search when the per-move deadline passes; unwinds to get_move."""


def _check_time() -> None:
    global _nodes
    _nodes += 1
    if _nodes & CLOCK_CHECK_MASK == 0 and time.monotonic() >= _deadline:
        raise _TimeUp


def _to_tt(score: int, ply: int) -> int:
    """Store mate scores relative to the root, not the current node."""
    if score > MATE_IN_MAX:
        return score + ply
    if score < -MATE_IN_MAX:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score > MATE_IN_MAX:
        return score - ply
    if score < -MATE_IN_MAX:
        return score + ply
    return score


# --- Move ordering ----------------------------------------------------------------------------


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUE[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    return PIECE_VALUE[victim] if victim else 0


def _score_move(board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int) -> int:
    if tt_move is not None and move == tt_move:
        return 30_000_000
    if board.is_capture(move):
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker else 0
        return 20_000_000 + _victim_value(board, move) * 16 - attacker_value
    if move.promotion:
        return 19_000_000 + PIECE_VALUE.get(move.promotion, 0)
    if ply < len(_killers):
        if move == _killers[ply][0]:
            return 18_000_000
        if move == _killers[ply][1]:
            return 17_000_000
    return _history.get((board.turn, move.from_square, move.to_square), 0)


def _order(
    board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
) -> list[chess.Move]:
    return sorted(moves, key=lambda m: _score_move(board, m, tt_move, ply), reverse=True)


def _register_killer(move: chess.Move, ply: int) -> None:
    if ply < len(_killers) and _killers[ply][0] != move:
        _killers[ply][1] = _killers[ply][0]
        _killers[ply][0] = move


def _bump_history(board: chess.Board, move: chess.Move, depth: int) -> None:
    key = (board.turn, move.from_square, move.to_square)
    _history[key] = _history.get(key, 0) + depth * depth


# --- Search -----------------------------------------------------------------------------------


def _is_draw(board: chess.Board) -> bool:
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return True
    # A repetition needs at least four plies with no capture or pawn move to close the loop.
    return board.halfmove_clock >= 4 and board.is_repetition(2)


def _quiesce(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    """Search only forcing moves to a quiet position so evaluation is never read mid-exchange."""
    _check_time()
    if ply >= MAX_PLY:
        return evaluate(board)

    if board.is_check():
        # In check the side to move is not free to stand pat; search every evasion.
        best = -INF
        for move in _order(board, list(board.legal_moves), None, ply):
            board.push(move)
            score = -_quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                return best
        return best if best > -INF else -(MATE - ply)  # no evasion means checkmate

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    for move in _order(board, captures, None, ply):
        # Delta pruning: if even winning the target plus a margin can't reach alpha, skip it.
        if not move.promotion and stand_pat + _victim_value(board, move) + 200 < alpha:
            continue
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, ply + 1)
        board.pop()
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


def _search(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    """Fail-soft negamax with alpha-beta and a transposition table."""
    _check_time()

    if ply > 0 and _is_draw(board):
        return 0

    alpha_orig = alpha
    key = board._transposition_key()
    entry = _tt.get(key)
    tt_move: chess.Move | None = None
    if entry is not None:
        e_depth, e_flag, e_score, e_move = entry
        tt_move = e_move
        if e_depth >= depth:
            value = _from_tt(e_score, ply)
            if e_flag == EXACT:
                return value
            if e_flag == LOWER and value > alpha:
                alpha = value
            elif e_flag == UPPER and value < beta:
                beta = value
            if alpha >= beta:
                return value

    if depth <= 0:
        return _quiesce(board, alpha, beta, ply)

    moves = list(board.legal_moves)
    if not moves:
        return -(MATE - ply) if board.is_check() else 0

    best = -INF
    best_move: chess.Move | None = None
    for move in _order(board, moves, tt_move, ply):
        board.push(move)
        score = -_search(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:  # fail-high: this move is too good, opponent avoids the line
            if not board.is_capture(move) and not move.promotion:
                _register_killer(move, ply)
                _bump_history(board, move, depth)
            break

    flag = UPPER if best <= alpha_orig else LOWER if best >= beta else EXACT
    _tt[key] = (depth, flag, _to_tt(best, ply), best_move)
    return best


def _search_root(board: chess.Board, depth: int) -> tuple[int, chess.Move | None]:
    alpha, beta = -INF, INF
    best = -INF
    best_move: chess.Move | None = None
    key = board._transposition_key()
    entry = _tt.get(key)
    tt_move = entry[3] if entry is not None else None
    for move in _order(board, list(board.legal_moves), tt_move, 0):
        board.push(move)
        score = -_search(board, depth - 1, -beta, -alpha, 1)
        board.pop()
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
    _tt[key] = (depth, EXACT, _to_tt(best, 0), best_move)
    return best, best_move


# --- Time budget ------------------------------------------------------------------------------


def _budget_seconds(time_left_ms: int) -> float:
    """Seconds to spend on this move, from the clock we were handed. Never risks a flag."""
    budget_ms = time_left_ms / 25.0 + INCREMENT_MS * 0.7
    budget_ms = min(budget_ms, time_left_ms * 0.4)  # never stake more than 40% on one move
    return max(0.0, (budget_ms - SAFETY_MS) / 1000.0)


# --- Entry point ------------------------------------------------------------------------------


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    global _nodes, _deadline, _killers, _history

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"  # the referee has already ended the game; never actually reached
    if len(legal) == 1:
        return legal[0].uci()

    _deadline = time.monotonic() + _budget_seconds(time_left_ms)
    _nodes = 0
    _killers = [[None, None] for _ in range(MAX_PLY + 1)]
    _history = {}

    # Order the root once so the fallback is a sane move even if depth 1 never completes.
    best_move = _order(board, legal, None, 0)[0]
    try:
        for depth in range(1, MAX_PLY):
            score, move = _search_root(board, depth)
            if move is not None:
                best_move = move
            if abs(score) > MATE_IN_MAX:  # forced mate found; deeper search cannot improve on it
                break
            if time.monotonic() >= _deadline:  # no time to start another full iteration
                break
    except _TimeUp:
        pass  # ran out mid-iteration; keep the best move from the last completed depth

    return best_move.uci()
