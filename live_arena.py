"""Local, throwaway arena with a live scoreboard and an optional live board. NOT for committing.

Two modes, both leaving the harness untouched:

* default (a benchmark): wraps the real `play_match` (same referee, clock, sandbox, so the score is
  honest), showing a progress bar, running +W =D -L, a per-game elapsed ticker and a result line.
* --watch (a spectator): drives the game itself so it can draw the board move-by-move as the engines
  compute, on a single full-screen board with box-drawing and colour-coded pieces. It reuses the
  harness's own decision/clock functions, so behaviour matches the referee; use it to watch a few
  games, not to produce the authoritative score. Best with a small --games.

Every game is always appended to --log as PGN, which opens in any chess GUI afterwards.

    uv run python live_arena.py --opponent baselines/our-old-agent --games 100        # scoreboard
    uv run python live_arena.py --opponent baselines/our-old-agent --games 4 --watch  # live board
"""

from __future__ import annotations

import argparse
import io
import itertools
import threading
import time
from pathlib import Path

import chess
import chess.pgn

from harness.referee import (
    FAILED_TERMINATIONS,
    Outcome,
    _adjudicate,
    _decide,
    _legal_move,
    _opponent_wins,
    _outcome,
    _start,
    play_match,
)
from harness.rules import PLY_CAP
from harness.sandbox import AgentFailure, local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100
SPINNER = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

# --- board rendering (used by --watch) --------------------------------------------------------

BLUE = "\033[1;94m"  # White's pieces
RED = "\033[1;91m"  # Black's pieces
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[3J\033[H"  # clear screen + scrollback, cursor home
FIGURINE = {"P": "♟", "N": "♞", "B": "♝", "R": "♜", "Q": "♛", "K": "♚"}


def _cell(board: chess.Board, square: int) -> str:
    piece = board.piece_at(square)
    if piece is None:
        return " "
    colour = BLUE if piece.color == chess.WHITE else RED
    return f"{colour}{FIGURINE[piece.symbol().upper()]}{RESET}"


def _render_board(board: chess.Board, orientation: bool) -> str:
    files = list(range(8) if orientation == chess.WHITE else range(7, -1, -1))
    ranks = list(range(7, -1, -1) if orientation == chess.WHITE else range(8))
    top = "   ┌" + "┬".join(["───"] * 8) + "┐"
    mid = "   ├" + "┼".join(["───"] * 8) + "┤"
    bottom = "   └" + "┴".join(["───"] * 8) + "┘"
    lines = [top]
    for index, rank in enumerate(ranks):
        cells = "│".join(f" {_cell(board, chess.square(f, rank))} " for f in files)
        lines.append(f" {rank + 1} │{cells}│")
        if index != 7:
            lines.append(mid)
    lines.append(bottom)
    lines.append("     " + "   ".join(chr(ord('a') + f) for f in files))
    return "\n".join(lines)


def _frame(header: list[str], board: chess.Board, orientation: bool, status: str) -> None:
    body = "\n".join([*header, "", _render_board(board, orientation), "", status])
    print(CLEAR + body, flush=True)


# --- spectator: drive a game and draw it live -------------------------------------------------


def _watch_game(
    white_path: Path,
    black_path: Path,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
    plays_white: bool,
    header: list[str],
    frame_pause: float,
) -> Outcome:
    """A faithful copy of harness.referee._play that draws the board after every move.

    Decision/clock/adjudication all come from the harness's own functions, so results match the
    referee; only the loop is inlined here to render each ply.
    """
    white, black = local(white_path), local(black_path)
    orientation = plays_white
    try:
        white_failure, black_failure = _start(white), _start(black)
        if white_failure is not None and black_failure is not None:
            return _outcome(chess.Board(), "void", "both_failed")
        if white_failure is not None:
            return _outcome(chess.Board(), "black", white_failure)
        if black_failure is not None:
            return _outcome(chess.Board(), "white", black_failure)

        board = chess.Board()
        agents = {chess.WHITE: white, chess.BLACK: black}
        clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
        _frame(header + _clocks(clock), board, orientation, "starting…")

        while True:
            finish = board.outcome(claim_draw=True)
            if finish is not None:
                return _outcome(board, _decide(finish), finish.termination.name.lower())
            if len(board.move_stack) >= ply_cap:
                return _outcome(board, _adjudicate(board), "adjudication")

            mover = board.turn
            name = "White" if mover == chess.WHITE else "Black"
            _frame(header + _clocks(clock), board, orientation, f"{name} to move — thinking…")
            started = time.monotonic()
            try:
                uci = agents[mover].move(board.fen(), int(clock[mover]))
            except AgentFailure as failure:
                return _outcome(board, _opponent_wins(mover), failure.reason)
            clock[mover] -= (time.monotonic() - started) * 1000.0
            if clock[mover] < 0:
                return _outcome(board, _opponent_wins(mover), "flag")

            move = _legal_move(board, uci)
            if move is None:
                return _outcome(board, _opponent_wins(mover), "illegal")
            san = board.san(move)
            board.push(move)
            clock[mover] += increment_ms
            _frame(header + _clocks(clock), board, orientation, f"{name} played {san}")
            time.sleep(frame_pause)
    finally:
        white.stop()
        black.stop()


def _clocks(clock: dict[bool, float]) -> list[str]:
    return [
        f"  {BLUE}White{RESET} {clock[chess.WHITE] / 1000:6.1f}s"
        f"    {RED}Black{RESET} {clock[chess.BLACK] / 1000:6.1f}s"
    ]


# --- scoreboard (default, honest benchmark via play_match) ------------------------------------


def ply_count(pgn: str) -> int:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return len(list(game.mainline_moves())) if game is not None else 0


def _bar(done: int, total: int, width: int = 24) -> str:
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _log_game(path: Path, number: int, outcome: Outcome, plays_white: bool) -> None:
    """Append the game's real PGN to the log; openable in any chess GUI (e.g. lichess import)."""
    seat = "white=us black=opp" if plays_white else "white=opp black=us"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"; game {number}  {seat}  result={outcome.result} "
            f"termination={outcome.termination}\n"
        )
        handle.write(outcome.pgn.strip() + "\n\n")


def _play_threaded(
    args: argparse.Namespace, white: Path, black: Path
) -> tuple[dict[str, Outcome], threading.Thread]:
    """Set up one honest game via play_match in a thread so the caller can animate a spinner."""
    box: dict[str, Outcome] = {}
    worker = threading.Thread(
        target=lambda: box.__setitem__(
            "o",
            play_match(local(white), local(black), args.base_ms, args.increment_ms,
                       ply_cap=args.ply_cap),
        )
    )
    return box, worker


def run() -> None:
    parser = argparse.ArgumentParser(description="Arena with a live scoreboard/board (local only).")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/our-old-agent"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--watch", action="store_true", help="Draw each game live, move by move.")
    parser.add_argument("--log", type=Path, default=Path("arena_games.pgn"))
    parser.add_argument("--frame-pause", type=float, default=0.25, help="Seconds per move (watch).")
    args = parser.parse_args()

    agent, opponent = args.agent.resolve(), args.opponent.resolve()
    us, them = agent.name or "agent", opponent.name
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    started = time.monotonic()

    if not args.watch:
        print(f"{us}  vs  {them}   ({args.games} games)\n")

    for game in range(args.games):
        plays_white = game % 2 == 0
        white, black = (agent, opponent) if plays_white else (opponent, agent)
        score = (wins + draws / 2) / game if game else 0.0

        if args.watch:
            seat = f"WHITE {BLUE}(blue){RESET}" if plays_white else f"BLACK {RED}(red){RESET}"
            header = [
                f"  {us}  vs  {them}",
                f"  game {game + 1}/{args.games}   we play {seat}"
                f"   running +{wins} ={draws} -{losses} ({score:.0%})",
            ]
            outcome = _watch_game(white, black, args.base_ms, args.increment_ms, args.ply_cap,
                                  plays_white, header, args.frame_pause)
        else:
            box, worker = _play_threaded(args, white, black)
            game_start = time.monotonic()
            worker.start()
            while worker.is_alive():
                elapsed = time.monotonic() - game_start
                side = "white" if plays_white else "black"
                print(
                    f"\r{next(SPINNER)} {_bar(game, args.games)} game {game + 1}/{args.games} "
                    f"(we play {side}, {elapsed:4.0f}s)  running +{wins} ={draws} -{losses} "
                    f"({score:.0%})   ",
                    end="",
                    flush=True,
                )
                time.sleep(0.2)
            worker.join()
            outcome = box["o"]

        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result in ("draw", "void"):
            draws += 1
            verdict = "draw"
        elif (outcome.result == "white") == plays_white:
            wins += 1
            verdict = "WIN "
        else:
            losses += 1
            verdict = "loss"

        flag = "  <-- FAILED" if outcome.termination in FAILED_TERMINATIONS else ""
        line = (
            f"game {game + 1:>3}/{args.games}  [{'W' if plays_white else 'B'}]  {verdict}  "
            f"{ply_count(outcome.pgn):>3} plies  by {outcome.termination}{flag}"
        )
        if args.watch:
            print(f"\n  {line}", flush=True)
            time.sleep(1.5)  # let the final position sit before the next game clears the screen
        else:
            print("\r" + line + " " * 20)
        if args.log:
            _log_game(args.log, game + 1, outcome, plays_white)

    played = wins + draws + losses
    final = (wins + draws / 2) / played if played else 0.0
    mins = (time.monotonic() - started) / 60
    print(f"\n{'=' * 44}")
    print(f"  +{wins} ={draws} -{losses}   score {final:.1%}   ({mins:.1f} min)")
    print("  terminations: " + ", ".join(f"{k} {v}" for k, v in terminations.items()))
    broken = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    if broken:
        print("  !! FAILED TERMINATIONS: " + ", ".join(f"{k} {v}" for k, v in broken.items()))
    print("=" * 44)


if __name__ == "__main__":
    run()
