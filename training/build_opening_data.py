"""Build an opening-move dataset from a human-game PGN corpus.

Stage P1 of the opening-selection plan (docs/OPENING_ML_PLAN.md). We read a PGN of human games,
keep the strong ones, and for the first few plies record which move was played from each position
and how the game turned out for the side to move. Positions are keyed by their EPD (piece
placement, side to move, castling, en passant), so different move orders that transpose into the
same position aggregate together.

The output maps each opening position to its human move distribution, weighted by result: a move
that is played often *and* scores well outranks a popular-but-losing move. A later stage distills
this into a small policy net; nothing here ships in the agent.

Only human games are used (no engine labels), which the contract permits as training data. A
lookup database is explicitly *not* allowed to ship at runtime, so this file stays in training/.

Lichess monthly dumps arrive as .pgn.zst; decompress first (`zstd -d file.pgn.zst`) and pass the
plain .pgn here:

    uv run python -m training.build_opening_data --pgn games.pgn --out data/opening_moves.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

# Openings only: the first this many plies (8 full moves) of each game.
DEFAULT_PLIES = 16
# Skip a game if either player is rated below this; keeps the moves strong.
DEFAULT_MIN_RATING = 2000
# Drop a position seen fewer than this many times across the corpus; prunes noise.
DEFAULT_MIN_COUNT = 1
# Time controls to skip: bullet openings are noisy and pre-move heavy.
SKIP_EVENT_SUBSTRINGS = ("bullet",)


@dataclass
class MoveStat:
    """How one move fared from one position: how often, and how well."""

    count: int = 0
    score: float = 0.0  # summed result for the side that played it (win=1, draw=0.5, loss=0)


def result_for_side(result: str, white_to_move: bool) -> float | None:
    """The game's result from the moving side's perspective, or None if unfinished."""
    if result == "1-0":
        white_score = 1.0
    elif result == "0-1":
        white_score = 0.0
    elif result == "1/2-1/2":
        white_score = 0.5
    else:
        return None
    return white_score if white_to_move else 1.0 - white_score


def elo(headers: chess.pgn.Headers, key: str) -> int | None:
    """Parse a rating header, tolerating '?' and missing values."""
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def game_is_eligible(headers: chess.pgn.Headers, min_rating: int) -> bool:
    """Keep only rated, non-bullet games where both players clear the rating floor."""
    event = headers.get("Event", "").lower()
    if any(tag in event for tag in SKIP_EVENT_SUBSTRINGS):
        return False
    white, black = elo(headers, "WhiteElo"), elo(headers, "BlackElo")
    if white is None or black is None:
        return False
    return white >= min_rating and black >= min_rating


def accumulate(
    stats: dict[str, dict[str, MoveStat]],
    game: chess.pgn.Game,
    plies: int,
) -> bool:
    """Fold one game's opening moves into ``stats``. Returns False if the game was unusable."""
    result = game.headers.get("Result", "*")
    board = game.board()
    seen_any = False
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= plies:
            break
        score = result_for_side(result, board.turn == chess.WHITE)
        if score is None:  # unfinished game, no usable label
            return seen_any
        moves = stats.setdefault(board.epd(), {})
        stat = moves.get(move.uci())
        if stat is None:
            stat = MoveStat()
            moves[move.uci()] = stat
        stat.count += 1
        stat.score += score
        board.push(move)
        seen_any = True
    return seen_any


def to_output(
    stats: dict[str, dict[str, MoveStat]],
    min_count: int,
) -> dict[str, list[dict[str, object]]]:
    """Serialise the aggregate: each position -> moves sorted best-first by score then count."""
    output: dict[str, list[dict[str, object]]] = {}
    for epd, moves in stats.items():
        if sum(stat.count for stat in moves.values()) < min_count:
            continue
        ranked = sorted(moves.items(), key=lambda kv: (kv[1].score, kv[1].count), reverse=True)
        output[epd] = [
            {"uci": uci, "count": stat.count, "score": round(stat.score, 2)}
            for uci, stat in ranked
        ]
    return output


def build(
    pgn_path: Path,
    plies: int,
    min_rating: int,
    max_games: int | None,
) -> dict[str, dict[str, MoveStat]]:
    """Stream the PGN and aggregate opening moves from eligible games.

    ``pgn_path`` of "-" reads stdin, so a Lichess dump can be piped straight from zstd without
    ever writing the ~200 GB decompressed file to disk:

        zstd -dc lichess_db_standard_rated_2026-07.pgn.zst | \
            uv run python -m training.build_opening_data --pgn - --max-games 2000000 ...
    """
    stats: dict[str, dict[str, MoveStat]] = {}
    read = kept = 0
    stream = str(pgn_path) == "-"
    handle = sys.stdin if stream else pgn_path.open(encoding="utf-8", errors="replace")
    try:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            read += 1
            if game_is_eligible(game.headers, min_rating) and accumulate(stats, game, plies):
                kept += 1
            if max_games is not None and read >= max_games:
                break
            if read % 10_000 == 0:
                print(f"  read {read:,} games, kept {kept:,}, positions {len(stats):,}")
    finally:
        if not stream:
            handle.close()
    print(f"read {read:,} games, kept {kept:,}, positions {len(stats):,}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an opening-move dataset from a PGN.")
    parser.add_argument("--pgn", type=Path, required=True, help="Plain-text PGN, or '-' for stdin.")
    parser.add_argument("--out", type=Path, default=Path("data/opening_moves.json"))
    parser.add_argument("--plies", type=int, default=DEFAULT_PLIES)
    parser.add_argument("--min-rating", type=int, default=DEFAULT_MIN_RATING)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--max-games", type=int, default=None, help="Cap games read (quick runs).")
    arguments = parser.parse_args()

    stats = build(arguments.pgn, arguments.plies, arguments.min_rating, arguments.max_games)
    output = to_output(stats, arguments.min_count)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=0)
    records = sum(len(moves) for moves in output.values())
    print(f"wrote {len(output):,} positions, {records:,} position-move records to {arguments.out}")


if __name__ == "__main__":
    main()
