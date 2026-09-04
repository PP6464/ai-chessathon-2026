"""Build the opening dataset from the Lichess Opening Explorer API.

Stage P1 of the opening-selection plan (docs/OPENING_ML_PLAN.md), low-bandwidth variant. Instead
of parsing a 29 GB monthly PGN dump, we query the Opening Explorer, which returns — for any
position — the moves humans played with their white/draw/black counts, aggregated over the whole
Lichess database and bucketed by rating. We walk the opening tree from the start, expanding popular
moves down to a ply limit, and write the same schema ``build_opening_data.py`` produces:

    { "<epd>": [ {"uci": "e2e4", "count": <games>, "score": <mover result sum>}, ... ], ... }

so the trainer (P2) consumes either source unchanged. Each response is a few KB and we only ever
touch opening positions, so a full crawl is a few MB, not 29 GB.

This is human game statistics used as training data, distilled later into a policy net — the
contract permits that. It is *not* shipped as a runtime lookup database.

    uv run python -m training.crawl_explorer --out data/opening_moves.json --plies 16
    uv run python -m training.crawl_explorer --self-test   # offline check, no network
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import chess
from tqdm import tqdm

ENDPOINT = "https://explorer.lichess.ovh/lichess"
USER_AGENT = "aichessathon-opening-crawler (training/crawl_explorer.py)"

DEFAULT_PLIES = 16
DEFAULT_RATINGS = "2000,2200,2500"
DEFAULT_SPEEDS = "blitz,rapid,classical"
# Only follow a move deeper if humans played it at least this many times; bounds the tree.
DEFAULT_EXPAND_MIN = 1000
# Only record a move at all if it has at least this many games; prunes noise.
DEFAULT_RECORD_MIN = 50
DEFAULT_MAX_POSITIONS = 20_000
DEFAULT_DELAY = 0.25


def move_count_score(move: dict[str, int], white_to_move: bool) -> tuple[int, float]:
    """Total games for a move, and its summed result from the moving side's perspective.

    Matches the PGN pipeline: ``count`` is games played, ``score`` sums win=1 / draw=0.5, so a
    move's average quality is ``score / count``.
    """
    white, draws, black = move["white"], move["draws"], move["black"]
    count = white + draws + black
    mover_wins = white if white_to_move else black
    return count, mover_wins + 0.5 * draws


def records_from_response(
    response: dict[str, object],
    white_to_move: bool,
    record_min: int,
) -> list[dict[str, object]]:
    """Turn one Explorer response into ranked, pruned move records for a position."""
    moves = response.get("moves", [])
    assert isinstance(moves, list)
    records: list[tuple[str, int, float]] = []
    for move in moves:
        count, score = move_count_score(move, white_to_move)
        if count >= record_min:
            records.append((move["uci"], count, score))
    records.sort(key=lambda record: (record[2], record[1]), reverse=True)
    return [{"uci": uci, "count": count, "score": round(score, 2)} for uci, count, score in records]


def query(
    fen: str, ratings: str, speeds: str, timeout: float, retries: int, token: str
) -> dict[str, object]:
    """One Explorer request, retrying with backoff on rate limits and transient errors.

    The endpoint now requires a Lichess API token (any personal token, no scopes needed), sent as
    a Bearer header; a 401 means the token is missing or invalid.
    """
    url = ENDPOINT + "?" + urlencode(
        {"variant": "standard", "fen": fen, "ratings": ratings, "speeds": speeds, "moves": 20}
    )
    headers = {"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}
    request = Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
                assert isinstance(result, dict)
                return result
        except HTTPError as error:
            if error.code == 429:  # rate limited: back off and retry
                time.sleep(2.0 * (attempt + 1))
                continue
            if error.code == 401:  # bad/missing token; retrying will not help
                raise RuntimeError(
                    "401 from the Explorer: set a valid Lichess API token via --token or "
                    "LICHESS_TOKEN (create one at lichess.org/account/oauth/token, no scopes)"
                ) from error
            raise
        except (URLError, TimeoutError):
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"gave up on {fen} after {retries} attempts")


def crawl(arguments: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    """Breadth-first walk of the opening tree, one Explorer query per unique position."""
    output: dict[str, list[dict[str, object]]] = {}
    start = chess.Board()
    frontier: deque[tuple[str, int]] = deque([(start.epd(), 0)])
    seen: set[str] = {start.epd()}

    progress = tqdm(total=arguments.max_positions, desc="crawling", unit="pos", mininterval=0.5)
    while frontier and len(output) < arguments.max_positions:
        epd, depth = frontier.popleft()
        board = chess.Board(epd + " 0 1")
        response = query(
            board.fen(), arguments.ratings, arguments.speeds, arguments.timeout, 5, arguments.token
        )
        records = records_from_response(response, board.turn == chess.WHITE, arguments.record_min)
        if not records:
            continue
        output[epd] = records
        progress.update(1)
        progress.set_postfix(queued=len(frontier))

        if depth + 1 < arguments.plies:
            for record in records:
                if int(record["count"]) < arguments.expand_min:  # too rare to follow deeper
                    continue
                child = board.copy()
                child.push(chess.Move.from_uci(str(record["uci"])))
                child_epd = child.epd()
                if child_epd not in seen:
                    seen.add(child_epd)
                    frontier.append((child_epd, depth + 1))

        if len(output) % 200 == 0:  # copy-pasteable status line + safety checkpoint
            checkpoint(output, arguments.out)
            elapsed = progress.format_dict["elapsed"]
            rate = len(output) / elapsed if elapsed else 0.0
            progress.write(
                f"[crawl] {len(output):,} positions, {len(frontier):,} queued, "
                f"{elapsed / 60:.1f} min elapsed, {rate:.1f} pos/s — checkpointed"
            )
        time.sleep(arguments.delay)

    progress.close()
    print(f"crawled {len(output):,} positions")
    return output


def checkpoint(output: dict[str, list[dict[str, object]]], out: Path) -> None:
    """Persist progress so a slow or interrupted crawl is never lost."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=0)


def self_test() -> None:
    """Validate the offline logic (scoring, perspective, pruning, ranking) without network."""
    sample: dict[str, object] = {
        "white": 500,
        "draws": 200,
        "black": 300,
        "moves": [
            {"uci": "e2e4", "san": "e4", "white": 300, "draws": 100, "black": 200},  # 600 games
            {"uci": "d2d4", "san": "d4", "white": 120, "draws": 60, "black": 120},  # 300 games
            {"uci": "a2a3", "san": "a3", "white": 5, "draws": 1, "black": 4},  # 10 games, pruned
        ],
    }
    # White to move: e4 score = 300 + 0.5*100 = 350 over 600; d4 = 120 + 30 = 150 over 300.
    white_records = records_from_response(sample, white_to_move=True, record_min=50)
    assert [r["uci"] for r in white_records] == ["e2e4", "d2d4"], white_records
    assert white_records[0] == {"uci": "e2e4", "count": 600, "score": 350.0}, white_records[0]
    # Black to move: mover wins are the "black" counts instead.
    black_records = records_from_response(sample, white_to_move=False, record_min=50)
    assert black_records[0] == {"uci": "e2e4", "count": 600, "score": 250.0}, black_records[0]
    print("self-test passed: scoring, perspective, pruning and ranking all correct")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl opening data from the Lichess Explorer.")
    parser.add_argument("--out", type=Path, default=Path("data/opening_moves.json"))
    parser.add_argument("--plies", type=int, default=DEFAULT_PLIES)
    parser.add_argument("--ratings", default=DEFAULT_RATINGS)  # rating buckets, e.g. 2000,2200
    parser.add_argument("--speeds", default=DEFAULT_SPEEDS)
    parser.add_argument("--expand-min", type=int, default=DEFAULT_EXPAND_MIN)
    parser.add_argument("--record-min", type=int, default=DEFAULT_RECORD_MIN)
    parser.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between calls.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--token", default=os.environ.get("LICHESS_TOKEN", ""))  # or LICHESS_TOKEN
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit.")
    arguments = parser.parse_args()

    if arguments.self_test:
        self_test()
        return

    if not arguments.token:
        raise SystemExit(
            "the Explorer now needs a Lichess API token: create one (no scopes) at "
            "lichess.org/account/oauth/token, then pass --token <t> or set LICHESS_TOKEN"
        )

    output = crawl(arguments)
    checkpoint(output, arguments.out)
    records = sum(len(moves) for moves in output.values())
    print(f"wrote {len(output):,} positions, {records:,} position-move records to {arguments.out}")


if __name__ == "__main__":
    main()
