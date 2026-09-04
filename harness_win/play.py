import argparse
from pathlib import Path

from harness_win.referee import play_match
from harness_win.rules import BASE_MS, INCREMENT_MS, PLY_CAP
from harness_win.sandbox import local


def main() -> None:
    parser = argparse.ArgumentParser(description="Play one game between two agent directories.")
    parser.add_argument("--white", type=Path, default=Path("."))
    parser.add_argument("--black", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--pgn", type=Path)
    arguments = parser.parse_args()

    white = local(arguments.white)
    black = local(arguments.black)
    outcome = play_match(
        white, black, arguments.base_ms, arguments.increment_ms, ply_cap=arguments.ply_cap
    )

    print(f"{arguments.white} vs {arguments.black}: {outcome.result} by {outcome.termination}")
    for name, agent in (("white", white), ("black", black)):
        if agent.stderr_tail:
            print(f"\n{name} wrote to stderr:\n{agent.stderr_tail.rstrip()}")
    if arguments.pgn:
        arguments.pgn.write_text(outcome.pgn + "\n")
        print(f"pgn written to {arguments.pgn}")


if __name__ == "__main__":
    main()