import argparse
from pathlib import Path

from harness_win.referee import FAILED_TERMINATIONS, play_match
from harness_win.rules import PLY_CAP
from harness_win.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an agent over several games.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}

    for game in range(arguments.games):
        plays_white = game % 2 == 0
        white, black = (agent, opponent) if plays_white else (opponent, agent)
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            ply_cap=arguments.ply_cap,
        )
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result == "draw" or outcome.result == "void":
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        print(f"game {game + 1}/{arguments.games}: {outcome.result} by {outcome.termination}")

    score = (wins + draws / 2) / arguments.games
    print(f"\n{arguments.agent} vs {arguments.opponent} over {arguments.games} games")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    broken = {name: count for name, count in terminations.items() if name in FAILED_TERMINATIONS}
    if broken:
        raise SystemExit(
            "your agent failed to finish a game: "
            + ", ".join(f"{name} {count}" for name, count in broken.items())
        )


if __name__ == "__main__":
    main()
