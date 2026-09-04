import argparse
import chess
import sys
import os
import importlib.util


def clear_screen():
    # Works for both Windows and Unix
    print("\033[H\033[J", end="")


def load_agent_module(module_path):
    """Dynamically loads a python file as a module."""
    # Ensure we have an absolute path
    abs_path = os.path.abspath(module_path)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register in sys.modules
    spec.loader.exec_module(module)
    return module


def play():
    parser = argparse.ArgumentParser(description="Play against any agent implementation")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Which color do you want to play?")
    parser.add_argument("--agent", default="agent.py", help="Path to the agent.py file you want to play against")
    parser.add_argument("--model", default="model", help="Path to the model directory (for NNUE agents)")
    args = parser.parse_args()

    # Set the environment variable for NNUE agents
    os.environ["MODEL_DIR"] = args.model

    try:
        agent_module = load_agent_module(args.agent)
    except Exception as e:
        print(f"Error loading agent: {e}")
        sys.exit(1)

    board = chess.Board()
    user_color = chess.WHITE if args.color == "white" else chess.BLACK
    ai_color = chess.BLACK if user_color == chess.WHITE else chess.WHITE

    print(f"Welcome! You are playing as {args.color}. Agent '{args.agent}' is playing as {'black' if ai_color == chess.BLACK else 'white'}.")

    while not board.is_game_over():
        clear_screen()
        print("=== Chess Agent Playtest ===")
        print(board)
        print("\n")

        if board.turn == user_color:
            # User move
            move_str = input("Enter your move (UCI, e.g., e2e4): ")
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    input("Press Enter to continue...")
                    continue
            except ValueError:
                print("Invalid UCI format! Try again.")
                input("Press Enter to continue...")
                continue
        else:
            # AI move
            print("AI is thinking...")
            # We give the AI 5 seconds for the playtest
            move_uci = agent_module.get_move(board.fen(), 5000)
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            print(f"AI played: {move_uci}")
            input("Press Enter to continue...")

    clear_screen()
    print("=== Game Over ===")
    print(board)
    print(f"\nResult: {board.result()}")


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        print("\nGame exited.")
        sys.exit(0)
