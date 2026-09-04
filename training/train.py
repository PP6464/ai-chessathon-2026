import math
import chess
import sys
from pathlib import Path

# Add the project root to sys.path so imports work regardless of how the script is run
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from model.nnue import NNUE, fen_to_nnue_input

# --- Configuration ---
C = 150.0  # Centipawn scaling factor for tanh normalization
BATCH_SIZE = 512
EPOCHS = 1
LEARNING_RATE = 0.001
MAX_SAMPLES = 2_000_000  # Increased to 2M samples


class ChessDataset(IterableDataset):
    def __init__(self, split="train", limit=None):
        super().__init__()
        self.ds = load_dataset("Lichess/chess-position-evaluations", split=split, streaming=True)
        self.limit = limit
        # Balancing counters
        self.counts = {
            "white": 0,
            "black": 0,
            "draw": 0,
            "opening": 0,
            "midgame": 0,
            "endgame": 0,
        }
        # Target limits per category to ensure balance
        self.max_per_cat = (limit // 3) if limit else 100_000

    def _get_phase(self, board):
        # Use the same weight logic as agent.py
        weights = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
        total = sum(weights[p.piece_type] for p in board.piece_map().values())
        if total > 18: return "opening"
        if total > 10: return "midgame"
        return "endgame"

    def __iter__(self):
        for i, entry in enumerate(self.ds):
            if self.limit and i >= self.limit * 10: # Search a larger pool to find balanced samples
                break

            board = chess.Board(entry['fen'])

            # 1. Calculate raw score
            if entry['mate'] is not None:
                sign = 1 if (entry['cp'] is None or entry['cp'] > 0) else -1
                y = sign * 1000.0
            else:
                y = float(entry['cp']) if entry['cp'] is not None else 0.0

            # 2. Determine Categories
            side = "white" if y > 50 else "black" if y < -50 else "draw"
            phase = self._get_phase(board)

            # 3. Balancing Filter: Skip if category is already full
            if self.counts[side] >= self.max_per_cat or self.counts[phase] >= self.max_per_cat:
                continue

            # Update counters
            self.counts[side] += 1
            self.counts[phase] += 1

            # 4. Feature Extraction
            x = fen_to_nnue_input(entry['fen'])
            y_norm = math.tanh(y / C)

            yield x, torch.tensor([y_norm], dtype=torch.float32)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training set
    train_loader = DataLoader(ChessDataset(limit=MAX_SAMPLES), batch_size=BATCH_SIZE)
    # Testing set (small slice from the streaming train set)
    test_loader = DataLoader(ChessDataset(split="train", limit=10000), batch_size=BATCH_SIZE)


    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        count = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            if count % 100 == 0:
                print(f"Batch {count}, Loss: {loss.item():.6f}")

        print(f"Epoch {epoch + 1} completed. Avg Loss: {total_loss / count:.6f}")

    # Evaluation
    model.eval()
    test_loss = 0
    test_count = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            test_loss += criterion(preds, y).item()
            test_count += 1

    print(f"Test MSE: {test_loss / test_count:.6f}")

    # Save weights
    torch.save(model.state_dict(), "model/weights.pth")
    print("Weights saved to model/weights.pth")


if __name__ == "__main__":
    train()
