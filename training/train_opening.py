"""Train the opening policy net (P2 of docs/OPENING_ML_PLAN.md).

Input is the dataset from P1 (`build_opening_data.py` or `crawl_explorer.py`): each opening
position mapped to the moves humans played, with a game count and a result-weighted score. We fit a
small MLP that, given a position, predicts a distribution over moves matching the human
result-weighted distribution — so it prefers moves that were played often *and* scored well.

The trained weights are saved as a plain ``.npz`` (not ONNX) so the agent can run inference with a
few numpy matmuls and no heavy runtime dependency; the exact same `encode` lives here and will be
mirrored in the agent at P3. Only human game statistics are used, which the contract permits as
training data.

    uv run python -m training.train_opening --data data/opening_moves.json --out model/opening.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

# Feature layout: 12 piece planes x 64 squares, then side-to-move + four castling rights.
N_PLANES = 12
N_SQUARES = 64
N_EXTRA = 5
N_FEATURES = N_PLANES * N_SQUARES + N_EXTRA  # 773
# Move space: from-square x to-square. Openings never promote, so promotion type is ignored.
N_MOVES = N_SQUARES * N_SQUARES  # 4096


def encode(board: chess.Board) -> np.ndarray:
    """A position as a flat float32 feature vector (shared verbatim with the agent at P3)."""
    features = np.zeros(N_FEATURES, dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color == chess.WHITE else 6) + (piece.piece_type - 1)
        features[plane * N_SQUARES + square] = 1.0
    base = N_PLANES * N_SQUARES
    features[base] = 1.0 if board.turn == chess.WHITE else 0.0
    features[base + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[base + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[base + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[base + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    return features


def move_index(uci: str) -> int:
    """Map a UCI move to its from*64+to index."""
    move = chess.Move.from_uci(uci)
    return move.from_square * N_SQUARES + move.to_square


class OpeningNet(nn.Module):
    """Two hidden layers to a per-move logit head."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_MOVES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_tensors(
    data: dict[str, list[dict[str, object]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn the JSON dataset into padded (features, candidate indices, targets) tensors.

    Targets are the score-normalised human distribution over each position's candidate moves; a
    position where every move lost (all scores 0) falls back to popularity (counts). Padding uses
    move index 0 (a1a1, never a legal move) with target weight 0, so it is inert.
    """
    max_candidates = max(len(moves) for moves in data.values())
    features: list[np.ndarray] = []
    candidates: list[list[int]] = []
    targets: list[list[float]] = []

    for epd, moves in data.items():
        board = chess.Board(epd + " 0 1")
        weights = np.array([float(m["score"]) for m in moves], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.array([float(m["count"]) for m in moves], dtype=np.float64)
        weights = weights / weights.sum()

        pad = max_candidates - len(moves)
        features.append(encode(board))
        candidates.append([move_index(str(m["uci"])) for m in moves] + [0] * pad)
        targets.append(weights.tolist() + [0.0] * pad)

    return (
        torch.tensor(np.array(features), dtype=torch.float32),
        torch.tensor(candidates, dtype=torch.int64),
        torch.tensor(targets, dtype=torch.float32),
    )


def resolve_device(name: str) -> str:
    """Turn 'auto' into the fastest available backend; otherwise honour the explicit choice."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple GPU; rarely faster than CPU for a net this small, but available
    return "cpu"


def train(
    net: OpeningNet,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    """Fit the net with cross-entropy over the *full* move space.

    Softmaxing over all moves (not just the candidates) is what makes inference work: every move
    the humans did not play is pushed down, so an argmax over legal moves surfaces the learned
    ones. The sparse per-position target is scattered into a dense move distribution per batch.
    """
    features, candidates, targets = (tensor.to(device) for tensor in tensors)
    positions = features.shape[0]
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    progress = tqdm(range(epochs), desc="training", unit="epoch", mininterval=0.5)
    for _epoch in progress:
        order = torch.randperm(positions, device=device)
        running = 0.0
        for start in range(0, positions, batch_size):
            batch = order[start : start + batch_size]
            logits = net(features[batch])  # (batch, N_MOVES)
            target_full = torch.zeros_like(logits).scatter_(1, candidates[batch], targets[batch])
            loss = -(target_full * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item() * len(batch)
        progress.set_postfix(loss=f"{running / positions:.4f}")


def save_npz(net: OpeningNet, out: Path, hidden: int) -> None:
    """Dump the linear layers as numpy arrays for the agent's numpy-only inference."""
    layers = [m for m in net.net if isinstance(m, nn.Linear)]
    arrays: dict[str, np.ndarray] = {"hidden": np.array(hidden)}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weight.detach().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().numpy().astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)


def preview(net: OpeningNet, board: chess.Board, label: str) -> None:
    """Print the net's top few legal moves from a position, as a sanity check."""
    with torch.no_grad():
        logits = net(torch.tensor(encode(board)).unsqueeze(0)).squeeze(0)
    legal = list(board.legal_moves)
    scored = sorted(
        legal,
        key=lambda mv: logits[mv.from_square * N_SQUARES + mv.to_square].item(),
        reverse=True,
    )
    top = ", ".join(board.san(mv) for mv in scored[:5])
    print(f"  {label}: {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the opening policy net.")
    parser.add_argument("--data", type=Path, default=Path("data/opening_moves.json"))
    parser.add_argument("--out", type=Path, default=Path("weights/opening.npz"))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")  # auto | cpu | mps | cuda
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    with arguments.data.open(encoding="utf-8") as handle:
        data = json.load(handle)
    print(f"loaded {len(data):,} positions from {arguments.data}")

    device = resolve_device(arguments.device)
    print(f"training on {device}")
    net = OpeningNet(arguments.hidden).to(device)
    train(net, build_tensors(data), arguments.epochs, arguments.lr, arguments.batch_size, device)
    net.to("cpu")  # weights come back to the CPU to serialise as numpy for the agent
    save_npz(net, arguments.out, arguments.hidden)
    print(f"saved weights to {arguments.out}")

    preview(net, chess.Board(), "from start")
    black = chess.Board()
    black.push_san("e4")
    preview(net, black, "after 1.e4 ")


if __name__ == "__main__":
    main()
