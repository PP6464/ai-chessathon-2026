import torch
import torch.nn as nn

class NNUE(nn.Module):
    """
    A simplified NNUE-style architecture.
    The input is a flattened board representation: 64 squares * 12 pieces = 768 features.
    """
    def __init__(self, input_dim=768, hidden_dim=256):
        super(NNUE, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

def fen_to_nnue_input(fen):
    """
    Converts a FEN string to a 768-dimensional one-hot vector.
    Structure: 64 squares, each with 12 possible pieces (6 white, 6 black).
    """
    import chess
    board = chess.Board(fen)
    input_vec = torch.zeros(768)

    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        # offset: 0-5 for white, 6-11 for black
        offset = piece_map[piece_type] + (6 if not piece.color else 0)
        # index = square * 12 + offset
        input_vec[square * 12 + offset] = 1.0

    return input_vec
