import torch
import torch.nn as nn

class NNUE(nn.Module):
    """
    A simplified NNUE-style architecture.
    The input is a flattened board representation: 64 squares * 12 pieces = 768 features.
    """
    def __init__(self, input_dim=768, hidden_dim=256):
        super(NNUE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # We separate the first layer to allow for incremental updates (True NNUE)
        self.l1 = nn.Linear(input_dim, hidden_dim)

        # The rest of the network
        self.rest = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Full forward pass (used for training)
        x = self.l1(x)
        x = self.rest(x)
        return x

    def get_l1_weights(self):
        """Returns the weights and bias of the first layer for incremental updates."""
        return self.l1.weight.data, self.l1.bias.data

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
