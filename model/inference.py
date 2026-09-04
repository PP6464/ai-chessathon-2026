import onnxruntime as ort
import numpy as np
import chess

class NNUEIncremental:
    def __init__(self, weights_path="model/weights_l1.npz", model_path="model/weights.onnx"):
        # Load L1 weights and bias
        l1_data = np.load(weights_path)
        self.weights = l1_data["weights"] # Shape: (256, 768)
        self.bias = l1_data["bias"]       # Shape: (256,)
        self.C = 150.0

        # Load the REST of the network (ONNX)
        self.session = ort.InferenceSession(model_path)

    def _get_piece_weight_index(self, piece: chess.Piece) -> int:
        if piece is None:
            return -1
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        offset = piece_map[piece.piece_type] + (6 if not piece.color else 0)
        return offset

    def get_initial_accumulator(self, board: chess.Board) -> np.ndarray:
        """
        Computes the starting accumulator from scratch.
        """
        accumulator = np.copy(self.bias).astype(np.float32)
        for square, piece in board.piece_map().items():
            idx = square * 12 + self._get_piece_weight_index(piece)
            accumulator += self.weights[:, idx]
        return accumulator

    def get_square_weight(self, square: int, piece: chess.Piece | None) -> np.ndarray:
        """Returns the weight vector for a piece at a specific square."""
        if piece is None:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        idx = square * 12 + self._get_piece_weight_index(piece)
        return self.weights[:, idx]

    def evaluate(self, accumulator: np.ndarray) -> int:
        """
        Runs the ONNX model on the current accumulator.
        """
        input_tensor = accumulator.reshape(1, -1).astype(np.float32)
        outputs = self.session.run(None, {"accumulator": input_tensor})
        prediction = outputs[0][0][0]

        # Inverse Normalization
        prediction = np.clip(prediction, -0.9999, 0.9999)
        score_cp = self.C * np.arctanh(prediction)

        return int(score_cp)
