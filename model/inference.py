import onnxruntime as ort
import numpy as np
import chess
import torch
from model.nnue import fen_to_nnue_input

class NNUEInference:
    def __init__(self, model_path="model/weights.onnx"):
        self.session = ort.InferenceSession(model_path)
        # The scaling factor used during training (C = 300.0)
        self.C = 300.0

    def evaluate(self, fen: str) -> int:
        """
        Evaluates a position and returns it in centipawns.
        """
        # 1. Feature Extraction
        # Convert fen_to_nnue_input output (torch.Tensor) to numpy
        input_vec = fen_to_nnue_input(fen).numpy().astype(np.float32)

        # Reshape for ONNX batch input: (1, 768)
        input_tensor = input_vec.reshape(1, -1)

        # 2. Run Inference
        outputs = self.session.run(None, {"input": input_tensor})
        prediction = outputs[0][0][0] # Get scalar value

        # 3. Inverse Normalization
        # y_norm = tanh(y / C)  =>  y = C * arctanh(y_norm)
        # np.arctanh only works for values in (-1, 1).
        # We clip to avoid NaN.
        prediction = np.clip(prediction, -0.9999, 0.9999)
        score_cp = self.C * np.arctanh(prediction)

        return int(score_cp)

# Example usage
if __name__ == "__main__":
    # To run this, you need a weights.onnx file
    try:
        engine = NNUEInference()
        test_fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
        score = engine.evaluate(test_fen)
        print(f"Position: {test_fen}\nScore: {score} cp")
    except Exception as e:
        print(f"Inference failed: {e}. Make sure you ran train.py and export.py first.")
