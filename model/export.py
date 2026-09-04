import numpy as np
import torch
from model.nnue import NNUE

def export_to_onnx():
    # 1. Load Model
    model = NNUE()
    model.load_state_dict(torch.load("model/weights.pth"))
    model.eval()

    # 2. Export First Layer Weights for Incremental Updates
    l1_weights, l1_bias = model.get_l1_weights()
    # Use savez to save multiple arrays in one file
    np.savez("model/weights_l1.npz", weights=l1_weights.numpy(), bias=l1_bias.numpy())
    print("First layer weights saved to model/weights_l1.npz")

    # 3. Export the REST of the network to ONNX
    dummy_input = torch.randn(1, model.hidden_dim)

    onnx_path = "model/weights.onnx"
    torch.onnx.export(
        model.rest,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['accumulator'],
        output_names=['output'],
        dynamic_axes={'accumulator': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model rest successfully exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
