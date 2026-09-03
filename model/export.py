import onnx
import torch
from model.nnue import NNUE


def export_to_onnx():
    # 1. Load Model
    model = NNUE()
    model.load_state_dict(torch.load("weights.pth"))
    model.eval()

    # 2. Create Dummy Input
    # Shape: (batch_size, input_dim) -> (1, 768)
    dummy_input = torch.randn(1, 768)

    # 3. Export
    onnx_path = "weights.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={ 'input': { 0: 'batch_size' }, 'output': { 0: 'batch_size' } }
    )
    m = onnx.load(onnx_path)
    onnx.save_model(m, onnx_path, save_as_external_data=False)
    print(f"Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    export_to_onnx()
