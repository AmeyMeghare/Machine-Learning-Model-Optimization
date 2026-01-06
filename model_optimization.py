"""
Model optimization techniques.
"""

import torch
import torch.quantization as quant
import onnx
from pathlib import Path


class ModelOptimizer:
    def __init__(self, model):
        self.model = model.eval()

    def apply_dynamic_quantization(self):
        return quant.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

    def convert_to_onnx(
        self,
        model,
        save_path: Path,
        input_shape=(1, 3, 224, 224),
        opset_version=18,
    ):
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)
