"""
Export a trained SpectraNet checkpoint to ONNX for edge inference.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from spectranet.models.zoo import build_model


def export_onnx(checkpoint_path, model_name, input_shape, num_classes, out_path, opset=18, dynamic_batch=True, verify=True):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(model_name, in_channels=input_shape[0], num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, *input_shape)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {out_path}")

    if verify:
        _verify_onnx(model, out_path, dummy_input)

    return out_path


def _verify_onnx(torch_model, onnx_path, dummy_input, atol=1e-4):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("onnx/onnxruntime not installed - skipping verification.")
        return

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    status = "OK" if max_diff < atol else "MISMATCH"
    print(f"ONNX verification: {status} (max abs diff vs PyTorch: {max_diff:.2e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-shape", type=str, default="1,128,128")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    input_shape = tuple(int(v) for v in args.input_shape.split(","))
    export_onnx(
        args.checkpoint, args.model_name, input_shape, args.num_classes,
        args.out, opset=args.opset, dynamic_batch=args.dynamic_batch, verify=args.verify,
    )


if __name__ == "__main__":
    main()
