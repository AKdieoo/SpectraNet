"""Edge-inference validation done entirely on this laptop: quantize FP32->INT8
(real edge-deployment optimization) and benchmark both under single-thread
CPU constraints (approximates a resource-constrained edge device)."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def get_total_model_size_mb(onnx_path):
    total_bytes = os.path.getsize(onnx_path)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        total_bytes += os.path.getsize(data_file)
    return total_bytes / (1024 ** 2)


def quantize_model(onnx_path, out_path):
    fp32_size_mb = get_total_model_size_mb(onnx_path)

    preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
    quant_pre_process(onnx_path, preprocessed_path)

    quantize_dynamic(model_input=preprocessed_path, model_output=out_path, weight_type=QuantType.QInt8)

    int8_size_mb = get_total_model_size_mb(out_path)
    return fp32_size_mb, int8_size_mb


def benchmark_single_thread(model_path, input_shape, n_warmup=10, n_iters=100):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, *input_shape).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    timings = []
    for _ in range(n_iters):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        timings.append((time.perf_counter() - start) * 1000)

    timings = np.array(timings)
    return {
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--input-shape", type=str, default="1,128,33")
    args = parser.parse_args()

    input_shape = tuple(int(v) for v in args.input_shape.split(","))

    print("Quantizing FP32 -> INT8 (dynamic quantization)...")
    fp32_size, int8_size = quantize_model(args.onnx, args.out)
    print(f"  FP32 size: {fp32_size:.3f} MB")
    print(f"  INT8 size: {int8_size:.3f} MB")
    print(f"  Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%\n")

    print("Benchmarking FP32 model (single-thread CPU, edge-constrained)...")
    fp32_stats = benchmark_single_thread(args.onnx, input_shape)
    print(f"  mean={fp32_stats['mean_ms']:.3f}ms  p50={fp32_stats['p50_ms']:.3f}ms  p95={fp32_stats['p95_ms']:.3f}ms\n")

    print("Benchmarking INT8 model (single-thread CPU, edge-constrained)...")
    int8_stats = benchmark_single_thread(args.out, input_shape)
    print(f"  mean={int8_stats['mean_ms']:.3f}ms  p50={int8_stats['p50_ms']:.3f}ms  p95={int8_stats['p95_ms']:.3f}ms\n")

    speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"]
    print(f"Summary: INT8 is {speedup:.2f}x the speed of FP32 under single-thread CPU constraints, "
          f"at {(1 - int8_size / fp32_size) * 100:.1f}% smaller size.")


if __name__ == "__main__":
    main()
