"""Static (calibrated) INT8 quantization - the officially-recommended method
for CNNs per ONNX Runtime docs. Uses real dataset samples as calibration
data, entirely on this laptop."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from spectranet.data.preprocessing import normalize_iq, spectrogram


def load_index_simple(root, index_file):
    rows = []
    with open(Path(root) / index_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["label"])))
    return rows


class RFCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_root, index_file, input_name, n_fft=128, hop_length=32, max_samples=100):
        rows = load_index_simple(data_root, index_file)
        self.paths = [str(Path(data_root) / p) for p, _ in rows][:max_samples]
        self.input_name = input_name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.iterator = iter(self.paths)

    def get_next(self):
        path = next(self.iterator, None)
        if path is None:
            return None
        iq = np.load(path)
        iq = normalize_iq(iq, mode="unit_energy")
        spec = spectrogram(iq, n_fft=self.n_fft, hop_length=self.hop_length)
        spec = spec[np.newaxis, np.newaxis, ...].astype(np.float32)
        return {self.input_name: spec}

    def rewind(self):
        self.iterator = iter(self.paths)


def get_input_name(onnx_path):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return session.get_inputs()[0].name


def get_total_model_size_mb(onnx_path):
    total_bytes = os.path.getsize(onnx_path)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        total_bytes += os.path.getsize(data_file)
    return total_bytes / (1024 ** 2)


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
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--index-file", type=str, default="train.csv")
    parser.add_argument("--input-shape", type=str, default="1,128,33")
    parser.add_argument("--n-calibration", type=int, default=100)
    args = parser.parse_args()

    input_shape = tuple(int(v) for v in args.input_shape.split(","))

    print("Pre-processing ONNX graph (shape inference)...")
    preprocessed_path = args.onnx.replace(".onnx", "_static_preprocessed.onnx")
    quant_pre_process(args.onnx, preprocessed_path)

    input_name = get_input_name(preprocessed_path)
    print(f"Model input name: {input_name}")

    print(f"Building calibration reader from {args.n_calibration} real samples in {args.data_root}/{args.index_file} ...")
    calibration_reader = RFCalibrationDataReader(args.data_root, args.index_file, input_name, max_samples=args.n_calibration)

    print("Running static (calibrated) INT8 quantization...")
    fp32_size = get_total_model_size_mb(args.onnx)
    quantize_static(
        model_input=preprocessed_path,
        model_output=args.out,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    int8_size = get_total_model_size_mb(args.out)

    print(f"  FP32 size: {fp32_size:.3f} MB")
    print(f"  Static INT8 size: {int8_size:.3f} MB")
    print(f"  Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%\n")

    print("Benchmarking FP32 model (single-thread CPU, edge-constrained)...")
    fp32_stats = benchmark_single_thread(args.onnx, input_shape)
    print(f"  mean={fp32_stats['mean_ms']:.3f}ms  p50={fp32_stats['p50_ms']:.3f}ms  p95={fp32_stats['p95_ms']:.3f}ms\n")

    print("Benchmarking static INT8 model (single-thread CPU, edge-constrained)...")
    int8_stats = benchmark_single_thread(args.out, input_shape)
    print(f"  mean={int8_stats['mean_ms']:.3f}ms  p50={int8_stats['p50_ms']:.3f}ms  p95={int8_stats['p95_ms']:.3f}ms\n")

    speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"]
    print(f"Summary: static INT8 is {speedup:.2f}x the speed of FP32 under single-thread CPU constraints, "
          f"at {(1 - int8_size / fp32_size) * 100:.1f}% smaller size.")


if __name__ == "__main__":
    main()
