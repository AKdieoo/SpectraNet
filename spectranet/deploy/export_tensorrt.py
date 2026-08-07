"""
Build a TensorRT engine from an exported ONNX model, for deployment on
edge accelerators (e.g. Jetson Orin/Xavier, or any TensorRT-capable GPU).

Supports FP32 / FP16 / INT8 (with a simple calibration hook) precision
modes and reports engine build info + a quick latency benchmark of the
built engine.

Note: requires `tensorrt` + `pycuda` installed and a CUDA-capable device
with TensorRT available — this is expected to run on the edge target or a
build machine matching its TensorRT version, not in a generic CPU sandbox.

Usage
-----
    python -m spectranet.deploy.export_tensorrt \
        --onnx exported/mobilenet_v3_small.onnx \
        --out exported/mobilenet_v3_small.trt \
        --precision fp16 --max-batch-size 8
"""

from __future__ import annotations

import argparse
import time

try:
    import tensorrt as trt
    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False

try:
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
    import pycuda.driver as cuda
    _HAS_PYCUDA = True
except ImportError:
    _HAS_PYCUDA = False

import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if _HAS_TRT else None


class Int8Calibrator:
    """
    Minimal entropy calibrator for INT8 quantization. Feed it a small
    representative set of preprocessed spectrograms (e.g. 200-500 samples
    spanning your SNR range and classes) via `calibration_data`.
    """

    def __init__(self, calibration_data: np.ndarray, cache_file: str = "calibration.cache"):
        if not (_HAS_TRT and _HAS_PYCUDA):
            raise RuntimeError("tensorrt and pycuda are required for INT8 calibration")

        self.data = calibration_data.astype(np.float32)
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

        class _Impl(trt.IInt8EntropyCalibrator2):
            def __init__(inner_self):
                trt.IInt8EntropyCalibrator2.__init__(inner_self)

            def get_batch_size(inner_self):
                return self.batch_size

            def get_batch(inner_self, names):
                if self.current_index >= len(self.data):
                    return None
                batch = np.ascontiguousarray(self.data[self.current_index])
                cuda.memcpy_htod(self.device_input, batch)
                self.current_index += 1
                return [int(self.device_input)]

            def read_calibration_cache(inner_self):
                try:
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                except FileNotFoundError:
                    return None

            def write_calibration_cache(inner_self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        self._impl = _Impl()

    def as_trt_calibrator(self):
        return self._impl


def build_engine(
    onnx_path: str,
    out_path: str,
    precision: str = "fp16",
    max_batch_size: int = 8,
    workspace_gb: float = 4.0,
    calibrator: "Int8Calibrator | None" = None,
) -> str:
    if not _HAS_TRT:
        raise RuntimeError(
            "tensorrt is not installed in this environment. "
            "Build/export TensorRT engines on your edge target or a machine "
            "with the matching TensorRT version installed."
        )

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse ONNX model:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("Warning: platform reports no fast FP16 support; proceeding anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("Warning: platform reports no fast INT8 support; proceeding anyway.")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            raise ValueError("INT8 precision requires a calibrator (see Int8Calibrator).")
        config.int8_calibrator = calibrator.as_trt_calibrator()
    elif precision != "fp32":
        raise ValueError(f"Unknown precision: {precision}")

    # Dynamic batch profile: 1 .. max_batch_size
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    shape = input_tensor.shape  # (-1, C, H, W) if exported with dynamic batch
    c, h, w = shape[1], shape[2], shape[3]
    profile.set_shape(input_tensor.name, (1, c, h, w), (max(1, max_batch_size // 2), c, h, w), (max_batch_size, c, h, w))
    config.add_optimization_profile(profile)

    print(f"Building TensorRT engine (precision={precision}, max_batch_size={max_batch_size}) ...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")
    build_time = time.time() - t0

    with open(out_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Built engine in {build_time:.1f}s -> {out_path}")
    return out_path


def benchmark_engine(engine_path: str, input_shape: tuple[int, int, int], n_iters: int = 100) -> dict:
    if not (_HAS_TRT and _HAS_PYCUDA):
        raise RuntimeError("tensorrt and pycuda are required to benchmark an engine")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    c, h, w = input_shape
    input_shape_full = (1, c, h, w)
    context.set_input_shape(engine.get_tensor_name(0), input_shape_full)

    host_input = np.random.randn(*input_shape_full).astype(np.float32)
    output_shape = tuple(context.get_tensor_shape(engine.get_tensor_name(1)))
    host_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(host_input.nbytes)
    d_output = cuda.mem_alloc(host_output.nbytes)
    stream = cuda.Stream()

    context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(d_output))

    # Warmup
    for _ in range(20):
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()

    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    timings = np.array(timings)
    return {
        "latency_mean_ms": float(timings.mean()),
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--input-shape", type=str, default="1,128,128", help="C,H,W, used only for --benchmark")
    args = parser.parse_args()

    build_engine(
        args.onnx, args.out, precision=args.precision,
        max_batch_size=args.max_batch_size, workspace_gb=args.workspace_gb,
    )

    if args.benchmark:
        input_shape = tuple(int(v) for v in args.input_shape.split(","))
        stats = benchmark_engine(args.out, input_shape)
        print(f"TensorRT engine latency: p50={stats['latency_p50_ms']:.2f}ms "
              f"p95={stats['latency_p95_ms']:.2f}ms mean={stats['latency_mean_ms']:.2f}ms")


if __name__ == "__main__":
    main()
