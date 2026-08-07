# SpectraNet

**AI platform for RF threat classification and edge inference.**

End-to-end pipeline: raw IQ → preprocessing/augmentation → CNN training with
experiment tracking → benchmarking across architectures → ONNX/TensorRT
export for edge deployment.

Stack: PyTorch, NumPy, ONNX, TensorRT, Weights & Biases, MLflow.

## Project layout

```
spectranet/
  data/
    preprocessing.py   # IQ normalization, FFT/STFT, spectrogram gen, RF augmentation
    dataset.py          # RFIQDataset (on-the-fly) + RFSpectrogramDataset (precomputed)
  models/
    zoo.py               # CustomCNN, ResNet18, VGG16, EfficientNet-B0, MobileNetV3-Small
  training/
    train.py             # custom training loop, W&B + MLflow tracking
    benchmark.py          # accuracy / latency / model-size comparison across architectures
  deploy/
    export_onnx.py         # PyTorch -> ONNX, with correctness verification
    export_tensorrt.py      # ONNX -> TensorRT engine (FP32/FP16/INT8), latency benchmark
configs/                   # per-architecture YAML training configs
scripts/
  build_index.py            # builds train.csv/val.csv from a class-labeled directory tree
tests/
  test_preprocessing.py      # NumPy-only unit tests for the preprocessing module
```

## Data format

Point `build_index.py` at a directory like:

```
data/processed/samples/<class_name>/*.npy
```

where each `.npy` holds a complex IQ vector (or a `(2, N)` real I/Q array).
This mirrors common RF datasets (RadioML-style captures, SigMF exports,
GNU Radio dumps) closely enough that swapping in real data is mostly a
matter of pointing at your files — `load_index()` in `dataset.py` is the
single place to override if your index format differs.

```bash
python scripts/build_index.py --data-root data/processed --val-fraction 0.15
```

## Preprocessing pipeline

`spectranet/data/preprocessing.py` implements, in pure NumPy:

- **IQ normalization** — z-score, unit-energy, or peak normalization
- **FFT / STFT** — hand-rolled short-time Fourier transform (Hann/Hamming/rect windows)
- **Spectrogram generation** — log-magnitude, min-max scaled to `[0, 1]`; optional
  2-channel magnitude+phase-derivative variant
- **RF-specific augmentation** — AWGN injection at a target SNR, random carrier
  frequency offset (CFO), random phase rotation, random time shift, and
  SpecAugment-style time/frequency masking on the spectrogram itself

These compose into `RFAugmentPipeline`, used directly inside `RFIQDataset`.

## Training

```bash
python -m spectranet.training.train --config configs/resnet18.yaml
```

Each config in `configs/` targets one architecture (`custom_cnn`, `resnet18`,
`vgg16`, `efficientnet_b0`, `mobilenet_v3_small`). The training loop logs to
**both** W&B and MLflow in parallel (either can be disabled independently via
`use_wandb` / `use_mlflow` in the config), checkpoints the best model by
validation accuracy, and supports early stopping and cosine/step LR schedules.

## Benchmarking architectures

Once you've trained checkpoints for each architecture, compare them head to
head on accuracy, parameter count, on-disk size, and inference latency:

```bash
python -m spectranet.training.benchmark \
  --models custom_cnn resnet18 vgg16 efficientnet_b0 mobilenet_v3_small \
  --checkpoints checkpoints/custom_cnn_best.pt checkpoints/resnet18_best.pt \
                checkpoints/vgg16_best.pt checkpoints/efficientnet_b0_best.pt \
                checkpoints/mobilenet_v3_small_best.pt \
  --data-root data/processed --val-index val.csv \
  --input-shape 1,128,128 --num-classes 11 \
  --out benchmark_results.csv
```

Results are written to CSV, printed as a table, and (if MLflow is available)
logged as nested runs under an `architecture_comparison` parent run.

## Edge deployment

**1. Export to ONNX:**

```bash
python -m spectranet.deploy.export_onnx \
  --checkpoint checkpoints/mobilenet_v3_small_best.pt \
  --model-name mobilenet_v3_small \
  --input-shape 1,128,128 --num-classes 11 \
  --out exported/mobilenet_v3_small.onnx --dynamic-batch --verify
```

`--verify` runs the exported graph through `onnxruntime` and checks outputs
match the PyTorch model within tolerance.

**2. Build a TensorRT engine** (run on your edge target / a machine with a
matching TensorRT install — e.g. Jetson Orin via JetPack):

```bash
python -m spectranet.deploy.export_tensorrt \
  --onnx exported/mobilenet_v3_small.onnx \
  --out exported/mobilenet_v3_small.trt \
  --precision fp16 --max-batch-size 8 --benchmark
```

Supports FP32/FP16/INT8 (INT8 via `Int8Calibrator`, which expects a small
representative calibration set spanning your class and SNR distribution).

## Tests

```bash
pip install -r requirements.txt
pytest tests/
```

`tests/test_preprocessing.py` covers the NumPy-only preprocessing module and
has no torch dependency, so it runs anywhere.

## Notes / assumptions

- Default spectrogram size in examples is `128x128` (`n_fft=128, hop_length` tuned
  to your burst length) — adjust `--input-shape` / config to match your actual data.
- `num_classes` defaults to `11` in the example configs (a common RadioML-style
  modulation count) — set this to match your actual label set.
- Backbone first-conv layers are automatically adapted from 3-channel RGB to
  1- or 2-channel spectrogram input, with pretrained ImageNet weights
  channel-averaged as the init (`models/zoo.py::_replace_first_conv`) so
  transfer learning still has a reasonable starting point.
