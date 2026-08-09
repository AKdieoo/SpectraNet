"""Grad-CAM explainability for SpectraNet models. Shows WHICH parts of the
spectrogram the model attended to when making a prediction."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom

from generate_threat_dataset import make_benign, make_threat
from spectranet.data.preprocessing import normalize_iq, spectrogram
from spectranet.models.zoo import build_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].detach().numpy()

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.numpy(), target_class, probs


def get_target_layer(model, model_name):
    if model_name == "custom_cnn":
        return model.layer3
    elif model_name == "resnet18":
        return model.layer4
    elif model_name == "vgg16":
        return model.features[-2]
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    elif model_name == "mobilenet_v3_small":
        return model.features[-1]
    else:
        raise ValueError(f"No Grad-CAM target layer defined for {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--kind", choices=["benign", "threat"], default="threat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="gradcam_explanation.png")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    model = build_model(cfg["model_name"], in_channels=cfg["in_channels"], num_classes=cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_names = {}
    with open(Path(cfg["data_root"]) / "class_names.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names[int(row["label"])] = row["class_name"]

    rng = np.random.default_rng(args.seed)
    iq = make_benign(rng) if args.kind == "benign" else make_threat(rng)
    iq_norm = normalize_iq(iq, mode="unit_energy")
    spec = spectrogram(iq_norm, n_fft=128, hop_length=32)
    input_tensor = torch.from_numpy(spec[np.newaxis, np.newaxis, ...].astype(np.float32))
    input_tensor.requires_grad_(False)

    target_layer = get_target_layer(model, cfg["model_name"])
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_class_idx, probs = gradcam.generate(input_tensor)

    pred_class = class_names.get(pred_class_idx, str(pred_class_idx))
    confidence = float(probs[pred_class_idx])

    zoom_h = spec.shape[0] / heatmap.shape[0]
    zoom_w = spec.shape[1] / heatmap.shape[1]
    heatmap_resized = zoom(heatmap, (zoom_h, zoom_w), order=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"Input Spectrogram\n(ground truth: {args.kind})")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    axes[1].imshow(heatmap_resized, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap\n(where the model looked)")
    axes[1].set_xlabel("Time")

    axes[2].imshow(spec, aspect="auto", origin="lower", cmap="gray")
    axes[2].imshow(heatmap_resized, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay\nPredicted: {pred_class} ({confidence*100:.1f}%)")
    axes[2].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Ground truth: {args.kind}")
    print(f"Predicted: {pred_class} ({confidence*100:.1f}% confidence)")
    print(f"All probabilities: {dict(zip([class_names[i] for i in range(len(probs))], probs.tolist()))}")
    print(f"\nSaved explanation to {args.out}")


if __name__ == "__main__":
    main()
