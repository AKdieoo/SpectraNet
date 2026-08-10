"""Open-set detection take 2, improved: Mahalanobis + PCA + honest AUC-ROC
evaluation (not just a single cherry-picked detection-rate number)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from generate_threat_dataset import make_threat
from spectranet.data.preprocessing import normalize_iq, spectrogram
from spectranet.models.zoo import build_model


def load_index_simple(root, index_file):
    rows = []
    with open(Path(root) / index_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["label"])))
    return rows


class FeatureExtractor:
    def __init__(self, model, model_name):
        self.features = None
        layer = self._get_feature_layer(model, model_name)
        layer.register_forward_hook(self._save_features)

    def _get_feature_layer(self, model, model_name):
        if model_name == "custom_cnn":
            return model.pool
        elif model_name in ("resnet18", "vgg16", "efficientnet_b0", "mobilenet_v3_small"):
            return model.avgpool
        else:
            raise ValueError(f"No feature layer defined for {model_name}")

    def _save_features(self, module, input, output):
        self.features = output.detach().flatten(1)

    def extract(self, model, spec):
        tensor = torch.from_numpy(spec[np.newaxis, np.newaxis, ...].astype(np.float32))
        with torch.no_grad():
            model(tensor)
        return self.features[0].numpy()


def extract_features_for_index(model, extractor, data_root, index_file, n_per_class=200):
    rows = load_index_simple(data_root, index_file)
    by_label = {}
    for path, label in rows:
        by_label.setdefault(label, []).append(path)

    features_by_class = {}
    for label, paths in by_label.items():
        feats = []
        for p in paths[:n_per_class]:
            iq = np.load(Path(data_root) / p)
            iq_norm = normalize_iq(iq, mode="unit_energy")
            spec = spectrogram(iq_norm, n_fft=128, hop_length=32)
            feats.append(extractor.extract(model, spec))
        features_by_class[label] = np.array(feats)
    return features_by_class


def fit_class_gaussians(features_by_class):
    means = {label: feats.mean(axis=0) for label, feats in features_by_class.items()}
    centered_all = np.vstack([feats - means[label] for label, feats in features_by_class.items()])
    dim = centered_all.shape[1]
    cov = np.cov(centered_all, rowvar=False)
    cov_reg = cov + np.eye(dim) * 1e-3
    cov_inv = np.linalg.inv(cov_reg)
    return means, cov_inv


def mahalanobis_min_distance(feature, means, cov_inv):
    distances = []
    for mu in means.values():
        diff = feature - mu
        d = np.sqrt(max(diff @ cov_inv @ diff, 0.0))
        distances.append(d)
    return min(distances)


def get_ood_features(model, extractor, n_samples=100, seed=999):
    rng = np.random.default_rng(seed)
    feats = []
    for _ in range(n_samples):
        iq = make_threat(rng)
        iq_norm = normalize_iq(iq, mode="unit_energy")
        spec = spectrogram(iq_norm, n_fft=128, hop_length=32)
        feats.append(extractor.extract(model, spec))
    return np.array(feats)


def run_pipeline(model, extractor, data_root, n_components, percentile):
    train_by_class = extract_features_for_index(model, extractor, data_root, "train.csv")
    val_by_class = extract_features_for_index(model, extractor, data_root, "val.csv")
    ood_features_raw = get_ood_features(model, extractor)

    if n_components is not None:
        all_train = np.vstack(list(train_by_class.values()))
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(all_train)
        train_by_class = {k: pca.transform(v) for k, v in train_by_class.items()}
        val_by_class = {k: pca.transform(v) for k, v in val_by_class.items()}
        ood_features = pca.transform(ood_features_raw)
        explained = pca.explained_variance_ratio_.sum()
    else:
        ood_features = ood_features_raw
        explained = None

    means, cov_inv = fit_class_gaussians(train_by_class)

    id_distances = []
    for feats in val_by_class.values():
        for f in feats:
            id_distances.append(mahalanobis_min_distance(f, means, cov_inv))

    ood_distances = [mahalanobis_min_distance(f, means, cov_inv) for f in ood_features]

    threshold = float(np.percentile(id_distances, percentile))
    flagged = sum(1 for d in ood_distances if d > threshold)
    detection_rate = flagged / len(ood_distances)

    labels = [0] * len(id_distances) + [1] * len(ood_distances)
    scores = id_distances + ood_distances
    auc = roc_auc_score(labels, scores)

    return {
        "n_components": n_components,
        "explained_variance": explained,
        "mean_id_distance": float(np.mean(id_distances)),
        "mean_ood_distance": float(np.mean(ood_distances)),
        "threshold": threshold,
        "detection_rate": detection_rate,
        "flagged": flagged,
        "n_ood": len(ood_distances),
        "auc_roc": float(auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg["model_name"], in_channels=cfg["in_channels"],
                         num_classes=cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    extractor = FeatureExtractor(model, cfg["model_name"])

    print("=== Full-dimensional features (256-dim), as before ===")
    result_full = run_pipeline(model, extractor, args.data_root, n_components=None, percentile=args.percentile)
    print(f"  AUC-ROC: {result_full['auc_roc']:.4f} (0.5=random, 1.0=perfect separation)")
    print(f"  Detection rate at {args.percentile}th percentile threshold: "
          f"{result_full['flagged']}/{result_full['n_ood']} ({result_full['detection_rate']*100:.1f}%)")
    print(f"  Mean distance - known: {result_full['mean_id_distance']:.3f}, "
          f"OOD: {result_full['mean_ood_distance']:.3f}\n")

    print("=== PCA-reduced features (30-dim), better-conditioned covariance ===")
    result_pca = run_pipeline(model, extractor, args.data_root, n_components=30, percentile=args.percentile)
    print(f"  Explained variance retained: {result_pca['explained_variance']*100:.1f}%")
    print(f"  AUC-ROC: {result_pca['auc_roc']:.4f} (0.5=random, 1.0=perfect separation)")
    print(f"  Detection rate at {args.percentile}th percentile threshold: "
          f"{result_pca['flagged']}/{result_pca['n_ood']} ({result_pca['detection_rate']*100:.1f}%)")
    print(f"  Mean distance - known: {result_pca['mean_id_distance']:.3f}, "
          f"OOD: {result_pca['mean_ood_distance']:.3f}\n")

    print("=== Summary ===")
    print(f"Full-dim AUC:  {result_full['auc_roc']:.4f}")
    print(f"PCA-dim AUC:   {result_pca['auc_roc']:.4f}")
    if result_pca['auc_roc'] > result_full['auc_roc']:
        print("PCA reduction genuinely improved separation.")
    else:
        print("PCA reduction did NOT improve separation - full-dim features were already better here.")


if __name__ == "__main__":
    main()
