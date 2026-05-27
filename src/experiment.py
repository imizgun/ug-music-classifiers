from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.config import Config
from src.dataset import load_dataset, split_dataset
from src.features.librosa_features import get_librosa_features, get_mfcc_sequences
from src.features.spectrograms import get_spectrograms
from src.features.embeddings import get_embeddings
from src.models.classical import RandomForestModel, KNNModel
from src.models.mlp import MLPModel
from src.models.cnn import CNNModel
from src.models.lstm import LSTMModel
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_comparison_table, plot_embeddings_tsne


def _resolve_device(cfg: Config) -> torch.device:
    spec = cfg.training.device
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def run(cfg: Config) -> dict:
    out_dir = Path(cfg.output.dir)
    device = _resolve_device(cfg)
    print(f"Device: {device}")

    # --- Data ---
    print("\n[1/3] Loading dataset...")
    df = load_dataset(cfg.data)
    print(f"  {len(df)} samples loaded")
    train_df, val_df, test_df = split_dataset(df, cfg.data.split)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # --- Feature extraction (с кэшем) ---
    print("\n[2/3] Extracting features...")
    features: dict = {}

    needs = _needed_inputs(cfg)
    if "librosa" in needs:
        features["librosa"] = get_librosa_features(train_df, val_df, test_df, cfg.features)
    if "mfcc_sequence" in needs:
        features["mfcc_sequence"] = get_mfcc_sequences(train_df, val_df, test_df, cfg.features)
    if "spectrogram" in needs:
        features["spectrogram"] = get_spectrograms(train_df, val_df, test_df, cfg.features)
    if "embeddings" in needs:
        features["embeddings"] = get_embeddings(train_df, val_df, test_df, cfg.features)

    # t-SNE визуализация — все 900 треков, оба источника фич
    for feat_name in ("embeddings", "librosa"):
        if feat_name not in features:
            continue
        f = features[feat_name]
        X_all = np.concatenate([f["train"][0], f["val"][0], f["test"][0]])
        y_all = np.concatenate([f["train"][1], f["val"][1], f["test"][1]])
        label = cfg.features.embeddings_model if feat_name == "embeddings" else "librosa"
        plot_embeddings_tsne(X_all, y_all, title=label, out_dir=out_dir)

    # --- Training & evaluation ---
    print("\n[3/3] Training models...")
    results: dict[str, dict] = {}
    models_cfg = cfg.models

    if models_cfg.random_forest.enabled:
        results["random_forest"] = _run_model(
            "random_forest", RandomForestModel(models_cfg.random_forest),
            features[models_cfg.random_forest.input],
            out_dir, cfg,
        )

    if models_cfg.knn.enabled:
        results["knn"] = _run_model(
            "knn", KNNModel(models_cfg.knn),
            features[models_cfg.knn.input],
            out_dir, cfg,
        )

    if models_cfg.knn_librosa.enabled:
        results["knn_librosa"] = _run_model(
            "knn_librosa", KNNModel(models_cfg.knn_librosa),
            features[models_cfg.knn_librosa.input],
            out_dir, cfg,
        )

    if models_cfg.mlp.enabled:
        results["mlp"] = _run_model(
            "mlp", MLPModel(models_cfg.mlp),
            features[models_cfg.mlp.input],
            out_dir, cfg, device=device,
        )

    if models_cfg.mlp_emb.enabled:
        results["mlp_emb"] = _run_model(
            "mlp_emb", MLPModel(models_cfg.mlp_emb),
            features[models_cfg.mlp_emb.input],
            out_dir, cfg, device=device,
        )

    if models_cfg.cnn.enabled:
        results["cnn"] = _run_model(
            "cnn", CNNModel(models_cfg.cnn),
            features["spectrogram"],
            out_dir, cfg, device=device,
        )

    if models_cfg.lstm.enabled:
        results["lstm"] = _run_model(
            "lstm", LSTMModel(models_cfg.lstm),
            features["mfcc_sequence"],
            out_dir, cfg, device=device,
        )

    plot_comparison_table(results, out_dir)
    return results


def _needed_inputs(cfg: Config) -> set[str]:
    needed = set()
    mc = cfg.models
    for m, input_type in [
        (mc.random_forest, mc.random_forest.input),
        (mc.knn, mc.knn.input),
        (mc.knn_librosa, mc.knn_librosa.input),
        (mc.mlp, mc.mlp.input),
        (mc.mlp_emb, mc.mlp_emb.input),
        (mc.cnn, "spectrogram"),
        (mc.lstm, "mfcc_sequence"),
    ]:
        if m.enabled:
            needed.add(input_type)
    return needed


def _run_model(name: str, model, feat: dict, out_dir: Path, cfg: Config, device=None) -> dict:
    print(f"\n  → {name}")

    X_tr, y_tr = feat["train"]
    X_val, y_val = feat["val"]
    X_te, y_te = feat["test"]

    fit_kwargs = {}
    if device is not None:
        fit_kwargs["device"] = device

    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, **fit_kwargs)

    y_pred = model.predict(X_te)
    metrics = compute_metrics(y_te, y_pred)
    print(f"     acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}")

    plot_confusion_matrix(y_te, y_pred, name, out_dir)

    if cfg.output.save_models:
        model.save(out_dir / "models" / f"{name}.pt")

    return metrics
