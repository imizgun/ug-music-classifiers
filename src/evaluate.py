from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from src.dataset import LABEL_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_dir: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_comparison_table(results: dict[str, dict], out_dir: Path) -> None:
    """results: {model_name: {"accuracy": ..., "f1_macro": ...}}"""
    import pandas as pd

    df = pd.DataFrame(results).T[["accuracy", "f1_macro", "f1_weighted"]]
    df = df.sort_values("f1_macro", ascending=False)
    df.columns = ["Accuracy", "F1 macro", "F1 weighted"]

    print("\n" + "=" * 55)
    print("MODEL COMPARISON (test set)")
    print("=" * 55)
    print(df.to_string(float_format="{:.4f}".format))
    print("=" * 55 + "\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    df[["Accuracy", "F1 macro"]].plot(kind="bar", ax=ax, rot=30)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "comparison.png", dpi=150)
    plt.close(fig)

    df.to_csv(out_dir / "results.csv")


def plot_embeddings_tsne(
    X: np.ndarray, y: np.ndarray, title: str, out_dir: Path
) -> None:
    from sklearn.manifold import TSNE

    print(f"  Computing t-SNE for {title}...")
    coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    for i, label in enumerate(LABEL_NAMES):
        mask = y == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i], label=label, alpha=0.6, s=20)
    ax.legend()
    ax.set_title(f"t-SNE — {title}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"tsne_{title.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
