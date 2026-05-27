from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DataConfig, SplitConfig

LABEL_NAMES = ["Q1_Happy", "Q2_Angry", "Q3_Sad", "Q4_Calm"]
# Quadrant string → int index 0-3
LABEL_MAP = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3,
             "1": 0, "2": 1, "3": 2, "4": 3,
             1: 0, 2: 1, 3: 2, 4: 3}


def load_dataset(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.labels_file)

    audio_dir = Path(cfg.audio_dir)

    def resolve_path(row) -> str:
        song_id = str(row[cfg.id_column])
        quadrant = str(row[cfg.label_column])
        # попробуем Q1/song_id.mp3, потом просто song_id.mp3
        for candidate in [
            audio_dir / quadrant / f"{song_id}.mp3",
            audio_dir / f"{song_id}.mp3",
            audio_dir / quadrant / f"{song_id}.wav",
            audio_dir / f"{song_id}.wav",
        ]:
            if candidate.exists():
                return str(candidate)
        return ""

    df["audio_path"] = df.apply(resolve_path, axis=1)
    missing = (df["audio_path"] == "").sum()
    if missing:
        print(f"[dataset] Warning: {missing} audio files not found, skipping")
    df = df[df["audio_path"] != ""].reset_index(drop=True)

    df["label"] = df[cfg.label_column].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["audio_path", "label"]].copy()


def split_dataset(
    df: pd.DataFrame, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=cfg.test, random_state=cfg.seed, stratify=df["label"]
    )
    val_ratio = cfg.val / (cfg.train + cfg.val)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=cfg.seed, stratify=train_val["label"]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
