from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from src.config import FeaturesConfig


def _cache_path(audio_path: str, suffix: str, cache_dir: Path) -> Path:
    key = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return cache_dir / f"{key}_{suffix}.npy"


def extract_librosa_vector(audio_path: str, cfg: FeaturesConfig) -> np.ndarray:
    """Возвращает плоский вектор: mean+std по каждой фиче."""
    y, sr = librosa.load(audio_path, sr=cfg.sr, duration=cfg.duration, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc,
                                  n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=cfg.hop_length)
    rms = librosa.feature.rms(y=y, hop_length=cfg.hop_length)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    features = []
    for mat in [mfcc, chroma, centroid, rolloff, zcr, rms]:
        features.extend([mat.mean(axis=1), mat.std(axis=1)])
    features.append(np.array([float(np.atleast_1d(tempo)[0])]))

    return np.concatenate(features).astype(np.float32)


def extract_mfcc_sequence(audio_path: str, cfg: FeaturesConfig) -> np.ndarray:
    """Возвращает матрицу (lstm_max_frames, n_mfcc) с паддингом/обрезкой."""
    y, sr = librosa.load(audio_path, sr=cfg.sr, duration=cfg.duration, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc,
                                  n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    mfcc = mfcc.T  # (T, n_mfcc)

    T = cfg.lstm_max_frames
    if mfcc.shape[0] >= T:
        mfcc = mfcc[:T]
    else:
        pad = np.zeros((T - mfcc.shape[0], cfg.n_mfcc), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])

    return mfcc.astype(np.float32)


def _extract_and_cache(
    df: pd.DataFrame,
    cfg: FeaturesConfig,
    mode: str,        # "vector" | "sequence"
    split_name: str,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_list, y_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"librosa [{split_name}]"):
        cp = _cache_path(row["audio_path"], mode, cache_dir)
        if cp.exists():
            feat = np.load(cp)
        else:
            if mode == "vector":
                feat = extract_librosa_vector(row["audio_path"], cfg)
            else:
                feat = extract_mfcc_sequence(row["audio_path"], cfg)
            np.save(cp, feat)

        if augment:
            feat = feat + np.random.normal(0, 0.01, feat.shape).astype(np.float32)

        X_list.append(feat)
        y_list.append(row["label"])

    return np.array(X_list), np.array(y_list, dtype=np.int64)


def get_librosa_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: FeaturesConfig
) -> dict:
    X_tr, y_tr = _extract_and_cache(train, cfg, "vector", "train", augment=True)
    X_val, y_val = _extract_and_cache(val, cfg, "vector", "val")
    X_te, y_te = _extract_and_cache(test, cfg, "vector", "test")
    return {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}


def get_mfcc_sequences(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: FeaturesConfig
) -> dict:
    X_tr, y_tr = _extract_and_cache(train, cfg, "sequence", "train", augment=True)
    X_val, y_val = _extract_and_cache(val, cfg, "sequence", "val")
    X_te, y_te = _extract_and_cache(test, cfg, "sequence", "test")
    return {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}
