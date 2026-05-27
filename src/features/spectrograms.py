from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from src.config import FeaturesConfig


def _cache_path(audio_path: str, cache_dir: Path) -> Path:
    key = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return cache_dir / f"{key}_spec.npy"


def extract_spectrogram(audio_path: str, cfg: FeaturesConfig) -> np.ndarray:
    """Возвращает Mel-спектрограмму (1, H, W) нормализованную в [0, 1]."""
    y, sr = librosa.load(audio_path, sr=cfg.sr, duration=cfg.duration, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg.n_mels, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Ресайз до нужного размера
    from PIL import Image
    H, W = cfg.spectrogram_size
    img = Image.fromarray(mel_db).resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Нормализация в [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr[np.newaxis, :, :]  # (1, H, W)


def _augment_spectrogram(spec: np.ndarray) -> np.ndarray:
    """SpecAugment: случайное маскирование по времени и частоте."""
    spec = spec.copy()
    _, H, W = spec.shape

    # Frequency masking
    f = np.random.randint(0, H // 8)
    f0 = np.random.randint(0, H - f)
    spec[:, f0:f0 + f, :] = 0.0

    # Time masking
    t = np.random.randint(0, W // 8)
    t0 = np.random.randint(0, W - t)
    spec[:, :, t0:t0 + t] = 0.0

    return spec


def get_spectrograms(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: FeaturesConfig
) -> dict:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_split(df: pd.DataFrame, name: str, augment: bool):
        X_list, y_list = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"spectrograms [{name}]"):
            cp = _cache_path(row["audio_path"], cache_dir)
            if cp.exists():
                spec = np.load(cp)
            else:
                spec = extract_spectrogram(row["audio_path"], cfg)
                np.save(cp, spec)

            if augment:
                spec = _augment_spectrogram(spec)

            X_list.append(spec)
            y_list.append(row["label"])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    return {
        "train": extract_split(train, "train", augment=True),
        "val": extract_split(val, "val", augment=False),
        "test": extract_split(test, "test", augment=False),
    }
