from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import FeaturesConfig

_MERT_MODEL_ID = "m-a-p/MERT-v1-95M"


def _load_mert():
    """Загружает модель один раз, кэширует в модуле."""
    import torch
    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained(_MERT_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(_MERT_MODEL_ID, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device


_mert_cache: tuple | None = None


def _get_mert():
    global _mert_cache
    if _mert_cache is None:
        print(f"  Loading MERT model ({_MERT_MODEL_ID})...")
        _mert_cache = _load_mert()
    return _mert_cache


def _cache_path(audio_path: str, model_name: str, cache_dir: Path) -> Path:
    key = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return cache_dir / f"{key}_emb_{model_name}.npy"


def _extract_mert(audio_path: str) -> np.ndarray:
    import torch
    import soundfile as sf

    processor, model, device = _get_mert()

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # MERT ожидает sr=24000
    if sr != 24000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

    inputs = processor(audio, sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Усредняем последний hidden state по временной оси → (D,)
    last_hidden = outputs.last_hidden_state.squeeze(0)  # (T, D)
    return last_hidden.mean(dim=0).cpu().numpy().astype(np.float32)


def get_embeddings(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: FeaturesConfig
) -> dict:
    if cfg.embeddings_model != "mert":
        raise ValueError(
            f"Unknown embeddings model: '{cfg.embeddings_model}'. "
            "Only 'mert' is supported (openl3 requires Python < 3.12)."
        )

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_split(df: pd.DataFrame, name: str):
        X_list, y_list = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"embeddings/mert [{name}]"):
            cp = _cache_path(row["audio_path"], "mert", cache_dir)
            if cp.exists():
                emb = np.load(cp)
            else:
                emb = _extract_mert(row["audio_path"])
                np.save(cp, emb)
            X_list.append(emb)
            y_list.append(row["label"])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    return {
        "train": extract_split(train, "train"),
        "val": extract_split(val, "val"),
        "test": extract_split(test, "test"),
    }
