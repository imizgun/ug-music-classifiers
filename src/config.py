from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel


class SplitConfig(BaseModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42


class DataConfig(BaseModel):
    audio_dir: str
    labels_file: str
    label_column: str = "Quadrant"
    id_column: str = "Song ID"
    split: SplitConfig = SplitConfig()


class FeaturesConfig(BaseModel):
    sr: int = 22050
    duration: int = 30
    n_mfcc: int = 20
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    spectrogram_size: tuple[int, int] = (128, 128)
    lstm_max_frames: int = 256
    cache_dir: str = "features/cache"
    embeddings_model: str = "openl3"


class RandomForestConfig(BaseModel):
    enabled: bool = True
    input: str = "librosa"
    n_estimators: int = 200
    max_depth: Optional[int] = None


class KNNConfig(BaseModel):
    enabled: bool = True
    input: str = "embeddings"
    k: int = 5
    metric: str = "cosine"


class MLPConfig(BaseModel):
    enabled: bool = True
    input: str = "librosa"
    hidden_layers: list[int] = [256, 128, 64]
    dropout: float = 0.3
    lr: float = 0.001
    epochs: int = 50
    batch_size: int = 32


class CNNConfig(BaseModel):
    enabled: bool = True
    input: str = "spectrogram"
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 32


class LSTMConfig(BaseModel):
    enabled: bool = True
    input: str = "mfcc_sequence"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 0.001
    epochs: int = 50
    batch_size: int = 32


class ModelsConfig(BaseModel):
    random_forest: RandomForestConfig = RandomForestConfig()
    knn: KNNConfig = KNNConfig()
    knn_librosa: KNNConfig = KNNConfig(input="librosa")
    mlp: MLPConfig = MLPConfig()
    mlp_emb: MLPConfig = MLPConfig(input="embeddings")
    cnn: CNNConfig = CNNConfig()
    lstm: LSTMConfig = LSTMConfig()


class TrainingConfig(BaseModel):
    device: str = "auto"


class OutputConfig(BaseModel):
    dir: str = "outputs"
    save_models: bool = True


class Config(BaseModel):
    data: DataConfig
    features: FeaturesConfig = FeaturesConfig()
    models: ModelsConfig = ModelsConfig()
    training: TrainingConfig = TrainingConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
