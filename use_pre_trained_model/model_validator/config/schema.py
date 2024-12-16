# FullSubNet_plus/config/schema.py

from pydantic import BaseModel
from dataset.audio_dataset import AudioDataSetConfig

from use_pre_trained_model.model_validator.model_validator import ModelValidatorConfig


# class AudioConfig(BaseModel):
#     """Audio processing configuration"""
#     n_fft: int = 512
#     win_length: int = 512
#     hop_length: int = 256
#     sr: int = 16000
#     batch_size: int = 8
#     num_workers: int = 4


class DataLoaderConfig(BaseModel):
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = False

class DataConfig(BaseModel):
    dataset: AudioDataSetConfig
    data_path: str


class Config(BaseModel):
    """Main configuration"""
    data_loader: DataLoaderConfig = DataLoaderConfig()
    data_config: DataConfig
    model_validator: ModelValidatorConfig
