# FullSubNet_plus/config/schema.py

from typing import List
from pydantic import BaseModel, Field
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNetPlusConfig
from dataset import AudioDataSetConfig
from use_pre_trained_model.model_validator import ModelValidatorConfig,ModelValidator


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 256
    sr: int = 16000
    batch_size: int = 8
    num_workers: int = 4

class DataConfig(BaseModel):
    dataset: AudioDataSetConfig
    data_path: str


class Config(BaseModel):
    """Main configuration"""
    audio: AudioConfig = AudioConfig()
    data_config: DataConfig
    model_validator: ModelValidatorConfig
