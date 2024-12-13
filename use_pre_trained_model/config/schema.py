# FullSubNet_plus/config/schema.py

from pydantic import BaseModel
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNetPlusConfig
from dataset.dataset import AudioDataSetConfig

class AudioConfig(BaseModel):
    """Audio processing configuration"""
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 256
    sr: int = 16000
    batch_size: int = 8
    num_workers: int = 4


class PreTrainedModelDataConfig(BaseModel):
    dataset: AudioDataSetConfig
    data_path: str
    enhanced_dir_path: str

class PreTrainedModelConfig(BaseModel):
    checkpoint_path: str
    device: str = "cuda"
    model: FullSubNetPlusConfig


class Config(BaseModel):
    """Main configuration"""
    audio: AudioConfig = AudioConfig()
    pre_trained_data_model: PreTrainedModelDataConfig
    pre_trained_model: PreTrainedModelConfig
