# FullSubNet_plus/config/schema.py

from typing import List
from pydantic import BaseModel, Field
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNetPlusConfig


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 256
    sr: int = 16000
    batch_size: int = 8
    num_workers: int = 4

class PreTrainedModelConfig(BaseModel):
    checkpoint_path: str
    device: str = "cuda"
    model: FullSubNetPlusConfig

class ModelConfig(BaseModel):
    """Model configuration"""
    checkpoint_path: str
    device: str = "cuda"
    sb_num_neighbors: int = 15
    fb_num_neighbors: int = 0
    num_freqs: int = 257
    look_ahead: int = 2
    sequence_model: str = "LSTM"
    fb_output_activate_function: str = "ReLU"
    sb_output_activate_function: bool = False
    channel_attention_model: str = "TSSE"
    fb_model_hidden_size: int = 512
    sb_model_hidden_size: int = 384
    weight_init: bool = False
    norm_type: str = "offline_laplace_norm"
    num_groups_in_drop_band: int = 2
    kersize: List[int] = Field(default_factory=lambda: [3, 5, 10])
    subband_num: int = 1

class Config(BaseModel):
    """Main configuration"""
    audio: AudioConfig = AudioConfig()
    pre_trained_model: PreTrainedModelConfig