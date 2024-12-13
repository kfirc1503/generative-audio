import pydantic
from typing import Optional
from nppc_audio.validator import NPPCAudioValidatorConfig, NPPCAudioValidator


class Config(pydantic.BaseModel):
    nppc_audio_validator_configuration: NPPCAudioValidatorConfig
    clean_wav_path: str
    noisy_wav_path: str
    save_dor: Optional[str]


