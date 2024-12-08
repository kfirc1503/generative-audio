import pydantic
from nppc_audio.nppc_model import NPPCModelConfig
from nppc_audio.trainer import NPPCAudioTrainerConfig

class Config(pydantic.BaseModel):
    nppc_audio_trainer_configuration: NPPCAudioTrainerConfig
    n_steps: int = 10
    n_epochs: int = None






