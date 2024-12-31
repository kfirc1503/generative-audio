import pydantic
from nppc_audio.inpainting.trainer.restoration_trainer import InpaintingTrainerConfig



class Config(pydantic.BaseModel):
    inpainting_training_configuration: InpaintingTrainerConfig
    checkpoint_dir: str
    n_steps: int = None
    n_epochs: int = None



