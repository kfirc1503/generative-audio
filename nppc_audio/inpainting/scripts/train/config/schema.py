import pydantic
from nppc_audio.inpainting.trainer.restoration_trainer import InpaintingTrainerConfig
from dataset.audio_dataset_inpainting import AudioInpaintingConfig


class Config(pydantic.BaseModel):
    inpainting_training_configuration: InpaintingTrainerConfig
    checkpoint_dir: str
    n_steps: int = None
    n_epochs: int = None
    validation_data_configuration: AudioInpaintingConfig = None



