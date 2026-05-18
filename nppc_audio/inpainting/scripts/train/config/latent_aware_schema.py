import pydantic
from nppc_audio.inpainting.trainer.latent_aware_restoration_trainer import LatentAwareRestorationTrainerConfig
from dataset.audio_dataset_inpainting import AudioInpaintingConfig


class LatentAwareConfig(pydantic.BaseModel):
    latent_aware_training_configuration: LatentAwareRestorationTrainerConfig
    checkpoint_dir: str
    n_steps: int = None
    n_epochs: int = None
    validation_data_configuration: AudioInpaintingConfig = None

