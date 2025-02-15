import pydantic
from dataset.audio_dataset_inpainting import AudioInpaintingConfig
from nppc_audio.inpainting.networks.unet import UNetConfig,RestorationWrapper
from nppc_audio.inpainting.nppc.nppc_model import NPPCModel,NPPCModelConfig
from nppc_audio.inpainting.trainer.nppc_trainer import NPPCAudioInpaintingTrainerConfig
from utils import DataLoaderConfig

class Config(pydantic.BaseModel):
    inpainting_nppc_training_configuration: NPPCAudioInpaintingTrainerConfig
    checkpoint_dir: str
    n_steps: int = None
    n_epochs: int = None
    validation_data_configuration: AudioInpaintingConfig = None
