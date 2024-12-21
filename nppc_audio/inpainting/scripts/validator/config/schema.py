import pydantic
from nppc_audio.inpainting.validator.validator_restoration_model import InpaintingModelValidator,InpaintingModelValidatorConfig
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset
from use_pre_trained_model.model_validator.config.schema import DataLoaderConfig




class Config(pydantic.BaseModel):
    model_validator_configuration: InpaintingModelValidatorConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig



