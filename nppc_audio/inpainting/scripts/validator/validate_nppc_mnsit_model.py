import hydra
from omegaconf import DictConfig
from pathlib import Path

import pydantic
import torch
import matplotlib.pyplot as plt
from nppc_audio.inpainting.validator.validator_nppc_mnist_model import NPPCMNISTValidator, NPPCMNISTValidatorConfig
from nppc_audio.inpainting.networks.unet import LatentEncoderConfig, UNetConfig

class Config(pydantic.BaseModel):
    model_validator_configuration: NPPCMNISTValidatorConfig


@hydra.main(version_base=None, config_path="config", config_name="config_nppc_mnist")
def main(cfg: DictConfig):
    # Create validator config
    config = Config(**cfg)

    # Create validator
    validator = NPPCMNISTValidator(config.model_validator_configuration)
    # validator.nppc_latent_model.latent_model_config = config.latent_model_config
    # validator.restoration_model.model_config = config.restoration_model_config

    print(f"\nValidating MNIST NPPC model from checkpoint: {config.model_validator_configuration.checkpoint_path}")
    # print(f"Number of samples to validate: {config.num_samples_to_validate}")
    print("\nProcessing samples...")

    save_dir = Path(config.model_validator_configuration.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Run validation
    validator.validate_samples()

    print(f"\nResults saved in: {save_dir}")


if __name__ == "__main__":
    main()