import hydra
from omegaconf import DictConfig
from pathlib import Path

import pydantic
import torch
import matplotlib.pyplot as plt
from dataset.audio_dataset_inpainting import AudioInpaintingDataset, AudioInpaintingConfig
from nppc_audio.inpainting.validator.validator_nppc_latent_model import (
    NPPCLatentModelValidator, 
    NPPCLatentModelValidatorConfig
)
from utils import DataLoaderConfig, collate_fn


class Config(pydantic.BaseModel):
    model_validator_configuration: NPPCLatentModelValidatorConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    num_samples_to_validate: int = 5


@hydra.main(version_base=None, config_path="config", config_name="config_nppc_latent_space")
def main(cfg: DictConfig):
    # Create validator config
    config = Config(**cfg)

    # Create validator
    validator = NPPCLatentModelValidator(config.model_validator_configuration)

    # Create dataset
    dataset = AudioInpaintingDataset(config.data_configuration)

    # Create dataloader for Validation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Always use batch size 1 for validation
        shuffle=config.dataloader_configuration.shuffle,
        num_workers=config.dataloader_configuration.num_workers,
        pin_memory=config.dataloader_configuration.pin_memory,
        collate_fn=collate_fn
    )

    # Validate multiple samples
    num_samples = min(cfg.num_samples_to_validate, len(dataset))

    print(f"\nValidating NPPC Latent Space model from checkpoint: {config.model_validator_configuration.checkpoint_path}")
    print(f"Number of samples to validate: {num_samples}")
    print(f"Number of PC directions to plot: {config.model_validator_configuration.max_dirs_to_plot if config.model_validator_configuration.max_dirs_to_plot else 'all'}")
    print("\nProcessing samples...")

    save_dir = Path(config.model_validator_configuration.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, (masked_spec, mask, clean_spec, masked_audio, metadata) in enumerate(dataloader):
        if i >= num_samples:
            break

        print(f"Processing sample {i + 1}/{num_samples}")

        results = validator.validate_sample(
            masked_spec,
            mask,
            clean_spec,
            masked_audio,
            metadata,
            config.data_configuration.sub_sample_length_seconds,
            i
        )

        # Save plot
        results['figure'].savefig(save_dir / f"pc_directions_sample_{i}.png")
        plt.close(results['figure'])

    print(f"\nResults saved in: {save_dir}")


if __name__ == "__main__":
    main()

