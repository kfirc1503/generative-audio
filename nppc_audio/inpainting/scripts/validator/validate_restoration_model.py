import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from nppc_audio.inpainting.validator.validator_restoration_model import InpaintingModelValidator,InpaintingModelValidatorConfig
from dataset.audio_dataset_inpainting import AudioInpaintingDataset
from nppc_audio.inpainting.scripts.validator.config.schema import Config

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Create validator config
    config = Config(**cfg)


    # Create validator
    validator = InpaintingModelValidator(config.model_validator_configuration)

    # Create dataset
    dataset = AudioInpaintingDataset(config.data_configuration)

    # Create dataloader for Validation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataloader_configuration.batch_size,  # Adjust based on your GPU memory
        shuffle=config.dataloader_configuration.shuffle,
        num_workers=config.dataloader_configuration.num_workers,
        pin_memory=config.dataloader_configuration.pin_memory
    )

    # Validate multiple samples
    total_mse = 0
    total_mae = 0
    num_samples = min(20, len(dataset))  # Validate on first 10 samples

    print(f"\nValidating model from checkpoint: {config.model_validator_configuration.checkpoint_path}")
    print(f"Number of samples to validate: {num_samples}")
    print("\nProcessing samples...")

    for i, (masked_spec, mask, clean_spec) in enumerate(dataloader):
        if i >= num_samples:
            break

        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, clean_spec.shape[1], clean_spec.shape[2], -1)

        results = validator.validate_sample(
            masked_spec,
            mask,
            clean_spec,
            config.data_configuration.sub_sample_length_seconds
        )

        total_mse += results['mse']
        total_mae += results['mae']

        # Save individual sample results
        save_dir = Path(config.model_validator_configuration.save_dir)
        save_dir.mkdir(exist_ok=True)
        results['figure'].savefig(save_dir / f"spectrogram_comparison_sample_{i}.png")

        print(f"Sample {i}:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print()

    # Calculate and print average metrics
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples

    print("\nValidation Results:")
    print(f"Average MSE across {num_samples} samples: {avg_mse:.6f}")
    print(f"Average MAE across {num_samples} samples: {avg_mae:.6f}")
    print(f"\nResults saved in: {config.model_validator_configuration.save_dir}")


if __name__ == "__main__":
    main()