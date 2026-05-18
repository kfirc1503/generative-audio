import torch
import hydra
from omegaconf import DictConfig
import pydantic
from pathlib import Path
import matplotlib.pyplot as plt

from nppc_audio.inpainting.validator.validator_comparison import ComparisonValidator, ComparisonValidatorConfig
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset
from utils import DataLoaderConfig, collate_fn


class Config(pydantic.BaseModel):
    comparison_validator_configuration: ComparisonValidatorConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    num_samples_to_validate: int = 5


@hydra.main(version_base=None, config_path="config", config_name="config_comparison")
def main(cfg: DictConfig):
    # Create config
    config = Config(**cfg)
    
    # Create validator
    print("\n" + "="*80)
    print("COMPARISON VALIDATOR - NPPC Regular vs NPPC Latent")
    print("="*80)
    validator = ComparisonValidator(config.comparison_validator_configuration)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = AudioInpaintingDataset(config.data_configuration)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Always use batch size 1 for validation
        shuffle=config.dataloader_configuration.shuffle,
        num_workers=config.dataloader_configuration.num_workers,
        pin_memory=config.dataloader_configuration.pin_memory,
        collate_fn=collate_fn
    )
    
    # Validate multiple samples
    num_samples = min(config.num_samples_to_validate, len(dataset))
    
    print(f"\n{'='*80}")
    print(f"Starting validation with {num_samples} samples")
    print(f"NPPC Regular checkpoint: {config.comparison_validator_configuration.nppc_checkpoint_path}")
    print(f"NPPC Latent checkpoint: {config.comparison_validator_configuration.nppc_latent_checkpoint_path}")
    print(f"Results will be saved to: {config.comparison_validator_configuration.save_dir}")
    print(f"Number of PC directions to plot: {config.comparison_validator_configuration.max_dirs_to_plot if config.comparison_validator_configuration.max_dirs_to_plot else 'all'}")
    print(f"{'='*80}\n")
    
    save_dir = Path(config.comparison_validator_configuration.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for i, (masked_spec, mask, clean_spec, masked_audio, metadata) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        print(f"\n{'='*80}")
        print(f"Processing sample {i + 1}/{num_samples}")
        print(f"{'='*80}")
        
        results = validator.validate_sample(
            masked_spec,
            mask,
            clean_spec,
            masked_audio,
            metadata,
            config.data_configuration.sub_sample_length_seconds,
            i
        )
        
        # Save plots for both models
        nppc_regular_fig = results['nppc_regular']['figure']
        nppc_latent_fig = results['nppc_latent']['figure']
        
        # Save to separate directories
        regular_save_path = save_dir / f"sample_{i}" / "nppc_regular" / f"comparison_sample_{i}.png"
        regular_save_path.parent.mkdir(parents=True, exist_ok=True)
        nppc_regular_fig.savefig(regular_save_path, dpi=150, bbox_inches='tight')
        plt.close(nppc_regular_fig)
        
        latent_save_path = save_dir / f"sample_{i}" / "nppc_latent" / f"comparison_sample_{i}.png"
        latent_save_path.parent.mkdir(parents=True, exist_ok=True)
        nppc_latent_fig.savefig(latent_save_path, dpi=150, bbox_inches='tight')
        plt.close(nppc_latent_fig)
        
        # Save multi-level figure if available
        multi_level_save_path = None
        if 'nppc_multi_level' in results:
            nppc_multi_level_fig = results['nppc_multi_level']['figure']
            # Use v2 folder if configured for v2 mode
            multi_level_folder = "nppc_multi_level_v2" if config.comparison_validator_configuration.multi_level_validation_mode == "v2" else "nppc_multi_level"
            multi_level_save_path = save_dir / f"sample_{i}" / multi_level_folder / f"comparison_sample_{i}.png"
            multi_level_save_path.parent.mkdir(parents=True, exist_ok=True)
            nppc_multi_level_fig.savefig(multi_level_save_path, dpi=150, bbox_inches='tight')
            plt.close(nppc_multi_level_fig)
        
        print(f"\nSample {i} results saved:")
        print(f"  - NPPC Regular: {regular_save_path.parent}")
        print(f"  - NPPC Latent: {latent_save_path.parent}")
        if multi_level_save_path:
            print(f"  - NPPC Multi-Level: {multi_level_save_path.parent}")
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE!")
    print(f"All results saved in: {save_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

