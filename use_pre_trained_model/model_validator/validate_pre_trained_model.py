from use_pre_trained_model.model_validator.model_validator import ModelValidator

# FullSubNet_plus/use_pre_model2.py

import hydra
from omegaconf import DictConfig
import torch

from use_pre_trained_model.model_validator.config.schema import Config
from dataset import AudioDataset
import utils
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)

        # Get device
    device = utils.get_device(config.model_validator.device)
    print(f"Using device: {device}")

    dataset = AudioDataset(config.data_config.dataset)

    print(f"Total sample pairs in dataset: {len(dataset)}")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data_loader.batch_size,  # Adjust based on your GPU memory
        shuffle=config.data_loader.shuffle,
        num_workers=config.data_loader.num_workers,
        pin_memory=config.data_loader.pin_memory
    )

    # Initialize validator
    validator = ModelValidator(config.model_validator)

    # Run validation
    metrics = validator.validate_dataloader(dataloader)

    # Save results
    validator.save_metrics(metrics, "validation_results.json")


if __name__ == "__main__":
    main()
