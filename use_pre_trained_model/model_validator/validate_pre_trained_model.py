from pre_trained_model.model_validator.model_validator import ModelValidator

# FullSubNet_plus/use_pre_model2.py

import hydra
from omegaconf import DictConfig
import torch

from pre_trained_model.model_validator.config.schema import Config
from dataset import AudioDataset

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)

    dataset = AudioDataset(config.data_config.dataset)

    print(f"Total sample pairs in dataset: {len(dataset)}")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.audio.batch_size,  # Adjust based on your GPU memory
        shuffle=False,
        num_workers=config.audio.num_workers,
        pin_memory=True
    )

    # Initialize validator
    validator = ModelValidator(config.model_validator)

    # Run validation
    metrics = validator.validate_dataloader(dataloader)

    # Save results
    validator.save_metrics(metrics, "validation_results.json")


if __name__ == "__main__":
    main()
