from model_validator import ModelValidator , ModelValidatorConfig


# FullSubNet_plus/use_pre_model2.py

import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader

from config.model_validator_schema import Config
from dataset import AudioDataset
from utils import prepare_input, model_outputs_to_waveforms, load_pretrained_model, prepare_input_from_waveform

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

@hydra.main(version_base=None, config_path="config", config_name="model_validator_config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)


    dataset = AudioDataset(config.data_config.dataset)

    print(f"Total sample pairs in dataset: {len(dataset)}")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= config.audio.batch_size,  # Adjust based on your GPU memory
        shuffle=False,
        num_workers= config.audio.num_workers,
        pin_memory=True
    )

    # Initialize validator
    validator = ModelValidator(config.model_validator)

    # Run validation
    # metrics_like_article = validator.validate_dataloader_like_article(dataloader)
    metrics = validator.validate_dataloader(dataloader)

    # Save results
    validator.save_metrics(metrics, "validation_results.json")
    # validator.save_metrics(metrics_like_article, "validation_results_like_article.json")


if __name__ == "__main__":
    main()