import hydra
from omegaconf import DictConfig
import torch

from dataset import AudioDataset
import utils
import os

from nppc_audio.trainer import NPPCAudioTrainer , NPPCAudioTrainerConfig
from nppc_audio.pc_wrapper import AudioPCWrapper , AudioPCWrapperConfig



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = NPPCAudioTrainerConfig(**cfg)
    # Get device
    device = utils.get_device(config.model_validator.device)
    print(f"Using device: {device}")

    trainer = NPPCAudioTrainer(config)
    trainer.train()



if __name__ == "__main__":
    main()
