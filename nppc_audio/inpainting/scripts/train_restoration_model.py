import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch

from nppc_audio.inpainting.networks.unet import UNetConfig
from dataset.audio_dataset_inpainting import AudioInpaintingConfig
from nppc_audio.inpainting.trainer.restoration_trainer import InpaintingTrainer, InpaintingTrainerConfig, OptimizerConfig
from nppc_audio.inpainting.scripts.config.schema import Config

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = Config(**cfg)
    trainer = InpaintingTrainer(config.inpainting_training_configuration)
    trainer.train(n_steps= config.n_steps,checkpoint_dir = config.checkpoint_dir)

if __name__ == "__main__":
    main()