import hydra
from omegaconf import DictConfig

from nppc_audio.inpainting.trainer.restoration_trainer import InpaintingTrainer
from nppc_audio.inpainting.scripts.train.config.schema import Config

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = Config(**cfg)
    trainer = InpaintingTrainer(config.inpainting_training_configuration)
    trainer.train(n_steps= config.n_steps, n_epochs= config.n_epochs,checkpoint_dir = config.checkpoint_dir)

if __name__ == "__main__":
    main()