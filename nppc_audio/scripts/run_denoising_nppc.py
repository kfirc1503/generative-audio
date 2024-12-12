import hydra
from omegaconf import DictConfig

# import sys
# import os
#
# # Add the parent directory of nppc_audio to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nppc_audio.trainer import NPPCAudioTrainer , NPPCAudioTrainerConfig
from config.schema import Config
from utils import get_device


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)
    # Get device
    device = get_device(config.nppc_audio_trainer_configuration.device)
    #device = utils.get_device(config.model_validator.device)
    print(f"Using device: {device}")

    trainer = NPPCAudioTrainer(config.nppc_audio_trainer_configuration)
    trainer.train(config.n_steps)



if __name__ == "__main__":
    main()
