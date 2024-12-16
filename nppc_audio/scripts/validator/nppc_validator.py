import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from dataset.audio_dataset import AudioDataSetConfig, AudioDataset
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from config.schema import Config
from utils import get_device
#from nppc_audio.scripts.validator.config.schema import Config
from nppc_audio.validator import NPPCAudioValidatorConfig, NPPCAudioValidator

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)
    # read the clean and the noising audio:
    clean_audio, sr = sf.read(config.clean_wav_path)
    noisy_audio, sr = sf.read(config.noisy_wav_path)
    # Convert to torch tensors and add channel dimension
    clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)
    noisy_tensor = torch.FloatTensor(noisy_audio).unsqueeze(0)

    nppc_model_validator = NPPCAudioValidator(config.nppc_audio_validator_configuration)
    spec_list, enhanced_spec_db = nppc_model_validator.visualize_pc_spectrograms(noisy_tensor,clean_tensor,config.save_dir)



if __name__ == "__main__":
    main()




