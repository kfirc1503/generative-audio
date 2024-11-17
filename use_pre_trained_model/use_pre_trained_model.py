# FullSubNet_plus/use_pre_model2.py

import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader

from config.schema import Config
from dataset import AudioDataset
from utils import prepare_input, model_outputs_to_waveforms, load_pretrained_model, prepare_input_from_waveform
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)

    # Initialize model
    # model = FullSubNet_Plus(config.pre_trained_model.model)
    # # load pre trained model
    # model = FullSubNet_Plus(**config.model.dict(exclude={'checkpoint_path', 'device'}))
    model_path:str = config.pre_trained_model.checkpoint_path
    sub_net_plus_config = config.pre_trained_model.model
    model = load_pretrained_model(model_path, sub_net_plus_config)

    # checkpoint = torch.load(config.model.checkpoint_path, map_location="cpu")
    # model.load_state_dict(checkpoint["model"])
    device = torch.device(config.pre_trained_model.device)
    model.to(device)
    model.eval()

    # Setup data paths
    data_path = "C:/Kfir/repos/generative-audio/FullSubNet_plus/data"
    data_dir = Path(data_path)
    noisy_dir = data_dir / "noisy"
    clean_dir = data_dir / "clean"
    enhanced_dir = data_dir / "enhanced2"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = AudioDataset(noisy_path=noisy_dir,clean_path=clean_dir, sample_rate=config.audio.sr)
    dataloader = DataLoader(
        dataset,
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        shuffle=False
    )

    # Process audio files
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clean_waveforms, noisy_waveforms = batch
            clean_waveforms = clean_waveforms.to(device)
            noisy_waveforms = noisy_waveforms.to(device)
            # Prepare input
            noisy_mag, noisy_real, noisy_imag = prepare_input_from_waveform(noisy_waveforms)
            model = model.to(device)
            # Forward pass
            enhanced_masks = model(noisy_mag, noisy_real, noisy_imag)

            # Convert to waveform
            enhanced_wavs = model_outputs_to_waveforms(
                enhanced_masks,
                noisy_real,
                noisy_imag,
                orig_length= 48000
            )

            # Save enhanced audio
            for i in range(enhanced_wavs.shape[0]):
                save_path = enhanced_dir / f"enhanced_batch{batch_idx}_sample{i}.wav"
                torchaudio.save(
                    save_path,
                    enhanced_wavs[i].cpu().unsqueeze(0),
                    config.audio.sr
                )
                print(f"Saved enhanced audio to {save_path}")


if __name__ == "__main__":
    main()