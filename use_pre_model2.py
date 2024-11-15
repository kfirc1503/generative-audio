from pydantic import BaseModel
import FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus as fullsubnet_plus
from pathlib import Path
import torch
import torchaudio
import tarfile
import io
import utils
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus,FullSubNetPlusConfig
from dataset import AudioDataset
from torch.utils.data import DataLoader

from use_pre_model import enhanced_waveform


def main():
    # Load the pre-trained model using the utility function
    subNetPlusConfig = FullSubNetPlusConfig()
    model_path = Path("./FullSubNet_plus/best_model.tar")
    model = utils.load_pretrained_model(model_path, subNetPlusConfig)

    # Define batch size and number of workers
    batch_size = 8
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths to clean and noisy audio directories
    clean_path = "./FullSubNet_plus/data/clean"
    noisy_path = "./FullSubNet_plus/data/noisy"
    orig_length = 48000
    # Create save directory if it doesn't exist
    enhanced_path = Path("./FullSubNet_plus/data/enhanced")
    enhanced_path.mkdir(parents=True, exist_ok=True)

    # Create an instance of the AudioDataset
    audio_dataset = AudioDataset(clean_path=clean_path, noisy_path=noisy_path)

    # Create DataLoader
    data_loader = DataLoader(
        dataset=audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    model.eval()  # Set model to evaluation mode

    # Process batches
    for batch_idx, batch in enumerate(data_loader):
        clean_waveforms, noisy_waveforms = batch
        
        # Move tensors to device if using GPU
        clean_waveforms = clean_waveforms.to(device)
        noisy_waveforms = noisy_waveforms.to(device)
        noisy_mag, noisy_real, noisy_imag = utils.prepare_input_from_waveform(noisy_waveforms)
        model = model.to(device)

        # Process batch through model
        with torch.no_grad():
            enhanced_masks = model(noisy_mag,noisy_real,noisy_imag)
            enhanced_waveforms = utils.model_outputs_to_waveforms(enhanced_masks,noisy_real,noisy_imag, orig_length)
            #TODO create a function that go back from stft to samples
        # Save enhanced audio files using the enhanced_path defined earlier
        for i in range(enhanced_waveforms.shape[0]):
            file_name = f"enhanced_batch{batch_idx}_sample{i}.wav"
            save_path = enhanced_path / file_name
            
            # Move back to CPU for saving
            enhanced_audio = enhanced_waveforms[i].cpu()
            enhanced_audio = enhanced_audio.unsqueeze(0)
            torchaudio.save(
                save_path.as_posix(),  # Convert Path to string using as_posix()
                enhanced_audio,
                sample_rate=16000
            )

if __name__ == "__main__":
    main()
