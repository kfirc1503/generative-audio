import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from pathlib import Path
import utils

def load_audio_from_folder(sample_folder, sample_rate=16000):
    """Load full audio files and cut to first 2 seconds"""
    audio_data = {}
    
    # Load clean audio (full version)
    clean_path = sample_folder / "clean_full.wav"
    if clean_path.exists():
        audio, sr = torchaudio.load(clean_path)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        # Cut to first 2 seconds
        samples_to_keep = 2 * sample_rate
        audio_data['clean'] = audio.squeeze(0)[:samples_to_keep]
    else:
        print(f"Warning: {clean_path} not found.")
    
    # Load PC variations (full versions)
    pc_dirs = sorted([d for d in sample_folder.iterdir() if d.is_dir() and d.name.startswith('pc_')])
    
    for pc_dir in pc_dirs:
        pc_num = pc_dir.name.split('_')[1]
        alpha_files = sorted([
            f for f in pc_dir.iterdir() 
            if f.is_file() and f.suffix == '.wav' and '_full' in f.stem
        ])
        
        for alpha_file in alpha_files:
            alpha_str = alpha_file.stem.replace('alpha_', '').replace('_full', '')
            try:
                alpha_val = float(alpha_str)
                audio, sr = torchaudio.load(alpha_file)
                if sr != sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, sample_rate)
                # Cut to first 2 seconds
                samples_to_keep = 2 * sample_rate
                key = f'pc{pc_num}_alpha{alpha_val:.1f}'
                audio_data[key] = audio.squeeze(0)[:samples_to_keep]
            except ValueError:
                print(f"Could not parse alpha value from {alpha_file.name}, skipping.")
                continue
    
    return audio_data

def calculate_spectrogram(audio, n_fft=255, hop_length=128):
    """Calculate spectrogram with consistent parameters"""
    window = torch.hann_window(n_fft)
    spec = torch.stft(audio, 
                     n_fft=n_fft, 
                     hop_length=hop_length,
                     win_length=n_fft,
                     window=window,
                     return_complex=False)  # Returns [F, T, 2]
    
    # Calculate magnitude spectrogram in dB scale
    mag = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)
    mag_db = torch.log10(mag + 1e-6)
    return mag_db

def plot_spectrograms(audio_data, sample_output, n_dirs=3):
    """Plot spectrograms for all audio variations with consistent scale"""
    sample_rate = 16000
    n_fft = 255
    hop_length = 128
    
    # Calculate spectrograms for all audio files
    specs = {}
    for key, audio in audio_data.items():
        specs[key] = calculate_spectrogram(audio, n_fft, hop_length)
    
    # Create directory for spectrograms
    spec_dir = sample_output / 'spectrograms'
    spec_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame indices for the focused region
    center_frame = int(1.0 * sample_rate / hop_length)  # Frame at 1 second
    mask_frames = 18  # Number of frames in masked region
    
    # Calculate mask boundaries
    mask_start_frame = center_frame - mask_frames//2
    mask_end_frame = center_frame + (mask_frames - mask_frames//2)
    
    # Calculate context window (same duration as mask on each side)
    mask_duration = mask_end_frame - mask_start_frame
    context_start_frame = max(0, mask_start_frame - mask_duration)
    context_end_frame = min(specs['clean'].shape[1], mask_end_frame + mask_duration)
    
    # Calculate time values for x-axis
    time_per_frame = hop_length / sample_rate
    plot_start_time = context_start_frame * time_per_frame
    plot_end_time = context_end_frame * time_per_frame
    
    # Calculate frequency values for y-axis
    freqs = np.linspace(0, sample_rate/2, specs['clean'].shape[0])
    
    # Plot clean spectrogram
    if 'clean' in specs:
        plt.figure(figsize=(10, 6))
        plt.imshow(specs['clean'][:, context_start_frame:context_end_frame], 
                  origin='lower', aspect='auto',
                  vmin=-3, vmax=3,
                  extent=[plot_start_time, plot_end_time, freqs[0]/1000, freqs[-1]/1000])
        
        # Add mask boundary lines
        mask_start_time = mask_start_frame * time_per_frame
        mask_end_time = mask_end_frame * time_per_frame
        plt.axvline(x=mask_start_time, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=mask_end_time, color='r', linestyle='--', alpha=0.5)
        
        plt.colorbar(label='Log Magnitude (Normalized)')
        plt.title('Clean Audio Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (kHz)')
        plt.savefig(spec_dir / 'clean_spec.png')
        plt.close()
    
    # Plot PC variations
    for i in range(n_dirs):
        pc_num = i + 1
        # Get all variations for this PC
        pc_specs = {k: v for k, v in specs.items() if f'pc{pc_num}_' in k}
        
        if pc_specs:
            n_variations = len(pc_specs)
            fig, axes = plt.subplots(1, n_variations, figsize=(5*n_variations, 5))
            
            for idx, (key, spec) in enumerate(sorted(pc_specs.items())):
                im = axes[idx].imshow(spec[:, context_start_frame:context_end_frame], 
                                    origin='lower', aspect='auto',
                                    vmin=-3, vmax=3,
                                    extent=[plot_start_time, plot_end_time, freqs[0]/1000, freqs[-1]/1000])
                
                # Add mask boundary lines
                axes[idx].axvline(x=mask_start_time, color='r', linestyle='--', alpha=0.5)
                axes[idx].axvline(x=mask_end_time, color='r', linestyle='--', alpha=0.5)
                
                axes[idx].set_title(key)
                axes[idx].set_xlabel('Time (s)')
                if idx == 0:
                    axes[idx].set_ylabel('Frequency (kHz)')
                plt.colorbar(im, ax=axes[idx], label='Log Magnitude (Normalized)')
            
            plt.tight_layout()
            plt.savefig(spec_dir / f'pc{pc_num}_variations.png')
            plt.close()

def process_sample(sample_folder, output_dir):
    """Process a single sample folder"""
    sample_name = sample_folder.name
    print(f"\nProcessing {sample_name}...")
    
    # Load audio data
    audio_data = load_audio_from_folder(sample_folder)
    if not audio_data:
        print(f"No audio files found in {sample_folder}")
        return
    
    # Create sample output directory
    sample_output = output_dir / sample_name
    sample_output.mkdir(parents=True, exist_ok=True)
    
    # Plot spectrograms
    plot_spectrograms(audio_data, sample_output)
    
    print(f"Completed {sample_name}")

def main():
    input_dir = Path(r"C:\Users\kfir\Downloads\audio_samples\audio_samples")
    output_dir = Path("audio_plots")
    specific_sample = None  # Set to "sample_3" to process only that sample, or None for all
    
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return
    
    # Find sample folders
    if specific_sample:
        sample_folders = [input_dir / specific_sample]
        if not sample_folders[0].exists():
            print(f"Error: {sample_folders[0]} does not exist")
            return
    else:
        sample_folders = [d for d in input_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('sample_')]
    
    if not sample_folders:
        print("No sample folders found")
        return
    
    print(f"Found {len(sample_folders)} sample folders")
    
    # Process each sample
    for sample_folder in sorted(sample_folders):
        try:
            process_sample(sample_folder, output_dir)
        except Exception as e:
            print(f"Error processing {sample_folder.name}: {e}")
    
    print(f"\nAll done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
