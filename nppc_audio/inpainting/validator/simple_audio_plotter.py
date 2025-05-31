import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from pathlib import Path
import utils
import librosa

def load_audio_from_folder(sample_folder, sample_rate=16000):
    """Load audio files (non-full versions) and calculate pitch."""
    audio_data = {}
    
    # Load clean audio (non-full version)
    clean_path = sample_folder / "clean.wav"
    if clean_path.exists():
        audio, sr = torchaudio.load(clean_path)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio_data['clean'] = audio.squeeze(0)
    else:
        print(f"Warning: {clean_path} not found.")
    
    # Load PC variations (non-full versions)
    pc_dirs = []
    for d in sample_folder.iterdir():
        if d.is_dir() and d.name.startswith('pc_'):
            try:
                # Extract PC number, ignoring any additional text
                pc_num = int(d.name.split('_')[1].split()[0].split('-')[0])
                pc_dirs.append((pc_num, d))
            except (IndexError, ValueError):
                print(f"Warning: Skipping malformed PC directory: {d.name}")
                continue
    
    # Sort by PC number
    pc_dirs.sort()
    
    for _, pc_dir in pc_dirs:
        alpha_files = sorted([
            f for f in pc_dir.iterdir() 
            if f.is_file() and f.suffix == '.wav' and '_full' not in f.stem
        ])
        
        for alpha_file in alpha_files:
            alpha_str = alpha_file.stem.replace('alpha_', '')
            try:
                alpha_val = float(alpha_str)
                audio, sr = torchaudio.load(alpha_file)
                if sr != sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, sample_rate)

                # Use the clean pc_num from our earlier parsing
                pc_num = int(pc_dir.name.split('_')[1].split()[0].split('-')[0])
                key = f'pc{pc_num}_alpha{alpha_val:.1f}'
                audio_data[key] = audio.squeeze(0)
            except ValueError:
                print(f"Could not parse alpha value from {alpha_file.name}, skipping.")
                continue
    
    return audio_data

def plot_pitch_comparison(audio_variations: dict, sample_rate: int = 16000, save_dir=None, sample_idx=None):
    """
    Plot pitch contours for each PC direction in subplots with a single shared legend.
    """
    if not audio_variations or 'clean' not in audio_variations:
        print("Clean audio not found in audio_variations. Skipping pitch plot.")
        return None

    # Get clean audio and calculate its pitch
    clean_audio = audio_variations['clean']
    clean_np = clean_audio.squeeze().numpy()
    if clean_np.ndim > 1:
        clean_np = clean_np[0]
    
    f0_clean, _, _ = librosa.pyin(
        clean_np,
        fmin=80,
        fmax=400,
        sr=sample_rate
    )
    times = librosa.times_like(f0_clean)

    # Get unique PC numbers and alphas
    pc_nums = sorted(list(set([int(k.split('pc')[1].split('_')[0]) for k in audio_variations.keys() if k.startswith('pc')])))
    if not pc_nums:
        print("No PC variations found.")
        return None

    alphas = sorted(list(set([float(k.split('alpha')[1]) for k in audio_variations.keys() if 'alpha' in k])))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    # Create figure with subplots
    n_pcs = len(pc_nums)
    fig, axes = plt.subplots(1, n_pcs, figsize=(5*n_pcs, 4))
    if n_pcs == 1:
        axes = [axes]

    # Store lines for legend
    legend_lines = []
    legend_labels = []

    # First line will be clean audio (black)
    clean_line = axes[0].plot(times, f0_clean, color='black', linewidth=2)[0]
    legend_lines.append(clean_line)
    legend_labels.append('Clean')

    # Plot each PC direction
    for idx, pc_num in enumerate(pc_nums):
        ax = axes[idx]
        
        # Plot clean reference (black)
        ax.plot(times, f0_clean, color='black', linewidth=2)
        
        # Plot variations
        for alpha_idx, alpha in enumerate(alphas):
            key = f'pc{pc_num}_alpha{alpha:.1f}'
            if key in audio_variations:
                audio = audio_variations[key]
                audio_np = audio.squeeze().numpy()
                if audio_np.ndim > 1:
                    audio_np = audio_np[0]

                f0, _, _ = librosa.pyin(
                    audio_np,
                    fmin=80,
                    fmax=400,
                    sr=sample_rate
                )
                line = ax.plot(times, f0, color=colors[alpha_idx], alpha=0.7, linewidth=2)[0]
                
                # Only add to legend from first subplot
                if idx == 0:
                    legend_lines.append(line)
                    legend_labels.append(f'α={alpha:.1f}')

        ax.set_title(f'PC {pc_num}', fontsize=14)
        ax.set_ylabel('Frequency (Hz)' if idx == 0 else '', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.grid(True)
        ax.tick_params(labelsize=10)

    # Add single legend outside plots
    fig.legend(legend_lines, legend_labels, 
              loc='center right', 
              bbox_to_anchor=(1.01, 0.5),  # Reduced spacing between plots and legend
              fontsize=12)

    # Adjust layout to make room for legend while keeping plots tight
    plt.tight_layout()
    plt.subplots_adjust(right=0.90, wspace=0.15)  # Adjusted right margin and reduced space between subplots

    # Save if directory provided
    if save_dir is not None:
        save_path = Path(save_dir) / f"sample_{sample_idx}" / "pitch_contours"
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f'pitch_comparison.png', bbox_inches='tight', dpi=300)

    return fig

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
    
    # Plot pitch comparison
    # Pass sample_name as sample_idx for directory naming inside plot_pitch_comparison
    pitch_fig = plot_pitch_comparison(audio_data, sample_rate=16000, save_dir=output_dir, sample_idx=sample_name)
    if pitch_fig:
        # If you want to save the main combined plot from process_sample as well:
        # main_pitch_plot_path = sample_output / "overall_pitch_comparison.png"
        # pitch_fig.savefig(main_pitch_plot_path)
        plt.close(pitch_fig) # Close the figure after saving or if not needed further
    
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
