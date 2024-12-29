import torch
import torch.nn as nn
import torch.optim as optim
import pydantic
import os
from datetime import datetime
import json
from tqdm.auto import tqdm
from nppc.auxil import LoopLoader
import matplotlib.pyplot as plt
from nppc_audio.inpainting.networks.unet import UNetConfig, RestorationWrapper
from dataset.audio_dataset_inpainting import AudioInpaintingDataset, AudioInpaintingConfig
from use_pre_trained_model.model_validator.config.schema import DataLoaderConfig
from utils import normalize_spectrograms, denormalize_spectrograms


# Preprocessing and Postprocessing utilities
def preprocess_log_magnitude(magnitude, eps=1e-6):
    """
    Convert magnitude spectrogram to normalized log-magnitude spectrogram.

    Args:
        magnitude (torch.Tensor): Input magnitude spectrogram.
        eps (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Normalized log-magnitude spectrogram.
        torch.Tensor: Mean of the log-magnitude spectrogram.
        torch.Tensor: Standard deviation of the log-magnitude spectrogram.
    """
    log_mag = torch.log(magnitude + eps)
    mean = log_mag.mean()
    std = log_mag.std()
    # normalized_log_mag = log_mag
    normalized_log_mag = (log_mag - mean) / std
    return normalized_log_mag, mean, std


class OptimizerConfig(pydantic.BaseModel):
    type: str
    args: dict


class InpaintingTrainerConfig(pydantic.BaseModel):
    """Configuration for Inpainting trainer"""
    model_configuration: UNetConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    # stft_configuration: StftConfig
    learning_rate: float = 1e-4
    device: str = "cuda"
    save_interval: int = 10
    log_interval: int = 100


class InpaintingTrainer(nn.Module):
    def __init__(self, config: InpaintingTrainerConfig):
        super().__init__()
        self.config = config
        # self.model = UNet(self.config.model_configuration).to(config.device)
        self.model = RestorationWrapper(self.config.model_configuration).to(config.device)
        self.device = config.device
        if config.device == 'cuda':
            # check if gpu is exist
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dataset and dataloader
        dataset = AudioInpaintingDataset(config.data_configuration)
        print(f"Total sample pairs in dataset: {len(dataset)}")

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.dataloader_configuration.batch_size,
            shuffle=config.dataloader_configuration.shuffle,
            num_workers=config.dataloader_configuration.num_workers,
            pin_memory=config.dataloader_configuration.pin_memory
        )
        self.step = 0

        # Initialize optimizer
        optimizer_class = getattr(optim, config.optimizer_configuration.type)
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **config.optimizer_configuration.args
        )

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints", save_flag=True):
        """Main training loop using LoopLoader"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize loss history
        loss_history = []

        # Create loop loader
        loop_loader = LoopLoader(
            dataloader=self.dataloader,
            n_steps=n_steps,
            n_epochs=n_epochs
        )

        # Training loop with progress bar
        pbar = tqdm(loop_loader, total=len(loop_loader))
        for batch in pbar:
            # Move batch to device
            masked_spec, mask, clean_spec = [x.to(self.device) for x in batch]
            batch = (masked_spec, mask, clean_spec)

            # Forward and backward pass
            loss, log_dict = self.base_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store loss
            loss_history.append(loss.item())

            # Update progress bar
            pbar.set_description(
                f'Loss: {loss.item():.4f}'
            )

            self.step += 1

        # Plot loss curve
        self.plot_loss_curve(loss_history)

        if save_flag:
            # Save final checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_final_{timestamp}.pt"
            )

            # Save metrics including loss history
            self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)
            self.save_checkpoint(final_checkpoint_path)

    def plot_loss_curve(self, loss_history):
        """Plot the training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()

    def base_step(self, batch):
        """Base training step for inpainting using a mask created from clean_spec."""
        # We now ignore the masked_spec from the dataset;
        # we generate our own masked_spec_norm from clean_spec_norm.

        # masked_spec, mask, clean_spec = batch
        masked_spec, mask, clean_spec = batch  # ignore masked_spec

        # 1) Normalize the clean spectrogram
        # clean_spec_norm, clean_mean, clean_std = normalize_spectrograms(clean_spec)

        # 2) Create the masked spectrogram from clean_spec_norm
        #    (instead of using the masked_spec from the dataset)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, clean_spec.shape[1], clean_spec.shape[2], -1)
        # calculate the mag of the clean and masked specs:
        masked_spec_mag = torch.sqrt(masked_spec[:, 0, :, :] ** 2 + masked_spec[:, 1, :, :] ** 2)
        masked_spec_mag = masked_spec_mag.unsqueeze(1)
        clean_spec_mag = torch.sqrt(clean_spec[:, 0, :, :] ** 2 + clean_spec[:, 1, :, :] ** 2)
        clean_spec_mag = clean_spec_mag.unsqueeze(1)
        clean_spec_mag_log, _, _ = preprocess_log_magnitude(clean_spec_mag)
        # clean_spec_mag_log = torch.log(clean_spec_mag)
        # normalized clean spec mag log
        masked_spec_mag_log = clean_spec_mag_log * mask[:, 0, :, :].unsqueeze(1)

        # Concatenate mask with masked spectrogram along channel dimension
        # model_input = torch.cat([masked_spec, mask[:,0,:,:].unsqueeze(1)], dim=1)

        # masked_spec_norm = clean_spec_norm * mask

        # 3) Forward pass
        # output = self.model(masked_spec, mask)
        output = self.model(masked_spec_mag_log, mask)
        # Compute loss in normalized space
        opposite_mask = 1 - mask
        masked_loss = (torch.abs(output - clean_spec_mag_log)) * opposite_mask
        # masked_loss = ((output - clean_spec_mag) ** 2) * mask
        loss = masked_loss.sum() / (opposite_mask.sum() + 1e-6)

        # Denormalize the output using clean spectrogram stats
        # output = denormalize_spectrograms(output_norm, clean_mean, clean_std)

        # Store logs
        log = {
            'clean_spec': clean_spec.detach(),
            'output': output.detach(),
            'loss': loss.detach()
        }

        return loss, log

    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint including model state, optimizer state, and training info

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _get_and_save_metrics(self, checkpoint_dir, log_dict, n_epochs, n_steps, timestamp):
        """Save training metrics to JSON file"""
        final_metrics = {
            'timestamp': timestamp,
            'total_steps': self.step,
            'final_loss': log_dict['loss'].item(),
            'training_config': {
                'n_steps': n_steps,
                'n_epochs': n_epochs,
                'learning_rate': self.config.learning_rate,
                'device': self.config.device,
                'batch_size': self.config.dataloader_configuration.batch_size
            }
        }
        metrics_path = os.path.join(
            checkpoint_dir,
            f"metrics_final_{timestamp}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
