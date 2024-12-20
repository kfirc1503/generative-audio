import torch
import torch.nn as nn
import torch.optim as optim
import pydantic
import os
from datetime import datetime
import json
from tqdm.auto import tqdm
from nppc.auxil import LoopLoader

from nppc_audio.inpainting.networks.unet import UNet, UNetConfig, RestorationWrapper
from dataset.audio_dataset_inpainting import AudioInpaintingDataset, AudioInpaintingConfig
from use_pre_trained_model.model_validator.config.schema import DataLoaderConfig
from utils import StftConfig, audio_to_stft

class OptimizerConfig(pydantic.BaseModel):
    type: str
    args: dict


class InpaintingTrainerConfig(pydantic.BaseModel):
    """Configuration for Inpainting trainer"""
    model_configuration: UNetConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    stft_configuration: StftConfig
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

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints"):
        """
        Main training loop using LoopLoader

        Args:
            checkpoint_dir: Directory to save checkpoints
            n_steps: Number of training steps (optional)
            n_epochs: Number of training epochs (optional)

        Note: Must provide either n_steps or n_epochs
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

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

            # Update progress bar
            pbar.set_description(
                f'Loss: {loss.item():.4f}'
            )

            self.step += 1

        # Save final checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_final_{timestamp}.pt"
        )
        self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)
        self.save_checkpoint(final_checkpoint_path)

    def base_step(self, batch):
        """
        Base training step for inpainting

        Args:
            batch: Tuple containing:
                masked_spec: [B, 2, F, T]
                mask: [B, 1, F, T]
                clean_spec: [B, 2, F, T]
        Returns:
            tuple: (loss, log_dict)
        """
        #masked_spec, mask, clean_spec = batch
        masked_waveform , mask , clean_waveform = batch
        masked_waveform = masked_waveform.squeeze(1)
        clean_waveform =  clean_waveform.squeeze(1)

        masked_spec = audio_to_stft(masked_waveform,self.config.stft_configuration,self.device)
        clean_spec = audio_to_stft(clean_waveform,self.config.stft_configuration,self.device)
        # convert the audio into specs
        # Forward pass through model
        output = self.model(masked_spec, mask)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, clean_spec)

        # Store logs
        log = {
            'masked_spec': masked_spec.detach(),
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
