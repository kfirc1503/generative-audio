import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pydantic
import os
import tempfile
from pathlib import Path
from datetime import datetime
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
from typing import Optional, List
from nppc_audio.inpainting.networks.unet import UNetConfig, RestorationWrapper, UNet
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset, AudioInpaintingSample
from use_pre_trained_model.model_validator.config.schema import DataLoaderConfig
import utils
from nppc.auxil import LoopLoader
torch.cuda.empty_cache()


class OptimizerConfig(pydantic.BaseModel):
    type: str
    args: dict


class WandbConfig(pydantic.BaseModel):
    """Wandb configuration for loading pretrained model"""
    entity: str
    project: str
    artifact_name: str
    artifact_version: str = "latest"
    checkpoint_filename: str = "checkpoint_final.pt"


class LatentAwareRestorationTrainerConfig(pydantic.BaseModel):
    """Configuration for Latent-Aware Restoration trainer"""
    model_configuration: UNetConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    
    # Pretrained model loading: either local path OR wandb config
    pretrained_restoration_model_path: Optional[str] = None
    pretrained_wandb_config: Optional[WandbConfig] = None
    
    # Lambda for latent regularization term
    latent_reg_lambda: float = 1.0
    
    # Validation settings
    enable_periodic_validation: bool = True
    validation_interval: int = 500
    
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project_name: Optional[str] = "generative-audio"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_artifact_name: str = "latent_aware_restoration_model"


class LatentAwareRestorationTrainer(nn.Module):
    def __init__(self, config: LatentAwareRestorationTrainerConfig):
        super().__init__()
        self.config = config

        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project_name,
                name=config.wandb_run_name,
                config=config.model_dump(),
                tags=config.wandb_tags
            )

        # Setup device
        self.device = config.device
        if config.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize NEW model (GREEN - trainable)
        base_network = UNet(self.config.model_configuration)
        self.model = RestorationWrapper(base_network)
        self.model.to(self.device)

        # Load PRETRAINED model (GRAY - frozen)
        if config.pretrained_wandb_config:
            self._load_pretrained_from_wandb()
        elif config.pretrained_restoration_model_path:
            self._load_pretrained_from_local()
        else:
            raise ValueError("Either pretrained_wandb_config or pretrained_restoration_model_path must be provided")

        # Create optimizer (only for trainable model)
        self.optimizer = getattr(optim, config.optimizer_configuration.type)(
            self.model.parameters(),
            **config.optimizer_configuration.args
        )

        # Initialize dataset
        dataset = AudioInpaintingDataset(config.data_configuration)
        print(f"Total sample pairs in dataset: {len(dataset)}")

        # Create dataloader with custom collate function
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.dataloader_configuration.batch_size,
            shuffle=config.dataloader_configuration.shuffle,
            num_workers=config.dataloader_configuration.num_workers,
            pin_memory=config.dataloader_configuration.pin_memory,
            collate_fn=utils.collate_fn
        )
        self.dataloader = dataloader
        self.step = 0

    def _load_pretrained_from_wandb(self):
        """Load pretrained model from wandb artifact (same as NPPC trainer)"""
        wconfig = self.config.pretrained_wandb_config
        artifact_path = f"{wconfig.entity}/{wconfig.project}/{wconfig.artifact_name}:{wconfig.artifact_version}"

        print(f"Loading pretrained model from wandb:")
        print(f"  Entity: {wconfig.entity}")
        print(f"  Project: {wconfig.project}")
        print(f"  Artifact: {wconfig.artifact_name}")
        print(f"  Version: {wconfig.artifact_version}")
        print(f"  Checkpoint: {wconfig.checkpoint_filename}")

        try:
            api = wandb.Api()
            artifact = api.artifact(artifact_path)

            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = artifact.download(root=temp_dir)
                checkpoint_path = Path(artifact_dir) / wconfig.checkpoint_filename
                
                if not checkpoint_path.exists():
                    available_files = list(Path(artifact_dir).glob("*.pt"))
                    raise FileNotFoundError(
                        f"Checkpoint '{wconfig.checkpoint_filename}' not found. "
                        f"Available files: {[f.name for f in available_files]}"
                    )

                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Create and load pretrained model
                pretrained_base_network = UNet(self.config.model_configuration)
                pretrained_base_network.load_state_dict(checkpoint['model_state_dict'])
                pretrained_base_network.to(self.device)

                self.pretrained_model = RestorationWrapper(pretrained_base_network)
                
                # Freeze pretrained model
                for param in self.pretrained_model.parameters():
                    param.requires_grad = False
                self.pretrained_model.eval()

                print("Successfully loaded pretrained model from wandb and frozen")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from wandb: {str(e)}")

    def _load_pretrained_from_local(self):
        """Load pretrained model from local path"""
        print(f"Loading pretrained model from local path: {self.config.pretrained_restoration_model_path}")
        
        try:
            checkpoint_path = Path(self.config.pretrained_restoration_model_path).absolute()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create and load pretrained model
            pretrained_base_network = UNet(self.config.model_configuration)
            pretrained_base_network.load_state_dict(checkpoint['model_state_dict'])
            pretrained_base_network.to(self.device)

            self.pretrained_model = RestorationWrapper(pretrained_base_network)
            
            # Freeze pretrained model
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            self.pretrained_model.eval()
            
            print("Successfully loaded pretrained model from local path and frozen")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from local path: {str(e)}")

    def _get_latent_mask(self, mask):
        """
        Downsample the spectrogram mask to latent space dimensions.
        The encoder does 4 downsampling operations (each /2), so latent is 1/16 of original.
        """
        # Apply max pooling 4 times to match encoder downsampling
        latent_mask = mask
        for _ in range(4):
            latent_mask = F.max_pool2d(latent_mask, kernel_size=2)
        return latent_mask

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints",
              save_flag=False, val_dataloader=None):
        """Main training loop"""
        assert n_steps is not None or n_epochs is not None, "Must specify either n_steps or n_epochs"

        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_history: List[float] = []
        reconst_loss_history: List[float] = []
        latent_loss_history: List[float] = []
        val_loss_history: List[float] = []
        best_val_loss = float('inf')

        loop_loader = LoopLoader(
            dataloader=self.dataloader,
            n_steps=n_steps,
            n_epochs=n_epochs
        )

        pbar = tqdm(loop_loader, total=len(loop_loader))
        for batch in pbar:
            # Unpack batch including metadata
            masked_spec, mask_frames, clean_spec, masked_audio, metadata = batch

            # Move tensors to device
            masked_spec = masked_spec.to(self.device)
            mask_frames = mask_frames.to(self.device)
            clean_spec = clean_spec.to(self.device)
            masked_audio = masked_audio.to(self.device)

            # Training step
            loss, log_dict = self.base_step((masked_spec, mask_frames, clean_spec, masked_audio))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            loss_history.append(loss.item())
            reconst_loss_history.append(log_dict['reconst_loss'].item())
            latent_loss_history.append(log_dict['latent_loss'].item())
            
            pbar.set_description(
                f'Loss: {loss.item():.4f} | Reconst: {log_dict["reconst_loss"].item():.4f} | Latent: {log_dict["latent_loss"].item():.4f}'
            )
            
            # Log to wandb
            if self.config.use_wandb and self.step % 100 == 0:
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/reconst_loss": log_dict['reconst_loss'].item(),
                    "train/latent_loss": log_dict['latent_loss'].item(),
                }, step=self.step)
            
            # Periodic validation to check for overfitting
            if (self.config.enable_periodic_validation and 
                val_dataloader and 
                self.step % self.config.validation_interval == 0):
                val_loss = self.validate(val_dataloader)
                val_loss_history.append(val_loss)
                print(f"\n[Step {self.step}] Validation Loss: {val_loss:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                    }, step=self.step)
                
                # Track best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.config.use_wandb:
                        wandb.log({"val/best_loss": best_val_loss}, step=self.step)
            
            self.step += 1

        # Final validation
        if val_dataloader:
            val_loss = self.validate(val_dataloader)
            val_loss_history.append(val_loss)
            print(f"Final Validation Loss: {val_loss:.4f}")

        # Plot loss curve
        fig = self.plot_loss_curve(loss_history, reconst_loss_history, latent_loss_history, val_loss_history)

        if save_flag:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"latent_aware_checkpoint_final_{timestamp}.pt"
            )
            self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)
            self.save_checkpoint(final_checkpoint_path)

        # Log everything to wandb at the end
        if self.config.use_wandb:
            wandb.log({
                "train/final_loss": loss_history[-1],
                "train/avg_loss": sum(loss_history) / len(loss_history),
                "loss_curve": wandb.Image(fig)
            })

            if val_dataloader:
                wandb.log({
                    "val/final_loss": val_loss,
                    "val/best_loss": min(val_loss_history) if val_loss_history else None
                })

            wandb.finish()

        plt.close(fig)

    def base_step(self, batch):
        """
        Training step with latent-aware loss.
        
        Loss = ||x̂ - x||² + λ||M'(h - h_m)||
        
        Where:
        - x̂ = output from GREEN model (trainable)
        - x = clean spectrogram
        - h = latent from GREEN encoder (trainable)
        - h_m = latent from PRETRAINED encoder (frozen)
        - M' = mask in latent space
        """
        # Unpack batch
        masked_spec, mask_frames, clean_spec, masked_audio = batch

        # Preprocess data
        clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(
            clean_spec, masked_spec, mask_frames
        )

        # ===== GREEN MODEL (trainable) =====
        # Get latent and skip connections from GREEN encoder
        h, skip_connections = self.model.net.encoder(masked_spec_mag_log)
        
        # Decode to get output
        decoded = self.model.net.decoder(h, skip_connections)
        
        # Apply restoration wrapper logic: x_in + decoded * mask
        output = masked_spec_mag_log + decoded * (1 - mask)

        # ===== PRETRAINED MODEL (frozen) =====
        with torch.no_grad():
            # Get latent from PRETRAINED encoder
            h_m, _ = self.pretrained_model.net.encoder(masked_spec_mag_log)

        # ===== COMPUTE LOSSES =====
        
        # Term 1: Reconstruction loss ||x̂ - x||²
        reconst_loss = ((output - clean_spec_mag_norm_log) ** 2).mean()

        # Term 2: Latent regularization ||M'(h - h_m)||
        # Get mask in latent space
        latent_mask = self._get_latent_mask(1 - mask)  # 1-mask because inpainting region is where mask=0
        
        # Compute latent difference in inpainting region
        latent_diff = h - h_m
        masked_latent_diff = latent_diff * latent_mask
        latent_loss = (masked_latent_diff ** 2).mean()

        # Total loss
        total_loss = reconst_loss + self.config.latent_reg_lambda * latent_loss

        # Store logs
        log = {
            'clean_spec': clean_spec.detach(),
            'output': output.detach(),
            'loss': total_loss.detach(),
            'reconst_loss': reconst_loss.detach(),
            'latent_loss': latent_loss.detach(),
            'masked_audio': masked_audio.detach(),
            'h': h.detach(),
            'h_m': h_m.detach(),
        }
        return total_loss, log

    def validate(self, val_dataloader):
        """Validation loop"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                masked_spec, mask_frames, clean_spec, masked_audio, metadata = batch

                # Move tensors to device
                masked_spec = masked_spec.to(self.device)
                mask_frames = mask_frames.to(self.device)
                clean_spec = clean_spec.to(self.device)
                masked_audio = masked_audio.to(self.device)

                loss, _ = self.base_step((masked_spec, mask_frames, clean_spec, masked_audio))
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        self.model.train()
        return avg_val_loss

    def plot_loss_curve(self, loss_history, reconst_loss_history, latent_loss_history, val_loss_history):
        """Plot the training and validation loss curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot total loss
        axes[0, 0].plot(loss_history, label='Total Loss', alpha=0.5)
        if val_loss_history:
            axes[0, 0].plot(val_loss_history, label='Validation Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot reconstruction loss
        axes[0, 1].plot(reconst_loss_history, label='Reconstruction Loss', alpha=0.5, color='green')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Reconstruction Loss ||x̂ - x||²')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot latent loss
        axes[1, 0].plot(latent_loss_history, label='Latent Loss', alpha=0.5, color='orange')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title("Latent Loss ||M'(h - h_m)||")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot smoothed total loss
        smoothed_loss = self._smooth_losses(loss_history)
        axes[1, 1].plot(smoothed_loss, label='Smoothed Total Loss')
        if val_loss_history:
            axes[1, 1].plot(val_loss_history, label='Validation Loss')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Smoothed Total Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        return fig

    def _smooth_losses(self, losses, window_size=100):
        """Smooth losses using moving average"""
        smoothed = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size + 1)
            smoothed.append(sum(losses[start_idx:(i + 1)]) / (i - start_idx + 1))
        return smoothed

    def save_checkpoint(self, checkpoint_path):
        """Save model checkpoint and add it to wandb artifact"""
        checkpoint = {
            'model_state_dict': self.model.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config.model_dump()
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if self.config.use_wandb:
            try:
                artifact = wandb.Artifact(
                    name=self.config.wandb_artifact_name,
                    type='model',
                    description='Collection of latent-aware restoration model checkpoints'
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
                print(f"Checkpoint added to wandb artifact '{self.config.wandb_artifact_name}'")
            except Exception as e:
                print(f"Error saving to wandb: {str(e)}")

    def _get_and_save_metrics(self, checkpoint_dir, log_dict, n_epochs, n_steps, timestamp):
        """Save training metrics to JSON file and wandb"""
        final_metrics = {
            'timestamp': timestamp,
            'total_steps': self.step,
            'final_loss': log_dict['loss'].item(),
            'final_reconst_loss': log_dict['reconst_loss'].item(),
            'final_latent_loss': log_dict['latent_loss'].item(),
            'training_config': {
                'n_steps': n_steps,
                'n_epochs': n_epochs,
                'learning_rate': self.config.optimizer_configuration.args.get('lr'),
                'device': self.config.device,
                'batch_size': self.config.dataloader_configuration.batch_size,
                'latent_reg_lambda': self.config.latent_reg_lambda,
                'audio_len': self.config.data_configuration.sub_sample_length_seconds,
                'missing_length_seconds': self.config.data_configuration.missing_length_seconds,
                'missing_start_seconds': self.config.data_configuration.missing_start_seconds,
                'length_audio_seconds': self.config.data_configuration.sub_sample_length_seconds,
                'nfft': self.config.data_configuration.stft_configuration.nfft
            }
        }

        metrics_path = os.path.join(checkpoint_dir, f"latent_aware_metrics_final_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

        if self.config.use_wandb:
            wandb.log({
                "final_metrics/total_steps": final_metrics['total_steps'],
                "final_metrics/final_loss": final_metrics['final_loss'],
                "final_metrics/final_reconst_loss": final_metrics['final_reconst_loss'],
                "final_metrics/final_latent_loss": final_metrics['final_latent_loss'],
                "config/learning_rate": final_metrics['training_config']['learning_rate'],
                "config/batch_size": final_metrics['training_config']['batch_size'],
                "config/latent_reg_lambda": final_metrics['training_config']['latent_reg_lambda'],
                "config/audio_len": final_metrics['training_config']['audio_len'],
                "config/missing_length": final_metrics['training_config']['missing_length_seconds'],
                "config/nfft": final_metrics['training_config']['nfft']
            })

            metrics_artifact = wandb.Artifact(
                name=f'latent_aware_metrics_{timestamp}',
                type='metrics',
                description=f'Training metrics at step {self.step}'
            )
            metrics_artifact.add_file(metrics_path)
            wandb.log_artifact(metrics_artifact)

