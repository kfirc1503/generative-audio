import torch
import torch.nn as nn
import pydantic
from nppc_audio.nppc_model import NPPCModelConfig, NPPCModel
# from nppc_model import NPPCModelConfig
from use_pre_trained_model.model_validator.config.schema import DataConfig, DataLoaderConfig
from dataset import AudioDataset
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.feature import drop_band
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import build_complex_ideal_ratio_mask
from tqdm.auto import tqdm
from nppc.auxil import LoopLoader


class NPPCAudioTrainerConfig(pydantic.BaseModel):
    """Configuration for NPPCAudio trainer"""
    nppc_model_configuration: NPPCModelConfig
    data_configuration: DataConfig
    data_loader_configuration: DataLoaderConfig
    # output_dir: str
    learning_rate: float = 1e-4
    device: str = "cuda"
    save_interval: int = 10
    log_interval: int = 100
    second_moment_loss_lambda: float = 1.0
    second_moment_loss_grace: int = 500


class NPPCAudioTrainer(nn.Module):
    def __init__(self, config: NPPCAudioTrainerConfig):
        super().__init__()
        self.config = config
        ## this is suppose to be the same thing
        # self.nppc_model = self.config.nppc_model_configuration.make_instance()
        self.nppc_model = NPPCModel(self.config.nppc_model_configuration)
        self.device = self.config.device
        # create data loader:
        dataset = AudioDataset(config.data_configuration.dataset)

        print(f"Total sample pairs in dataset: {len(dataset)}")

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.data_loader_configuration.batch_size,  # Adjust based on your GPU memory
            shuffle=config.data_loader_configuration.shuffle,
            num_workers=config.data_loader_configuration.num_workers,
            pin_memory=config.data_loader_configuration.pin_memory
        )
        self.dataloader = dataloader
        self.step = 0

    def train(self, n_steps=None, n_epochs=None):
        """
        Main training loop using LoopLoader

        Args:
            n_steps: Number of training steps (optional)
            n_epochs: Number of training epochs (optional)

        Note: Must provide either n_steps or n_epochs
        """
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
            if isinstance(batch, (tuple, list)):
                batch = tuple(x.to(self.device) for x in batch)
            else:
                batch = batch.to(self.device)

            # Forward and backward pass
            objective, _ = self.base_step(batch)

            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

            # Update progress bar
            pbar.set_description(f'Loss: {objective.item():.4f}')

            self.step += 1

    def base_step(self, batch):
        """
        Base training step adapted from NPPC for audio enhancement using CRM directions.
        The main differences are:
        1. Our PC directions are CRM masks (complex-valued)
        2. We work with complex masks instead of real images
        3. Error is computed in CRM domain

        Args:
            batch: Dictionary containing:
                - noisy_waveform: [B,T]
                - clean_waveform: [B,T]
        Returns:
            tuple: (objective, log_dict)
        """
        model = self.nppc_model
        # Get batch data
        noisy_waveform, clean_waveform = batch
        # Get predicted CRM directions (our PC directions)
        w_mat = model(noisy_waveform)  # [B, n_dirs, 2, F, T]
        B, n_dirs, _, F, T = w_mat.shape
        # Flatten frequency and time dimensions for PC computation
        w_mat_flat = w_mat.reshape(B, n_dirs, 2, -1)  # [B, n_dirs, 2, F*T]
        # get CRMS
        num_groups_in_drop_band = self.config.nppc_model_configuration.audio_pc_wrapper_configuration.multi_direction_configuration.num_groups_in_drop_band

        gt_crm, pred_crm = self._get_true_and_pred_crm(B, clean_waveform, model, noisy_waveform,
                                                                           num_groups_in_drop_band)
        pred_crm_flat = pred_crm.reshape(B, 2, -1)  # [B, 2, F*T]
        gt_crm_flat = gt_crm.reshape(B, 2, -1)  # [B,2,F*T]

        # Compute norms of each CRM direction
        w_norms = torch.norm(w_mat_flat, dim=(2, 3))  # [B, n_dirs]
        # Normalize CRM directions
        w_hat_mat = w_mat_flat / (w_norms[..., None, None] + 1e-8)  # [B, n_dirs, 2, F*T]
        # Compute error in CRM domain
        err = (gt_crm_flat - pred_crm_flat)  # [B, 2, F*T]
        # Normalize error
        err_norm = torch.norm(err, dim=(1, 2))  # [B]
        err = err / (err_norm[:, None, None] + 1e-8)  # [B, 2, F*T]
        w_norms = w_norms / (err_norm[:, None] + 1e-8)
        # Compute projections of error onto normalized CRM directions
        err_proj = torch.sum(w_hat_mat * err[:, None], dim=(2, 3))  # [B, n_dirs]
        # Reconstruction error (how well CRM directions explain the error)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)  # [B]
        # Second moment loss (align CRM norms with projection magnitudes)
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
        # Compute final objective with adaptive weighting
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)

        # Store logs
        log = {
            'noisy_complex': noisy_waveform,
            'clean_complex': clean_waveform,
            'pred_crm': pred_crm.detach(),
            'w_mat': w_mat.detach(),

            'err_norm': err_norm.detach(),
            'err_proj': err_proj.detach(),
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),

            'objective': objective.detach()
        }

        return objective, log

    def _calculate_final_objective(self, reconst_err, second_moment_mse):
        second_moment_loss_lambda = -1 + 2 * self.step / self.config.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
        second_moment_loss_lambda *= self.config.second_moment_loss_lambda
        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
        return objective

    def _get_true_and_pred_crm(self, clean_waveform, model, noisy_waveform, num_groups_in_drop_band):
        nfft = self.config.nppc_model_configuration.stft_configuration.nfft
        hop_length = self.config.nppc_model_configuration.stft_configuration.hop_length
        win_length = self.config.nppc_model_configuration.stft_configuration.win_length

        window = torch.hann_window(win_length).to(self.device)

        clean_complex = torch.stft(clean_waveform, nfft, hop_length=hop_length, win_length=win_length,
                                        window=window, return_complex=True)

        noisy_complex = torch.stft(noisy_waveform, nfft, hop_length=hop_length, win_length=win_length, window=window,
                                        return_complex=True)

        gt_crm = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
        gt_crm = drop_band(
            gt_crm.permute(0, 3, 1, 2),  # [B, 2, F ,T]
            num_groups_in_drop_band
        )
        # not sure if necessary to turn it back to [B,F,T,2]
        # gt_cIRM = gt_cIRM.permute(0,2,3,1) # [B,F,T,2]
        # Get enhanced CRM from model's prediction
        pred_crm = model.get_pred_crm(noisy_waveform)  # [B, 2, F, T]
        return gt_crm, pred_crm
