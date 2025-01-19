# import torch
# import torch.nn as nn
# from typing import Literal
# import pydantic
# from pathlib import Path
#
# from nppc_audio.inpainting.nppc.pc_wrapper import AudioInpaintingPCWrapperConfig, AudioInpaintingPCWrapper
# from nppc_audio.inpainting.networks.unet import UNet,UNetConfig,RestorationWrapper
#
#
# class NPPCModelConfig(pydantic.BaseModel):
#     pretrained_restoration_model_configuration: UNetConfig
#     pretrained_restoration_model_path: str
#     audio_pc_wrapper_configuration: AudioInpaintingPCWrapperConfig
#     device: Literal['cpu', 'cuda'] = 'cuda'
#
#
#
# class NPPCModel(nn.Module):
#     def __init__(self, config: NPPCModelConfig):
#         super().__init__()
#         self.config = config
#         self.device = config.device
#         if config.device == 'cuda':
#             # check if gpu is exist
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         checkpoint_path = Path(config.pretrained_restoration_model_path).absolute()
#         checkpoint = torch.load(checkpoint_path, map_location="cpu")
#         # self.model = RestorationWrapper(checkpoint['model'])
#         # self.model = RestorationWrapper(checkpoint['model_config']).to(self.device)
#         base_net = UNet(self.config.pretrained_restoration_model_configuration)
#         base_net.load_state_dict(checkpoint['model_state_dict'])
#         base_net.to(self.device)
#         self.pretrained_restoration_model = RestorationWrapper(base_net)
#         self.pretrained_restoration_model.eval()
#         self.pc_wrapper = AudioInpaintingPCWrapper(self.config.audio_pc_wrapper_configuration)
#         self.pc_wrapper.to(self.device)
#
#     def forward(self, masked_spec_mag_norm:torch.Tensor , mask: torch.Tensor):
#         # both size of [B,1,F,T]
#         #first get the pred spec mag norm from the pretrain model:
#         pred_spec_mag_norm = self.get_pred_spec_mag_norm(masked_spec_mag_norm, mask)
#         # now let's concatinate the masked spec into the pred_spec
#         masked_with_pred_spec_mag_norm = torch.cat((masked_spec_mag_norm, pred_spec_mag_norm), dim=1)
#         w_mat = self.pc_wrapper(masked_with_pred_spec_mag_norm , mask)
#         return w_mat # [B,n_dirs,F,T]
#
#     def get_pred_spec_mag_norm(self, masked_spec_mag_norm, mask):
#         with torch.no_grad():
#             pred_spec_mag_norm = self.pretrained_restoration_model(masked_spec_mag_norm, mask)
#         return pred_spec_mag_norm


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
import pydantic
from pathlib import Path
import wandb
import tempfile

from nppc_audio.inpainting.nppc.pc_wrapper import AudioInpaintingPCWrapperConfig, AudioInpaintingPCWrapper
from nppc_audio.inpainting.networks.unet import UNet, UNetConfig, RestorationWrapper


class WandbConfig(pydantic.BaseModel):
    """Configuration for wandb artifact loading"""
    entity: str = "kfirc-tel-aviv-university"
    project: str = "generative-audio"
    artifact_name: str
    artifact_version: str = "latest"
    checkpoint_filename: str = "checkpoint_final.pt"


class NPPCModelConfig(pydantic.BaseModel):
    """Configuration for NPPC model"""
    pretrained_restoration_model_configuration: UNetConfig
    pretrained_restoration_model_path: Optional[str] = None
    wandb_config: Optional[WandbConfig] = None
    audio_pc_wrapper_configuration: AudioInpaintingPCWrapperConfig
    device: Literal['cpu', 'cuda'] = 'cuda'


class NPPCModel(nn.Module):
    def __init__(self, config: NPPCModelConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        if config.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model (either from wandb or local path)
        if config.wandb_config:
            self._load_from_wandb()
        elif config.pretrained_restoration_model_path:
            self._load_from_local()
        else:
            raise ValueError("Either wandb_config or pretrained_restoration_model_path must be provided")

        self.pc_wrapper = AudioInpaintingPCWrapper(self.config.audio_pc_wrapper_configuration)
        self.pc_wrapper.to(self.device)

    def _load_from_wandb(self):
        """Load pretrained model from wandb artifact"""
        wconfig = self.config.wandb_config
        artifact_path = f"{wconfig.entity}/{wconfig.project}/{wconfig.artifact_name}:{wconfig.artifact_version}"

        print(f"Loading pretrained model from wandb:")
        print(f"  Entity: {wconfig.entity}")
        print(f"  Project: {wconfig.project}")
        print(f"  Artifact: {wconfig.artifact_name}")
        print(f"  Version: {wconfig.artifact_version}")
        print(f"  Checkpoint: {wconfig.checkpoint_filename}")

        try:
            # Initialize wandb API
            api = wandb.Api()

            # Get the artifact
            artifact = api.artifact(artifact_path)

            # Create a temporary directory to download the artifact
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the artifact
                artifact_dir = artifact.download(root=temp_dir)

                # Look for specific checkpoint file
                checkpoint_path = Path(artifact_dir) / wconfig.checkpoint_filename
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint file '{wconfig.checkpoint_filename}' not found in artifact. "
                        f"Available files: {list(Path(artifact_dir).glob('*.pt'))}"
                    )

                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Initialize model
                base_net = UNet(self.config.pretrained_restoration_model_configuration)
                base_net.load_state_dict(checkpoint['model_state_dict'])
                base_net.to(self.device)

                self.pretrained_restoration_model = RestorationWrapper(base_net)
                self.pretrained_restoration_model.eval()

                print("Successfully loaded pretrained model from wandb")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from wandb: {str(e)}")

    def _load_from_local(self):
        """Load pretrained model from local path"""
        print(f"Loading pretrained model from local path: {self.config.pretrained_restoration_model_path}")
        try:
            checkpoint_path = Path(self.config.pretrained_restoration_model_path).absolute()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            base_net = UNet(self.config.pretrained_restoration_model_configuration)
            base_net.load_state_dict(checkpoint['model_state_dict'])
            base_net.to(self.device)

            self.pretrained_restoration_model = RestorationWrapper(base_net)
            self.pretrained_restoration_model.eval()

            print("Successfully loaded pretrained model from local path")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from local path: {str(e)}")


    def forward(self, masked_spec_mag_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NPPC model.

        Args:
            masked_spec_mag_norm: Masked normalized magnitude spectrogram [B,1,F,T]
            mask: Binary mask indicating masked regions [B,1,F,T]
                1 represents unmasked regions
                0 represents masked regions to be inpainted

        Returns:
            w_mat: Principal components matrix [B,n_dirs,F,T]
                Contains the principal components for each time-frequency bin
        """
        # Get predicted spectrogram from pretrained model
        pred_spec_mag_norm = self.get_pred_spec_mag_norm(masked_spec_mag_norm, mask)

        # Concatenate masked and predicted spectrograms along channel dimension
        masked_with_pred_spec_mag_norm = torch.cat(
            (masked_spec_mag_norm, pred_spec_mag_norm),
            dim=1
        )

        # Generate principal components using PC wrapper
        w_mat = self.pc_wrapper(masked_with_pred_spec_mag_norm, mask)

        return w_mat


    def get_pred_spec_mag_norm(self, masked_spec_mag_log, mask):
        """
        Get the predicted normalized log magnitude spectrogram
        Args:
            masked_spec_mag_log: Masked log magnitude spectrogram [B,1,F,T]
            mask: Binary mask [B,1,F,T]

        Returns:
            pred_spec_mag_norm_log: Predicted normalized log magnitude spectrogram [B,1,F,T]
        """
        with torch.no_grad():
            pred_spec_mag_log = self.pretrained_restoration_model(masked_spec_mag_log, mask)
        return pred_spec_mag_log