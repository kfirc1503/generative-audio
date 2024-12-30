import torch
import torch.nn as nn
from typing import Literal
import pydantic
from pathlib import Path

from nppc_audio.inpainting.nppc.pc_wrapper import AudioInpaintingPCWrapperConfig, AudioInpaintingPCWrapper
from nppc_audio.inpainting.networks.unet import UNet,UNetConfig,RestorationWrapper


class NPPCModelConfig(pydantic.BaseModel):
    pretrained_restoration_model_configuration: UNetConfig
    pretrained_restoration_model_path: str
    audio_pc_wrapper_configuration: AudioInpaintingPCWrapperConfig
    device: Literal['cpu', 'cuda'] = 'cuda'



class NPPCModel(nn.Module):
    def __init__(self, config: NPPCModelConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        if config.device == 'cuda':
            # check if gpu is exist
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(config.pretrained_restoration_model_path).absolute()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # self.model = RestorationWrapper(checkpoint['model'])
        self.pretrained_restoration_model = RestorationWrapper(self.config.pretrained_restoration_model_configuration)
        # self.model = RestorationWrapper(checkpoint['model_config']).to(self.device)
        self.pretrained_restoration_model.load_state_dict(checkpoint['model_state_dict'])
        self.pretrained_restoration_model.to(self.device)
        self.pretrained_restoration_model.eval()
        self.pc_wrapper = AudioInpaintingPCWrapper(self.config.audio_pc_wrapper_configuration)
        self.pc_wrapper.to(self.device)

    def forward(self, masked_spec_mag_norm:torch.Tensor , mask: torch.Tensor):
        # both size of [B,1,F,T]
        #first get the pred spec mag norm from the pretrain model:
        with torch.no_grad():
            pred_spec_mag_norm = self.pretrained_restoration_model(masked_spec_mag_norm, mask)
        # now let's concatinate the masked spec into the pred_spec
        masked_with_pred_spec_mag_norm = torch.cat((masked_spec_mag_norm, pred_spec_mag_norm), dim=1)
        w_mat = self.pc_wrapper(masked_with_pred_spec_mag_norm , mask)
        return w_mat # [B,n_dirs,F,T]




