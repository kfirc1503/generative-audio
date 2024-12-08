import torch
import torch.nn as nn
from typing import Literal
import pydantic
from nppc_audio.pc_wrapper import AudioPCWrapper,AudioPCWrapperConfig
#from pc_wrapper import AudioPCWrapper , AudioPCWrapperConfig
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNetPlusConfig
import utils
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM



class NPPCModelConfig(pydantic.BaseModel):
    pretrained_restoration_model_configuration: FullSubNetPlusConfig
    pretrained_restoration_model_path: str
    audio_pc_wrapper_configuration: AudioPCWrapperConfig
    stft_configuration: utils.StftConfig
    device: Literal['cpu', 'cuda'] = 'cuda'

    def make_instance(self):
        # Create and return an instance of Model using this config
        return AudioPCWrapper(self)


class NPPCModel(nn.Module):
    def __init__(self,config: NPPCModelConfig):
        """
        this model uses the pretrained model for resturation, and then the pc wrapper model
        he will receive the noisy audio,
        will turn him into a spectrum mag,real,imag
        and then we got the crm from the pretrained model
        and from their we can got to the enhanced audio and we are going to keep it in stft form
        so we got enhanced_mag, enhanced_real, enhanced_imag
        and we are going to input that into the pc_wrapper and that's it,
        maybe we will add a method of got restoration audio which be
        to got the ouput of the pretrained model after converting it with istft


        Args:
            config:
        """
        super().__init__()
        self.config = config
        self.pretrained_restoration_model = utils.load_pretrained_model(config.pretrained_restoration_model_path , config.pretrained_restoration_model_configuration)
        self.device = config.device
        if config.device == 'cuda':
            # check if gpu is exist
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_restoration_model.to(self.device)
        self.pretrained_restoration_model.eval()

        # same lines!!
        self.audio_pc_wrapper = AudioPCWrapper(config.audio_pc_wrapper_configuration)
        #self.audio_pc_wrapper = self.config.audio_pc_wrapper_configuration.make_instance()
        self.audio_pc_wrapper.to(self.device)


    def forward(self, noisy_waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NPPC audio model. The process follows these steps:
        1. Convert input waveform to STFT components (mag, real, imag)
        2. Pass through pretrained restoration model to get CRM
        3. Apply CRM to get enhanced STFT components
        4. Pass both noisy and enhanced components through PC wrapper to get principal components

        Args:
            noisy_waveform: Input noisy audio waveform tensor of shape [B, T]
                where B is batch size and T is number of time samples

        Returns:
            w_mat: Principal component directions tensor of shape [B, n_dirs, 2, F, T]
                where:
                - B: batch size
                - n_dirs: number of principal component directions
                - 2: real and imaginary components
                - F: number of frequency bins from STFT
                - T: number of time frames from STFT

        Note:
            The pretrained model generates a complex ratio mask (CRM) which is used to enhance
            the noisy input. Both the noisy and enhanced spectrograms are then used by the
            PC wrapper to compute orthogonal directions in the complex spectrogram space.
        """
        # Get STFT components from waveform
        nfft = self.config.stft_configuration.nfft
        hop_length = self.config.stft_configuration.hop_length
        win_length = self.config.stft_configuration.win_length
        noisy_mag, noisy_real, noisy_imag = utils.prepare_input_from_waveform(
            noisy_waveform, nfft, hop_length, win_length, self.device
        )
        noisy_complex = torch.complex(noisy_mag, noisy_real)

        # Get CRM from pretrained model
        pred_crm = self.pretrained_restoration_model(noisy_mag, noisy_real, noisy_imag)
        pred_crm = pred_crm.permute(0, 2, 3, 1)
        pred_crm = decompress_cIRM(pred_crm)

        # Get enhanced STFT components using CRM
        enhanced_mag, enhanced_real, enhanced_imag = utils.crm_to_stft_components(
            pred_crm, noisy_real , noisy_imag
        )
        # add the channel dim back,[B,F,T] -> [B,1,F,T]
        enhanced_mag = enhanced_mag.unsqueeze(1)
        enhanced_real = enhanced_real.unsqueeze(1)
        enhanced_imag = enhanced_imag.unsqueeze(1)


        # Get principal component directions from PC wrapper
        w_mat = self.audio_pc_wrapper(
            noisy_mag, noisy_real, noisy_imag,
            enhanced_mag, enhanced_real, enhanced_imag
        )

        return w_mat  # [B, n_dirs, 2, F, T]

    def get_pred_crm(self , noisy_waveform: torch.Tensor) -> torch.Tensor:
        # Get STFT components from waveform
        nfft = self.config.stft_configuration.nfft
        hop_length = self.config.stft_configuration.hop_length
        win_length = self.config.stft_configuration.win_length
        noisy_mag, noisy_real, noisy_imag = utils.prepare_input_from_waveform(
            noisy_waveform, nfft, hop_length, win_length, self.device
        )
        noisy_complex = torch.complex(noisy_mag, noisy_real)

        # Get CRM from pretrained model
        pred_crm = self.pretrained_restoration_model(noisy_mag, noisy_real, noisy_imag)
        #pred_crm = pred_crm.permute(0, 2, 3, 1)
        pred_crm = decompress_cIRM(pred_crm)
        return pred_crm



