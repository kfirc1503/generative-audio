from pydantic import BaseModel
import FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus as fullsubnet_plus
from pathlib import Path
import torch
import tarfile
import io
import utils
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus


class FullSubNetPlusConfig(BaseModel):
    num_freqs = 257
    look_ahead = 2
    sequence_model = "LSTM"
    sb_num_neighbors = 15
    fb_num_neighbors = 0
    fb_output_activate_function = "ReLU"
    sb_output_activate_function = False
    fb_model_hidden_size = 512
    sb_model_hidden_size = 384
    channel_attention_model = "TSSE"
    norm_type = "offline_laplace_norm"
    num_groups_in_drop_band = 2
    output_size = 2
    subband_num = 1
    kersize = [3, 5, 10]
    weight_init = False


tmp_config = FullSubNetPlusConfig()

# ## model args:
# sb_num_neighbors = 15
# fb_num_neighbors = 0
# num_freqs = 257
# look_ahead = 2
# sequence_model = "LSTM"
# fb_output_activate_function = "ReLU"
# sb_output_activate_function = False
# channel_attention_model = "TSSE"
# fb_model_hidden_size = 512
# sb_model_hidden_size = 384
# weight_init = False
# norm_type = "offline_laplace_norm"
# num_groups_in_drop_band = 2
# kersize=[3, 5, 10]
# subband_num = 1

def preload_model(model_path,model):
    """
    Preload model parameters (in "*.tar" format) at the start of experiment.

    Args:
        model_path (Path): The file path of the *.tar file
    """
    model_path = model_path.expanduser().absolute()
    assert model_path.exists(), f"The file {model_path.as_posix()} is not exist. please check path."

    model_checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
    model.load_state_dict(model_checkpoint["model"], strict=False)
    return model
    # self.model.to(self.rank)
    #
    # if self.rank == 0:
    #     print(f"Model preloaded successfully from {model_path.as_posix()}.")

def load_pretrained_model(tar_path, model):
    """
    Load pretrained model weights from a tar file containing pickle/state dict
    
    Args:
        tar_path (str): Path to the tar file containing model weights
        model (nn.Module): Initialized model to load weights into
        
    Returns:
        model: Model with loaded pretrained weights
    """
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            # Find checkpoint file in tar
            checkpoint_file = None
            for member in tar.getmembers():
                if member.name.endswith('.pth') or member.name.endswith('.pkl'):
                    checkpoint_file = member
                    break
            
            if checkpoint_file is None:
                raise ValueError("No .pth or .pkl file found in tar archive")
                
            # Extract and load checkpoint
            f = tar.extractfile(checkpoint_file)
            state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
            
            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
            # Remove 'module.' prefix if present (happens with DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
                
            # Load the cleaned state dict
            model.load_state_dict(new_state_dict, strict=False)
            return model
            
    except Exception as e:
        print(f"Error loading pretrained model: {str(e)}")
        raise
#
# import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
#tar_path = "./FullSubNet_plus/best_model.tar"
tar_path = Path("./FullSubNet_plus/best_model.tar")
pre_train_model = fullsubnet_plus.FullSubNet_Plus(**tmp_config.dict())
pre_train_model = preload_model(tar_path , pre_train_model)
#load_pretrained_model(tar_path, pre_train_model)


pre_train_model.eval()  # Set model to evaluation mode
with torch.no_grad():
    # Prepare input
    path_to_audio = "./FullSubNet_plus/data/noisy/_1eSheWjfJQ.wav"
    noisy_mag, noisy_real, noisy_imag = utils.prepare_input(path_to_audio)

    # Forward pass through model
    enhanced_magnitude = pre_train_model(noisy_mag,noisy_real, noisy_imag)

    # Convert back to audio if needed
    # ... additional processing to convert enhanced magnitude back to waveform





print(pre_train_model)
