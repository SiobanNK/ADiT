import torch
from torch import nn
import loralib as lora
import sys
sys.modules['fmas'] = sys.modules['adit']


def replace_linear_with_lora(model, rank, alpha=1.0):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features, out_features = module.in_features, module.out_features
            use_bias = module.bias is not None
            lora_layer = lora.Linear(in_features, out_features, rank, alpha, merge_weights=False, bias=use_bias)
            setattr(model, name, lora_layer)
        else:
            replace_linear_with_lora(module, rank, alpha)
    lora.mark_only_lora_as_trainable(model)
    return model


def load_model_checkpoint(model, ckpt_path, load_weight_only=True):
    """Load state dict from checkpoint file.

    :param model: The model to load the state dict into.
    :param ckpt_path: The path to the checkpoint file.
    """
    if ckpt_path is None:
        return model, None
    if not load_weight_only:
        return model, ckpt_path
    
    # The ckpt_path ending with .ckpt is a checkpoint file saved by pytorch-lightning.
    # If the ckpt_path is a .pth file, it is viewed as a checkpoint file saved by pytorch
    # such that only net parameters are loaded. 
    # (This may avoid the ambiguity of loading #epochs/lr for finetuning)
    if ckpt_path.endswith(".pth"):  
        net_params = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        net_params = {k.replace('net.', ''): v for k, v in net_params.items()}
        net_params = {k.replace('fmas.', 'adit.'): v for k, v in net_params.items()}
        model.net.load_state_dict(net_params)
        ckpt_path = None
    elif ckpt_path.endswith(".ckpt"):
        # will be handled later by the trainer
        net_params = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        net_params = {k.replace('net.', ''): v for k, v in net_params.items()}
        model.net.load_state_dict(net_params, strict=False)
        ckpt_path = None
    else:
        # suffix check
        raise ValueError(f"ckpt_path {ckpt_path} is not a valid checkpoint file.")
    
    return model, ckpt_path

