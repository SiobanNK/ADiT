from typing import Optional, Tuple
import torch


class StructureNoiser:
    """
    Add Gaussian noise to protein structures
    Following the implementation of https://github.com/NVlabs/edm/blob/main/training/loss.py
    """
    def __init__(self, P_mean=0.05, P_std=0.02, sigma_min=0.01, sigma_max=0.2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def add_noise(self, batch: Tuple[torch.Tensor, torch.Tensor], loss_type: str):
        """
        Random mask aatype in batch
        Args:
            batch: data dict

        Returns:
        Dict contains:
            aa_mask: [..., N] random mask
            gt_aatype: [..., N] ground_truth aa type
        """
        device = batch["seq_mask"].device
        batch_size = batch["seq_mask"].shape[0]
        atom_mask = batch["atom_mask"].bool()   # shape: [4, 503, 37]
        atom_positions = batch["atom_positions"]    # shape: [batch_size, num_residue, 37, 3]

        if loss_type == "gaussian":
            # EDM loss
            rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
            sigma = (rnd_normal * self.P_std + self.P_mean).clamp(min=self.sigma_min, max=self.sigma_max)
        elif loss_type == "exp":
            # VE loss
            rnd_uniform = torch.rand([batch_size, 1, 1, 1], device=device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        elif loss_type == "logit-normal":
            rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
            rnd_logit = (rnd_normal * self.P_std + self.P_mean).sigmoid()
            sigma = self.sigma_min + rnd_logit * (self.sigma_max - self.sigma_min)
        
        # weight = 1 / sigma ** 2
        n = torch.randn_like(atom_positions) * sigma
        batch["gt_atom_positions"] = batch["atom_positions"].clone()
        batch["atom_positions"][atom_mask] = (atom_positions + n)[atom_mask]
        batch["sigma"] = sigma      # shape: [batch_size, 1, 1, 1]
        
        return batch
