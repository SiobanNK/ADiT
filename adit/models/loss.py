# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


#################### Training Losses ####################

class DenoisingLoss(nn.Module):

    def __init__(self, config):
        super(DenoisingLoss, self).__init__()
        self.config = config    # config.loss

    def forward(self, out, batch, _return_breakdown=False):
        # Configure masks.
        atom_mask = batch["atom_mask"].bool()
        batch_size = atom_mask.shape[0]
        gt_atom_positions = batch["gt_atom_positions"][atom_mask]
        pred_atom_positions = out["pred_atom_positions"]

        num_atoms = atom_mask.sum((-1, -2), keepdim=True).expand_as(atom_mask)   # (batch_size, num_res, 37)
        mse_loss = (pred_atom_positions - gt_atom_positions) ** 2      # (num_atom, 3)
        mse_loss = mse_loss.sum(-1)    # (num_atom, )
        mse_loss = mse_loss.sum() / batch_size

        cum_loss = 0.
        cum_loss = cum_loss + mse_loss
        
        losses = {}
        losses["denoising loss"] = mse_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss
        
        return cum_loss, losses


class MseLoss(nn.Module):
    """
    Used by skempi, lba, and davis settings
    """

    def __init__(self):
        super(MseLoss, self).__init__()

    def forward(self, pred, target):
        mse_loss = torch.nn.functional.mse_loss(pred, target)

        cum_loss = 0.
        cum_loss = cum_loss + mse_loss
        
        losses = {}
        losses["mse loss"] = mse_loss.detach().clone()
        
        return cum_loss, losses
