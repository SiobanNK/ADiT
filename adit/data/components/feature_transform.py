import tree
from typing import Optional
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation as R

from adit.common import residue_constants
from adit.common.ligand import element_dict
from adit.data.components.data_utils import continues_crop, spatial_crop

CA_IDX = residue_constants.atom_order['CA']
N_IDX = residue_constants.atom_order['N']
C_IDX = residue_constants.atom_order['C']

DTYPE_MAPPING = {
    'aatype': torch.long,
    'atom_positions': torch.float,
    'atom_mask': torch.float,
}


class FeatureTransform:
    def __init__(self, 
                 unit: Optional[str] = 'angstrom', 
                 truncate_length: Optional[int] = None,
                 strip_missing_residues: bool = True,
                 recenter_and_scale: bool = True,
                 eps: float = 1e-8,
                 pretrain: bool = False,
                 label_name: Optional[str] = None,
                 AA: bool = True
    ):
        if unit == 'angstrom':
            self.coordinate_scale = 1.0
        elif unit in ('nm', 'nanometer'):
            self.coordinate_scale = 0.1
        else:
            raise ValueError(f"Invalid unit: {unit}")
        
        if truncate_length is not None:
            assert truncate_length > 0, f"Invalid truncate_length: {truncate_length}"
        self.truncate_length = truncate_length
        
        self.strip_missing_residues = strip_missing_residues
        self.recenter_and_scale = recenter_and_scale
        self.eps = eps
        self.pretrain = pretrain
        self.label_name = label_name
        self.AA = AA
    
    @staticmethod
    def mask_side_chain(protein_feat):
        protein_feat['atom_mask'][:, 3:] = 0
        return protein_feat
        
    def __call__(self, chain_feats, val_stage = False):
        protein_feat, ligand_feat, label = self.split_feats(chain_feats, self.label_name)

        if not self.AA:
            protein_feat = self.mask_side_chain(protein_feat)
        
        protein_feat = self.patch_feats(protein_feat)
        
        if self.strip_missing_residues:
            protein_feat = self.strip_ends(protein_feat)
        
        if self.truncate_length is not None:
            protein_feat = self.random_truncate(protein_feat, max_len=self.truncate_length)
        
        # Recenter and scale atom positions
        if self.recenter_and_scale:
            protein_feat, ligand_feat = self.recenter_and_scale_coords(protein_feat, ligand_feat, coordinate_scale=self.coordinate_scale, eps=self.eps)
        
        if self.pretrain and not val_stage:
            protein_feat, ligand_feat = self.random_rotate_coords(protein_feat, ligand_feat)

        # Map to torch Tensor
        protein_feat = self.map_to_tensors(protein_feat)
        ligand_feat = self.map_to_tensors(ligand_feat)

        unified_feats = self.merge_feats(protein_feat, ligand_feat)

        if self.label_name is not None:
            unified_feats.update(label)
        
        return unified_feats

    @staticmethod
    def split_feats(chain_feats, label_name):
        # split feats into protein feat, ligand feat, and ground truth label.
        return chain_feats, None, None
    
    @staticmethod
    def patch_feats(protein_feat):
        seq_mask = np.ones((protein_feat['atom_mask'].shape[0],), dtype=int)
        patch_feats = {
            'seq_mask': seq_mask,
        }
        protein_feat.update(patch_feats)
        return protein_feat
    
    @staticmethod
    def strip_ends(protein_feat):
        # Strip missing residues on both ends
        modeled_idx = np.where(protein_feat['aatype'] != 20)[0]
        min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)
        protein_feat = tree.map_structure(
                lambda x: x[min_idx : (max_idx+1)], protein_feat)
        return protein_feat
    
    @staticmethod
    def random_truncate(protein_feat, max_len):
        L = protein_feat['aatype'].shape[0]
        if L > max_len:
            # Randomly truncate
            start = np.random.randint(0, L - max_len + 1)
            end = start + max_len
            protein_feat = tree.map_structure(
                    lambda x: x[start : end], protein_feat)
        return protein_feat
    
    @staticmethod
    def map_to_tensors(chain_feats):
        if chain_feats is None:
            return chain_feats
        
        chain_feats = {k: torch.as_tensor(v) for k, v in chain_feats.items()}
        # Alter dtype 
        for k, dtype in DTYPE_MAPPING.items():
            if k in chain_feats:
                chain_feats[k] = chain_feats[k].type(dtype)
        return chain_feats
    
    @staticmethod
    def recenter_and_scale_coords(protein_feat, ligand_feat, coordinate_scale, eps=1e-8):
        # recenter and scale atom positions
        bb_pos = protein_feat['atom_positions'][:, CA_IDX]
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(protein_feat['seq_mask']) + eps)
        centered_pos = protein_feat['atom_positions'] - bb_center[None, None, :]
        scaled_pos = centered_pos * coordinate_scale
        protein_feat['atom_positions'] = scaled_pos * protein_feat['atom_mask'][..., None]

        if ligand_feat is not None:
            ligand_feat["ligand_atom_positions"] = (
                ligand_feat["ligand_atom_positions"] - bb_center[None, :]
            ) * coordinate_scale

        return protein_feat, ligand_feat
    
    @staticmethod
    def random_rotate_coords(protein_feat, ligand_feat):
        rot_vec = np.random.randn(3)
        rot_vec = rot_vec / ((rot_vec ** 2).sum() ** 0.5 + 1e-9)

        rot_sigma = np.pi * 2 * np.random.rand()
        rot = rot_vec * rot_sigma
        rot_matrix = R.from_rotvec(rot).as_matrix()
        protein_feat['atom_positions'] = protein_feat['atom_positions'] @ rot_matrix

        if ligand_feat is not None:
            ligand_feat["ligand_atom_positions"] = ligand_feat["ligand_atom_positions"] @ rot_matrix

        return protein_feat, ligand_feat
    
    @staticmethod
    def merge_feats(protein_feat, ligand_feat):
        # merge residue tokens and atom tokens into unified tokens
        unified_feat = {}

        # chain-level: chain_index
        if ligand_feat is not None:
            protein_chain_index = protein_feat["chain_index"]
            ligand_chain_index = torch.ones_like(ligand_feat["ligand_atom_type"]) * (protein_feat["chain_index"].max() + 1)
            unified_feat["chain_index"] = torch.concat([protein_chain_index, ligand_chain_index], dim = -1)
        else:
            unified_feat["chain_index"] = protein_feat["chain_index"]

        # token-level: seq_mask, protein_mask, ligand_mask, token_idx, token_type, aatype(padding for ligand)
        unified_feat["seq_mask"] = torch.ones_like(unified_feat["chain_index"], dtype=torch.long)
        unified_feat["protein_mask"] = torch.zeros_like(unified_feat["seq_mask"])
        unified_feat["protein_mask"][:protein_feat["chain_index"].shape[0]] = 1
        unified_feat["ligand_mask"] = 1 - unified_feat["protein_mask"]

        residue_index = protein_feat["residue_index"]
        ligand_idx_start = protein_feat["residue_index"].max() + 1
        ligand_index = torch.arange(ligand_idx_start, ligand_idx_start + unified_feat["ligand_mask"].sum())
        unified_feat["token_idx"] = torch.concat([residue_index, ligand_index], dim = -1)

        unified_feat["token_type"] = torch.zeros_like(unified_feat["seq_mask"])
        unified_feat["token_type"][unified_feat["ligand_mask"].bool()] = 1

        unified_feat["aatype"] = torch.zeros_like(unified_feat["seq_mask"])
        unified_feat["aatype"][unified_feat["protein_mask"].bool()] = protein_feat["aatype"]

        # atom-level: atom_mask, atom_positions, atomic_number
        if ligand_feat is None:
            unified_feat["atom_mask"] = protein_feat["atom_mask"]
            unified_feat["atom_positions"] = protein_feat["atom_positions"]
            unified_feat["atomic_number"] = torch.zeros(unified_feat["atom_positions"].shape[:-1], dtype=torch.long)
        else:
            ligand_token_atom_mask = torch.Tensor([[1.] + [0.] * 36 for i in range(unified_feat["ligand_mask"].sum())])
            unified_feat["atom_mask"] = torch.vstack([protein_feat["atom_mask"], ligand_token_atom_mask])
            ligand_token_num = unified_feat["ligand_mask"].sum()
            ligand_token_atom_position = torch.zeros((ligand_token_num, 37, 3), dtype=torch.float)
            ligand_token_atom_position[:, 0, :] = ligand_feat["ligand_atom_positions"]
            unified_feat["atom_positions"] = torch.cat([protein_feat["atom_positions"], ligand_token_atom_position], dim = 0)

            unified_feat["atomic_number"] = torch.zeros(unified_feat["atom_positions"].shape[:-1], dtype=torch.long)
            unified_feat["atomic_number"][torch.where(unified_feat["ligand_mask"])[0], 0] = ligand_feat["ligand_atom_type"]
        
        atom_types_37 = [
            'N', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'O', 'O', 'S', 'C',
            'C', 'C', 'N', 'N', 'O', 'O', 'S', 'C', 'C', 'C', 'C',
            'N', 'N', 'N', 'O', 'O', 'C', 'N', 'N', 'O', 'C', 'C',
            'C', 'N', 'O'
        ]
        unified_feat["atomic_number"][torch.where(unified_feat["protein_mask"])[0], :] = torch.LongTensor([[
            element_dict[atom]
            for atom in atom_types_37
        ]] * unified_feat["protein_mask"].sum())
        unified_feat["atomic_number"][(1 - unified_feat["atom_mask"]).bool()] = 0
        if "esm_repr" in protein_feat:
            unified_feat["esm_repr"] = torch.zeros(unified_feat["chain_index"].shape[0], protein_feat["esm_repr"].shape[-1], dtype=torch.float)
            unified_feat["esm_repr"][unified_feat["protein_mask"].bool(), :] = protein_feat["esm_repr"]

        return unified_feat
    

class AditDataFeatureTransform(FeatureTransform):

    def __call__(self, chain_feats, ref_chains, crop_methods):
        protein_feat, ligand_feat, label = self.split_feats(chain_feats, self.label_name)
        protein_feat, ligand_feat = self.get_ref_chain(protein_feat, ligand_feat, ref_chains)
        
        if not self.AA:
            protein_feat = self.mask_side_chain(protein_feat)
        protein_feat = self.patch_feats(protein_feat)
        
        if self.strip_missing_residues:
            protein_feat = self.strip_ends(protein_feat)
        
        if self.truncate_length is not None:
            protein_feat, ligand_feat = self.crop_data(
                protein_feat, ligand_feat, crop_methods, crop_size=self.truncate_length
            )

        # Recenter and scale atom positions
        if self.recenter_and_scale:
            protein_feat, ligand_feat = self.recenter_and_scale_coords(
                protein_feat, ligand_feat, coordinate_scale=self.coordinate_scale, eps=self.eps
            )
        
        if self.pretrain:
            protein_feat, ligand_feat = self.random_rotate_coords(protein_feat, ligand_feat)

        # Map to torch Tensor
        protein_feat = self.map_to_tensors(protein_feat)
        ligand_feat = self.map_to_tensors(ligand_feat)

        unified_feats = self.merge_feats(protein_feat, ligand_feat)

        if self.label_name is not None:
            unified_feats.update(label)
        
        return unified_feats

    @staticmethod
    def crop_data(protein_feat, ligand_feat, crop_methods, crop_size):
        random.shuffle(crop_methods)
        crop_method = crop_methods[0]
        if crop_method == 'continues':
            assert ligand_feat is None, 'only single chain prot uses continues crop'
            protein_feat = continues_crop(protein_feat, crop_size)

        elif crop_method == 'spatial':
            assert ligand_feat is None, 'only single chain prot uses spatial crop'
            protein_feat, ligand_feat = spatial_crop(
                protein_feat, ligand_feat, crop_size, interface = False
            )
            
        elif crop_method == 'spatial_interface':
            protein_feat, ligand_feat = spatial_crop(
                protein_feat, ligand_feat, crop_size, interface = True
            )

        else:
            raise ValueError('unknown crop method')

        return protein_feat, ligand_feat

    @staticmethod
    def split_feats(chain_feats, label_name):
        protein_feat = {
            k: v
            for k, v in chain_feats.items() if k not in [
                "ligand_atom_type", "ligand_atom_positions"
            ]
        }

        ligand_feat = {
            k: v
            for k, v in chain_feats.items() if k in [
                "ligand_atom_type", "ligand_atom_positions"
            ]
        }

        return protein_feat, ligand_feat, None

    @staticmethod
    def get_ref_chain(protein_feat, ligand_feat, ref_chains):
        prot_ref_chain = [c for c in ref_chains[:-1] if c != -1]
        ligand_ref_chain = ref_chains[-1]

        token_mask_prot = np.zeros_like(protein_feat['chain_index'])
        for c in prot_ref_chain:
            token_mask_prot[protein_feat['chain_index'] == c] = 1
        
        def get_subgraph(protein_feat, res_mask):
            for k in protein_feat.keys():
                protein_feat[k] = protein_feat[k][res_mask]
            return protein_feat
        
        protein_feat = get_subgraph(protein_feat, np.where(token_mask_prot)[0])
        if ligand_ref_chain == -1:
            ligand_feat = None
        else:
            ligand_feat = {
                k: v[ligand_ref_chain]
                for k, v in ligand_feat.items()
            }

        return protein_feat, ligand_feat


class LBAFeatureTransform(FeatureTransform):

    @staticmethod
    def split_feats(chain_feats, label_name):
        protein_feat = {
            k: v
            for k, v in chain_feats.items() if k not in [
                label_name, "ligand_atom_type", "ligand_atom_positions"
            ]
        }

        ligand_feat = {
            k: v
            for k, v in chain_feats.items() if k in [
                "ligand_atom_type", "ligand_atom_positions"
            ]
        }
        
        label = {
            label_name: torch.FloatTensor(np.array([chain_feats[label_name]])).view(-1)
        }

        return protein_feat, ligand_feat, label


class DavisFeatureTransform(LBAFeatureTransform):
    
    @staticmethod
    def split_feats(chain_feats, label_name):

        protein_feat, ligand_feat, label_value = chain_feats
        label = {
            label_name: torch.FloatTensor([label_value]).view(-1)
        }

        return protein_feat, ligand_feat, label
    
    @staticmethod
    def recenter_and_scale_coords(protein_feat, ligand_feat, coordinate_scale, eps=1e-8):
        # recenter and scale atom positions
        bb_pos = protein_feat['atom_positions'][:, CA_IDX]
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(protein_feat['seq_mask']) + eps)
        centered_pos = protein_feat['atom_positions'] - bb_center[None, None, :]
        scaled_pos = centered_pos * coordinate_scale
        protein_feat['atom_positions'] = scaled_pos * protein_feat['atom_mask'][..., None]

        ligand_center = np.sum(ligand_feat["ligand_atom_positions"], axis=0) / ligand_feat["ligand_atom_positions"].shape[0]
        ligand_feat["ligand_atom_positions"] = (ligand_feat["ligand_atom_positions"] - ligand_center[None, :]) * coordinate_scale

        return protein_feat, ligand_feat
    
    @staticmethod
    def random_rotate_coords(protein_feat, ligand_feat):
        rot_vec = np.random.randn(3)
        rot_vec = rot_vec / ((rot_vec ** 2).sum() ** 0.5 + 1e-9)

        rot_sigma = np.pi * 2 * np.random.rand()
        rot = rot_vec * rot_sigma
        rot_matrix = R.from_rotvec(rot).as_matrix()
        protein_feat['atom_positions'] = protein_feat['atom_positions'] @ rot_matrix

        rot_vec = np.random.randn(3)
        rot_vec = rot_vec / ((rot_vec ** 2).sum() ** 0.5 + 1e-9)

        rot_sigma = np.pi * 2 * np.random.rand()
        rot = rot_vec * rot_sigma
        rot_matrix = R.from_rotvec(rot).as_matrix()
        ligand_feat["ligand_atom_positions"] = ligand_feat["ligand_atom_positions"] @ rot_matrix

        return protein_feat, ligand_feat
