import torch
from torch import nn
from torch_scatter import scatter_mean
from adit.models.net.adit.common import LinearNoBias

def create_one_hot_encoding(x, class_count):
    return torch.nn.functional.one_hot(x, num_classes=class_count).type(torch.float)

def compute_token_coordinates(atom_coordinates, atom2token, scaling = 0.1):
    """
    Atom coordinates are already centered and rescaled to the unit : angstrom (cf file data/components/feature_transform.py)
    Taille d'une protéine : 100 - 1000 angtroms.
    Pour avoir 64 classes : tokens classés de distance n°0 à distance n°64, puis classe n°65 correspond aux tokens plus éloignés dans une même chaîne.
    Si les 65 premières classes couvrent les distances 0 - 500 angstroms:
        500 / 65 = environ 8 = environ 10
    Donc diviser par 10 les coordonnées (déjà centrées il me semble) et prendre l'int des distances.
    Pb: adaptable à d'autres molécules ?
    """
    centroid = scatter_mean(atom_coordinates, atom2token, dim=0)
    return centroid * scaling # output shape : (num_tokens, 3)

class RelativePositionEncoding(nn.Module):

    def __init__(self, token_pair_dim, r_max = 32, s_max = 2, dropout = 0.0):
        super(RelativePositionEncoding, self).__init__()
        self.q_max = 2 * r_max
        self.r_max = r_max
        self.s_max = s_max
        self.token_pair_dim = token_pair_dim
        self.linear_no_bias = LinearNoBias((self.q_max + 1) + (2 * self.r_max + 2) + (2 * self.s_max + 2), token_pair_dim)

        self.layer_norm = nn.LayerNorm((self.q_max + 1) + (2 * self.r_max + 2) + (2 * self.s_max + 2))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_idx, token2chain, edge_token, atom_coordinates, atom2token):
        same_chain = token2chain[edge_token[0]] == token2chain[edge_token[1]]

        token_coordinates = compute_token_coordinates(atom_coordinates, atom2token) # shape : (num_tokens,3)
        dist = ((token_coordinates[edge_token[0]] - token_coordinates[edge_token[1]]) ** 2).sum(dim=-1).sqrt()
        # d_ij_3d = torch.where(
        #     same_chain, # condition
        #     torch.clamp(dist.floor().long(), 0, self.q_max),
        #     (self.q_max + 1) * torch.ones_like(same_chain, device=token_idx.device, dtype=torch.long)
        # )
        d_ij_3d = torch.clamp(dist.floor().long(), 0, self.q_max)   # distances entre tous les tokens, même de chaînes différentes.
        a_ij_rel_3d = create_one_hot_encoding(d_ij_3d, self.q_max + 1)

        d_ij_token = torch.where(   # signed, symmetric relative distance as a non-negative bucket index for one-hot encoding
            same_chain, # condition
            torch.clamp(token_idx[edge_token[0]] - token_idx[edge_token[1]] + self.r_max, 0, 2 * self.r_max),   # input. pourquoi clamp ?
            (2 * self.r_max + 1) * torch.ones_like(same_chain, device=token_idx.device, dtype=torch.long)       # other if condition not met
        )
        a_ij_rel_token = create_one_hot_encoding(d_ij_token, 2 * self.r_max + 2)

        d_ij_chain = torch.where(
            ~same_chain,
            torch.clamp(token2chain[edge_token[0]] - token2chain[edge_token[1]] + self.s_max, 0, 2 * self.s_max),
            (2 * self.s_max + 1) * torch.ones_like(same_chain, device=token_idx.device, dtype=torch.long)
        )
        a_ij_rel_chain = create_one_hot_encoding(d_ij_chain, 2 * self.s_max + 2)

        p_ij = torch.cat([a_ij_rel_3d, a_ij_rel_token, a_ij_rel_chain], dim=-1)
        p_ij = self.layer_norm(p_ij)
        p_ij = self.linear_no_bias(p_ij)
        p_ij = self.activation(p_ij)
        p_ij = self.dropout(p_ij)
        return p_ij
