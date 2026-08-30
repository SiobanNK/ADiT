import torch
from torch import nn
from torch_scatter import scatter_mean
from adit.models.net.adit.common import LinearNoBias

def create_one_hot_encoding(x, class_count):
    return torch.nn.functional.one_hot(x, num_classes=class_count).type(torch.float)


class RelativePositionEncoding(nn.Module):

    def __init__(self, token_pair_dim, q_max = 64, r_max = 32, s_max = 2, dropout = 0.0, token_coord_encoder = None, d_min = 0.2, d_max = 2.2):
        super(RelativePositionEncoding, self).__init__()
        self.q_max = q_max if token_coord_encoder else -1
        self.r_max = r_max
        self.s_max = s_max
        self.token_pair_dim = token_pair_dim
        self.linear_no_bias = LinearNoBias((self.q_max + 1) + (2 * self.r_max + 2) + (2 * self.s_max + 2), token_pair_dim)

        self.layer_norm = nn.LayerNorm((self.q_max + 1) + (2 * self.r_max + 2) + (2 * self.s_max + 2))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.token_coord_encoder = token_coord_encoder
        self.d_min = d_min
        self.d_max = d_max

    def rbf(self, dist, device="cpu"):
        rbf_dim = self.q_max + 1

        d_mu = torch.linspace(self.d_min, self.d_max, rbf_dim, device=device)
        d_mu = d_mu.view([1, -1])
        d_sigma = (self.d_max - self.d_min) / rbf_dim
        d_expand = torch.unsqueeze(dist, -1)

        rbf = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
        return rbf

    def forward(self, token_idx, token2chain, edge_token, token_coordinates):
        same_chain = token2chain[edge_token[0]] == token2chain[edge_token[1]]

        if self.token_coord_encoder :
            dist = ((token_coordinates[edge_token[0]] - token_coordinates[edge_token[1]]) ** 2).sum(dim=-1).sqrt()

            if self.token_coord_encoder == "onehot" :
                d_ij_3d = torch.clamp(dist.floor().long(), 0, self.d_max)   # distances entre tous les tokens, même de chaînes différentes.
                a_ij_rel_3d = create_one_hot_encoding(d_ij_3d, self.q_max + 1)

            elif self.token_coord_encoder == "rbf" :
                a_ij_rel_3d = self.rbf(dist, device=token_idx.device)

            else:
                raise ValueError(
                        f"Valid token_coord_encoder values are 'onehot' or 'rbf'. "
                        f"Got: {self.token_coord_encoder!r} (type: {type(self.token_coord_encoder).__name__})"
                    )

        d_ij_token = torch.where(
            same_chain,
            torch.clamp(token_idx[edge_token[0]] - token_idx[edge_token[1]] + self.r_max, 0, 2 * self.r_max),
            (2 * self.r_max + 1) * torch.ones_like(same_chain, device=token_idx.device, dtype=torch.long)
        )
        a_ij_rel_token = create_one_hot_encoding(d_ij_token, 2 * self.r_max + 2)

        d_ij_chain = torch.where(
            ~same_chain,
            torch.clamp(token2chain[edge_token[0]] - token2chain[edge_token[1]] + self.s_max, 0, 2 * self.s_max),
            (2 * self.s_max + 1) * torch.ones_like(same_chain, device=token_idx.device, dtype=torch.long)
        )
        a_ij_rel_chain = create_one_hot_encoding(d_ij_chain, 2 * self.s_max + 2)

        if self.token_coord_encoder :
            p_ij = torch.cat([a_ij_rel_3d, a_ij_rel_token, a_ij_rel_chain], dim=-1)
        else:
            p_ij = torch.cat([a_ij_rel_token, a_ij_rel_chain], dim=-1)
        p_ij = self.layer_norm(p_ij)
        p_ij = self.linear_no_bias(p_ij)
        p_ij = self.activation(p_ij)
        p_ij = self.dropout(p_ij)
        return p_ij
