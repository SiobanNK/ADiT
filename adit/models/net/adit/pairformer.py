from torch.nn import LayerNorm
import torch
from torch import nn
from adit.models.net.adit.common import LinearNoBias


class SimplePairFormer(nn.Module):
    def __init__(self, token_dim, token_pair_dim, dropout = 0.0):
        super(SimplePairFormer, self).__init__()
        self.token_dim = token_dim
        self.token_pair_dim = token_pair_dim

        self.token_norm = LayerNorm(self.token_dim)
        self.token_pair_norm = LayerNorm(self.token_pair_dim)
        self.linear_no_bias_0 = LinearNoBias(token_dim, token_dim)
        self.linear_no_bias_1 = LinearNoBias(token_dim * 2, token_pair_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_feat_compressed, edge_token, relative_position_embedding):
        normed_token_feat = self.token_norm(token_feat_compressed)
        s_i_init = self.linear_no_bias_0(normed_token_feat)
        s_i_init = self.relu(s_i_init)

        token_0 = normed_token_feat[edge_token[0]]
        token_1 = normed_token_feat[edge_token[1]]

        combined_tokens = torch.cat([token_0, token_1], dim=-1)
        z_ij_init = self.linear_no_bias_1(combined_tokens)
        z_ij_init = self.relu(z_ij_init)
        z_ij_init = self.dropout(z_ij_init)
        z_ij_init = z_ij_init + relative_position_embedding

        z_ij_init = self.token_pair_norm(z_ij_init)

        return s_i_init, z_ij_init
