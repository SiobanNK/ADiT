import torch
from torch import nn
from adit.models.net.adit.common import LinearNoBias

def create_one_hot_encoding(x, class_count):
    return torch.nn.functional.one_hot(x, num_classes=class_count).type(torch.float)

class RelativePositionEncoding(nn.Module):

    def __init__(self, token_pair_dim, r_max = 32, s_max = 2, dropout = 0.0):
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.token_pair_dim = token_pair_dim
        self.linear_no_bias = LinearNoBias((2 * self.r_max + 2) + (2 * self.s_max + 2), token_pair_dim)

        self.layer_norm = nn.LayerNorm((2 * self.r_max + 2) + (2 * self.s_max + 2))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_idx, token2chain, edge_token):
        # token_mask = batch["seq_mask"].bool()
        # token_edges_matrix, edge_token = generate_dense_attention_edge_batch(num_tokens)
        # token2chain = batch["chain_index"][token_mask],
        # token_idx = batch["token_idx"][token_mask]
        same_chain = token2chain[edge_token[0]] == token2chain[edge_token[1]]       # c'est quoi token2chain ? matrice ? et edge_token ?

        d_ij_token = torch.where(
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

        p_ij = torch.cat([a_ij_rel_token, a_ij_rel_chain], dim=-1)
        p_ij = self.layer_norm(p_ij)
        p_ij = self.linear_no_bias(p_ij)
        p_ij = self.activation(p_ij)
        p_ij = self.dropout(p_ij)
        return p_ij
