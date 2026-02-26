import torch
from torch import nn

from adit.models.net.adit.common import Transition, LinearNoBias
from adit.models.net.adit.atom_attention import AtomAttentionEncoder, AtomAttentionDecoder
from adit.models.net.adit.diffusion_transformer import DiffusionTransformer


class DiffusionModule(nn.Module):

    def __init__(self, atom_dim, atom_pair_dim, token_dim, token_pair_dim, 
                 N_block_atom, N_head_atom, N_block_token, N_head_token, dropout=0.0):
        super(DiffusionModule, self).__init__()
        self.diffusion_conditioning = DiffusionConditioning(token_dim, token_pair_dim)
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_dim, atom_pair_dim, N_block_atom, N_head_atom, token_dim, token_pair_dim, dropout=dropout
        )
        
        self.linear_no_bias_0 = LinearNoBias(token_dim, token_dim)
        self.layernorm_0 = nn.LayerNorm(token_dim)
        self.diffusion_transformer = DiffusionTransformer(
            token_dim, token_dim, token_pair_dim, N_block_token, N_head_token
        )
        self.layernorm_1 = nn.LayerNorm(token_dim)
        
        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_dim, atom_pair_dim, N_block_atom, N_head_atom, token_dim
        )

    def forward(self, token_feat, token_pair_feat,
                atom_name, atomic_number, atom_coordinates, atom2token, atom_belong_to_protein, num_atoms, 
                edge, edge_token, edge_matrix_token):
        # conditioning
        token_feat, token_pair_feat = self.diffusion_conditioning(token_feat, token_pair_feat)
        
        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a_i, q_l_skip, c_l_skip, p_lm_skip = self.atom_attention_encoder(
            atom_name, atomic_number, atom_belong_to_protein, atom_coordinates, 
            atom2token, num_atoms, edge, token_feat, token_pair_feat, edge_matrix_token
        )

        # Full self-attention on token level
        a_i = a_i + self.linear_no_bias_0(self.layernorm_0(token_feat))
        a_i = self.layernorm_1(self.diffusion_transformer(
            a_i, token_feat, token_pair_feat, edge_token
        ))

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        q_l = self.atom_attention_decoder(a_i, q_l_skip, c_l_skip, p_lm_skip, atom2token, edge)
        return q_l


class DiffusionConditioning(nn.Module):

    def __init__(self, res_feat_dim, res_pair_feat_dim):
        super(DiffusionConditioning, self).__init__()
        self.layernorm_res_pair = nn.LayerNorm(res_pair_feat_dim)
        self.linear_no_bias_0 = nn.Linear(res_pair_feat_dim, res_pair_feat_dim)
        self.transition_0 = Transition(res_pair_feat_dim, 2)
        self.transition_1 = Transition(res_pair_feat_dim, 2)

        self.layernorm_res = nn.LayerNorm(res_feat_dim)
        self.linear_no_bias_1 = nn.Linear(res_feat_dim, res_feat_dim)
        self.transition_2 = Transition(res_feat_dim, 2)
        self.transition_3 = Transition(res_feat_dim, 2)

    def forward(self, res_feat, res_pair_feat):
        # pair conditioning
        # res_pair_feat = torch.concat([res_pair_feat, relative_position_embedding], dim = -1)
        res_pair_feat = self.linear_no_bias_0(self.layernorm_res_pair(res_pair_feat))
        res_pair_feat = res_pair_feat + self.transition_0(res_pair_feat)
        res_pair_feat = res_pair_feat + self.transition_1(res_pair_feat)

        # single conditioning
        # res_feat = torch.concat([res_feat, res_feat_trunk], dim=-1)
        res_feat = self.linear_no_bias_1(self.layernorm_res(res_feat))
        res_feat = res_feat + self.transition_2(res_feat)
        res_feat = res_feat + self.transition_3(res_feat)
        return res_feat, res_pair_feat
