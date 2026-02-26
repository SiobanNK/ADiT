import torch
from torch import nn
from torch_scatter import scatter_mean

from adit.models.net.adit.common import LinearNoBias, MLP_P_LM
from adit.models.net.adit.diffusion_transformer import DiffusionTransformer
from adit.common import residue_constants


class AtomAttentionEncoder(nn.Module):

    def __init__(
        self, 
        atom_dim, 
        atom_pair_dim, 
        N_block, 
        N_head, 
        token_dim, 
        token_pair_dim, 
        atom_emb_dim = 64,
        dropout = 0.0,
    ):
        super(AtomAttentionEncoder, self).__init__()
        self.atom_name_embedding = nn.Embedding(num_embeddings=residue_constants.atom_type_num, embedding_dim=atom_emb_dim)
        self.atom_type_embedding = nn.Embedding(num_embeddings=120, embedding_dim=atom_emb_dim)
        self.linear_atom_input = LinearNoBias(atom_emb_dim * 2, atom_dim)
        self.linear_no_bias_coords = LinearNoBias(3, atom_dim)

        self.rbf_dim = 16
        
        self.linear_D_lm = LinearNoBias(3, atom_pair_dim)
        self.linear_dist_D_lm = LinearNoBias(self.rbf_dim, atom_pair_dim)
        self.linear_v_lm = LinearNoBias(1, atom_pair_dim)

        self.linear_c_l = LinearNoBias(atom_dim, atom_pair_dim)
        self.linear_c_m = LinearNoBias(atom_dim, atom_pair_dim)
        self.mlp_P_lm = MLP_P_LM(atom_pair_dim)

        self.diffusion_transformer = DiffusionTransformer(atom_dim, atom_dim, atom_pair_dim, N_block, N_head)

        self.linear_no_bias_Q_l = LinearNoBias(atom_dim, token_dim)
        self.relu = nn.ReLU()

        self.linear_no_bias_token2atom = LinearNoBias(token_dim, atom_dim)
        self.linear_no_bias_tokenpair2atompair = LinearNoBias(token_pair_dim, atom_pair_dim)
        self.layernorm_token = nn.LayerNorm(token_dim)
        self.layernorm_tokenpair = nn.LayerNorm(token_pair_dim)
        self.dropout = nn.Dropout(dropout)

    def rbf(self, d, d_min=0.0, d_max=2.0, device="cpu"):
        # We'll use nm instead of angstorm here.
        d_mu = torch.linspace(d_min, d_max, self.rbf_dim, device=device)
        d_mu = d_mu.view([1, -1])
        d_sigma = (d_max - d_min) / self.rbf_dim
        d_expand = torch.unsqueeze(d, -1)

        rbf = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
        return rbf

    def forward(
        self, atom_name, atomic_number, atom_belong_to_protein, atom_coordinates, 
        atom2token, num_atoms, edge, token_feat_trunk = None, token_feat_pair = None, edge_matrix_token = None
    ):
        '''
        edge_token is a matrix, not indices
        '''
        _atom_name_emb = self.atom_name_embedding(atom_name)
        atom_name_emb = torch.where(
            atom_belong_to_protein.unsqueeze(-1),
            _atom_name_emb,
            torch.zeros_like(_atom_name_emb, device = _atom_name_emb.device)
        )
        atom_type_emb = self.atom_type_embedding(atomic_number)

        atom_name_xyz_emb = torch.cat([atom_name_emb, atom_type_emb], dim=-1)
        C_l = self.linear_atom_input(atom_name_xyz_emb)

        v_lm = (atom2token[edge[0]] == atom2token[edge[1]]).view(-1, 1) 
        D_lm = atom_coordinates[edge[0]] - atom_coordinates[edge[1]]
        dist = (D_lm ** 2).sum(dim=-1).sqrt()
        edge_attr_dist = self.rbf(dist, device=atom_name_emb.device)

        P_lm = self.linear_D_lm(D_lm) + self.linear_dist_D_lm(edge_attr_dist) + self.linear_v_lm(v_lm.float())

        Q_l = C_l.clone()
        if token_feat_trunk is not None:
            C_l = C_l + self.linear_no_bias_token2atom(self.dropout(self.layernorm_token(token_feat_trunk[atom2token])))
            P_lm = P_lm + self.linear_no_bias_tokenpair2atompair(
                self.layernorm_tokenpair(
                    self.dropout(token_feat_pair[edge_matrix_token[atom2token[edge[0]], atom2token[edge[1]]]])
                )
            )
        Q_l = Q_l + self.linear_no_bias_coords(atom_coordinates)

        P_lm = P_lm + self.linear_c_l(C_l[edge[0]]) + self.linear_c_m(C_l[edge[1]])
        P_lm = P_lm + self.mlp_P_lm(P_lm)

        Q_l = self.diffusion_transformer(Q_l, C_l, P_lm, edge)  # (num_atom, atom_dim)

        # Mean pooling for each residue
        A_i = scatter_mean(self.relu(self.linear_no_bias_Q_l(Q_l)), atom2token, dim=0, dim_size=num_atoms.shape[0])  # (num_res, token_dim)
        return A_i, Q_l, C_l, P_lm
    

class AtomAttentionDecoder(nn.Module):

    def __init__(self, atom_dim, atom_pair_dim, N_block, N_head, token_dim):
        super(AtomAttentionDecoder, self).__init__()
        self.diffusion_transformer = DiffusionTransformer(atom_dim, atom_dim, atom_pair_dim, N_block, N_head)
        
        self.linear_no_bias_token2atom = LinearNoBias(token_dim, atom_dim)

    def forward(self, a_i, q_l_skip, c_l_skip, p_lm_skip, atom2residue, edge):
        q_l = self.linear_no_bias_token2atom(a_i[atom2residue]) + q_l_skip
        q_l = self.diffusion_transformer(q_l, c_l_skip, p_lm_skip, edge)

        return q_l
