import torch
from torch import nn
from torch_scatter import scatter_sum
from torch_scatter.composite import scatter_softmax

from adit.models.net.adit.common import LinearNoBias, AdaLN

class AttentionPairBias(nn.Module):

    def __init__(self, single_dim_1, single_dim_2, pair_dim, N_head):
        super(AttentionPairBias, self).__init__()
        self.N_head = N_head
        self.single_dim = single_dim_1
        self.ada_ln = AdaLN(single_dim_1, single_dim_2)
        head_dim = single_dim_1 // N_head
        self.head_dim = head_dim
        self.linear_q = nn.Linear(single_dim_1, single_dim_1)
        self.linear_no_bias_k = LinearNoBias(single_dim_1, single_dim_1)
        self.linear_no_bias_v = LinearNoBias(single_dim_1, single_dim_1)
        self.linear_no_bias_g = LinearNoBias(single_dim_1, single_dim_1)
        self.linear_no_bias_zij = LinearNoBias(pair_dim, N_head)
        
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.linear_no_bias_a_i = LinearNoBias(single_dim_1, single_dim_1)
        self.output_projection_linear = nn.Linear(single_dim_2, single_dim_1)
        
    def forward(self, a_i, s_i, z_ij, edge):
        a_i = self.ada_ln(a_i, s_i) 
        N_token, d = a_i.shape
        assert d == self.N_head * self.head_dim
        multihead_shape = (N_token, self.N_head, self.head_dim)
        
        # Linear layers
        q_i = self.linear_q(a_i).view(*multihead_shape)
        k_i = self.linear_no_bias_k(a_i).view(*multihead_shape)
        v_i = self.linear_no_bias_v(a_i).view(*multihead_shape)
        g_i = self.linear_no_bias_g(a_i).sigmoid().view(*multihead_shape)
        b_ij = self.linear_no_bias_zij(z_ij).view(-1, self.N_head)    # (num_edge, N_head)

        # Attention scores
        A_ij_1_step = torch.einsum('mhd,mhd->mh', q_i[edge[0]], k_i[edge[1]]) / self.head_dim ** 0.5
        A_ij_2_step = A_ij_1_step + b_ij      # (num_edge, N_head)
        A_ij = scatter_softmax(src=A_ij_2_step, index=edge[0], dim=0, dim_size=a_i.shape[0])     # (num_edge, N_head)

        # Message aggregation
        # Memory-expensive, as we have to construct the message explicitly
        m_ij = A_ij[..., None] * v_i[edge[1]]
        u_i = scatter_sum(src=m_ij, index=edge[0], dim=0, dim_size=a_i.shape[0])     # (num_token, N_head, head_dim)
        '''
        # Memory-efficient, but much slower
        # (2, num_edge) -> (2, num_edge * N_head)
        indices = edge.repeat_interleave(self.N_head, dim=1) * self.N_head 
        indices = indices + torch.arange(self.N_head, device=edge.device).repeat(2, edge.shape[1])
        values = A_ij.flatten()     # (num_edge, )
        attn_weight = torch.sparse_coo_tensor(indices, values, size=(N_token * self.N_head, N_token * self.N_head))   # (num_token * N_head, num_token * N_head), sparse tensor
        message = v_i.view(N_token * self.N_head, self.head_dim)    # (num_token * N_head, head_dim)
        u_i = torch.sparse.mm(attn_weight, message)   # (num_token * N_head, head_dim)
        u_i = u_i.view(N_token, self.N_head, self.head_dim)      # (num_token, N_head, head_dim)
        '''
        attn_output = (g_i * u_i).view(N_token, d)

        a_i = self.linear_no_bias_a_i(attn_output)
        a_i = torch.mul(self.sigmoid(self.output_projection_linear(s_i)), a_i)
        return a_i


class ConditionedTransitionBlock(nn.Module):

    def __init__(self, single_dim_1, single_dim_2):
        super(ConditionedTransitionBlock, self).__init__()
        self.ada_ln = AdaLN(single_dim_1, single_dim_2)
        self.swish = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.linear_no_bias_0 = LinearNoBias(single_dim_1, 2 * single_dim_1)
        self.linear_no_bias_1 = LinearNoBias(single_dim_1, 2 * single_dim_1)
        self.linear_no_bias_2 = LinearNoBias(2 * single_dim_1, single_dim_1)
        self.linear = nn.Linear(single_dim_2, single_dim_1)

    def forward(self, a, s):
        a = self.ada_ln(a, s)
        b = torch.mul(self.swish(self.linear_no_bias_0(a)), self.linear_no_bias_1(a))
        a = torch.mul(self.sigmoid(self.linear(s)), self.linear_no_bias_2(b))
        return a
    

class DiffusionTransformer(nn.Module):

    def __init__(self, single_dim_1, single_dim_2, pair_dim, N_block, N_head):
        super(DiffusionTransformer, self).__init__()
        self.N_block = N_block
        self.N_head = N_head
        for i in range(0, N_block):
            self.add_module("attention_pair_bias_%d" % i, AttentionPairBias(single_dim_1, single_dim_2, pair_dim, N_head))
            self.add_module("conditioned_transition_block_%d" % i, ConditionedTransitionBlock(single_dim_1, single_dim_2))
        
    def forward(self, a_i, s_i, z_ij, edge):
        '''
            a_i: single representation
            s_i: condition representation
            z_ij: pair representation
        '''
        for i in range(self.N_block):
            b_i = a_i + self._modules["attention_pair_bias_%d" % i](a_i, s_i, z_ij, edge)
            a_i = b_i + self._modules["conditioned_transition_block_%d" % i](b_i, s_i)
        return a_i
