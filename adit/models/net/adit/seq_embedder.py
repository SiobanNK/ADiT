# sequence embedder
import torch
from torch import nn
from torch.nn import LayerNorm, Dropout
from torch_scatter import scatter_add

from adit.common import residue_constants
from adit.models.net.esm2 import esm
from adit.models.net.adit.common import LinearNoBias


def calculate_num_residues_per_chain(residue2chain, num_residues, token_type):
    device = residue2chain.device
    num_residues_per_chain = []
    right = num_residues.cumsum(-1)
    left = right - num_residues
    for batch_idx in range(num_residues.shape[0]):
        left_batch = left[batch_idx]
        right_batch = right[batch_idx]
        condition_chain = torch.cat(
            [
                torch.tensor([True], device=device), 
                residue2chain[left_batch:right_batch - 1] != residue2chain[left_batch + 1:right_batch]
            ], dim = -1
        )
        condition_token = torch.cat(
            [
                torch.tensor([True], device=device), 
                token_type[left_batch:right_batch - 1] != token_type[left_batch + 1:right_batch]
            ], dim = -1
        )
        strat_points = condition_chain | condition_token
        strat_points = torch.where(strat_points)[0]
        end_points = torch.cat([strat_points[1:], torch.tensor([right_batch - left_batch], device=device)])
        num_residues_per_chain.append(end_points - strat_points)
    num_residues_per_chain = torch.cat(num_residues_per_chain)
    return num_residues_per_chain.long()


class SeqEmbedder(nn.Module):

    def __init__(self, token_dim, dropout = 0.0, esm_weight_path = None, esm_model = None):
        super(SeqEmbedder, self).__init__()

        if esm_weight_path is not None:
            self.esm_encoder = esm.ESM(path = esm_weight_path, model = esm_model)
            # Freeze ESM parameters
            for p in self.esm_encoder.parameters():
                p.requires_grad_(False)
            self.linearNoBias = LinearNoBias(self.esm_encoder.output_dim, token_dim)
        else:
            self.esm_encoder = None
            self.res_embed = nn.Embedding(
                num_embeddings = residue_constants.restype_num + 2, 
                embedding_dim = token_dim
            )
        
        self.token_type_embed = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = 16
        )
        self.linearNoBias_out = LinearNoBias(token_dim + 16, token_dim)

        self.layer_norm1 = LayerNorm(token_dim + 16)
        # self.layer_norm2 = LayerNorm(token_dim)
        self.activation = nn.ReLU()
        self.dropout = Dropout(dropout)
                
    def forward(self, seq_mask, token_type, aatype, chain_index, esm_repr = None):
        token_mask = seq_mask.bool()
        token_type_compressed = token_type[token_mask]
        token_type_embeddings = self.token_type_embed(token_type_compressed)

        num_tokens = seq_mask.sum(-1)
        num_residues_per_chain = calculate_num_residues_per_chain(chain_index[token_mask], num_tokens, token_type_compressed)

        if self.esm_encoder:
            with torch.no_grad():
                if esm_repr is None:
                    esm_feat = self.esm_encoder(
                        aatype[token_mask], num_residues_per_chain
                    )
                else:
                    esm_feat = esm_repr[token_mask]
            res_emb_feat = self.linearNoBias(esm_feat)
        else:
            res_emb_feat = self.res_embed(aatype[token_mask])
        
        # for atom token, remove res embedding
        _res_emb_feat = torch.where(
            token_type_compressed.eq(1).unsqueeze(-1),
            torch.zeros_like(res_emb_feat),
            res_emb_feat
        )

        token_embedding = torch.cat([token_type_embeddings, _res_emb_feat], dim=-1)
        token_embedding = self.layer_norm1(token_embedding)
        token_embedding = self.linearNoBias_out(token_embedding)
        token_embedding = self.activation(token_embedding)
        token_embedding = self.dropout(token_embedding)
        # token_embedding = self.layer_norm2(token_embedding)

        return token_embedding
