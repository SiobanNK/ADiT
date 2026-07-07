import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_sum, scatter_max

from adit.common import residue_constants
from adit.models.net.adit.pairformer import SimplePairFormer
from adit.models.net.adit.seq_embedder import SeqEmbedder
from adit.models.net.adit.relative_position_encoder import RelativePositionEncoding
from adit.models.net.adit.diffusion_module import DiffusionModule
from adit.models.net.adit.utils import generate_sparse_attention_edge_batch, generate_dense_attention_edge_batch


class ADiT(nn.Module):

    def __init__(
        self,
        token_dim, token_pair_dim, atom_dim, atom_pair_dim, # some hidden dim and final output dim
        N_block_atom, N_head_atom, N_block_token, N_head_token, # for diffusion transformer
        N_query = 32, N_key = 128, dropout = 0.0,
        esm_weight_path = None, esm_model = None,
        remove_protein_ligand_edge = False,
    ):
        super(ADiT, self).__init__()
        # basic
        self.token_dim = token_dim
        self.token_pair_dim = token_pair_dim
        self.atom_dim = atom_dim
        self.atom_pair_dim = atom_pair_dim
        self.remove_protein_ligand_edge = remove_protein_ligand_edge    # Used for unbound structures

        # atom local attention
        self.N_query = N_query
        self.N_key = N_key

        # modules
        self.seq_embedder = SeqEmbedder(self.token_dim, dropout = dropout, esm_weight_path = esm_weight_path, esm_model = esm_model)
        self.relative_position_encoder = RelativePositionEncoding(self.token_pair_dim, dropout = dropout)
        self.simple_pair_former = SimplePairFormer(token_dim, token_pair_dim)
        self.diffusion_module = DiffusionModule(
            atom_dim, atom_pair_dim, token_dim, token_pair_dim,
            N_block_atom, N_head_atom, N_block_token, N_head_token
        )

    def forward(self, batch):
        device = batch["aatype"].device
        batch_size = batch["aatype"].shape[0]

        token_mask = batch["seq_mask"].bool()
        num_tokens = batch["seq_mask"].sum(-1)
        num_atoms_per_token = batch["atom_mask"].sum(dim=-1).int()[token_mask]

        # edges
        atom_edges = generate_sparse_attention_edge_batch(num_tokens, num_atoms_per_token, self.N_query, self.N_key)
        if self.remove_protein_ligand_edge:
            token_type = batch["token_type"]
            token_type = token_type[token_mask]     # (num_token,)
            atom_mask = batch["atom_mask"].bool()
            atom_mask = atom_mask[token_mask]   # (num_token, 37)
            atom_token_type = token_type[..., None].expand_as(atom_mask)[atom_mask]     # (num_atom, )
            edge_in, edge_out = atom_edges[0], atom_edges[1]
            is_same_entity = atom_token_type[edge_in] == atom_token_type[edge_out]
            atom_edges = atom_edges[:, is_same_entity]
        token_edges_matrix, token_edges = generate_dense_attention_edge_batch(num_tokens)

        # token init feat
        token_feat = self.seq_embedder(
            batch["seq_mask"], batch["token_type"], batch["aatype"], batch["chain_index"], batch.get("esm_repr")
        )

        # relative position encoding: token2chain, token_idx
        token_idx = batch["token_idx"][token_mask]
        token2chain = batch["chain_index"][token_mask]
        relative_position_embedding = self.relative_position_encoder(token_idx, token2chain, token_edges, )

        # token pair repr
        token_feat, token_pair_feat = self.simple_pair_former(
            token_feat, token_edges, relative_position_embedding
        )

        # diffusion module
        atom_mask = batch["atom_mask"].bool()
        atom_name = torch.arange(residue_constants.atom_type_num, device=device)[None, None, :].expand_as(batch["atom_mask"])
        atom_name = atom_name[atom_mask]
        atom_belong_to_protein = (batch["protein_mask"].unsqueeze(-1) & atom_mask)[atom_mask].bool()
        atomic_number = batch["atomic_number"][atom_mask]
        atom_coordinates = batch["atom_positions"][atom_mask]
        num_tokens = batch["seq_mask"].sum(dim=-1)
        num_atoms = batch["atom_mask"].sum(-1)[token_mask].int()
        atom2token = torch.arange(num_tokens.sum(), device=device).repeat_interleave(num_atoms)

        out_atom_feat = self.diffusion_module(
            token_feat, token_pair_feat,
            atom_name, atomic_number, atom_coordinates, atom2token, atom_belong_to_protein, num_atoms,
            atom_edges, token_edges, token_edges_matrix
        )
        batch["atom_feat"] = out_atom_feat

        out_token_feat = scatter_mean(
            out_atom_feat,
            atom2token, dim=0, dim_size=num_tokens.sum()
        )
        batch["token_feat"] = out_token_feat

        token2complex = torch.arange(batch_size, device=device).repeat_interleave(num_tokens)
        out_complex_feat = scatter_sum(
            out_token_feat,
            token2complex, dim=0, dim_size=batch_size
        )
        batch["complex_feat"] = out_complex_feat
        return batch
