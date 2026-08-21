# utils
import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_cluster import radius_graph

from adit.common import residue_constants, protein
CA_IDX = residue_constants.atom_order['CA']
CB_IDX = residue_constants.atom_order['CB']


def generate_sparse_attention_matrix(num_atom, N_query, N_key, device):
    """
    Generate a sparse attention matrix.
    """
    adj = torch.zeros((num_atom, num_atom), device=device, dtype=torch.bool)
    for i in range(num_atom // N_query + 1):
        left = max(i * N_query + N_query // 2 - N_key // 2, 0)
        right = min(i * N_query + N_query // 2 + N_key // 2, num_atom)
        adj[i * N_query: (i + 1) * N_query, left:right] = 1
    return adj

def generate_sparse_attention_edge_batch(num_tokens, num_atoms_per_token, N_query, N_key):
    """
    Generate sparse attention edge batch.
    """
    device = num_tokens.device
    batch_size = num_tokens.shape[0]
    num_cum_tokens = num_tokens.cumsum(dim=0)

    def create_adj_matrix(i):
        start = num_cum_tokens[i] - num_tokens[i]
        end = num_cum_tokens[i]
        num_atom = num_atoms_per_token[start:end].sum()
        return generate_sparse_attention_matrix(num_atom, N_query, N_key, device)

    adjs = (create_adj_matrix(i) for i in range(batch_size))
    adj = torch.block_diag(*adjs)
    edge = torch.nonzero(adj).transpose(0, 1)
    return edge





def generate_euclidian_edge_index(num_nodes, node_coordinates, neighbour_radius):
    # batch vector: which sample each atom belongs to, e.g. [0,0,0,1,1,2,2,2,2,...]
    batch = torch.repeat_interleave(
        torch.arange(num_nodes.shape[0], device=num_nodes.device), num_nodes
    )
    edge_index = radius_graph(
        node_coordinates,
        r=neighbour_radius,
        batch=batch,
        loop=True,       # keep self-loops, to match your original `dist < max_neighbour_dist` on the diagonal (dist=0)
        max_num_neighbors=32,
    )

    source, destination = edge_index
    edge_distance = (node_coordinates[source] - node_coordinates[destination]).norm(dim=-1)
    return edge_index, edge_distance


def generate_token_coordinates(atom_coordinates, atom2token, atom_positions, atom_mask, token_mask, T:str = "centroid"):
    if T == "CB":
        cb_poses = atom_positions[:, :, CB_IDX, :]        # (B, Lmax, 3)
        cb_mask = atom_mask[:, :, CB_IDX].unsqueeze(-1)   # (B, Lmax, 1), 1 si CB présent
        ca_poses = atom_positions[:, :, CA_IDX, :]
        res_poses = torch.where(cb_mask.bool(), cb_poses, ca_poses) # Calpha position if glycine / no Cbeta
        return res_poses[token_mask]    # (Ltot,3)

    elif T == "CA":
        ca_poses = atom_coordinates[:, :, CA_IDX, :]
        return res_poses[token_mask]

    elif T == "centroid": # centroids of heavy atoms
        return scatter_mean(atom_coordinates, atom2token, dim=0)

    else:
        raise ValueError('unknown token coordinate type')



def generate_dense_attention_matrix(num_token, device):
    """
    Generate a dense attention matrix.
    """
    return torch.ones((num_token, num_token), device=device, dtype=torch.bool)


def generate_dense_attention_edge_batch(num_tokens):
    """
    Generate dense attention edge batch.
    """
    device = num_tokens.device
    batch_size = num_tokens.shape[0]
    adjs = [generate_dense_attention_matrix(num_tokens[i], device) for i in range(batch_size)]
    adj = torch.block_diag(*adjs)
    edge = torch.nonzero(adj).transpose(0, 1)
    adj = torch.sparse_coo_tensor(indices=edge,
                                   values=torch.arange(edge.shape[1], device=device),
                                   size=[adj.shape[0], adj.shape[1]]).to_dense()
    return adj, edge



class LinearNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
