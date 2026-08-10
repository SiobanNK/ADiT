# utils
import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_cluster import radius_graph


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
        max_num_neighbors=num_nodes.max().item(),  # avoid silently truncating dense regions; tune if memory-bound
    )
    return edge_index


def generate_token_coordinates(atom_coordinates, atom2token):
    return scatter_mean(atom_coordinates, atom2token, dim=0)


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
