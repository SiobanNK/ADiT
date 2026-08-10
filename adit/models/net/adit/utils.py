# utils
import torch
from torch import nn

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


def generate_sparse_euclidian_attention_matrix(dist_matrix, max_neighbour_dist, device):
    """
    Generate a sparse binary adjacency matrix.
    """
    num_nodes = dist_matrix.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
    adj[dist_matrix < max_neighbour_dist] = 1
    return adj

def generate_sparse_euclidian_attention_edge_batch(dist_matrix, max_neighbour_dist):
    """
    Generate sparse attention edge batch.
    Nodes are atoms or tokens.
    """
    device = dist_matrix.device
    batch_size = dist_matrix.shape[0]

    adjs = (generate_sparse_euclidian_attention_matrix(dist_matrix[i], max_neighbour_dist, device) for i in range(batch_size))
    adj = torch.block_diag(*adjs)
    edge = torch.nonzero(adj).transpose(0, 1)
    return edge

def atom_euclidian_edge_index(num_atoms, atom_coordinates, max_neighbour_dist: float = 5.):
    dense_edges = generate_dense_attention_edge_batch(num_atoms)
    dist_matrix = ((atom_coordinates[dense_edges[0]] - atom_coordinates[dense_edges[1]]) ** 2).sum(dim=-1).sqrt()
    return generate_sparse_euclidian_attention_edge_batch(dist_matrix, max_neighbour_dist)

def token_euclidian_edge_index(token_edges, atom_coordinates, atom2token, max_neighbour_dist: float = 11.):
    token_coordinates = scatter_mean(atom_coordinates, atom2token, dim=0)
    dist_matrix = ((token_coordinates[token_edges[0]] - token_coordinates[token_edges[1]]) ** 2).sum(dim=-1).sqrt()
    return generate_sparse_euclidian_attention_edge_batch(dist_matrix, max_neighbour_dist)


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
