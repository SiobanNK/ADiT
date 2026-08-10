"""
Graph Convolutional Network pour l'encodage des tokens d'ADiT.

Un token = 1 residu ou 1 atome de biomolecule. Les tokens sont vus comme les
noeuds d'un graphe, connectes s'ils sont a moins de `threshold` Angstroms
l'un de l'autre.

References:
  - Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
    Networks", ICLR 2017 (arXiv:1609.02907) : regle de propagation
    renormalisee, Eq. 8 : A_hat = D~^{-1/2} (A + I) D~^{-1/2}.
  - Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017
    (arXiv:1704.01212) : framework MPNN (message / update / readout), et
    readout avec skip-connections sur tous les pas de temps (cf. Duvenaud
    et al. 2015, repris section 2).

Convention de seuil (a adapter selon le niveau de tokenisation) :
  - 5 Angstroms pour un graphe d'atomes.
  - 10 Angstroms pour un graphe de tokens/residus.
"""
import torch
from torch import nn
from adit.models.net.adit import scatter_mean
from torch.nn import LayerNorm, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv

from adit.models.net.adit.common import LinearNoBias


def generate_sparse_adjacency_matrix(dist_matrix, max_neighbour_dist, device):
    """
    Generate a sparse binary adjacency matrix.
    """
    num_nodes = dist_matrix.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
    adj[dist_matrix < max_neighbour_dist] = 1
    return adj

def generate_sparse_adjacency_edge_batch(num_nodes, max_neighbour_dist, dist_matrix):
    """
    Generate sparse attention edge batch.
    Nodes are atoms or tokens.
    """
    device = num_nodes.device
    batch_size = num_nodes.shape[0]
    num_cum_tokens = num_nodes.cumsum(dim=0)

    def create_adj_matrix(i):
        return generate_sparse_adjacency_matrix(dist_matrix, max_neighbour_dist, device)

    adjs = (create_adj_matrix(i) for i in range(batch_size))
    adj = torch.block_diag(*adjs)
    edge = torch.nonzero(adj).transpose(0, 1)
    return edge


class GAT(nn.Module):
    def __init__(
        self,
        in_channels = -1,
        out_channels,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = False,
        share_weights: bool = True,
        residual: bool = True   # learnable skip connection
    ):
        super.__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads, concat, dropout, add_self_loops, bias, share_weights, residual)



class GCN(nn.Module):
    """
    Encodeur GCN multi-couches pour les tokens d'ADiT (residus ou atomes).

    Chaque pas de propagation suit le renormalization trick de Kipf & Welling :
        H^{t+1} = ReLU( A_hat @ (H^t W_t) )
    avec LayerNorm + Dropout pour stabiliser l'entrainement en profondeur
    (cf. Kipf & Welling, Appendix B, sur l'interet des connexions residuelles
    au-dela de ~5-7 couches).

    Le readout final concatene les etats de chaque pas de temps (y compris
    l'etat initial projete), a la maniere des skip-connections utilisees par
    Duvenaud et al. (2015) et discutees dans Gilmer et al. (2017, Section 2),
    puis reprojette vers `token_dim`.
    """

    def __init__(
        self,
        token_dim: int, # in_channels: Size of each input sample (-1 to derive the size from the first forward method)
        message_dim: int,   # out_channels : Size of each output sample
        time_steps: int = 3,
        dist_threshold: float = 10.0,
        dropout: float = 0.1,
        cached: bbol = True,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = False,
        tie_weights: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.dist_threshold = dist_threshold
        self.tie_weights = tie_weights
        self.use_residual = use_residual

        self.gcn = GCNConv(token_dim, 3)

    def forward(
        self,
        node_features: torch.Tensor,          # (B, N, token_dim)
        node_distance_matrix: torch.Tensor,   # (B, N, N)
        seq_mask: torch.Tensor,                # (B, N)
    ) -> torch.Tensor:
        """
        Args:
            node_features: features initiales par node (par ex. embeddings
                ESM par residu, ou features atomiques construites en amont a
                partir de aatype / token_type / chain_index).
            node_distance_matrix: distances euclidiennes par paire de
                nodes (Angstrom) — CA-CA pour des residus, coordonnees
                atomiques pour des atomes.
            seq_mask: 1 pour un token valide, 0 pour le padding.

        Returns:
            (B, N, node_dim) embeddings de tokens mis a jour.
        """
        edge_index = generate_sparse_adjacency_edge_batch(num_tokens, num_atoms_per_token, max_neighbour_dist)
        embedding = self.gcn(node_features, edge_index) # edge_weight optional
        return embedding



class AtomGAT(GAT):
    pass


class TokenGAT(GAT):
    def compute_token_dist_matrix(atom_coordinates, atom2token, edge_token):
        """
        Atom coordinates are already centered and rescaled to the unit : angstrom (cf file data/components/feature_transform.py)
        """
        token_coordinates = scatter_mean(atom_coordinates, atom2token, dim=0)
        dist = ((token_coordinates[edge_token[0]] - token_coordinates[edge_token[1]]) ** 2).sum(dim=-1).sqrt()
        return dist # output shape : (num_tokens, 3)

    def forward(self, token_features, edge_index, edge_attr, return_attentiuon_weights):
        edge_index = generate_sparse_adjacency_edge_batch(num_tokens, num_atoms_per_token, max_neighbour_dist)
        embedding = self.gat(token_features, edge_index)
        return embedding
