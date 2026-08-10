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

from adit.models.net.adit.common import LinearNoBias

def compute_token_dist_matrix(atom_coordinates, atom2token, edge_token):
    """
    Atom coordinates are already centered and rescaled to the unit : angstrom (cf file data/components/feature_transform.py)
    """
    token_coordinates = scatter_mean(atom_coordinates, atom2token, dim=0)
    dist = ((token_coordinates[edge_token[0]] - token_coordinates[edge_token[1]]) ** 2).sum(dim=-1).sqrt()
    return dist # output shape : (num_tokens, 3)

def normalized_looped_adjacency_matrix(
    token_distance_matrix: torch.Tensor,
    seq_mask: torch.Tensor,
    threshold: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Construit la matrice d'adjacence renormalisee de Kipf & Welling (2017),
    Eq. 8 : A_hat = D~^{-1/2} (A + I) D~^{-1/2}

    Args:
        token_distance_matrix: (B, N, N) distances euclidiennes par paire.
        seq_mask: (B, N) 1 pour un token valide, 0 pour le padding.
        threshold: distance (Angstrom) au-dela de laquelle il n'y a pas
            d'arete. 5.0 pour un graphe d'atomes, 10.0 pour un graphe de
            tokens/residus.
        eps: stabilite numerique pour la normalisation par le degre.

    Returns:
        (B, N, N) matrice d'adjacence normalisee avec self-loops, mise a
        zero sur les lignes/colonnes correspondant au padding.
    """
    B, N, _ = token_distance_matrix.shape
    seq_mask = seq_mask.to(token_distance_matrix.dtype)
    pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # (B, N, N)

    # adjacence binaire : token connectes s'ils sont strictement en dessous
    # du seuil, hors diagonale (les self-loops sont ajoutes explicitement)
    eye = torch.eye(N, device=token_distance_matrix.device, dtype=token_distance_matrix.dtype)
    eye = eye.unsqueeze(0)  # (1, N, N), broadcast sur le batch

    adjacency = (token_distance_matrix < threshold).to(token_distance_matrix.dtype)
    adjacency = adjacency * (1.0 - eye)

    # A~ = A + I
    looped_adjacency = adjacency + eye

    # on retire toute arete touchant un token de padding
    looped_adjacency = looped_adjacency * pair_mask

    # D~_ii = somme_j A~_ij
    degree = looped_adjacency.sum(dim=-1)  # (B, N)
    degree_inv_sqrt = torch.pow(degree.clamp(min=eps), -0.5)
    degree_inv_sqrt = degree_inv_sqrt * seq_mask  # annule le degre des tokens de padding

    normalized_adjacency = (
        degree_inv_sqrt.unsqueeze(-1) * looped_adjacency * degree_inv_sqrt.unsqueeze(-2)
    )
    return normalized_adjacency


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
        token_dim: int,
        message_dim: int,
        time_steps: int = 3,
        threshold: float = 10.0,
        dropout: float = 0.1,
        tie_weights: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.threshold = threshold
        self.tie_weights = tie_weights
        self.use_residual = use_residual

        # projection des features d'entree dans l'espace de message
        self.input_proj = LinearNoBias(token_dim, message_dim)

        # message passing : une couche par pas de temps, ou une seule couche
        # partagee (weight tying, convention GG-NN / Kipf & Welling)
        n_layers = 1 if tie_weights else time_steps
        self.linearNoBias_message = nn.ModuleList(
            [LinearNoBias(message_dim, message_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([LayerNorm(message_dim) for _ in range(n_layers)])
        self.activation = nn.ReLU()
        self.dropout = Dropout(dropout)

        # readout : concatenation des T+1 etats (skip connections) puis
        # reprojection vers token_dim
        self.linearNoBias_readout = LinearNoBias(message_dim * (time_steps + 1), token_dim)
        self.readout_norm = LayerNorm(token_dim)

    def _layer(self, t: int):
        idx = 0 if self.tie_weights else t
        return self.linearNoBias_message[idx], self.norms[idx]

    def forward(
        self,
        token_features: torch.Tensor,          # (B, N, token_dim)
        token_distance_matrix: torch.Tensor,   # (B, N, N)
        seq_mask: torch.Tensor,                # (B, N)
    ) -> torch.Tensor:
        """
        Args:
            token_features: features initiales par token (par ex. embeddings
                ESM par residu, ou features atomiques construites en amont a
                partir de aatype / token_type / chain_index).
            token_distance_matrix: distances euclidiennes par paire de
                tokens (Angstrom) — CA-CA pour des residus, coordonnees
                atomiques pour des atomes.
            seq_mask: 1 pour un token valide, 0 pour le padding.

        Returns:
            (B, N, token_dim) embeddings de tokens mis a jour.
        """
        seq_mask_f = seq_mask.to(token_features.dtype)
        adjacency = normalized_looped_adjacency_matrix(
            token_distance_matrix, seq_mask, self.threshold
        )

        h = self.input_proj(token_features)
        h = h * seq_mask_f.unsqueeze(-1)
        skip_connections = [h]

        for t in range(self.time_steps):
            linear, norm = self._layer(t)

            # message : projection lineaire des features
            message = linear(h)
            # agregation : multiplication par la matrice d'adjacence normalisee
            message = torch.bmm(adjacency, message)

            updated = self.activation(norm(message))
            updated = self.dropout(updated)

            h = (h + updated) if self.use_residual else updated
            h = h * seq_mask_f.unsqueeze(-1)
            skip_connections.append(h)

        readout_input = torch.cat(skip_connections, dim=-1)
        token_embedding = self.linearNoBias_readout(readout_input)
        token_embedding = self.readout_norm(token_embedding)
        token_embedding = token_embedding * seq_mask_f.unsqueeze(-1)

        return token_embedding
