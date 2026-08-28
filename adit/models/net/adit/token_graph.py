import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv


class OptionalGNN(nn.Module):
    def __init__(
        self,
        active: bool,
        gnn_type: str,
        in_channels: int,
        neighbour_radius: float,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        num_rbf: int = 16,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_rbf = num_rbf
        self.gcn_sigma = neighbour_radius / 3

        self.norm = nn.LayerNorm(in_channels)

        if active and gnn_type == "gat":
            if concat:
                assert in_channels % heads == 0, (
                    f"in_channels ({in_channels}) must be divisible by heads ({heads})."
                )
                out_channels = in_channels // heads
            else:
                out_channels = in_channels

            self.gnn = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_self_loops=True,
                bias=False,
                share_weights=True,
                residual=False,
                edge_dim=num_rbf,  # RBF multi-centres
            )
            # fixed RBF centers, not learnt (buffer instead of Parameter)
            self.register_buffer(
                "rbf_centers", torch.linspace(0, neighbour_radius, num_rbf)
            )
            self.rbf_width = neighbour_radius / num_rbf

        elif active and gnn_type == "gcn":
            self.gnn = GCNConv(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=True,
            )
        else:
            self.gnn = None

        self.relu = nn.ReLU()

    def _rbf_expand(self, distance):
        # distance: (E,) -> (E, num_rbf)
        diff = distance.unsqueeze(-1) - self.rbf_centers
        return torch.exp(-(diff ** 2) / (2 * self.rbf_width ** 2))

    def forward(self, node_features, edge_index, edge_distance=None):
        if not self.gnn:
            return node_features

        x = self.norm(node_features)

        if self.gnn_type == "gat":
            edge_attr = self._rbf_expand(edge_distance) if edge_distance is not None else None
            message_passing = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:  # gcn
            edge_weight = torch.exp(-edge_distance / self.gcn_sigma) if edge_distance is not None else None
            message_passing = self.gnn(x, edge_index, edge_weight=edge_weight)

        update = self.relu(message_passing)
        return x + update
