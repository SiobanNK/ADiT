import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv


# class OptionalGNN(nn.Module):
#     def __init__(
#         self,
#         active: bool,
#         gnn_type: str,
#         in_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         dropout: float = 0.0
#     ):
#         super().__init__()
#         self.gnn_type = gnn_type

#         self.norm = nn.LayerNorm(in_channels)

#         if active and gnn_type == "gat":
#             # Handle out_channels logic dynamically based on concat mode
#             if concat:
#                 assert in_channels % heads == 0, (
#                     f"in_channels ({in_channels}) must be divisible by heads ({heads})."
#                 )
#                 out_channels = in_channels // heads
#             else:
#                 out_channels = in_channels

#             self.gnn = GATv2Conv(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 heads=heads,
#                 concat=concat,
#                 dropout=dropout,
#                 add_self_loops=True,
#                 bias=False,
#                 share_weights=True,
#                 residual=False,
#                 edge_dim=1
#             )

#         elif active and gnn_type == "gcn":
#             self.gnn = GCNConv(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 cached=True,
#                 bias=True
#             )
#         else:
#             self.gnn = None

#         self.relu = nn.ReLU()


#     def forward(self, node_feat, edge_index, edge_feat):
#         if not self.gnn:
#             return node_feat

#         x = self.norm(node_feat)
#         if self.gnn_type == "gat":
#             x_gnn = self.gnn(x, edge_index, edge_attr=edge_feat.unsqueeze(-1) if edge_feat is not None else None)
#         else:   # gcn
#             x_gnn = self.gnn(x, edge_index, edge_weight=edge_feat)
#         x_gnn = self.relu(x_gnn)

#         return x + x_gnn



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




class OptionalGAT(nn.Module):
    def __init__(
        self,
        active: bool,
        in_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = False,
        share_weights: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.active = active

        if active:
            # Handle out_channels logic dynamically based on concat mode
            if concat:
                assert in_channels % heads == 0, (
                    f"in_channels ({in_channels}) must be divisible by heads ({heads})."
                )
                out_channels = in_channels // heads
            else:
                out_channels = in_channels

            self.gat = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_self_loops=add_self_loops,
                bias=bias,
                share_weights=share_weights,
                residual=residual
            )
        else:
            self.gat = None


    def forward(self, node_features, edge_index):
        if not self.active:
            return node_features

        return self.gat(node_features, edge_index)
