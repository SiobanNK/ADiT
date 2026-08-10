from torch import nn
from torch_geometric.nn import GATv2Conv


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
