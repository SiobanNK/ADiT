import torch
import torch.nn as nn
from torch.nn import Dropout


class MLP4downstream(nn.Module):
    
    def __init__(self, input_dim, hiddim = 128, outdim = 1):
        super(MLP4downstream, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hiddim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddim, hiddim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddim, outdim),
        )

    def forward(self, Plm):
        x = self.mlp(Plm)
        return x
    

class MLP(nn.Module):
    
    def __init__(self, input_dim, hiddim = 256, dropout = 0.0):
        super(MLP, self).__init__()
        self.dropout = Dropout(dropout)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hiddim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddim, hiddim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddim, 1),
        )
        
    def forward(self, Plm):
        x = self.mlp(self.dropout(Plm))
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
