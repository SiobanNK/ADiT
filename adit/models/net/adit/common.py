import torch
from torch import nn


class LinearNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)


class Transition(nn.Module):

    def __init__(self, indim, n = 4):
        super(Transition, self).__init__()
        self.linear_no_bias_0 = LinearNoBias(indim, indim * n)
        self.linear_no_bias_1 = LinearNoBias(indim, indim * n)
        self.linear_no_bias_2 = LinearNoBias(indim * n, indim)

        self.swish = nn.SiLU()
        self.layer_norm = nn.LayerNorm(indim)
        self.zero_init()

    def zero_init(self):
        nn.init.zeros_(self.linear_no_bias_2.weight)

    def forward(self, x):
        x = self.layer_norm(x)
        a = self.linear_no_bias_0(x)
        b = self.linear_no_bias_1(x)
        x = self.linear_no_bias_2(torch.mul(self.swish(a), b))
        return x


class AdaLN(nn.Module):
    """
    Implements Algorithm 26 in AF3
    refer to https://github.com/bytedance/Protenix
    """

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        """
        super(AdaLN, self).__init__()
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=False, bias=False)
        # The pytorch version should be newer than 2.1
        # self.layernorm_s = nn.LayerNorm(c_s, bias=False)
        self.layernorm_s = nn.LayerNorm(c_s)
        self.linear_s = nn.Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def zero_init(self):
        nn.init.zeros_(self.linear_s.weight)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_nobias_s.weight)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


class MLP_P_LM(nn.Module):
    
    def __init__(self, indim, hiddim=128):
        super(MLP_P_LM, self).__init__()
        self.linear_1 = LinearNoBias(indim, hiddim)
        self.linear_2 = LinearNoBias(hiddim, hiddim)
        self.linear_3 = LinearNoBias(hiddim, indim)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)

    def forward(self, Plm):
        x = self.relu(self.linear_1(Plm))

        x = self.relu(self.linear_2(x))

        x = self.linear_3(x)
        return x