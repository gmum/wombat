import torch
from torch_geometric.nn import MessagePassing


class CheckAllNonZero(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # return torch.min(torch.relu(x), dim=self.dim).values
        return torch.prod(torch.relu(x), dim=self.dim)


def deal_with_pyg(*args, **kwargs):
    # PyG explainer loves to mess up order of arguments...
    if len(args) == 2:
        node_features, edge_index = args
        edge_features = kwargs["edge_features"]
    elif len(args) == 3:
        node_features, edge_features, edge_index = args
    else:
        raise ValueError("Wrong number of arguments")

    return node_features, edge_features, edge_index


class MPNN(MessagePassing):
    def __init__(self, x_i_transform: torch.Tensor, x_n_transform: torch.Tensor, e_transform: torch.Tensor,
                 bias: torch.Tensor = None):
        super().__init__(aggr="sum")

        self.x_i_transform = torch.nn.Parameter(x_i_transform, requires_grad=False)
        self.x_n_transform = torch.nn.Parameter(x_n_transform, requires_grad=False)
        self.e_transform = torch.nn.Parameter(e_transform, requires_grad=False)
        self.nonzero_checker = CheckAllNonZero(dim=0)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = torch.nn.Parameter(torch.zeros(x_i_transform.shape[1]), requires_grad=False)

    def forward(self, x, edge_features, edge_index):
        # x (N, in_channels)
        # edge_features (E, edge channels)
        # edge_index (2, E)

        edge_index = edge_index.long()

        pre_bias = self.propagate(edge_index, x=x, e=edge_features)

        return torch.nn.functional.relu(pre_bias + self.bias)

    def message(self, x_i, x_j, e):
        # x_i (E, in_channels)
        # x_j ^

        # x_i_transform (in_channels, out_channels)
        # x_j_transform ^

        x_i = x_i @ self.x_i_transform
        x_j = x_j @ self.x_n_transform
        e = e @ self.e_transform

        x = torch.stack([x_i, x_j, e], dim=0)
        x = self.nonzero_checker(x)

        return x


class AllNonZeroMaxReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nonzero_checker = CheckAllNonZero(dim=-1)

    def forward(self, x):
        # answer = torch.unsqueeze(torch.max(check_all_nonzero(x, dim=1)), dim=0)
        # return answer

        nonzero = self.nonzero_checker(x)

        return torch.max(nonzero, -1).values


class AllNonZeroReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nonzero_checker = CheckAllNonZero(dim=-1)

    def forward(self, x):
        # answer = torch.unsqueeze(torch.max(check_all_nonzero(x, dim=1)), dim=0)
        # return answer

        nonzero = self.nonzero_checker(x)
        zero = torch.relu(1 - nonzero)

        nonzero = torch.prod(zero, dim=-1)  # If any is NONZERO then zero will have a zero.

        return torch.relu(1 - nonzero)


class AllNonZeroSumReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nonzero_checker = CheckAllNonZero(dim=-1)

    def forward(self, x):
        nonzero = self.nonzero_checker(x)

        return torch.sum(nonzero, dim=-1)
