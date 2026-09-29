import torch
import torch_geometric

from mpnn import one_hot_encode, mol_to_torch, smiles_to_torch, deal_with_pyg


class FluCarbGINE(torch.nn.Module):

    @staticmethod
    def get_embedding_weights() -> tuple[torch.Tensor, torch.Tensor]:
        carbon_embedding = [
            0.0,  # Not Fluorine
            0.0,  # Not Fluorine
            -10.0,  # Not Oxygen
            1.0,  # Carbon
            0.0  # Not a double bond,
        ]

        oxygen_embedding = [
            0.0,
            0.0,
            0.0,  # Oxygen
            0.0,  # Not carbon, but we want it to be connected to carbon
            0.0,
        ]

        fluorine_embedding = [
            1.0,  # Fluorine
            -1.0,
            # Also fluorine, but we don't want F's activation to happen in Fluorine as we want to make sure single bond connection happens
            -10.0,
            0.0,
            0.0,
        ]

        double_bond_embedding = [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,  # Double bond
        ]

        something_else_embedding = [0.0] * len(carbon_embedding)
        something_else_embedding[3] = -10.0

        x_embeddings = [something_else_embedding] * 100
        x_embeddings[6] = carbon_embedding
        x_embeddings[8] = oxygen_embedding
        x_embeddings[9] = fluorine_embedding

        edge_features_embeddings = [something_else_embedding] * 4
        edge_features_embeddings[1] = double_bond_embedding

        return torch.tensor(x_embeddings).T, torch.tensor(edge_features_embeddings).T

    @staticmethod
    def get_mpnn_mlp_weights():
        oxygen_vec = [0.0, 0.0, 1.0, 1.0, 1.0]
        oxygen_bias = [-1.0]
        # If this passes, oxygen activation after MLP is 1.0. Otherwise, it's 0.0.

        fluoride_vec = [1.0, 1.0, 0.0, 0.0, 0.0]
        fluoride_bias = [0.0]
        # If this passes, fluoride activation after MLP is 1.0. Otherwise, it's 0.0.

        mlp_w = [oxygen_vec, fluoride_vec]
        mlp_b = [oxygen_bias, fluoride_bias]

        mlp_w = torch.tensor(mlp_w)
        mlp_b = torch.tensor(mlp_b)

        mlp_b = mlp_b.squeeze()

        return mlp_w, mlp_b

    @staticmethod
    def get_readout_mlp_weights():
        final_w = [[1.0, 1.0]]
        final_b = [-1.51]

        return torch.tensor(final_w), torch.tensor(final_b)

    def __init__(self):
        super().__init__()

        x_embedding_weights, edge_features_embedding_weights = self.get_embedding_weights()

        self.x_embedding = torch.nn.Linear(100, 5, bias=False)
        self.e_embedding = torch.nn.Linear(4, 5, bias=False)

        assert self.x_embedding.weight.shape == x_embedding_weights.shape
        assert self.e_embedding.weight.shape == edge_features_embedding_weights.shape

        self.x_embedding.weight = torch.nn.Parameter(x_embedding_weights)
        self.e_embedding.weight = torch.nn.Parameter(edge_features_embedding_weights)

        mpnn_mlp_w, mpnn_mlp_b = self.get_mpnn_mlp_weights()

        mpnn_mlp = torch.nn.Linear(5, 2)

        mpnn_model = torch.nn.Sequential(
            mpnn_mlp, torch.nn.Tanh()
        )

        self.gine = torch_geometric.nn.GINEConv(mpnn_model)

        assert mpnn_mlp.weight.shape == mpnn_mlp_w.shape, f"Expected weight shape {mpnn_mlp.weight.shape}, but got {mpnn_mlp_w.shape}"
        assert mpnn_mlp.bias.shape == mpnn_mlp_b.shape, f"Expected bias shape {mpnn_mlp.bias.shape}, but got {mpnn_mlp_b.shape}"

        mpnn_mlp.weight = torch.nn.Parameter(mpnn_mlp_w)
        mpnn_mlp.bias = torch.nn.Parameter(mpnn_mlp_b)

        readout_mlp_w, readout_mlp_b = self.get_readout_mlp_weights()

        self.readout_mlp = torch.nn.Linear(2, 1)
        assert self.readout_mlp.weight.shape == readout_mlp_w.shape, f"Expected weight shape {self.readout_mlp.weight.shape}, but got {readout_mlp_w.shape}"
        assert self.readout_mlp.bias.shape == readout_mlp_b.shape, f"Expected bias shape {self.readout_mlp.bias.shape}, but got {readout_mlp_b.shape}"

        self.readout_mlp.weight = torch.nn.Parameter(readout_mlp_w)
        self.readout_mlp.bias = torch.nn.Parameter(readout_mlp_b)

        self.pyg_mode = True

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.max(x, dim=-2, keepdim=False).values  # Global max pooling
        x = self.readout_mlp(x)
        x = torch.sigmoid(x).squeeze(dim=-1)
        return x

    def forward(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, list]:
        if self.pyg_mode:
            x, edge_features, edge_index = deal_with_pyg(*args, **kwargs)
        else:
            x, edge_features, edge_index = args

        dump_activations = kwargs.get("dump_activations", False)

        # If x has hydrogen info, drop it
        if x.shape[1] > 100:
            x = x[:, :100]

        x = self.x_embedding(x)

        edge_features = self.e_embedding(edge_features)

        x = self.gine(x, edge_index, edge_features)
        acts = x.clone()  # Save activations for debugging

        x = self.readout(x)

        if not dump_activations:
            return x
        else:
            return x, [acts]


def main():
    from rdkit import Chem

    model = FluCarbGINE()

    # mol = Chem.MolFromSmiles("O.O.O.[F][Fe]([F])[F]")
    mol = Chem.MolFromSmiles("FCOC")
    mol = Chem.RemoveAllHs(mol)

    x, edge_index, edge_features = mol_to_torch(mol)

    print(
        model(x, edge_features, edge_index)
    )


if __name__ == "__main__":
    main()
