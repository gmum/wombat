import torch

from dig.xgraph.method.subgraphx import SubgraphX, find_closest_node_result

from xai_methods import AttributionMethod, BatchDimensionRemover


class ModelDataBatchWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, edge_features: torch.Tensor):
        super().__init__()
        self.model = model
        self.edge_features = edge_features

    def forward(self, data) -> torch.Tensor:
        batch_size = data.batch_size
        node_number_per_graph = data.x.shape[0] // batch_size

        assert data.x.shape[0] % batch_size == 0, "Expected the number of nodes to be divisible by the batch size"

        edge_index = data.edge_index
        x = data.x

        # This is the same graph, so we can just repeat the edge features for each graph in the batch
        edge_features = self.edge_features.repeat(batch_size, 1)  # shape (batch_size * num_edges, edge_feature_dim)

        output, activation_list = self.model(x, edge_features, edge_index, dump_activations=True)

        last_layer_activations = activation_list[-1]

        # We need to do readout per graph.
        readout_module = self.model.readout

        last_layer_activations = torch.stack(last_layer_activations.split(
            node_number_per_graph), dim=0)

        readouts = readout_module(last_layer_activations)

        return readouts


class SubgraphXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, edge_features: torch.Tensor):
        super().__init__()
        self.model = model
        self.edge_features = edge_features

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 2:
            x, edge_index = args
            class_1_probs = self.model(x, self.edge_features, edge_index)
            class_0_probs = 1 - class_1_probs

            return torch.stack([class_0_probs, class_1_probs], dim=-1)

        else:
            model = ModelDataBatchWrapper(self.model, self.edge_features)

            class_1_probs = model(**kwargs)
            # class_1_probs = torch.sigmoid(torch.randn(kwargs["data"].batch_size, ))

            class_0_probs = 1 - class_1_probs

            return torch.stack([class_0_probs, class_1_probs], dim=-1)


class SubgraphXAttributionMethod(AttributionMethod):
    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        previous_pyg_mode = self.model.pyg_mode

        self.model.pyg_mode = False

        model = SubgraphXWrapper(self.model, example_e)

        method = SubgraphX(
            model=model,
            num_classes=2,
            device="cpu"
        )

        _, explanation_results, related_preds = method(example_x, edge_index, max_nodes=example_x.shape[0])

        self.model.pyg_mode = previous_pyg_mode

        scores_and_coalitions = [(result["P"], result["coalition"]) for result in explanation_results[1]]
        scores_and_coalitions = sorted(scores_and_coalitions, key=lambda x: (x[0], -len(x[1])), reverse=True)

        score, coalition = scores_and_coalitions[0]

        node_attrs = torch.zeros(example_x.shape[0], dtype=torch.float)
        edge_attrs = torch.zeros(edge_index.shape[1], dtype=torch.float)

        for node_idx in coalition:
            node_attrs[node_idx] = 1.0

        for edge_idx, (x_idx, y_idx) in enumerate(edge_index.T):
            if x_idx in coalition and y_idx in coalition:
                edge_attrs[edge_idx] = 1.0

        node_attrs = node_attrs.unsqueeze(1)  # shape (num_nodes, 1)
        edge_attrs = edge_attrs.unsqueeze(1)  # shape (num_edges, 1)

        return node_attrs, edge_attrs


def main():
    from matplotlib import pyplot as plt
    from rdkit import Chem
    from mpnn import mol_to_torch, MoleculeFDetector
    from datavis import visualize_atom_importance_from_mol
    from dig.version import __version__

    print("VERSION:", __version__)

    model = MoleculeFDetector()
    smiles = "CCCCCCCC(=O)NNC(=O)C1COC2=CC=CC=C2O1"

    mol = Chem.MolFromSmiles(smiles)
    x, edge_index, edge_features = mol_to_torch(mol)

    attribution_method = SubgraphXAttributionMethod(model, "", [smiles], [])
    node_attrs, edge_attrs = attribution_method.explain(x, edge_features, edge_index)

    node_attrs = node_attrs.squeeze().detach().numpy().tolist()

    img = visualize_atom_importance_from_mol(mol, node_attrs, length=1000)
    img.show()
    exit()

    plt.gcf().set_dpi(200)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
