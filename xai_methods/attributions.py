import torch


class AttributionMethod:
    def __init__(self, model: torch.nn.Module, model_smarts: str, positive_smiles: list[str],
                 negative_smiles: list[str]):
        self.model = model
        self.model_smarts = model_smarts
        self.positive_smiles = positive_smiles
        self.negative_smiles = negative_smiles

    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        raise NotImplementedError
