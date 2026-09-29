import pandas as pd
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from torch_geometric.nn import MessagePassing
from captum.attr import IntegratedGradients, Saliency, DeepLift, ShapleyValueSampling, InputXGradient

from mpnn import one_hot_encode, smiles_to_torch
from datavis import visualize_atom_importance
from xai_methods import AttributionMethod, BatchDimensionRemover


class IGAttributionMethod(AttributionMethod):
    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        baseline_x = torch.zeros_like(example_x)
        baseline_e = torch.zeros_like(example_e)

        # Add batch dimension
        baseline_x = baseline_x.unsqueeze(0)
        baseline_e = baseline_e.unsqueeze(0)
        edge_index = edge_index.unsqueeze(0)
        example_x = example_x.unsqueeze(0)
        example_e = example_e.unsqueeze(0)

        model = BatchDimensionRemover(self.model)
        ig = IntegratedGradients(model)

        node_attrs, edge_attrs = ig.attribute(
            inputs=(example_x, example_e),
            baselines=(baseline_x, baseline_e),
            additional_forward_args=(edge_index,),
            internal_batch_size=1,
        )

        # Squeeze batch dimension
        node_attrs = node_attrs.squeeze(0)
        edge_attrs = edge_attrs.squeeze(0)

        return node_attrs, edge_attrs


class SaliencyAttributionMethod(AttributionMethod):
    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        model = BatchDimensionRemover(self.model)
        saliency = Saliency(model)

        example_x = example_x.unsqueeze(0)
        example_e = example_e.unsqueeze(0)
        edge_index = edge_index.unsqueeze(0)

        node_attrs, edge_attrs = saliency.attribute(
            inputs=(example_x, example_e),
            additional_forward_args=(edge_index,),
        )

        node_attrs = node_attrs.squeeze(0)
        edge_attrs = edge_attrs.squeeze(0)

        return node_attrs, edge_attrs


class InputXGradientAttributionMethod(AttributionMethod):
    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        model = BatchDimensionRemover(self.model)
        input_x_gradient = InputXGradient(model)

        example_x = example_x.unsqueeze(0)
        example_e = example_e.unsqueeze(0)

        node_attrs, edge_attrs = input_x_gradient.attribute(
            inputs=(example_x, example_e),
            additional_forward_args=(edge_index,),
        )

        node_attrs = node_attrs.squeeze(0)
        edge_attrs = edge_attrs.squeeze(0)

        return node_attrs, edge_attrs


class ShapleyValueSamplingAttributionMethod(AttributionMethod):
    def __init__(self, *args, n_samples: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        model = BatchDimensionRemover(self.model)
        shap = ShapleyValueSampling(model)

        baseline_x = torch.zeros_like(example_x)
        baseline_e = torch.zeros_like(example_e)

        node_mask = torch.arange(example_x.shape[0])
        edge_mask = torch.arange(example_e.shape[0])

        node_mask = node_mask.unsqueeze(1)
        edge_mask = edge_mask.unsqueeze(1)

        node_mask = torch.zeros_like(example_x) + node_mask
        edge_mask = torch.zeros_like(example_e) + edge_mask
        edge_mask = edge_mask + torch.max(node_mask) + 1

        baseline_x = baseline_x.unsqueeze(0)
        baseline_e = baseline_e.unsqueeze(0)
        edge_index = edge_index.unsqueeze(0)

        example_x = example_x.unsqueeze(0)
        example_e = example_e.unsqueeze(0)

        node_attrs, edge_attrs = shap.attribute(
            inputs=(example_x, example_e),
            baselines=(baseline_x, baseline_e),
            additional_forward_args=(edge_index,),
            feature_mask=(node_mask, edge_mask),
            n_samples=self.n_samples,
        )

        node_attrs = node_attrs.squeeze(0).squeeze(0)
        edge_attrs = edge_attrs.squeeze(0).squeeze(0)

        return node_attrs, edge_attrs
