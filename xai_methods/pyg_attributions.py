import random
import statistics

import pandas as pd
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from torch_geometric import debug
from torch_geometric.explain import Explainer, CaptumExplainer, ExplainerConfig, ModelConfig, DummyExplainer, \
    GNNExplainer, PGExplainer, GraphMaskExplainer
from torch_geometric.nn import MessagePassing
from captum.attr import IntegratedGradients
from tqdm import trange

from mpnn import one_hot_encode, smiles_to_torch, mol_to_torch
from datavis import visualize_atom_importance
from xai_methods import AttributionMethod, BatchDimensionRemover


class GNNExplainerAttributionMethod(AttributionMethod):
    def __init__(self,
                 *args,
                 lr: float = 0.01,
                 epochs: int = 100,
                 edge_size: float = 0.005,
                 edge_reduction: str = 'sum',
                 node_feat_size: float = 1.0,
                 node_feat_reduction: str = 'mean',
                 edge_ent: float = 1.0,
                 node_feat_ent: float = 0.1,
                 EPS: float = 1e-15,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.epochs = epochs
        self.edge_size = edge_size
        self.edge_reduction = edge_reduction
        self.node_feat_size = node_feat_size
        self.node_feat_reduction = node_feat_reduction
        self.edge_ent = edge_ent
        self.node_feat_ent = node_feat_ent
        self.EPS = EPS

    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        explainer = GNNExplainer(
            epochs=self.epochs,
            lr=self.lr,
            edge_size=self.edge_size,
            edge_reduction=self.edge_reduction,
            node_feat_size=self.node_feat_size,
            node_feat_reduction=self.node_feat_reduction,
            edge_ent=self.edge_ent,
            node_feat_ent=self.node_feat_ent,
            EPS=self.EPS,
        )

        explainer.connect(
            explainer_config=ExplainerConfig(
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
            ),
            model_config=ModelConfig(
                mode="regression",
                task_level="graph",
                return_type="raw",
            )
        )

        target = self.model(example_x, example_e, edge_index).item()

        attrs = explainer(
            model=self.model,
            x=example_x,
            edge_features=example_e,
            edge_index=edge_index,
            target=torch.Tensor([target]),
        )

        return attrs.node_mask, attrs.edge_mask


class PGExplainerAttributionMethod(AttributionMethod):
    def _train(self):
        ds = self.positive_smiles + self.negative_smiles

        mols = [Chem.MolFromSmiles(smiles) for smiles in ds]  # type: list[Chem.Mol]

        for mol, smiles in zip(mols, ds):
            if mol is None:
                raise ValueError(f"Invalid SMILES in dataset: {smiles}")

        preprocessed_ds = [mol_to_torch(mol, add_hydrogen_ohe=True) for mol in mols]

        bar = trange(self.train_epochs, desc="Training PGExplainer")

        for epoch in bar:
            random.shuffle(preprocessed_ds)
            losses = []
            for x, edge_index, edge_features in preprocessed_ds:
                target = self.model(x, edge_features, edge_index).item()
                target = torch.Tensor([target])

                loss: float = self.explainer.algorithm.train(
                    epoch,
                    self.model,
                    x,
                    edge_index,
                    target=target,
                    edge_features=edge_features
                )  # type: ignore

                losses.append(loss)

            bar.set_postfix({"mean_loss": statistics.mean(losses)})

    def __init__(self, *args, train_lr: float = 0.003, train_epochs: int = 10, infer_node_mask: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_lr = train_lr
        self.train_epochs = train_epochs
        self.infer_node_mask = infer_node_mask

        self.explainer = Explainer(
            model=self.model,
            algorithm=PGExplainer(epochs=train_epochs, lr=train_lr),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(
                mode="regression",
                task_level="graph",
                return_type="raw",
            )
        )

        self._train()

    def explain(self, example_x: torch.Tensor, example_e: torch.Tensor, edge_index: torch.Tensor) -> tuple[
        None | torch.Tensor, torch.Tensor]:

        pred = self.model(example_x, example_e, edge_index).item()

        edge_attrs = self.explainer(
            x=example_x,
            edge_features=example_e,
            edge_index=edge_index,
            target=torch.Tensor([pred]),
        ).edge_mask

        if not self.infer_node_mask:
            return None, edge_attrs
        else:
            node_attrs = torch.zeros(example_x.shape[0])

            for val, (x_idx, y_idx) in zip(edge_attrs, edge_index.T):
                node_attrs[x_idx] += val
                node_attrs[y_idx] += val

            node_attrs = node_attrs.unsqueeze(1)

            return node_attrs, edge_attrs
