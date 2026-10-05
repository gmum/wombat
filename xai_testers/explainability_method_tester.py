import pandas as pd
import numpy as np
import torch
import statistics
import pathlib
import warnings

from mpnn.mpnn_arch import AllNonZeroMaxReadout, AllNonZeroReadout
from xai_methods.subgraph_x_attributions import SubgraphXAttributionMethod

warnings.filterwarnings("ignore")

from random import Random

from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from ground_truth.highlight_smarts import highlight_atoms_in_mol
from krfp_models import krfp_models, model_name_to_publication_name
from mpnn import mol_to_torch
from dataset import PubchemProcessedSMILESDataset
from xai_methods import IGAttributionMethod, PGExplainerAttributionMethod, GNNExplainerAttributionMethod, \
    InputXGradientAttributionMethod, SaliencyAttributionMethod, AttributionMethod
from xai_methods.captum_attributions import ShapleyValueSamplingAttributionMethod


class NegativeExplainabilityMethodTester:
    def __init__(self, model: torch.nn.Module, smiles_list: list[str], pattern_smarts: str):
        self.model = model
        self.pattern_smarts = pattern_smarts

        self.pattern_mol = Chem.MolFromSmarts(pattern_smarts)

        if self.pattern_mol is None:
            raise ValueError(f"Invalid pattern SMARTS: {pattern_smarts}")

        self.smiles_list = smiles_list
        self.disable_tqdm = len(smiles_list) < 10

    def assert_no_lit_up_atoms(self, mol: Chem.Mol):
        lit_up_atoms = set(highlight_atoms_in_mol(mol, self.pattern_smarts))

        assert len(
            lit_up_atoms) == 0, f"Expected no lit up atoms for molecule {Chem.MolToSmiles(mol)}, but got {lit_up_atoms}"

    def check_iqr_criterion(self, attribution: torch.Tensor) -> bool:
        q1 = torch.quantile(attribution, 0.25, interpolation='lower')
        q3 = torch.quantile(attribution, 0.75, interpolation='higher')

        eps = 1e-6
        iqr = q3 - q1 + eps

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (attribution < lower_bound) | (attribution > upper_bound)

        return not torch.any(outliers)

    def evaluate_explainability_method(self, explainability_method: AttributionMethod, pyg_mode: bool = True,
                                       add_hydrogen_ohe: bool = False, cutoff: int | None = None):

        iqr_successes = []
        node_attrs_all = []
        edge_attrs_all = []

        self.model.pyg_mode = pyg_mode

        if cutoff is None:
            cutoff = len(self.smiles_list)

        for smile, _ in tqdm(zip(self.smiles_list, range(cutoff)), desc="Evaluating", total=cutoff,
                             disable=self.disable_tqdm):  # type: Chem.Mol
            mol = Chem.MolFromSmiles(smile)

            try:
                mol = Chem.RemoveAllHs(mol)
            except Chem.KekulizeException:
                iqr_successes.append(None)
                node_attrs_all.append(None)
                edge_attrs_all.append(None)
                continue

            assert mol is not None, f"Invalid SMILES: {smile}"
            self.assert_no_lit_up_atoms(mol)

            example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=add_hydrogen_ohe)

            node_attrs, edge_attrs = explainability_method.explain(
                example_x=example_x,
                example_e=example_e,
                edge_index=edge_index
            )

            unprocessed_node_attrs = node_attrs.clone()

            node_attrs = torch.sum(torch.abs(node_attrs), dim=1)

            iqr_success = self.check_iqr_criterion(node_attrs)

            iqr_successes.append(iqr_success)
            node_attrs_all.append(unprocessed_node_attrs)
            edge_attrs_all.append(edge_attrs)

        return iqr_successes, node_attrs_all, edge_attrs_all


class PositiveExplainabilityMethodTester:
    def __init__(self, model: torch.nn.Module, smiles_list: list[str], pattern_smarts: str):
        self.model = model
        self.pattern_smarts = pattern_smarts

        self.pattern_mol = Chem.MolFromSmarts(pattern_smarts)

        if self.pattern_mol is None:
            raise ValueError(f"Invalid pattern SMARTS: {pattern_smarts}")

        self.smiles_list = smiles_list
        self.disable_tqdm = len(smiles_list) < 10

    def _get_lit_up_atoms(self, mol: Chem.Mol) -> set[int]:
        return set(highlight_atoms_in_mol(mol, self.pattern_smarts))

    def get_stats_for_molecule(self, mol: Chem.Mol, explainability_method: AttributionMethod,
                               add_hydrogen_ohe: bool = False):
        lit_up_atoms = self._get_lit_up_atoms(mol)
        prediction_mask = [1 if i in lit_up_atoms else 0 for i in range(mol.GetNumAtoms())]
        prediction_mask = np.array(prediction_mask)

        if np.sum(prediction_mask != 1) == 0:
            # Can't calculate auROC or AP
            return None

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=add_hydrogen_ohe)

        node_attrs, edge_attrs = explainability_method.explain(
            example_x=example_x,
            example_e=example_e,
            edge_index=edge_index
        )

        unprocessed_node_attrs = node_attrs.clone()

        node_attrs = torch.sum(torch.abs(node_attrs), dim=1)

        auroc = roc_auc_score(prediction_mask, node_attrs.squeeze().detach().numpy())
        ap = average_precision_score(prediction_mask, node_attrs.squeeze().detach().numpy())
        ap_baseline = len(lit_up_atoms) / mol.GetNumAtoms()

        return auroc, ap, ap_baseline, unprocessed_node_attrs, edge_attrs

    def evaluate_explainability_method(self, explainability_method: AttributionMethod, pyg_mode: bool = True,
                                       add_hydrogen_ohe: bool = True, cutoff: int | None = None):
        aurocs = []
        aps = []
        ap_baselines = []
        node_attrs_all = []
        edge_attrs_all = []

        self.model.pyg_mode = pyg_mode

        if cutoff is None:
            cutoff = len(self.smiles_list)

        for smile, _ in tqdm(zip(self.smiles_list, range(cutoff)), total=cutoff, desc="Evaluating",
                             disable=self.disable_tqdm):  # type: Chem.Mol
            mol = Chem.MolFromSmiles(smile)

            try:
                mol = Chem.RemoveAllHs(mol)
            except Chem.KekulizeException:
                aurocs.append(None)
                aps.append(None)
                ap_baselines.append(None)
                node_attrs_all.append(None)
                edge_attrs_all.append(None)
                continue

            assert mol is not None, f"Invalid SMILES: {smile}"

            result = self.get_stats_for_molecule(mol, explainability_method,
                                                 add_hydrogen_ohe=add_hydrogen_ohe)

            if result is None:
                aurocs.append(None)
                aps.append(None)
                ap_baselines.append(None)
                node_attrs_all.append(None)
                edge_attrs_all.append(None)
                continue

            auroc, ap, ap_baseline, node_attrs, edge_attrs = result

            aurocs.append(auroc)
            aps.append(ap)
            ap_baselines.append(ap_baseline)
            node_attrs_all.append(node_attrs)
            edge_attrs_all.append(edge_attrs)

        return aurocs, aps, ap_baselines, node_attrs_all, edge_attrs_all


def get_split_into_positive_and_negative_smiles(df: pd.DataFrame):
    positive_smiles = df[df["ORIGIN"] == "POSITIVE"]["SMILES"]
    tversky_smiles = df[df["ORIGIN"] == "TVERSKY"]["SMILES"]
    negative_smiles = df[df["ORIGIN"] == "NEGATIVE"]["SMILES"]

    positive_len = len(positive_smiles)
    tversky_len = len(tversky_smiles)
    negative_len = len(negative_smiles)

    positive_sample_len = min(positive_len, 10000)
    tversky_sample_len = min(tversky_len, 5000)
    negative_sample_len = min(negative_len, 10000 - tversky_sample_len)

    positive_smiles = positive_smiles.sample(positive_sample_len, random_state=42).tolist()
    tversky_smiles = tversky_smiles.sample(tversky_sample_len, random_state=42).tolist()
    negative_smiles = negative_smiles.sample(negative_sample_len, random_state=42).tolist()

    negative_smiles = tversky_smiles + negative_smiles
    rng = Random(42)

    rng.shuffle(negative_smiles)

    return positive_smiles, negative_smiles, (positive_sample_len, tversky_sample_len, negative_sample_len)
