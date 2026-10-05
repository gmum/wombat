import pandas as pd
import torch
import pathlib
import warnings

from highlight_smarts import highlight_atoms_in_mol
from mpnn.gine_experiments import FluCarbGINE
from mpnn.molecule_o_variants import LeafActivatedMoleculeODetector, ActivationMovingMoleculeODetector, \
    SmoothingMoleculeODetector
from mpnn.mpnn_arch import AllNonZeroMaxReadout, AllNonZeroReadout
from xai_methods.subgraph_x_attributions import SubgraphXAttributionMethod

warnings.filterwarnings("ignore")

from rdkit import RDLogger, Chem

RDLogger.DisableLog("rdApp.*")

from krfp_models import krfp_models, model_name_to_publication_name
from xai_methods import IGAttributionMethod, PGExplainerAttributionMethod, GNNExplainerAttributionMethod, \
    InputXGradientAttributionMethod, SaliencyAttributionMethod, AttributionMethod
from xai_methods.captum_attributions import ShapleyValueSamplingAttributionMethod
from xai_testers.explainability_method_tester import PositiveExplainabilityMethodTester, \
    NegativeExplainabilityMethodTester, \
    get_split_into_positive_and_negative_smiles
from test_flucarb_gine import get_labels


class MultipleSmartsPositiveExplainabilityMethodTester(PositiveExplainabilityMethodTester):
    def __init__(self, model: torch.nn.Module, smiles_list: list[str], pattern_smarts: list[str]):
        super().__init__(model, smiles_list, "C")  # Placeholder smarts to keep compatiblity
        # ^ this is a hack.

        self.pattern_smarts = pattern_smarts

        for smart in pattern_smarts:
            if Chem.MolFromSmarts(smart) is None:
                raise ValueError(f"Invalid pattern SMARTS: {smart}")

    def _get_lit_up_atoms(self, mol: Chem.Mol) -> set[int]:
        lit_up_atoms = set()
        for smart in self.pattern_smarts:
            lit_up_atoms.update(highlight_atoms_in_mol(mol, smart))
        return lit_up_atoms


class MultipleSmartsNegativeExplainabilityMethodTester(NegativeExplainabilityMethodTester):
    def __init__(self, model: torch.nn.Module, smiles_list: list[str], pattern_smarts: list[str]):
        super().__init__(model, smiles_list, "C")  # Placeholder smarts to keep compatiblity
        # ^ this is a hack.

        self.pattern_smarts = pattern_smarts

        for smart in pattern_smarts:
            if Chem.MolFromSmarts(smart) is None:
                raise ValueError(f"Invalid pattern SMARTS: {smart}")

    def assert_no_lit_up_atoms(self, mol: Chem.Mol):
        for smart in self.pattern_smarts:
            lit_up_atoms = set(highlight_atoms_in_mol(mol, smart))

            if len(lit_up_atoms) == 0:
                return

        raise AssertionError(f"All SMARTS match molecule {Chem.MolToSmiles(mol)}. This should not happen.")


def main():
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    artifacts_path = pathlib.Path(f"artifacts/GINEConvFluCarb/{timestamp}")
    artifacts_path.mkdir(exist_ok=False, parents=True)

    methods = [
        # ("PG Explainer", PGExplainerAttributionMethod),
        # ("Integrated Gradients", IGAttributionMethod),
        # ("Saliency", SaliencyAttributionMethod),
        # ("SHAP Sampling", ShapleyValueSamplingAttributionMethod),
        # ("Input x Gradient", InputXGradientAttributionMethod),
        # ("GNN Explainer", GNNExplainerAttributionMethod),
        ("SubgraphX", SubgraphXAttributionMethod)
    ]

    model = FluCarbGINE()
    pattern_smarts = ["[!#1]F", "[#6]=O"]

    model_name = model.__class__.__name__
    df = pd.read_parquet(
        f"wombat-smiles/validation_datasets/Pattern 12/validation_molecules.parquet")
    df = df.sample(n=int(1e6), random_state=42)

    smiles, labels = get_labels(df["SMILES"].tolist(), pattern_smarts)

    positive_smiles = [smiles[i] for i in range(len(smiles)) if labels[i]]
    negative_smiles = [smiles[i] for i in range(len(smiles)) if not labels[i]]

    positive_smiles = positive_smiles[:10000]
    negative_smiles = negative_smiles[:10000]

    negative_tester = MultipleSmartsNegativeExplainabilityMethodTester(model, negative_smiles, pattern_smarts)
    positive_tester = MultipleSmartsPositiveExplainabilityMethodTester(model, positive_smiles, pattern_smarts)

    method_iqr_results_dict = {"SMILES": negative_smiles}
    method_positives_results_dict = {"SMILES": positive_smiles}

    subpath = artifacts_path / model_name
    subpath.mkdir(exist_ok=False)

    for method_name, method_cls in methods:
        method = method_cls(
            model=model,
            model_smarts="",  # Unused anyway
            positive_smiles=positive_smiles,
            negative_smiles=negative_smiles,
        )

        cutoff = 100 if (method_name == "SHAP Sampling" or method_name == "SubgraphX") else None

        aurocs, aps, ap_baselines, node_attrs_all_positive, edge_attrs_all_positive = positive_tester.evaluate_explainability_method(
            method, add_hydrogen_ohe=True,
            cutoff=cutoff)

        subdataframe_positive = pd.DataFrame({
            "SMILES": positive_smiles[:len(aurocs)],
            f"{method_name}_auroc": aurocs,
            f"{method_name}_ap": aps,
            f"{method_name}_ap_baseline": ap_baselines,
        })

        subdataframe_positive.to_csv(subpath / f"{method_name}_positive_explainability_results.csv", index=False)
        exit()  # FIXME

        iqr_successes, node_attrs_all_negative, edge_attrs_all_negative = negative_tester.evaluate_explainability_method(
            method, add_hydrogen_ohe=True, cutoff=cutoff)

        subdataframe_negative = pd.DataFrame({
            "SMILES": negative_smiles[:len(iqr_successes)],
            f"{method_name}_iqr_success": iqr_successes,
        })

        method_iqr_results_dict[method_name] = iqr_successes

        method_positives_results_dict[f"{method_name}_auroc"] = aurocs
        method_positives_results_dict[f"{method_name}_ap"] = aps
        method_positives_results_dict[f"{method_name}_ap_baseline"] = ap_baselines

        subdataframe_negative.to_csv(subpath / f"{method_name}_negative_explainability_results.csv", index=False)

        torch.save(method, subpath / f"{method_name}_explainer.pt")
        torch.save(node_attrs_all_negative, subpath / f"{method_name}_node_attrs_negative.pt")
        torch.save(edge_attrs_all_negative, subpath / f"{method_name}_edge_attrs_negative.pt")
        torch.save(node_attrs_all_positive, subpath / f"{method_name}_node_attrs_positive.pt")
        torch.save(edge_attrs_all_positive, subpath / f"{method_name}_edge_attrs_positive.pt")

    # Pad results for SHAP to match it with other methods in the DataFrame (e.g. for IG)

    baseline_len_iqr = len(method_iqr_results_dict["Integrated Gradients"])
    shap_len_iqr = len(method_iqr_results_dict["SHAP Sampling"])
    diff_iqr = baseline_len_iqr - shap_len_iqr

    method_iqr_results_dict["SHAP Sampling"] = method_iqr_results_dict["SHAP Sampling"] + [None] * diff_iqr
    method_iqr_results_dict["SubgraphX"] = method_iqr_results_dict["SubgraphX"] + [None] * diff_iqr

    baseline_len_positives = len(method_positives_results_dict["Integrated Gradients_auroc"])

    shap_len_positives = len(method_positives_results_dict["SHAP Sampling_auroc"])
    diff_positives = baseline_len_positives - shap_len_positives

    method_positives_results_dict["SHAP Sampling_auroc"] = method_positives_results_dict[
                                                               "SHAP Sampling_auroc"] + [None] * diff_positives
    method_positives_results_dict["SHAP Sampling_ap"] = method_positives_results_dict["SHAP Sampling_ap"] + [
        None] * diff_positives
    method_positives_results_dict["SHAP Sampling_ap_baseline"] = method_positives_results_dict[
                                                                     "SHAP Sampling_ap_baseline"] + [
                                                                     None] * diff_positives

    method_positives_results_dict["SubgraphX_auroc"] = method_positives_results_dict[
                                                           "SubgraphX_auroc"] + [None] * diff_positives
    method_positives_results_dict["SubgraphX_ap"] = method_positives_results_dict["SubgraphX_ap"] + [
        None] * diff_positives
    method_positives_results_dict["SubgraphX_ap_baseline"] = method_positives_results_dict[
                                                                 "SubgraphX_ap_baseline"] + [
                                                                 None] * diff_positives

    iqr_results_df = pd.DataFrame(method_iqr_results_dict)
    iqr_results_df.to_csv(subpath / f"negative_explainability_results.csv", index=False)

    positives_results_df = pd.DataFrame(method_positives_results_dict)
    positives_results_df.to_csv(subpath / f"positive_explainability_results.csv", index=False)

    with open(subpath / "model.txt", "w") as f:
        f.write(str(model))

    with open(subpath / "model.pth", "wb") as f:
        torch.save(model.state_dict(), f)

    with open(subpath / "ds_info.txt", "w") as f:
        f.write(f"Positive samples: {len(positive_smiles)}\n")
        f.write(f"Random negative samples: {len(negative_smiles)}\n")


if __name__ == "__main__":
    main()
