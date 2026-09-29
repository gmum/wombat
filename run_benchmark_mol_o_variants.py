import pandas as pd
import torch
import pathlib
import warnings

from mpnn.molecule_o_variants import LeafActivatedMoleculeODetector, ActivationMovingMoleculeODetector, \
    SmoothingMoleculeODetector
from mpnn.mpnn_arch import AllNonZeroMaxReadout, AllNonZeroReadout
from xai_methods.subgraph_x_attributions import SubgraphXAttributionMethod

warnings.filterwarnings("ignore")

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from krfp_models import krfp_models, model_name_to_publication_name
from xai_methods import IGAttributionMethod, PGExplainerAttributionMethod, GNNExplainerAttributionMethod, \
    InputXGradientAttributionMethod, SaliencyAttributionMethod, AttributionMethod
from xai_methods.captum_attributions import ShapleyValueSamplingAttributionMethod
from xai_testers.explainability_method_tester import PositiveExplainabilityMethodTester, \
    NegativeExplainabilityMethodTester, \
    get_split_into_positive_and_negative_smiles


def main():
    readout = AllNonZeroReadout()
    readout_str = "R1" if isinstance(readout, AllNonZeroMaxReadout) else "R2"
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    artifacts_path = pathlib.Path(f"artifacts/atypical_models_explainability_method_tester/{readout_str}-{timestamp}")
    artifacts_path.mkdir(exist_ok=False, parents=True)

    methods = [
        ("PG Explainer", PGExplainerAttributionMethod),
        ("Integrated Gradients", IGAttributionMethod),
        ("Saliency", SaliencyAttributionMethod),
        ("SHAP Sampling", ShapleyValueSamplingAttributionMethod),
        ("Input x Gradient", InputXGradientAttributionMethod),
        ("GNN Explainer", GNNExplainerAttributionMethod),
        ("SubgraphX", SubgraphXAttributionMethod)
    ]

    models = [
        # (LeafActivatedMoleculeODetector(), krfp_models[11][2]),
        # (ActivationMovingMoleculeODetector(), krfp_models[11][2]),
        (SmoothingMoleculeODetector(), krfp_models[11][2])
    ]

    for model, pattern_smarts in models:
        model_name = model.__class__.__name__
        model.readout = readout
        df = pd.read_parquet(
            f"wombat-smiles/validation_datasets/Pattern 12/validation_molecules.parquet")

        positive_smiles, negative_smiles, (
            positive_count, tversky_negative_count,
            random_negative_count) = get_split_into_positive_and_negative_smiles(df)

        # negative_smiles = negative_smiles[:2]
        # positive_smiles = positive_smiles[:2]

        print(f"Testing model {model.__class__.__name__} with pattern SMARTS {pattern_smarts}")
        print(
            f"Positive samples: {positive_count}, Tversky negative samples: {tversky_negative_count}, Random negative samples: {random_negative_count}")

        negative_tester = NegativeExplainabilityMethodTester(model, negative_smiles, pattern_smarts)
        positive_tester = PositiveExplainabilityMethodTester(model, positive_smiles, pattern_smarts)

        method_iqr_results_dict = {"SMILES": negative_smiles}
        method_positives_results_dict = {"SMILES": positive_smiles}

        subpath = artifacts_path / model_name
        subpath.mkdir(exist_ok=False)

        for method_name, method_cls in methods:
            method = method_cls(
                model=model,
                model_smarts=pattern_smarts,
                positive_smiles=positive_smiles,
                negative_smiles=negative_smiles,
            )

            cutoff = 1 if (method_name == "SHAP Sampling" or method_name == "SubgraphX") else None

            iqr_successes, node_attrs_all_negative, edge_attrs_all_negative = negative_tester.evaluate_explainability_method(
                method, add_hydrogen_ohe=True, cutoff=cutoff)

            aurocs, aps, ap_baselines, node_attrs_all_positive, edge_attrs_all_positive = positive_tester.evaluate_explainability_method(
                method, add_hydrogen_ohe=True,
                cutoff=cutoff)

            method_iqr_results_dict[method_name] = iqr_successes

            method_positives_results_dict[f"{method_name}_auroc"] = aurocs
            method_positives_results_dict[f"{method_name}_ap"] = aps
            method_positives_results_dict[f"{method_name}_ap_baseline"] = ap_baselines

            torch.save(method, subpath / f"{method_name}_explainer.pt")
            torch.save(node_attrs_all_negative, subpath / f"{method_name}_node_attrs_negative.pt")
            torch.save(edge_attrs_all_negative, subpath / f"{method_name}_edge_attrs_negative.pt")
            torch.save(node_attrs_all_positive, subpath / f"{method_name}_node_attrs_positive.pt")
            torch.save(edge_attrs_all_positive, subpath / f"{method_name}_edge_attrs_positive.pt")

        # Pad results for SHAP to match it with other methods in the DataFrame (eg for IG)

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
            f.write(f"Positive samples: {positive_count}\n")
            f.write(f"Tversky negative samples: {tversky_negative_count}\n")
            f.write(f"Random negative samples: {random_negative_count}\n")


if __name__ == "__main__":
    main()
