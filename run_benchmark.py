import pandas as pd
import torch
import pathlib
import warnings

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
    artifacts_path = pathlib.Path(f"artifacts/explainability_method_tester/{readout_str}-{timestamp}")
    artifacts_path.mkdir(exist_ok=False, parents=True)

    def ShapSamplingMoreSamples(model, model_smarts, positive_smiles, negative_smiles):
        return ShapleyValueSamplingAttributionMethod(
            model=model,
            model_smarts=model_smarts,
            positive_smiles=positive_smiles,
            negative_smiles=negative_smiles,
            n_samples=500
        )

    methods = [
        # ("PG Explainer", PGExplainerAttributionMethod),
        # ("Integrated Gradients", IGAttributionMethod),
        # ("Saliency", SaliencyAttributionMethod),
        ("SHAP Sampling", ShapSamplingMoreSamples),
        # ("Input x Gradient", InputXGradientAttributionMethod),
        # ("GNN Explainer", GNNExplainerAttributionMethod),
        # ("SubgraphX", SubgraphXAttributionMethod)
    ]

    for model, _, pattern_smarts in krfp_models:
        model_name = model.__class__.__name__
        model.readout = readout
        df = pd.read_parquet(
            f"wombat-smiles/validation_datasets/{model_name_to_publication_name[model_name]}/validation_molecules.parquet")

        positive_smiles, negative_smiles, (
            positive_count, tversky_negative_count,
            random_negative_count) = get_split_into_positive_and_negative_smiles(df)

        # negative_smiles = negative_smiles[:10]
        # positive_smiles = positive_smiles[:10]

        print(f"Testing model {model.__class__.__name__} with pattern SMARTS {pattern_smarts}")
        print(
            f"Positive samples: {positive_count}, Tversky negative samples: {tversky_negative_count}, Random negative samples: {random_negative_count}")

        negative_tester = NegativeExplainabilityMethodTester(model, negative_smiles, pattern_smarts)
        positive_tester = PositiveExplainabilityMethodTester(model, positive_smiles, pattern_smarts)

        positive_results_df = [pd.DataFrame({"SMILES": positive_smiles})]
        negative_results_dfs = [pd.DataFrame({"SMILES": negative_smiles})]

        subpath = artifacts_path / model_name
        subpath.mkdir(exist_ok=False)

        for method_name, method_cls in methods:
            method = method_cls(
                model=model,
                model_smarts=pattern_smarts,
                positive_smiles=positive_smiles,
                negative_smiles=negative_smiles,
            )

            positive_cutoff = 100 if (method_name == "SHAP Sampling" or method_name == "SubgraphX") else None
            negative_cutoff = 1 if (method_name == "SHAP Sampling" or method_name == "SubgraphX") else None

            iqr_successes, node_attrs_all_negative, edge_attrs_all_negative = negative_tester.evaluate_explainability_method(
                method, add_hydrogen_ohe=True, cutoff=negative_cutoff)

            negative_df = pd.DataFrame({
                f"{method_name}_iqr_success": iqr_successes,
            })

            negative_df.to_csv(subpath / f"{method_name}_negative_results.csv", index=False)

            aurocs, aps, ap_baselines, node_attrs_all_positive, edge_attrs_all_positive = positive_tester.evaluate_explainability_method(
                method, add_hydrogen_ohe=True,
                cutoff=positive_cutoff)

            positive_df = pd.DataFrame({
                f"{method_name}_auroc": aurocs,
                f"{method_name}_ap": aps,
                f"{method_name}_ap_baseline": ap_baselines,
            })

            positive_df.to_csv(subpath / f"{method_name}_positive_results.csv", index=False)

            positive_results_df.append(positive_df)
            negative_results_dfs.append(negative_df)

            torch.save(method, subpath / f"{method_name}_explainer.pt")
            torch.save(node_attrs_all_negative, subpath / f"{method_name}_node_attrs_negative.pt")
            torch.save(edge_attrs_all_negative, subpath / f"{method_name}_edge_attrs_negative.pt")
            torch.save(node_attrs_all_positive, subpath / f"{method_name}_node_attrs_positive.pt")
            torch.save(edge_attrs_all_positive, subpath / f"{method_name}_edge_attrs_positive.pt")

        combined_positive_df = pd.concat(positive_results_df, axis="columns")
        combined_negative_df = pd.concat(negative_results_dfs, axis="columns")

        combined_positive_df.to_csv(subpath / "positive_explainability_results.csv", index=False)
        combined_negative_df.to_csv(subpath / "negative_explainability_results.csv", index=False)

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
