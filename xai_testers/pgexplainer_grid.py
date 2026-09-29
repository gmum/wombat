import pandas as pd
import pathlib

from functools import partial

import torch

from krfp_models import krfp_models
from xai_methods import PGExplainerAttributionMethod
from xai_testers.explainability_method_tester import get_split_into_positive_and_negative_smiles, \
    NegativeExplainabilityMethodTester, PositiveExplainabilityMethodTester


def main():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = pathlib.Path(f"../artifacts/pgexplainer_grid/{timestamp}")
    artifacts_path.mkdir(exist_ok=False, parents=True)

    def get_pgexplainer(model, model_smarts, positive_smiles, negative_smiles, train_lr, train_epochs):
        return PGExplainerAttributionMethod(
            model=model,
            model_smarts=model_smarts,
            positive_smiles=positive_smiles,
            negative_smiles=negative_smiles,
            train_lr=train_lr,
            train_epochs=train_epochs,
        )

    methods = []

    for train_lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        for train_epochs in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            method_name = f"PGExplainer_lr_{train_lr}_epochs_{train_epochs}"
            method_cls = partial(get_pgexplainer, train_lr=train_lr, train_epochs=train_epochs)
            methods.append((method_name, method_cls))

    models = [krfp_models[-1]]

    for model, _, pattern_smarts in models:
        model_name = model.__class__.__name__
        df = pd.read_csv(f"../data/validation_datasets/{model_name}.csv")

        positive_smiles, negative_smiles, (
            positive_count, tversky_negative_count,
            random_negative_count) = get_split_into_positive_and_negative_smiles(df)

        negative_tester = NegativeExplainabilityMethodTester(model, negative_smiles, pattern_smarts)
        positive_tester = PositiveExplainabilityMethodTester(model, positive_smiles, pattern_smarts)

        subpath = artifacts_path / model_name
        subpath.mkdir(exist_ok=False)

        method_iqr_results_dict = {"SMILES": negative_smiles}
        method_positives_results_dict = {"SMILES": positive_smiles}

        for method_name, method_fn in methods:
            method = method_fn(
                model=model,
                model_smarts=pattern_smarts,
                positive_smiles=positive_smiles,
                negative_smiles=negative_smiles,
            )

            iqr_successes = negative_tester.evaluate_explainability_method(method, add_hydrogen_ohe=True)

            aurocs, aps, ap_baselines = positive_tester.evaluate_explainability_method(method, add_hydrogen_ohe=True)

            method_iqr_results_dict[method_name] = iqr_successes

            method_positives_results_dict[f"{method_name}_auroc"] = aurocs
            method_positives_results_dict[f"{method_name}_ap"] = aps
            method_positives_results_dict[f"{method_name}_ap_baseline"] = ap_baselines

            torch.save(method, subpath / f"{method_name}_explainer.pt")

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
