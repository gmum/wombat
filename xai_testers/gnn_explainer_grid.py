import pandas as pd
import pathlib

from functools import partial
import itertools
import torch
import random

from krfp_models import krfp_models
from xai_methods import PGExplainerAttributionMethod, GNNExplainerAttributionMethod
from xai_testers.explainability_method_tester import get_split_into_positive_and_negative_smiles, \
    NegativeExplainabilityMethodTester, PositiveExplainabilityMethodTester

from tqdm import tqdm


def main():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = pathlib.Path(f"../artifacts/gnnexplainer_grid/{timestamp}")
    artifacts_path.mkdir(exist_ok=False, parents=True)

    """
                lr: float = 0.01,
                 epochs: int = 100,
                 edge_size: float = 0.005,
                 edge_reduction: str = 'sum',
                 node_feat_size: float = 1.0,
                 node_feat_reduction: str = 'mean',
                 edge_ent: float = 1.0,
                 node_feat_ent: float = 0.1,
                 EPS: float = 1e-15,
    """

    def get_gnnexplainer(model, model_smarts, positive_smiles, negative_smiles,
                         train_lr, train_epochs, edge_size, edge_reduction, node_feat_size, node_feat_reduction,
                         edge_ent, node_feat_ent, EPS):
        return GNNExplainerAttributionMethod(
            model=model,
            model_smarts=model_smarts,
            positive_smiles=positive_smiles,
            negative_smiles=negative_smiles,
            lr=train_lr,
            epochs=train_epochs,
            edge_size=edge_size,
            edge_reduction=edge_reduction,
            node_feat_size=node_feat_size,
            node_feat_reduction=node_feat_reduction,
            edge_ent=edge_ent,
            node_feat_ent=node_feat_ent,
            EPS=EPS,
        )

    methods = []

    lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    epochs = [10, 50, 100, 500, 1000]
    edge_sizes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    edge_reductions = ['sum', 'mean']
    node_feat_sizes = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    node_feat_reductions = ['sum', 'mean']
    edge_ents = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    node_feat_ents = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    EPSs = [1e-15]

    iterator = itertools.product(
        lrs, epochs, edge_sizes, edge_reductions, node_feat_sizes, node_feat_reductions, edge_ents, node_feat_ents, EPSs
    )

    for lr, epoch, edge_size, edge_reduction, node_feat_size, node_feat_reduction, edge_ent, node_feat_ent, EPS in iterator:
        method_name = f"GNNExplainer_lr_{lr}_epochs_{epoch}_edge_size_{edge_size}_edge_reduction_{edge_reduction}_node_feat_size_{node_feat_size}_node_feat_reduction_{node_feat_reduction}_edge_ent_{edge_ent}_node_feat_ent_{node_feat_ent}_EPS_{EPS}"
        method_cls = partial(
            get_gnnexplainer,
            train_lr=lr,
            train_epochs=epoch,
            edge_size=edge_size,
            edge_reduction=edge_reduction,
            node_feat_size=node_feat_size,
            node_feat_reduction=node_feat_reduction,
            edge_ent=edge_ent,
            node_feat_ent=node_feat_ent,
            EPS=EPS,
        )
        methods.append((method_name, method_cls))

    rng = random.Random(42)
    rng.shuffle(methods)

    methods = rng.sample(methods, 50000)

    models = [krfp_models[-1]]

    print("Total methods to evaluate:", len(methods))

    for model, _, pattern_smarts in models:
        model_name = model.__class__.__name__
        df = pd.read_csv(f"../data/validation_datasets/{model_name}.csv")

        positive_smiles, negative_smiles, (
            positive_count, tversky_negative_count,
            random_negative_count) = get_split_into_positive_and_negative_smiles(df)

        positive_smiles = positive_smiles[:1]
        negative_smiles = negative_smiles[:1]

        # negative_tester = NegativeExplainabilityMethodTester(model, negative_smiles, pattern_smarts)
        positive_tester = PositiveExplainabilityMethodTester(model, positive_smiles, pattern_smarts)

        subpath = artifacts_path / model_name
        subpath.mkdir(exist_ok=False)

        # method_iqr_results_dict = {"SMILES": negative_smiles}
        method_positives_results_dict = {"SMILES": positive_smiles}

        for method_name, method_fn in tqdm(methods):
            method = method_fn(
                model=model,
                model_smarts=pattern_smarts,
                positive_smiles=positive_smiles,
                negative_smiles=negative_smiles,
            )

            # iqr_successes = negative_tester.evaluate_explainability_method(method, add_hydrogen_ohe=True)

            aurocs, aps, ap_baselines = positive_tester.evaluate_explainability_method(method, add_hydrogen_ohe=True)

            # method_iqr_results_dict[method_name] = iqr_successes

            method_positives_results_dict[f"{method_name}_auroc"] = aurocs
            method_positives_results_dict[f"{method_name}_ap"] = aps
            method_positives_results_dict[f"{method_name}_ap_baseline"] = ap_baselines

            # iqr_results_df = pd.DataFrame(method_iqr_results_dict)
            # iqr_results_df.to_csv(subpath / f"negative_explainability_results.csv", index=False)

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
