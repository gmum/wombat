import pathlib

import pandas as pd
import torch

from rdkit import RDLogger

from mpnn import validate_whitebox
from krfp_models import krfp_models

RDLogger.DisableLog("rdApp.*")


def main():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    validation_artifact_path = pathlib.Path(f"testing_artifacts/{timestamp}")
    validation_artifact_path.mkdir(exist_ok=True, parents=True)

    for model, _, smarts in krfp_models:
        model_name = model.__class__.__name__
        df = pd.read_csv(pathlib.Path("data/validation_datasets") / f"{model_name}.csv")
        smiles = df["SMILES"].tolist()
        labels = df[model_name].tolist()
        origins = df["ORIGIN"]

        assert type(labels[0]) is bool, f"Labels for {model_name} must be boolean, but got {type(labels[0])}"

        print(f"Validating {model.__class__.__name__} on {len(smiles)} molecules")
        artifact_df = validate_whitebox(model, smarts, smiles, labels=labels)

        artifact_df["ORIGIN"] = origins

        artifact_df.to_csv(
            validation_artifact_path / f"{model_name}_w_{model.readout.__class__.__name__}_validation_artifact.csv",
            index=False)
        torch.save(model.state_dict(),
                   validation_artifact_path / f"{model_name}_w_{model.readout.__class__.__name__}_state_dict.pt")


if __name__ == "__main__":
    main()
