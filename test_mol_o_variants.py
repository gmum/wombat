import pathlib

import pandas as pd
import torch

from rdkit import RDLogger

from mpnn import validate_whitebox
from krfp_models import krfp_models
from mpnn.molecule_o_variants import LeafActivatedMoleculeODetector, ActivationMovingMoleculeODetector, \
    SmoothingMoleculeODetector

RDLogger.DisableLog("rdApp.*")


def main():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    validation_artifact_path = pathlib.Path(f"testing_artifacts/{timestamp}")
    validation_artifact_path.mkdir(exist_ok=True, parents=True)

    models = [
        (LeafActivatedMoleculeODetector(), krfp_models[11][2]),
        (ActivationMovingMoleculeODetector(), krfp_models[11][2]),
        (SmoothingMoleculeODetector(), krfp_models[11][2])
    ]

    for model, smarts in models:
        model_name = model.__class__.__name__
        df = pd.read_parquet(pathlib.Path("wombat-smiles/validation_datasets/Pattern 12/validation_molecules.parquet"))
        df = df.sample(n=int(1e6), random_state=42)
        smiles = df["SMILES"].tolist()
        labels = df["MoleculeODetector"].tolist()
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
