import pathlib

import pandas as pd
import torch
from matplotlib import pyplot as plt
from mpnn.validate_whiteboxes import drop_molecule

from rdkit import RDLogger, Chem
from rdkit.Chem import Draw

from mpnn import validate_whitebox
from mpnn.vibe_coding import VibeCodedDetector

RDLogger.DisableLog("rdApp.*")


def main():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    validation_artifact_path = pathlib.Path(f"testing_artifacts/{timestamp}")
    validation_artifact_path.mkdir(exist_ok=True, parents=True)

    model = VibeCodedDetector()
    post_smarts = "[CHD2]=NN=[CHD2]"

    model_name = model.__class__.__name__
    df = pd.read_parquet(pathlib.Path("wombat-smiles/validation_datasets/Pattern 12/validation_molecules.parquet"))
    df = df.sample(n=int(1e6), random_state=42)
    df_positives = pd.read_csv("data/vibecoded_mol/positives.csv")

    df = pd.concat([df, df_positives], ignore_index=True)
    df = df.drop_duplicates(subset="SMILES", keep="first")

    smiles = df["SMILES"].tolist()

    artifact_df = validate_whitebox(model, post_smarts, smiles, labels=None)

    artifact_df.to_csv(
        validation_artifact_path / f"{model_name}_w_{model.readout.__class__.__name__}_validation_artifact.csv",
        index=False)

    torch.save(model.state_dict(),
               validation_artifact_path / f"{model_name}_w_{model.readout.__class__.__name__}_state_dict.pt")


if __name__ == "__main__":
    main()
