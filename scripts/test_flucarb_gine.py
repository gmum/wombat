import pathlib

import pandas as pd
import torch

from wombat.mpnn.gine_experiments import FluCarbGINE
from wombat.mpnn.validate_whiteboxes import drop_molecule

from rdkit import RDLogger, Chem

from wombat.mpnn import validate_whitebox

RDLogger.DisableLog("rdApp.*")


def get_labels(smiles_list: list[str], pattern_smarts: list[str]) -> tuple[list[str], list[bool]]:
    patterns = [Chem.MolFromSmarts(smart) for smart in pattern_smarts]

    ok_smiles = []
    labels = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        if drop_molecule(mol):
            continue

        try:
            mol = Chem.RemoveAllHs(mol)
        except Chem.KekulizeException:
            continue

        ok_smiles.append(smiles)

        sublabels = []
        for smarts_mol in patterns:
            sublabel = len(mol.GetSubstructMatches(smarts_mol)) != 0
            sublabels.append(sublabel)

        if all(sublabels):
            labels.append(True)
        else:
            labels.append(False)

    return ok_smiles, labels


def main():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    validation_artifact_path = pathlib.Path(f"testing_artifacts/{timestamp}")
    validation_artifact_path.mkdir(exist_ok=True, parents=True)

    model = FluCarbGINE()
    post_smarts = ["[!#1]F", "[#6]=O"]

    model_name = model.__class__.__name__
    df = pd.read_parquet(pathlib.Path("wombat-smiles/validation_datasets/Pattern 12/validation_molecules.parquet"))
    df = df.sample(n=int(1e6), random_state=42)

    smiles = df["SMILES"].tolist()
    # smiles = ["O.O.O.[F][Fe]([F])[F]"]

    smiles, labels = get_labels(smiles, post_smarts)

    artifact_df = validate_whitebox(model, None, smiles, labels=labels)

    artifact_df.to_csv(
        validation_artifact_path / f"{model_name}_validation_artifact.csv",
        index=False)

    torch.save(model.state_dict(),
               validation_artifact_path / f"{model_name}_state_dict.pt")


if __name__ == "__main__":
    main()
