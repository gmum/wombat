import pandas as pd
import time
import pathlib
import numpy as np

from multiprocessing import Pool
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from krfp_models import krfp_models
from mpnn.validate_whiteboxes import drop_molecule


def process_chunk(chunk: pd.DataFrame, chunk_idx: int, base_path: pathlib.Path):
    print("Processing chunk", chunk_idx)
    smiles = chunk["SMILES"].tolist()

    patterns = [Chem.MolFromSmarts(smart) for (_, _, smart) in krfp_models]
    model_names = [model.__class__.__name__ for (model, _, _) in krfp_models]

    all_data = []
    fps = []

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)

        if drop_molecule(mol):
            continue

        try:
            mol = Chem.RemoveAllHs(mol)
        except Chem.KekulizeException:
            continue

        smile = Chem.MolToSmiles(mol, canonical=True)

        fp = fpgen.GetFingerprint(mol)
        fp = list(fp)
        fps.append(np.array(fp, dtype=bool))

        data = [smile] + [mol.HasSubstructMatch(pattern) for pattern in patterns]

        all_data.append(data)

    all_data = pd.DataFrame(all_data, columns=["SMILES"] + model_names)
    fps = np.array(fps)

    output_path = base_path / f"chunk_{chunk_idx}.csv"
    all_data.to_csv(output_path, index=False)
    fps_output_path = base_path / f"chunk_{chunk_idx}_fps.npz"
    np.savez_compressed(fps_output_path, fps)


def main():
    if not pathlib.Path("data/pubchem_smiles_deduplicated.csv").exists():
        df = pd.read_csv("data/pubchem_smiles.csv")
        print("Original dataset size:", len(df))
        df = df.drop_duplicates(subset=["SMILES"])
        print("Dataset size after dropping duplicates:", len(df))
        df.to_csv("data/pubchem_smiles_deduplicated.csv", index=False)

    RDLogger.DisableLog("rdApp.*")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_path = pathlib.Path(f"pubchem_processed_{timestamp}/")
    data_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/pubchem_smiles_deduplicated.csv", chunksize=int(1e6))

    print("Starting processing chunks...")

    with Pool(processes=20) as pool:
        results = []

        for chunk_idx, chunk in enumerate(df):
            result = pool.apply_async(
                process_chunk,
                args=(chunk, chunk_idx, data_path)
            )

            results.append(result)

        for result in tqdm(results, desc="Processing chunks"):
            result.get()


if __name__ == "__main__":
    main()
