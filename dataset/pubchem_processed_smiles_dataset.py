import random
import numpy as np
import pandas as pd
import pathlib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from krfp_models import krfp_models, name_to_post_smarts, name_to_pre_smarts


class PubchemProcessedSMILESDataset:
    def _load_csvs(self, nchunks: int | None = None):
        smiles_csvs = list(self.path.rglob("*.csv"))

        def get_key(name):
            assert name.startswith("chunk")

            return int(name.split("_")[1])

        sorted_smiles_csvs = sorted(smiles_csvs, key=lambda x: get_key(x.stem))

        for i, path in enumerate(sorted_smiles_csvs):
            if nchunks is not None and i >= nchunks:
                break

            yield pd.read_csv(path)

    def _load_fps(self, nchunks: int | None = None):
        fps_csvs = list(self.path.rglob("*.npy")) + list(self.path.rglob("*.npz"))

        def get_key(name):
            assert name.startswith("chunk")

            return int(name.split("_")[1])

        sorted_fps_csvs = sorted(fps_csvs, key=lambda x: get_key(x.stem))

        for i, path in enumerate(sorted_fps_csvs):
            if nchunks is not None and i >= nchunks:
                break

            if path.suffix == ".npy":
                yield np.load(path).astype(bool)
            elif path.suffix == ".npz":
                yield np.load(path)["arr_0"].astype(bool)

    def __init__(self, path: pathlib.Path, nchunks: int | None = None):
        self.path = path
        self.nchunks = nchunks

        all_dfs = list(self._load_csvs(nchunks=nchunks))

        self.df = pd.concat(all_dfs, ignore_index=True)

    def _filter_for_pattern(self, pattern_name: str, val: bool) -> list[str]:
        if pattern_name not in self.df.columns:
            raise ValueError(f"Pattern name {pattern_name} not found in dataset columns")

        return self.df[self.df[pattern_name] == val]["SMILES"].tolist()

    def get_positives_for_pattern(self, pattern_name: str) -> list[str]:
        return self._filter_for_pattern(pattern_name, True)

    def get_negatives_for_pattern(self, pattern_name: str) -> list[str]:
        return self._filter_for_pattern(pattern_name, False)

    def filter_by_fp_score(self, scoring_fn: callable, threshold: float) -> pd.DataFrame:
        for fps, df in zip(self._load_fps(nchunks=self.nchunks), self._load_csvs(nchunks=self.nchunks)):
            scores = scoring_fn(fps)

            mask = scores >= threshold

            yield df[mask]

    def filter_by_tversky_index(self, target_fp: np.ndarray, alpha: float, beta: float, threshold: float):
        assert target_fp.dtype == bool

        if target_fp.ndim == 1:
            target_fp = np.expand_dims(target_fp, axis=0)
        else:
            assert target_fp.ndim == 2
            assert target_fp.shape[0] == 1

        def scoring_fn(fps):
            intersection = fps & target_fp
            only_in_prototype = target_fp & ~fps
            only_in_candidate = fps & ~target_fp

            intersection_sum = np.sum(intersection, axis=1)
            only_in_prototype_sum = np.sum(only_in_prototype, axis=1)
            only_in_candidate_sum = np.sum(only_in_candidate, axis=1)

            tversky_index = intersection_sum / (
                    intersection_sum + alpha * only_in_prototype_sum + beta * only_in_candidate_sum + 1e-8)

            return tversky_index

        return self.filter_by_fp_score(scoring_fn, threshold)


def get_dataset_for_pattern(pattern_name: str, path: pathlib.Path, nchunks: int | None = None,
                            max_bag: int = int(5e6)) -> pd.DataFrame:
    ds = PubchemProcessedSMILESDataset(path=path, nchunks=nchunks)

    # Step 1: All positives for the pattern, if there are more than max_bag, sample max_bag of them
    df = ds.df
    df = df.drop_duplicates(subset=["SMILES"])

    if pattern_name not in df.columns:
        raise ValueError(f"Pattern name {pattern_name} not found in dataset columns")

    positives_df = df[df[pattern_name] == True].copy()

    if len(positives_df) > max_bag:
        positives_df = positives_df.sample(n=max_bag, random_state=42)

    positives_df["ORIGIN"] = "POSITIVE"

    # Step 2: Sample negatives; similar story as positivies.
    negatives_df = df[df[pattern_name] == False]

    if len(negatives_df) > max_bag:
        negatives_df = negatives_df.sample(n=max_bag, random_state=42)

    negatives_df["ORIGIN"] = "NEGATIVE"

    # Step 3: Get the fingerprint for SMARTS and filter by Tversky index, keeping only those with Tversky index >= 0.5
    # (again, apply max_bag if there are too many)

    pattern_smarts = name_to_pre_smarts[pattern_name]

    mol = Chem.MolFromSmarts(pattern_smarts)

    # Because this is from SMARTS, we need to clean this up
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)

    smiles = smiles.replace("*", "C")
    mol = Chem.MolFromSmiles(smiles)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp = fpgen.GetFingerprint(mol)
    fp = list(fp)
    target_fp = np.array(fp, dtype=bool)

    sub_dfs = []

    for sub_df in ds.filter_by_tversky_index(
            target_fp=target_fp,
            alpha=1.0,
            beta=0.0,
            threshold=0.5
    ):
        sub_df = sub_df[sub_df[pattern_name] == False]  # We have enough positives already
        sub_dfs.append(sub_df)

    filtered_df = pd.concat(sub_dfs, ignore_index=True)
    filtered_df = filtered_df.drop_duplicates(subset=["SMILES"])

    if len(filtered_df) > max_bag:
        filtered_df = filtered_df.sample(n=max_bag, random_state=42)
    filtered_df["ORIGIN"] = "TVERSKY"

    # Note: Order is important here. If something is Tversky negative and random negative, we want to label it as Tversky negative.
    final_df = pd.concat([positives_df, filtered_df, negatives_df], ignore_index=True)
    final_df = final_df[["SMILES", pattern_name, "ORIGIN"]]

    print("Positives:", len(positives_df))
    print("Negatives:", len(negatives_df))
    print("Tversky-filtered:", len(filtered_df))

    print("Final dataset size:", len(final_df))

    # Drop duplicates if any
    final_df = final_df.drop_duplicates(subset=["SMILES"])

    print("Post deduplication dataset size Tversky:", len(final_df[final_df["ORIGIN"] == 'TVERSKY']))
    print("Post deduplication dataset size POSITIVE:", len(final_df[final_df["ORIGIN"] == 'POSITIVE']))
    print("Post deduplication dataset size NEGATIVE:", len(final_df[final_df["ORIGIN"] == 'NEGATIVE']))

    return final_df


def main():
    import warnings
    warnings.filterwarnings("ignore")

    for model, smarts, _ in tqdm(krfp_models):
        name = model.__class__.__name__

        df = get_dataset_for_pattern(name, path=pathlib.Path("../data/pubchem_processed/"), max_bag=int(5e6))
        df.to_csv(f"../data/validation_datasets_new/{name}.csv", index=False)


if __name__ == "__main__":
    main()
