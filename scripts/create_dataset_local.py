import os
import pathlib
import gzip
import re
import pandas as pd

from tqdm import tqdm


def main():
    gz_file_path = pathlib.Path("data/pubchem")
    regex = re.compile(r"\".*\"")

    all_smiles = []

    for gz_file in tqdm(gz_file_path.glob("*.ttl.gz"), desc="Extracting SMILES from gz files",
                        total=len(list(gz_file_path.glob("*.ttl.gz")))):
        with gzip.open(gz_file, 'rt') as smiles_file:
            contents = smiles_file.read()
            matches = regex.findall(contents)
            smiles = [match.strip('"') for match in matches]

            all_smiles += smiles

    df = pd.DataFrame({"SMILES": all_smiles})
    df.to_csv("data/pubchem_smiles.csv", index=False)


if __name__ == "__main__":
    main()
