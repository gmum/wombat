# This is meant to check if KRFP SMARTS patterns we
import json

from rdkit import Chem

with open("krfp_smarts.txt", "r") as f:
    data = f.read()

smarts = data.split()[1:]

krfps = []

for i, smart in enumerate(smarts):
    mol = Chem.MolFromSmarts(smart)

    assert mol is not None, f"Invalid SMARTS pattern: {smart}"

with open("krfp_smarts.json", "w") as f:
    json.dump(smarts, f, indent=4)
    