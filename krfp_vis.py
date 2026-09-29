import pandas as pd
import json

from rdkit import Chem
from rdkit.Chem import Draw

from matplotlib import pyplot as plt
from tqdm import tqdm


def main():
    smarts = list(json.load(open("krfp_smarts.json", "r")))

    imgs = []
    img_smarts = []
    img_smiles = []
    hit_number = []
    krfp_numbers = []

    for smart, i in tqdm(zip(smarts, range(1000))):
        mol = Chem.MolFromSmarts(smart)

        if mol is None:
            continue

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

        if mol is None or mol.GetRingInfo().NumRings() > 0:
            continue

        if mol.GetNumAtoms() > 10:
            continue

        smarts_pattern = Chem.MolFromSmarts(smart)
        # hits_number = sum(1 for bxaic_mol in bxaic_mols if bxaic_mol.HasSubstructMatch(smarts_pattern))
        hits_number = 0  # FIXME
        img = Draw.MolToImage(mol, size=(500, 500))
        img_smarts.append(smart)
        img_smiles.append(Chem.MolToSmiles(mol))
        imgs.append(img)
        hit_number.append(hits_number)
        krfp_numbers.append(i + 1)

        # if _ > 50:
        #    break

    # print("Number of possible smarts", len(imgs))

    batch_size = 20

    # Split the images and hit numbers into batches of size batch_size
    img_batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
    hit_batches = [hit_number[i:i + batch_size] for i in range(0, len(hit_number), batch_size)]
    krfp_batches = [krfp_numbers[i:i + batch_size] for i in range(0, len(krfp_numbers), batch_size)]

    for imgs, krfp_numbers in zip(img_batches, krfp_batches):

        fig, axs = plt.subplots(nrows=5, ncols=batch_size // 5, figsize=(20, 10))

        for krfp_number, img, ax in zip(krfp_numbers, imgs, axs.flatten()):
            ax.set_title(f"KRFP #{krfp_number}", fontsize=9)
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.gcf().set_dpi(300)
        plt.show()


if __name__ == "__main__":
    main()
