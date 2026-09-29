from itertools import zip_longest

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from mpnn import mol_to_torch, visualize_activations, visualize_all_activations
from matplotlib import pyplot as plt

from krfp_models import krfp_models, model_name_to_publication_name


def main():
    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(2, 7, figsize=(45, 15), squeeze=False)

    print(len(krfp_models))

    for ax in axs.flatten():
        ax.axis('off')

    for ax, (model, smarts, _) in zip(axs.flatten(), krfp_models):
        mol = Chem.MolFromSmarts(smarts)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(smiles)  # type: Chem.Mol
        # smiles = smiles.replace("*", "C")

        x, edge_index, edge_features = mol_to_torch(mol, add_hydrogen_ohe=True)
        _, activations = model(x, edge_features, edge_index, dump_activations=True)

        drawing_options = Draw.rdMolDraw2D.MolDrawOptions()
        drawing_options.bondLineWidth = 4
        img = Draw.MolToImage(mol, size=(600, 600), options=drawing_options)

        name = model.__class__.__name__

        publication_name = model_name_to_publication_name[name]
        # publication = publication_name.removeprefix("Pattern ")
        # publication_name = f"Motif {publication_name}"

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{publication_name}\n No. layers: {len(activations)}", fontsize=60)

    plt.gcf().set_dpi(300)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    # plt.show()

    plt.savefig("krfp_patterns.pdf")

    exit()

    for model, smarts, _ in krfp_models:
        mol = Chem.MolFromSmarts(smarts)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        smiles = smiles.replace("*", "C")

        mol = Chem.MolFromSmiles(smiles)

        x, edge_index, edge_features = mol_to_torch(mol, add_hydrogen_ohe=True)
        _, activations = model(x, edge_features, edge_index, dump_activations=True)

        imgs = visualize_all_activations(mol, activations, bond_line_width=8, annotation_font_scale=1.5)

        img_number = len(imgs)

        ncols = 3
        nrows = img_number // ncols + (1 if img_number % ncols != 0 else 0)

        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

        for i, (ax, img, activation) in enumerate(zip_longest(axs.flatten(), imgs, activations, fillvalue=None),
                                                  start=1):
            if img is None:
                ax.axis('off')
                continue
            ax.imshow(img)
            ax.set_title(f"Activations after MPNN layer {i} \n Latent size = {activation.shape[1]}",
                         fontsize=16)
            ax.axis('off')

        plt.gcf().set_dpi(300)
        plt.suptitle(f"Activations for {model.__class__.__name__}", fontsize=20)
        plt.tight_layout(rect=(0, 0, 1, 0.9))
        plt.show()


if __name__ == "__main__":
    main()
