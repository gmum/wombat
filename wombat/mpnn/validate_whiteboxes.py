import pandas as pd
import torch.nn
from rdkit import Chem
from matplotlib import pyplot as plt
from tqdm import tqdm

from wombat.mpnn import visualize_all_activations, MPNN
from wombat.mpnn import mol_to_torch


def inform_about_problem(model, mol: Chem.Mol, problem_name: str, add_hs: bool = True):
    print(f"{problem_name} found!")
    print(Chem.MolToSmiles(mol))

    x, edge_index, edge_features = mol_to_torch(mol, add_hydrogen_ohe=add_hs)
    output, activations = model(x, edge_features, edge_index, dump_activations=True)
    activation_imgs = visualize_all_activations(mol, activations)

    fig, axs = plt.subplots(nrows=1, ncols=len(activation_imgs), figsize=(10 * len(activation_imgs), 10), squeeze=False)

    for img, ax, i in zip(activation_imgs, axs.flatten(), range(len(activation_imgs))):
        ax.imshow(img)
        ax.set_title(f"Activations after MPNN layer {i + 1}; latent size {activations[i].shape[1]}", fontsize=16)
        ax.axis('off')

    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()


def drop_molecule(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return True

    if len(mol.GetBonds()) == 0 or len(mol.GetAtoms()) == 0:
        return True

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetTotalValence() > 2:
            return True
        elif atom.GetSymbol() == "N" and atom.GetTotalValence() > 4:
            return True
        elif atom.GetSymbol() == "C" and atom.GetTotalValence() > 4:
            return True
        elif atom.GetSymbol() == "S" and atom.GetTotalValence() > 6:
            return True

    return False


def validate_whitebox(model: torch.nn.Module, pattern_smarts: str | None, smiles_list: list[str], add_hs: bool = True,
                      labels: list[bool] | None = None) -> pd.DataFrame:
    if pattern_smarts is not None:
        smarts_mol = Chem.MolFromSmarts(pattern_smarts)

    positive_count, negative_count = 0, 0
    preds = []

    if labels is not None and len(labels) != len(smiles_list):
        raise ValueError("Length of labels must match length of smiles_list")

    if labels is None:
        labels = [None] * len(smiles_list)

    bar = tqdm(zip(smiles_list, labels), desc="Validating model", total=len(smiles_list))

    for smiles, label in bar:
        mol = Chem.MolFromSmiles(smiles)

        if drop_molecule(mol):
            preds.append(None)
            continue

        try:
            mol = Chem.RemoveAllHs(mol)
        except Chem.KekulizeException:
            preds.append(None)
            continue

        if label is None:
            if pattern_smarts is None:
                raise ValueError("Either pattern_smarts or labels must be provided")
            is_positive = len(mol.GetSubstructMatches(smarts_mol)) != 0
            is_negative = not is_positive
        else:
            is_positive = label
            is_negative = not label

        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

        x, edge_index, edge_features = mol_to_torch(mol, add_hydrogen_ohe=add_hs)
        output = model(x, edge_features, edge_index)

        preds.append(output.item())

        if is_positive and output.item() < 0.5:
            inform_about_problem(
                model=model,
                mol=mol,
                problem_name=f"False negative for {pattern_smarts}",
                add_hs=add_hs
            )
        elif is_negative and output.item() > 0.5:
            inform_about_problem(
                model=model,
                mol=mol,
                problem_name=f"False positive for {pattern_smarts}",
                add_hs=add_hs
            )

        bar.set_postfix({
            "positives": positive_count,
            "negatives": negative_count
        })

    print(
        f"Evaluated {model.__class__.__name__} ({pattern_smarts}) on {positive_count} positives and {negative_count} negatives.")

    return pd.DataFrame({
        "SMILES": smiles_list,
        "PRED": preds,
        "LABEL": labels
    })
