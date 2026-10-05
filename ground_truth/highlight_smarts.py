from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

from datavis import visualize_atom_importance_from_mol


# Don't use these functions if SMARTS are complicated, and you don't know what this does. Our SMARTS are, thankfully, "boring" so this should work fine.

def find_atoms_with_d_property(smarts: str) -> set[int]:
    smarts_mol = Chem.MolFromSmarts(smarts)

    if not smarts_mol:
        raise ValueError("Invalid SMARTS")

    atoms = set()

    for i, atom in enumerate(smarts_mol.GetAtoms()):
        if "D" in atom.GetSmarts():
            atoms.add(i)

    return atoms


def highlight_atoms_in_mol(mol: Chem.Mol, smarts: str) -> list[int]:
    """Highlights atoms in a molecule that match the given SMARTS pattern AND all neighbours of atoms that have the 'D' property in the SMARTS."""

    d_idxs = find_atoms_with_d_property(smarts)
    smarts_mol = Chem.MolFromSmarts(smarts)

    matches = mol.GetSubstructMatches(smarts_mol)

    if not matches:
        return []

    highlighted_atoms = set()

    for match in matches:
        for i, atom_idx in enumerate(match):
            highlighted_atoms.add(atom_idx)

            if i in d_idxs:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    highlighted_atoms.add(neighbor.GetIdx())

    return list(highlighted_atoms)


# This basically shows our ground truth for the explainers. This is why we had to look out for the "D" property, as our net needs to "look" at them (hence they should get a non-zero attribution).
def visualize_smarts_match(mol: Chem.Mol, smarts: str, size: int = 1000, bond_line_width: int = 2,
                           annotation_font_scale: float = 0.9, precision: int = 2, radius: float = 0.7) -> Image.Image:
    highlighted_atoms = highlight_atoms_in_mol(mol, smarts)
    attributions = [1.0 if i in highlighted_atoms else 0.0 for i in range(mol.GetNumAtoms())]

    return visualize_atom_importance_from_mol(mol, attributions, length=size, bond_line_width=bond_line_width,
                                              annotation_font_scale=annotation_font_scale, precision=precision,
                                              radius=radius)


def main():
    mol = Chem.MolFromSmiles("C1=CC=C(C=C1)/C=C/C(=O)NC(C(Cl)(Cl)Cl)NC(=S)NC2=CC=C(C=C2)Cl")
    smarts = "[CHD3]([NH][CD3](=O))C(Cl)(Cl)Cl"

    img = visualize_smarts_match(mol, smarts)
    img.show()


if __name__ == "__main__":
    main()
