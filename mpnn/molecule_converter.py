import torch

from rdkit import Chem
from rdkit.Chem import BondType
from torch.nn.functional import one_hot


def one_hot_encode(n: int, i: int | None):
    if i is not None:
        ohe = torch.zeros(n)
        ohe[i] = 1
        return ohe.tolist()
    else:
        return torch.zeros(n).tolist()


def mol_to_torch(mol: Chem.Mol, add_hydrogen_ohe: bool = True):
    atom_ohe_size = 100

    if mol is None:
        raise ValueError("Invalid SMILES")

    one_hot_atomic_nums = []

    mol = Chem.Mol(mol)
    mol = Chem.RemoveAllHs(mol) # Removes all Hs, also stuff like deuterium

    for atom in mol.GetAtoms():
        atom: Chem.Atom

        atomic_num = atom.GetAtomicNum()

        if atomic_num > atom_ohe_size - 1:
            atomic_num = None

        one_hot_atom = one_hot_encode(atom_ohe_size, atomic_num)
        one_hot_atomic_nums.append(one_hot_atom)

    if add_hydrogen_ohe:
        hydrogen_ohe_size = 10
        for atom, atom_ohe in zip(mol.GetAtoms(), one_hot_atomic_nums):
            atom: Chem.Atom

            num_hydrogens = atom.GetTotalNumHs()
            if num_hydrogens > hydrogen_ohe_size - 1:
                raise ValueError(
                    f"Too many hydrogens: {num_hydrogens} hydrogens won't fit into OHE of size {hydrogen_ohe_size}")

            one_hot_hydrogens = one_hot_encode(hydrogen_ohe_size, num_hydrogens)
            atom_ohe.extend(one_hot_hydrogens)

    bonds = []
    bonds_features = []
    bond_ohe_size = 4

    for edge in mol.GetBonds():
        edge: Chem.Bond

        bonds.append([edge.GetBeginAtomIdx(), edge.GetEndAtomIdx()])
        bonds.append([edge.GetEndAtomIdx(), edge.GetBeginAtomIdx()])

        bond_feature = edge.GetBondType()

        try:
            bond_int = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC].index(
                bond_feature)
            ohe_bond_int = one_hot_encode(bond_ohe_size, bond_int)
        except ValueError:
            ohe_bond_int = one_hot_encode(bond_ohe_size, None)

        bonds_features.append(ohe_bond_int)
        bonds_features.append(ohe_bond_int)

    return torch.Tensor(one_hot_atomic_nums), torch.Tensor(bonds).long().T, torch.Tensor(bonds_features)


def smiles_to_torch(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    return mol_to_torch(mol)
