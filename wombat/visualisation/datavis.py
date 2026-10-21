import rdkit
import re

from PIL import Image
from pymupdf import pymupdf
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point2D



def visualize_atom_importance_from_mol(mol: rdkit.Chem.Mol, attributions: list, length: int = 1000,
                                       bond_line_width: int = 2, annotation_font_scale: float = 0.9, precision: int = 2,
                                       radius: float = 0.7, show_atom_idx: bool = False
                                       ) -> Image.Image:
    mol = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    attributions = list(attributions)

    d = rdMolDraw2D.MolDraw2DSVG(length, length)
    options = d.drawOptions()
    options.bondLineWidth = bond_line_width
    options.annotationFontScale = annotation_font_scale

    assert len(attributions) == mol.GetNumAtoms(), "Number of attributions must match number of atoms in the molecule."

    for i, atom in enumerate(mol.GetAtoms()):
        if i < len(attributions):
            if show_atom_idx:
                atom.SetProp('atomNote', f"{i}: {attributions[i]:.{precision}f}")
            else:
                atom.SetProp('atomNote', f"{attributions[i]:.{precision}f}")

    # empty canvas!
    d.DrawMolecule(
        mol,
    )

    # don't draw ellipses if all attributions are zero
    if max(attributions) == 0:
        drawing_text = d.GetDrawingText()
        doc = pymupdf.open("svg", drawing_text.encode("utf-8"))
        page = doc[0]
        pix = page.get_pixmap(dpi=300, alpha=True)
        img = pix.pil_image()

        return img

    d.ClearDrawing()

    max_attrib = max(attributions)

    for i, atom in enumerate(mol.GetAtoms()):
        atom_coords_1 = Point2D(conf.GetAtomPosition(i).x - radius, conf.GetAtomPosition(i).y - radius)
        atom_coords_2 = Point2D(conf.GetAtomPosition(i).x + radius, conf.GetAtomPosition(i).y + radius)

        d.SetFillPolys(True)
        d.SetColour((1.0, 0.0, 0.0))
        d.DrawEllipse(atom_coords_1, atom_coords_2)

    d.DrawMolecule(
        mol,
    )

    d.FinishDrawing()

    # ugly hack because rdkit does not really care
    drawing_text = d.GetDrawingText()

    drawing_text = drawing_text.replace("style='fill:#FF0000", "style='fill:#FF0000;fill-opacity:MARK")

    drawing_text = drawing_text.replace(
        "style='fill:#FF0000;fill-opacity:MARK;",
        "fill='#FF0000' fill-opacity='MARK' style='"
    )

    # Now we go through drawing text and every time we find fill-opacity=MARK we substitute it with what we want
    while len(attributions) > 0:
        drawing_text = drawing_text.replace("MARK",
                                            f"{0.3 * (attributions.pop(0) / max_attrib if max_attrib != 0 else 0):.3f}",
                                            1)

    # And now change stroke opacity to zero for all ellipses and only ellipses
    # Regex!

    drawing_text = re.sub(
        r"(<ellipse[^>]*style='[^']*)stroke-opacity\s*:\s*1;?([^']*')",
        r"\1\2",
        drawing_text
    )

    drawing_text = re.sub(
        r"(<ellipse[^>]*style='[^']*)stroke:#FF0000;?([^']*')",
        r"\1\2",
        drawing_text
    )

    doc = pymupdf.open("svg", drawing_text.encode("utf-8"))
    page = doc[0]
    pix = page.get_pixmap(dpi=300, alpha=True)
    img = pix.pil_image()

    return img


def visualize_atom_importance(smiles: str, attributions: list, length: int = 4000, bond_line_width: int = 2,
                              annotation_font_scale: float = 0.9, precision: int = 2,
                              radius: float = 0.7, show_atom_idx: bool = False) -> Image.Image:
    mol = rdkit.Chem.MolFromSmiles(smiles)

    return visualize_atom_importance_from_mol(mol, attributions, length, bond_line_width, annotation_font_scale,
                                              precision, radius, show_atom_idx)
