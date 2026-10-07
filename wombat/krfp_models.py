import torch.nn

from wombat.mpnn import MPNN
from wombat.mpnn.molecule_c_detector import MoleculeCDetector
from wombat.mpnn.molecule_d_detector import MoleculeDDetector
from wombat.mpnn.molecule_e_detector import MoleculeEDetector
from wombat.mpnn.molecule_f_detector import MoleculeFDetector
from wombat.mpnn.molecule_g_detector import MoleculeGDetector
from wombat.mpnn.molecule_h_detector import MoleculeHDetector
from wombat.mpnn.molecule_i_detector import MoleculeIDetector
from wombat.mpnn.molecule_j_detector import MoleculeJDetector
from wombat.mpnn.molecule_k_detector import MoleculeKDetector
from wombat.mpnn.molecule_l_detector import MoleculeLDetector
from wombat.mpnn.molecule_m_detector import MoleculeMDetector
from wombat.mpnn.molecule_n_detector import MoleculeNDetector
from wombat.mpnn.molecule_o_detector import MoleculeODetector
from wombat.mpnn.molecule_p_detector import MoleculePDetector
from wombat.mpnn.molecule_q_detector import MoleculeQDetector

krfp_models: list[tuple[torch.nn.Module, str, str]] = [
    (MoleculeCDetector(), "[!#1][CH]([NH]C(=O)[!#1])C(Cl)(Cl)Cl", "[CHD3]([NH][CD3](=O))C(Cl)(Cl)Cl"),
    (MoleculeEDetector(), "[!#1][CH]([!#1])C(=O)[CH2]C([CH3])([CH3])[CH2]C([!#1])([!#1])[!#1]",
     "[CHD3]C(=O)[CH2]C([CH3])([CH3])[CH2][CD4]"),
    (MoleculeFDetector(), "[!#1][CH]([!#1])C(=O)[NH][NH]C(=O)[CH2][CH2][CH2][CH2][CH2][CH2][CH3]",
     "[CHD3]C(=O)[NH][NH]C(=O)[CH2][CH2][CH2][CH2][CH2][CH2][CH3]"),
    # NOTE: Technically, whitebox recognizes [CHD3]C(=O)[N;H1,H2][N;H1,H2]C(=O)[CH2][CH2][CH2][CH2][CH2][CH2][CH3] ([NH] -> [N;H1,H2]) as a match, but both of these SMARTS are equivalent in terms of the GT for our entire dataset, so for validation purposes we have used this formulation.
    (MoleculeGDetector(), "[!#1][CH]([!#1])C(=O)O[CH2][CH2]N([CH2][CH3])[CH2][CH3]",
     "[CHD3]C(=O)O[CH2][CH2]N([CH2][CH3])[CH2][CH3]"),
    (MoleculeHDetector(), "[!#1][CH]([CH2]C(=N[NH]C(=O)[!#1])[!#1])C(=O)[OH]",
     "[CHD3]([CH2][CD3](=N[NH][CD3](=O)))C(=O)[OH]"),
    # NOTE: Technically, whitebox recognizes [CHD3]([CH2][CD3](=N[N;H1,H2][CD3](=O)))C(=O)[OH] ([NH] -> [N;H1,H2]) as a match, but both of these SMARTS are equivalent in terms of the GT for our entire dataset, so for validation purposes we have used this formulation.
    (MoleculeIDetector(), "[!#1][CH]([!#1])[CH]([!#1])[!#1]", "[CHD3][CHD3]"),
    (MoleculeJDetector(), "[!#1][CH]([!#1])[CH2][CH2][CH3]", "[CHD3][CH2][CH2][CH3]"),
    (MoleculeKDetector(), "[!#1][CH]([!#1])[CH2]C(=O)[!#1]", "[CHD3][CH2][CD3](=O)"),
    (MoleculeLDetector(), "[!#1][CH]([!#1])S(=O)(=O)[!#1]", "[CHD3][SD4](=O)(=O)"),
    (MoleculeMDetector(), "[!#1][CH]([!#1])[!#1]", "[!#1][CH]([!#1])[!#1]"),
    (MoleculeNDetector(), "[!#1][CH]([!#1])[CH3]", "[!#1][CH]([!#1])[CH3]"),
    (MoleculeODetector(), "[!#1][CH]=O", "[!#1][CH]=O"),
    (MoleculePDetector(), "[!#1][N+]([CH3])([CH3])[CH3]", "[!#1][N+]([CH3])([CH3])[CH3]"),
    (MoleculeQDetector(), "[!#1][CH2]Cl", "[!#1][CH2]Cl"),
]

name_to_model = {model.__class__.__name__: model for model, _, _ in krfp_models}
name_to_pre_smarts = {model.__class__.__name__: pre for model, pre, _ in krfp_models}
name_to_post_smarts = {model.__class__.__name__: post for model, _, post in krfp_models}

# During the research process before publication, we've used different names for patterns internally.
# Those names would look strange in the paper, so we map them to be reader friendly.
# We retain the original names in the codebase to avoid confusion and chaos.
model_name_to_publication_name = {model.__class__.__name__: f"Pattern {i}" for i, (model, _, _) in
                                  enumerate(krfp_models, start=1)}

# Add mappings from Pattern names as well, so that we can use them interchangeably.
for model_name, publication_name in list(model_name_to_publication_name.items()):
    name_to_model[publication_name] = name_to_model[model_name]
    name_to_pre_smarts[publication_name] = name_to_pre_smarts[model_name]
    name_to_post_smarts[publication_name] = name_to_post_smarts[model_name]
    model_name_to_publication_name[publication_name] = publication_name