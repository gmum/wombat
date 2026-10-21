from .visualize_activations import visualize_activations, visualize_all_activations
from .mpnn_arch import MPNN, AllNonZeroReadout, deal_with_pyg
from .molecule_converter import mol_to_torch, smiles_to_torch, one_hot_encode

from .validate_whiteboxes import validate_whitebox

from .molecule_c_detector import MoleculeCDetector
from .molecule_e_detector import MoleculeEDetector
from .molecule_f_detector import MoleculeFDetector
from .molecule_g_detector import MoleculeGDetector
from .molecule_h_detector import MoleculeHDetector
from .molecule_i_detector import MoleculeIDetector
from .molecule_j_detector import MoleculeJDetector
from .molecule_k_detector import MoleculeKDetector
from .molecule_l_detector import MoleculeLDetector
from .molecule_m_detector import MoleculeMDetector
from .molecule_n_detector import MoleculeNDetector
from .molecule_o_detector import MoleculeODetector
from .molecule_p_detector import MoleculePDetector
from .molecule_q_detector import MoleculeQDetector

# Alias naming
from .molecule_c_detector import MoleculeCDetector as Pattern1Detector
from .molecule_e_detector import MoleculeEDetector as Pattern2Detector
from .molecule_f_detector import MoleculeFDetector as Pattern3Detector
from .molecule_g_detector import MoleculeGDetector as Pattern4Detector
from .molecule_h_detector import MoleculeHDetector as Pattern5Detector
from .molecule_i_detector import MoleculeIDetector as Pattern6Detector
from .molecule_j_detector import MoleculeJDetector as Pattern7Detector
from .molecule_k_detector import MoleculeKDetector as Pattern8Detector
from .molecule_l_detector import MoleculeLDetector as Pattern9Detector
from .molecule_m_detector import MoleculeMDetector as Pattern10Detector
from .molecule_n_detector import MoleculeNDetector as Pattern11Detector
from .molecule_o_detector import MoleculeODetector as Pattern12Detector
from .molecule_p_detector import MoleculePDetector as Pattern13Detector
from .molecule_q_detector import MoleculeQDetector as Pattern14Detector