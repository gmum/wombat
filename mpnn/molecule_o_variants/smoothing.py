import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class SmoothingMoleculeODetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        no_hydrogens_check_weights = [-10] * 10
        no_hydrogens_check_weights[0] = 0

        exactly_one_hydrogen_check_weights = [-10] * 10
        exactly_one_hydrogen_check_weights[1] = 0

        exactly_two_hydrogens_check_weights = [-10] * 10
        exactly_two_hydrogens_check_weights[2] = 0

        exactly_three_hydrogens_check_weights = [-10] * 10
        exactly_three_hydrogens_check_weights[3] = 0

        # A.1: Carbon with one hydrogen connected to something with a single bond.
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            one_hot_encode(4, 0),
            0
        )

        # B.1: Oxygen connected to something.
        substructure_b_1_detector_weights = (
            one_hot_encode(110, 8),
            np.ones(100).tolist() + np.zeros(10).tolist(),
            np.ones(4).tolist(),
            0
        )

        # C.1: Anything connected to anything via any bond.
        substructure_c_1_detector_weights = (
            np.ones(100).tolist() + np.zeros(10).tolist(),
            np.ones(100).tolist() + np.zeros(10).tolist(),
            np.ones(4).tolist(),
            0
        )

        params_zipped = zip(
            substructure_a_1_detector_weights,
            substructure_b_1_detector_weights,
            substructure_c_1_detector_weights
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn1 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

        # A.2: A.1 with C.1, regardless of bond type
        substructure_a_2_detector_weights = (
            one_hot_encode(3, 0),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # B.2: B.1 with C.1, regardless of bond type
        substructure_b_2_detector_weights = (
            one_hot_encode(3, 1),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # C.2: C.1 with C.1 via any bond
        substructure_c_2_detector_weights = (
            one_hot_encode(3, 2),
            one_hot_encode(3, 2),
            torch.ones(4).tolist(),
            0
        )

        params_zipped = zip(
            substructure_a_2_detector_weights,
            substructure_b_2_detector_weights,
            substructure_c_2_detector_weights
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn2 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

        # A.3: A.2 with C.2, regardless of bond type
        substructure_a_3_detector_weights = (
            one_hot_encode(3, 0),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # B.3: B.2 with C.2, regardless of bond type
        substructure_b_3_detector_weights = (
            one_hot_encode(3, 1),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # C.3: C.2 with C.2 via any bond
        substructure_c_3_detector_weights = (
            one_hot_encode(3, 2),
            one_hot_encode(3, 2),
            torch.ones(4).tolist(),
            0
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
            substructure_b_3_detector_weights,
            substructure_c_3_detector_weights
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

        # A.4: A.3 with C.3, regardless of bond type
        substructure_a_4_detector_weights = (
            one_hot_encode(3, 0),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # B.4: B.3 with C.3, regardless of bond type
        substructure_b_4_detector_weights = (
            one_hot_encode(3, 1),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # C.4: C.3 with C.3 via any bond
        substructure_c_4_detector_weights = (
            one_hot_encode(3, 2),
            one_hot_encode(3, 2),
            torch.ones(4).tolist(),
            0
        )

        params_zipped = zip(
            substructure_a_4_detector_weights,
            substructure_b_4_detector_weights,
            substructure_c_4_detector_weights
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn4 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

        # A.5: A.4 with C.4, regardless of bond type
        substructure_a_5_detector_weights = (
            one_hot_encode(3, 0),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # B.5: B.4 with C.4, regardless of bond type
        substructure_b_5_detector_weights = (
            one_hot_encode(3, 1),
            one_hot_encode(3, 2),
            np.ones(4).tolist(),
            0
        )

        # C.5: C.4 with C.4 via any bond
        substructure_c_5_detector_weights = (
            one_hot_encode(3, 2),
            one_hot_encode(3, 2),
            torch.ones(4).tolist(),
            0
        )

        params_zipped = zip(
            substructure_a_5_detector_weights,
            substructure_b_5_detector_weights,
            substructure_c_5_detector_weights
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn5 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

        # A.6: A.5 with B.5, double bond
        substructure_a_6_detector_weights = (
            one_hot_encode(3, 0),
            one_hot_encode(3, 1),
            one_hot_encode(4, 1),
            0
        )

        params_zipped = zip(
            substructure_a_6_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn6 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

        self.readout = AllNonZeroReadout()
        self.pyg_mode = True

    def forward(self, *args, **kwargs):
        if self.pyg_mode:
            x, edge_features, edge_index = deal_with_pyg(*args, **kwargs)
        else:
            x, edge_features, edge_index = args

        x1 = self.mpnn1(x, edge_features, edge_index)
        x2 = self.mpnn2(x1, edge_features, edge_index)
        x3 = self.mpnn3(x2, edge_features, edge_index)
        x4 = self.mpnn4(x3, edge_features, edge_index)
        x5 = self.mpnn5(x4, edge_features, edge_index)
        x6 = self.mpnn6(x5, edge_features, edge_index)

        dump_activations = kwargs.get("dump_activations", False)

        if not dump_activations:
            return self.readout(x6)
        else:
            return self.readout(x6), [x1, x2, x3, x4, x5, x6]
