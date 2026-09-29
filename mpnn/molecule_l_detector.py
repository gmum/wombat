import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeLDetector(torch.nn.Module):
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

        # A.1: Sulphur with >= 4 neighbours (whatever-bond) and no hydrogens. 
        # Note that activation here can be >1 (!) iff the sulphur has >4 neighbours => at most 1 double bond
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 16) + no_hydrogens_check_weights,
            np.ones(100).tolist() + [0] * 10,
            np.ones(4).tolist(),
            -3
        )

        # B.1: Sulphur with >= 3 neighbours WITH SINGLE BOND and no hydrogens. This implies sulphur without 2x double bonds
        substructure_b_1_detector_weights = (
            one_hot_encode(100, 16) + no_hydrogens_check_weights,
            np.ones(100).tolist() + [0] * 10,
            one_hot_encode(4, 0),
            -2
        )

        # C.1: Oxygen with double bond to sulphur (and no hydrogens)
        substructure_c_1_detector_weights = (
            one_hot_encode(100, 8) + no_hydrogens_check_weights,
            one_hot_encode(110, 16),
            one_hot_encode(4, 1),
            0
        )

        # D.1: Carbon with 3 single bond neighbours, exactly one hydrogen
        substructure_d_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + [0] * 10,
            one_hot_encode(4, 0),
            -2
        )

        params_zipped = zip(
            substructure_a_1_detector_weights,
            substructure_b_1_detector_weights,
            substructure_c_1_detector_weights,
            substructure_d_1_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn1 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn1 = x_i_transform.shape[1]

        # Note: Any A.1 that is not B.1 will have its activation = 1. A.1 that is also B.1 may have bigger activations -- this would mess up counting.
        # B.1 however should not match with our pattern.
        # A.2: A.1 that is NOT B.1 (!) connected with double bond to at least 2xC.1 
        substructure_a_2_detector_weights = (
            [1, -10, 0, 0],
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(4, 1),
            -1
        )

        # B.2: D.1 connected to A.1
        substructure_b_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_2_detector_weights,
            substructure_b_2_detector_weights
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn2 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn2 = x_i_transform.shape[1]

        # A.3: A.2 connected to B.2
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

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

        dump_activations = kwargs.get("dump_activations", False)

        if not dump_activations:
            return self.readout(x3)
        else:
            return self.readout(x3), [x1, x2, x3]
