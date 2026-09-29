import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeGDetector(torch.nn.Module):
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

        # A.1: Carbon with >=1 carbon neighbour and exactly 2 hydrogens. Said carbon neighbour must have exactly 3 hydrogens, i.e. be CH3.
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_two_hydrogens_check_weights,
            one_hot_encode(100, 6) + exactly_three_hydrogens_check_weights,
            one_hot_encode(4, 0),
            0
        )

        # B.1: Nitrogen with >= 3 carbon neighbours
        substructure_b_1_detector_weights = (
            one_hot_encode(110, 7),
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -2
        )

        # C.1: Because we need to check 2 same branches for nitrogen, we will need to count -- and because of that, we need to ensure 0-1 activations for B.1 and A.1 activations. Unfortunately, if we get a N+ with 4 carbon neighbours, B.1 will activate with value of 2. As such, we need a separate neuron for N+ with 4 carbon neighbours.

        substructure_c_1_detector_weights = (
            one_hot_encode(110, 7),
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -3
        )

        # D.1: Carbon with >= 1 carbon neighbour, exactly 2 hydrogens.
        substructure_d_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_two_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            0
        )

        # E.1: Oxygen with >= 1 carbon neighbour, no hydrogens.
        substructure_e_1_detector_weights = (
            one_hot_encode(100, 8) + no_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            0
        )

        # F.1: Carbon with no hydrogen, double bond to oxygen atom.
        substructure_f_1_detector_weights = (
            one_hot_encode(100, 6) + no_hydrogens_check_weights,
            one_hot_encode(110, 8),
            one_hot_encode(4, 1),
            0
        )

        # G.1: Carbon with >= 3 neighbours and exactly 1 hydrogen ( => must have exactly 3 neighbours)
        substructure_g_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            one_hot_encode(4, 0),
            -2
        )

        params_zipped = zip(
            substructure_a_1_detector_weights,
            substructure_b_1_detector_weights,
            substructure_c_1_detector_weights,
            substructure_d_1_detector_weights,
            substructure_e_1_detector_weights,
            substructure_f_1_detector_weights,
            substructure_g_1_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn1 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn1 = x_i_transform.shape[1]

        # A.2: B.1 with >=2 A.1s AND IS NOT C.1. If it is C.1, the reading will be incorrect, as B.1 will have a value of 2
        # and during multiplication with 1 A.1 we would get activated as we would expect from 2 A.2s.
        allow_b_1_and_disallow_c1 = [0] * dim_out_mpnn1
        allow_b_1_and_disallow_c1[1] = 1  # allow B.1
        allow_b_1_and_disallow_c1[2] = -10  # disallow C.1

        substructure_a_2_detector_weights = (
            allow_b_1_and_disallow_c1,
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 0),
            -1
        )

        # B.2: C.1 with >= 2 A.1s. Note that C.1 can only be within range of 0 to 1 (we work under assumption there won't be pentavalent N++, which I hope is correct)
        substructure_b_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 0),
            -1
        )

        # C.2: D.1 w/ D.1
        substructure_c_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(4, 0),
            0
        )

        # D.2: E.1 w/ D.1
        substructure_d_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 4),
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(4, 0),
            0
        )

        # E.2: F.1 w/ G.1
        substructure_e_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(dim_out_mpnn1, 6),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_2_detector_weights,
            substructure_b_2_detector_weights,
            substructure_c_2_detector_weights,
            substructure_d_2_detector_weights,
            substructure_e_2_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn2 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn2 = x_i_transform.shape[1]

        # A.3: C.2 with A.2 OR B.2. Note that A.2 and B.2 activations are mutually exclusive. We basically have 2 "paths" for N+ case and N case.
        allow_a_2_and_b_2 = [0] * dim_out_mpnn2
        allow_a_2_and_b_2[0] = 1  # allow A.2
        allow_a_2_and_b_2[1] = 1  # allow B.2

        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 2),
            allow_a_2_and_b_2,
            one_hot_encode(4, 0),
            0
        )

        # B.3: C.2 with D.2
        substructure_b_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 2),
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(4, 0),
            0
        )

        # C.3: D.2 with E.2
        substructure_c_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(dim_out_mpnn2, 4),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
            substructure_b_3_detector_weights,
            substructure_c_3_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn3 = x_i_transform.shape[1]

        # A.4: B.3 connected to A.3
        substructure_a_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(dim_out_mpnn3, 0),
            one_hot_encode(4, 0),
            0
        )

        # B.4: B.3 connected to C.3
        substructure_b_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(dim_out_mpnn3, 2),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_4_detector_weights,
            substructure_b_4_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn4 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

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

        dump_activations = kwargs.get("dump_activations", False)

        if not dump_activations:
            return self.readout(x4)
        else:
            return self.readout(x4), [x1, x2, x3, x4]
