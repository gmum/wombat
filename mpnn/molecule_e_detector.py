import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeEDetector(torch.nn.Module):
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

        # A.1: Carbon with 4 neighbours and no hydrogens. Value in this neuron cannot be greater than 1, obviously.
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 6) + no_hydrogens_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            one_hot_encode(4, 0),
            -3
        )

        # B.1: Carbon with 2 carbon neighbours and exactly 2 hydrogens
        substructure_b_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_two_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -1
        )

        # C.1: Carbon with 3 neighbours, exactly 1 hydrogen
        substructure_c_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            one_hot_encode(4, 0),
            -2
        )

        # D.1: Carbon with 1 carbon neighbour, 3 hydrogens
        substructure_d_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_three_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            0
        )

        # E.1: Oxygen with a double bond to carbon (and no hydrogens)
        substructure_e_1_detector_weights = (
            one_hot_encode(100, 8) + no_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 1),
            0
        )

        # F.1: Carbon with >= 2 carbon neighbours, no hydrogens. Note that some A.1s may be F.1s
        substructure_f_1_detector_weights = (
            one_hot_encode(100, 6) + no_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -1
        )

        params_zipped = zip(
            substructure_a_1_detector_weights,
            substructure_b_1_detector_weights,
            substructure_c_1_detector_weights,
            substructure_d_1_detector_weights,
            substructure_e_1_detector_weights,
            substructure_f_1_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn1 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn1 = x_i_transform.shape[1]

        # A.2: A.1 that is connected to 2x D.1; this works because D.1 activations are 0-1 only, so we can put bias to -1.
        substructure_a_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(4, 0),
            -1
        )

        # B.2: B.1 that is connected to 2x A.1; this works because, again, A.1 activations are 0-1 only.
        substructure_b_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 0),
            -1
        )

        # C.2: F.1 connected to C.1
        substructure_c_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(4, 0),
            0
        )

        # D.2: B.1 connected to F.1
        substructure_d_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(4, 0),
            0
        )

        # E.2: E.1 connected to F.1
        substructure_e_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 4),
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(4, 1),
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

        # A.3: A.2 connected to B.2
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(4, 0),
            0
        )

        # B.3: C.2 double bond to E.2
        substructure_b_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 2),
            one_hot_encode(dim_out_mpnn2, 4),
            one_hot_encode(4, 1),
            0
        )

        # C.3: D.2 to A.2
        substructure_c_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(dim_out_mpnn2, 0),
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

        # Now we need to make sure that we have 2 nodes that are both (A.3, B.3)-activated at the same time.

        # A.4: C.3 connected to B.3
        substructure_a_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 2),
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(4, 0),
            0
        )

        # B.4: C.3 connected to A.3
        substructure_b_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 2),
            one_hot_encode(dim_out_mpnn3, 0),
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
