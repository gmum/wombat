import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeFDetector(torch.nn.Module):
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

        # A.1: Carbon with 1 carbon neighbour (single bond), exactly 3 hydrogens.
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_three_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            0
        )

        # B.1: Carbon with 2 single bonds to carbons, exactly 2 hydrogens
        substructure_b_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_two_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -1
        )

        # C.1: Carbon with a double bond to oxygen, no hydrogens
        substructure_c_1_detector_weights = (
            one_hot_encode(100, 6) + no_hydrogens_check_weights,
            one_hot_encode(110, 8),
            one_hot_encode(4, 1),
            0
        )

        # D.1: Nitrogen with 1 single bond to other nitrogen, 1 or 2 hydrogens
        one_or_two_hydrogens_check_weights = [-10] * 10
        one_or_two_hydrogens_check_weights[1] = 0
        one_or_two_hydrogens_check_weights[2] = 0

        substructure_d_1_detector_weights = (
            one_hot_encode(100, 7) + one_or_two_hydrogens_check_weights,
            one_hot_encode(110, 7),
            one_hot_encode(4, 0),
            0
        )

        # E.1: Carbon with 3 single bonds to anything, exactly 1 hydrogen
        substructure_e_1_detector_weights = (
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
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn1 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn1 = x_i_transform.shape[1]

        # A.2: B.1 with A.1, single bond
        substructure_a_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 0),
            0
        )

        # B.2: B.1 with B.1 AND NOT A.1, single bond
        enforce_b1_without_a1_weights = one_hot_encode(dim_out_mpnn1, 1)
        enforce_b1_without_a1_weights[0] = -10

        substructure_b_2_detector_weights = (
            enforce_b1_without_a1_weights,
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(4, 0),
            0
        )

        # C.2: C.1 with D.1
        substructure_c_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(4, 0),
            0
        )

        # D.2: D.1 with D.1
        substructure_d_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(4, 0),
            0
        )

        # E.2: C.1 with E.1
        substructure_e_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(dim_out_mpnn1, 4),
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

        # (We will repeat the same trick with A.3 and B.3 as we did with A.2 and B.2)
        # (we will do it for A LOT of layers)
        # A.3: B.2 w/ A.2
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(4, 0),
            0
        )

        # B.3: B.2 w/ B.2, AND NOT A.2
        enforce_b2_without_a2_weights = one_hot_encode(dim_out_mpnn2, 1)
        enforce_b2_without_a2_weights[0] = -10

        substructure_b_3_detector_weights = (
            enforce_b2_without_a2_weights,
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(4, 0),
            0
        )

        # C.3: C.2 w/ D.2
        substructure_c_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 2),
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(4, 0),
            0
        )

        # D.3: D.2 w/ D.2
        substructure_d_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(4, 0),
            0
        )

        # E.3: D.2 w/ E.2
        substructure_e_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(dim_out_mpnn2, 4),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
            substructure_b_3_detector_weights,
            substructure_c_3_detector_weights,
            substructure_d_3_detector_weights,
            substructure_e_3_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn3 = x_i_transform.shape[1]

        # A.4: B.3 w/ A.3
        substructure_a_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(dim_out_mpnn3, 0),
            one_hot_encode(4, 0),
            0
        )

        # B.4: B.3 w/ B.3, AND NOT A.3
        enforce_b3_without_a3_weights = one_hot_encode(dim_out_mpnn3, 1)
        enforce_b3_without_a3_weights[0] = -10

        substructure_b_4_detector_weights = (
            enforce_b3_without_a3_weights,
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(4, 0),
            0
        )

        # C.4: C.3 w/ D.3
        substructure_c_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 2),
            one_hot_encode(dim_out_mpnn3, 3),
            one_hot_encode(4, 0),
            0
        )

        # D.4: D.3 w/ E.3
        substructure_d_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 3),
            one_hot_encode(dim_out_mpnn3, 4),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_4_detector_weights,
            substructure_b_4_detector_weights,
            substructure_c_4_detector_weights,
            substructure_d_4_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn4 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn4 = x_i_transform.shape[1]

        # A.5: B.4 w/ A.4
        substructure_a_5_detector_weights = (
            one_hot_encode(dim_out_mpnn4, 1),
            one_hot_encode(dim_out_mpnn4, 0),
            one_hot_encode(4, 0),
            0
        )

        # B.5: B.4 w/ B.4, AND NOT A.4
        enforce_b4_without_a4_weights = one_hot_encode(dim_out_mpnn4, 1)
        enforce_b4_without_a4_weights[0] = -10

        substructure_b_5_detector_weights = (
            enforce_b4_without_a4_weights,
            one_hot_encode(dim_out_mpnn4, 1),
            one_hot_encode(4, 0),
            0
        )

        # C.5: C.4 w/ D.4
        substructure_c_5_detector_weights = (
            one_hot_encode(dim_out_mpnn4, 2),
            one_hot_encode(dim_out_mpnn4, 3),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_5_detector_weights,
            substructure_b_5_detector_weights,
            substructure_c_5_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn5 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn5 = x_i_transform.shape[1]

        # A.6: B.5 w/ A.5
        substructure_a_6_detector_weights = (
            one_hot_encode(dim_out_mpnn5, 1),
            one_hot_encode(dim_out_mpnn5, 0),
            one_hot_encode(4, 0),
            0
        )

        # Finally we've collapsed the whole pattern!
        # B.6: B.5 w/ C.5. We could add check to make sure A.5 is not present, but it would be redundant in this case.
        substructure_b_6_detector_weights = (
            one_hot_encode(dim_out_mpnn5, 1),
            one_hot_encode(dim_out_mpnn5, 2),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_6_detector_weights,
            substructure_b_6_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn6 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn6 = x_i_transform.shape[1]

        # A.7: A.6 w/ B.6. Note that if A.6 and B.6 are in the same node this will NOT activate (as we want).
        # This is why we didn't have to add "No A.6" checks to B.6
        substructure_a_7_detector_weights = (
            one_hot_encode(dim_out_mpnn6, 0),
            one_hot_encode(dim_out_mpnn6, 1),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_7_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn7 = MPNN(x_i_transform, x_j_transform, e_transform, bias)

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
        x7 = self.mpnn7(x6, edge_features, edge_index)

        dump_activations = kwargs.get("dump_activations", False)

        if not dump_activations:
            return self.readout(x7)
        else:
            return self.readout(x7), [x1, x2, x3, x4, x5, x6, x7]
