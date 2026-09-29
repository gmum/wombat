import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeHDetector(torch.nn.Module):
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

        # A.1: Oxygen with double bond to a carbon; no hydrogens in oxygen.
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 8) + no_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 1),
            0
        )

        # B.1: Oxygen with exactly 1 hydrogen, 1-bonded to carbon.
        substructure_b_1_detector_weights = (
            one_hot_encode(100, 8) + exactly_one_hydrogen_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            0
        )

        # C.1: Carbon atom with 3 neighbours and 1 hydrogen.
        substructure_c_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + [0] * 10,
            one_hot_encode(4, 0),
            -2
        )

        # D.1: Carbon atom with 2 carbon neighbours and 2 hydrogens.
        substructure_d_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_two_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -1
        )

        # E.1: Carbon with >=3 neighbours and 0 hydrogens.
        substructure_e_1_detector_weights = (
            one_hot_encode(100, 6) + no_hydrogens_check_weights,
            np.ones(100).tolist() + [0] * 10,
            np.ones(4).tolist(),
            -2
        )

        # F.1: Nitrogen, double bond to carbon, 1 or 0 hydrogens allowed
        substructure_f_1_detector_weights = (
            one_hot_encode(100, 7) + [0, 0] + [-10] * 8,
            one_hot_encode(110, 6),
            one_hot_encode(4, 1),
            0
        )

        # G.1: Nitrogen, bond to another nitrogen, 1 or 2 hydrogens allowed
        substructure_g_1_detector_weights = (
            one_hot_encode(100, 7) + [-10, 0, 0] + [-10] * 7,
            one_hot_encode(110, 7),
            one_hot_encode(4, 0),
            0
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

        # A.2: E.1 w/ A.1
        substructure_a_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 4),
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 1),
            0
        )

        # B.2: B.1 w/ E.1
        substructure_b_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(dim_out_mpnn1, 4),
            one_hot_encode(4, 0),
            0
        )

        # C.2: C.1 w/ D.1
        substructure_c_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(4, 0),
            0
        )

        # D.2: D.1 w/ C.1
        substructure_d_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(4, 0),
            0
        )

        # E.2: E.1 w/ F.1
        substructure_e_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 4),
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(4, 1),
            0
        )

        # F.2: F.1 w/ G.1
        substructure_f_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(dim_out_mpnn1, 6),
            one_hot_encode(4, 0),
            0
        )

        # G.2: G.1 w/ F.1
        substructure_g_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 6),
            one_hot_encode(dim_out_mpnn1, 5),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_2_detector_weights,
            substructure_b_2_detector_weights,
            substructure_c_2_detector_weights,
            substructure_d_2_detector_weights,
            substructure_e_2_detector_weights,
            substructure_f_2_detector_weights,
            substructure_g_2_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn2 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn2 = x_i_transform.shape[1]

        # A.3: A.2 w/ B.2
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(4, 0),
            0
        )

        # B.3: C.2 w/ A.2
        substructure_b_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 2),
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(4, 0),
            0
        )

        # C.3: D.2 w/ C.2
        substructure_c_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(dim_out_mpnn2, 2),
            one_hot_encode(4, 0),
            0
        )

        # D.3: E.2 w/ D.2
        substructure_d_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 4),
            one_hot_encode(dim_out_mpnn2, 3),
            one_hot_encode(4, 0),
            0
        )

        # E.3: F.2 w/ E.2
        substructure_e_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 5),
            one_hot_encode(dim_out_mpnn2, 4),
            one_hot_encode(4, 1),
            0
        )

        # F.3: G.2 w/ A.2
        substructure_f_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 6),
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(4, 0),
            0
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
            substructure_b_3_detector_weights,
            substructure_c_3_detector_weights,
            substructure_d_3_detector_weights,
            substructure_e_3_detector_weights,
            substructure_f_3_detector_weights,
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

        # B.4: C.3 w/ B.3
        substructure_b_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 2),
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(4, 0),
            0
        )

        # C.4: D.3 w/ C.3
        substructure_c_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 3),
            one_hot_encode(dim_out_mpnn3, 2),
            one_hot_encode(4, 0),
            0
        )

        # D.4: E.3 w/ F.3
        substructure_d_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 4),
            one_hot_encode(dim_out_mpnn3, 5),
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

        # B.5: C.4 w/ D.4
        substructure_b_5_detector_weights = (
            one_hot_encode(dim_out_mpnn4, 2),
            one_hot_encode(dim_out_mpnn4, 3),
            one_hot_encode(4, 1),
            0
        )

        params_zipped = zip(
            substructure_a_5_detector_weights,
            substructure_b_5_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn5 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn5 = x_i_transform.shape[1]

        # A.6: A.5 w/ B.5
        substructure_a_6_detector_weights = (
            one_hot_encode(dim_out_mpnn5, 0),
            one_hot_encode(dim_out_mpnn5, 1),
            one_hot_encode(4, 0),
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
