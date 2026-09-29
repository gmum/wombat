import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeCDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Carbon bonded to >= 3 Chlorine atoms
        substructure_a_1_detector_weights = (one_hot_encode(110, 6), one_hot_encode(110, 17), one_hot_encode(4, 0), -2)
        # Carbon bonded to nitrogen 
        substructure_b_1_detector_weights = (one_hot_encode(110, 6), one_hot_encode(110, 7), one_hot_encode(4, 0), 0)
        # Nitrogen bonded to >= 2 carbons 
        substructure_c_1_detector_weights = (one_hot_encode(110, 7), one_hot_encode(110, 6), one_hot_encode(4, 0), -1)
        # Carbon with double bond to oxygen
        substructure_d_1_detector_weights = (one_hot_encode(110, 6), one_hot_encode(110, 8), one_hot_encode(4, 1), 0)
        # Carbon with >= 3 non-H neighbours (unknown type) with single bonds
        substructure_e_1_detector_weights = (
            one_hot_encode(110, 6), np.ones(100).tolist() + np.zeros(10).tolist(), np.ones(4).tolist(), -2
        )
        # Oxygen with double bond to carbon
        substructure_f_1_detector_weights = (one_hot_encode(110, 8), one_hot_encode(110, 6), one_hot_encode(4, 1), 0)
        # Atom with EXACTLY 1 hydrogen neighbour
        substructure_g_1_detector_weights = (one_hot_encode(110, 101), np.ones(110).tolist(), one_hot_encode(4, 0), 0)

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

        # E.1 bonded to A.1: Carbon with >= 3 neighbours bonded to a carbon with >= 3 chlorine neighbours
        substructure_a_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 4), one_hot_encode(dim_out_mpnn1, 0), one_hot_encode(4, 0), 0
        )
        # C.1 bonded to D.1: Nitrogen with at least 2 carbon neighbours, one of which has double bond to oxygen.
        substructure_b_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 2), one_hot_encode(dim_out_mpnn1, 3), one_hot_encode(4, 0), 0
        )
        # E.1 bonded to F.1: Carbon with >=3 neighbours bonded to an oxygen with a double bond to carbon
        substructure_c_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 4), one_hot_encode(dim_out_mpnn1, 5), one_hot_encode(4, 1), 0
        )
        # A.1 bonded to G.1: Carbon with >= 3 chlorine neighbours bonded to an atom with exactly 1 hydrogen neighbour
        substructure_d_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 0), one_hot_encode(dim_out_mpnn1, 6), one_hot_encode(4, 0), 0
        )
        # G.1 bonded to B.1: Atom with exactly 1 hydrogen neighbour, bonded to a carbon bonded to nitrogen
        substructure_e_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 6), one_hot_encode(dim_out_mpnn1, 1), one_hot_encode(4, 0), 0
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

        # B.2 bonded to A.2
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 1), one_hot_encode(dim_out_mpnn2, 0), one_hot_encode(4, 0), 0)

        # C.2 bonded to B.2
        substructure_b_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 2), one_hot_encode(dim_out_mpnn2, 1), one_hot_encode(4, 0), 0)

        # A.2 bonded to D.2: Carbon with >= 3 neighbours bonded to a carbon with >= 3 chlorine neighbours, which is bonded to an atom with exactly 1 hydrogen neighbour
        # This in turn means that A.2 has a connection to hydrogen itself. (we had to do this, because we can't do AND for features in the same node)
        # (I came to regret this choice, but that's the way it is)
        substructure_c_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 0), one_hot_encode(dim_out_mpnn2, 3), one_hot_encode(4, 0), 0
        )

        # E.2 bonded to A.2: this is to keep info about the nitrogen with hydrogen neighbour alive.
        substructure_d_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 4), one_hot_encode(dim_out_mpnn2, 0), one_hot_encode(4, 0), 0
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
            substructure_b_3_detector_weights,
            substructure_c_3_detector_weights,
            substructure_d_3_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn3 = x_i_transform.shape[1]

        # A.3 bonded to B.3
        substructure_a_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 0), one_hot_encode(dim_out_mpnn3, 1), one_hot_encode(4, 0), 0
        )
        # A.3 bonded to C.3
        substructure_b_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 0), one_hot_encode(dim_out_mpnn3, 2), one_hot_encode(4, 0), 0
        )

        # D.3 bonded to B.3 (should flare up in A.3 if it's the nitrogen we are looking for)
        substructure_c_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 3), one_hot_encode(dim_out_mpnn3, 1), one_hot_encode(4, 0), 0
        )

        params_zipped = zip(
            substructure_a_4_detector_weights,
            substructure_b_4_detector_weights,
            substructure_c_4_detector_weights,
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
