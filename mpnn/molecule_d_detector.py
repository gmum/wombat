import torch
import numpy as np

from mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class MoleculeDDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        no_hydrogens_check_weights = [-10] * 10
        no_hydrogens_check_weights[0] = 0

        exactly_one_hydrogen_check_weights = [-10] * 10
        exactly_one_hydrogen_check_weights[1] = 0

        exactly_two_hydrogens_check_weights = [-10] * 10
        exactly_two_hydrogens_check_weights[2] = 0

        # A.1: Carbon with >= 3 neighbours, EXACTLY 1 hydrogen (this implies EXACTLY 3 neighbours that are not hydrogen)
        substructure_a_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            one_hot_encode(4, 0),
            -2
        )

        # B.1: Carbon with = 2 carbon neighbours and exactly 2 hydrogens 
        substructure_b_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_two_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -1
        )

        # C.1: Carbon with = 3 carbon neighbours, exactly 1 hydrogen (note that C.1 => A.1)
        substructure_c_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 0),
            -2
        )

        # D.1: Carbon with >= 3 neighbours (don't care about bond arity) and NO hydrogens.
        substructure_d_1_detector_weights = (
            one_hot_encode(100, 6) + no_hydrogens_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            np.ones(4).tolist(),
            -2
        )

        # E.1: Oxygen with a double bond to carbon (and no hydrogens, no surprises there)
        substructure_e_1_detector_weights = (
            one_hot_encode(100, 8) + no_hydrogens_check_weights,
            one_hot_encode(110, 6),
            one_hot_encode(4, 1),
            0
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

        # A.2: B.1 that is connected to (exactly) 2 A.1s. Note that this normally would be very hard to do, BUT
        # thanks to A.1 definition (exactly 1 hydrogen, exactly 3 neighbours) we know it is either 0 or 1.
        # Since carbon cannot be pentavalent there cannot be A.1 that is signaled by any other positive integer than 1 => we can use bias to count
        substructure_a_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(4, 0),
            -1
        )

        # B.2: C.1 that is a neighbour of B.1
        substructure_b_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 2),
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(4, 0),
            0
        )

        # C.2: D.1 that is a neighbour of E.1 (double bond) (i.e. carbon with >= 3 neighbours, no hydrogens, that is bonded to an oxygen with a double bond to carbon)
        # Because we assume that oxygen cannot  form >2 bonds, this means that C.2 is basically a carbon with 2 outgoing connections and 1 double bond to oxygen
        substructure_c_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 3),
            one_hot_encode(dim_out_mpnn1, 4),
            one_hot_encode(4, 1),
            0
        )

        params_zipped = zip(
            substructure_a_2_detector_weights,
            substructure_b_2_detector_weights,
            substructure_c_2_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn2 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn2 = x_i_transform.shape[1]

        # A.3: B.2 that is connected to A.2 with a single bond
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(4, 0),
            0
        )

        # B.3: B.2 that is connected to C.2 with a single bond
        substructure_b_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 1),
            one_hot_encode(dim_out_mpnn2, 2),
            one_hot_encode(4, 0),
            0
        )

        # Note that atoms with A.2-activations cannot be C.2-activated (and vice-versa) since (for example) C.2 requires no hydrogens and A.2 requires 2
        # Furthermore, B.2-activation also mutually exclusive with the two (only one hydrogen can be present)

        # The implication is that if a node is A.3 or B.3-activated, only one of its neighbours can be A.3 or B.3 activated. And this is the neighbour we are looking for.

        params_zipped = zip(
            substructure_a_3_detector_weights,
            substructure_b_3_detector_weights,
        )

        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(x_i_transform, x_j_transform, e_transform, bias)
        dim_out_mpnn3 = x_i_transform.shape[1]

        # Now we need to make sure that we have 2 nodes that are both (A.3, B.3)-activated at the same time.

        # A.4: A.3 that is connected to B.3 with a single bond
        substructure_a_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 0),
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(4, 0),
            0
        )

        # B.4: B.3 that is connected to A.3 with a single bond
        substructure_b_4_detector_weights = (
            one_hot_encode(dim_out_mpnn3, 1),
            one_hot_encode(dim_out_mpnn3, 0),
            one_hot_encode(4, 0),
            0
        )

        # If we have 2 adjacent nodes with A.3 and B.3 activations, then we've found the pattern and we will have activations in both A.4 and B.4.
        # If only 1 such neighbour was present, it won't have right neighbour for activation -> signal will die down.
        # Hence, after this layer we can do readout.

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
