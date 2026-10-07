# This was vibe-coded using GPT-5.6 Sol with "very hard" thinking setting to assess if LLMs are capable of generating
# whitebox weights themselves. Model received code for other whiteboxes, instruction about relaxed SMARTS, was told to ignore mol d (faulty)
# Furthermore it received work-in-progress version of our manuscript.

# The only change to this file on our side is changing the class name and adding this comment to properly reflect authorship.
# As such, it may contain errors. The model did not validate experimentally the whitebox on its own.

import numpy as np
import torch

from wombat.mpnn import MPNN, AllNonZeroReadout, deal_with_pyg, one_hot_encode


class VibeCodedDetector(torch.nn.Module):
    """Detect the relaxed SMARTS pattern ``[CHD2]=NN=[CHD2]``.

    Under WOMBAT's atom features and valence assumptions, a ``[CHD2]`` atom
    can be recognized as a carbon with exactly one hydrogen which:

    1. is double-bonded to nitrogen; and
    2. has at least one single-bonded non-hydrogen neighbour.

    Those conditions already consume carbon's full valence, so they force
    heavy-atom degree two. The network uses the second condition as a canary
    rather than treating ``C(H1)=N`` as sufficient.
    """

    def __init__(self):
        super().__init__()

        exactly_one_hydrogen_check_weights = [-10] * 10
        exactly_one_hydrogen_check_weights[1] = 0

        # A.1: Nitrogen double-bonded to a carbon with exactly one hydrogen.
        substructure_a_1_detector_weights = (
            one_hot_encode(110, 7),
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            one_hot_encode(4, 1),
            0,
        )

        # B.1: Carbon with exactly one hydrogen and a single bond to some
        # non-hydrogen atom. Together with A.1, this is the D2 canary.
        substructure_b_1_detector_weights = (
            one_hot_encode(100, 6) + exactly_one_hydrogen_check_weights,
            np.ones(100).tolist() + np.zeros(10).tolist(),
            one_hot_encode(4, 0),
            0,
        )

        params_zipped = zip(
            substructure_a_1_detector_weights,
            substructure_b_1_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn1 = MPNN(
            x_i_transform,
            x_j_transform,
            e_transform,
            bias,
        )
        dim_out_mpnn1 = x_i_transform.shape[1]

        # A.2: A.1 nitrogen double-bonded to the same B.1 carbon.
        # This recognizes one complete [CHD2]=N endpoint.
        substructure_a_2_detector_weights = (
            one_hot_encode(dim_out_mpnn1, 0),
            one_hot_encode(dim_out_mpnn1, 1),
            one_hot_encode(4, 1),
            0,
        )

        params_zipped = zip(
            substructure_a_2_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn2 = MPNN(
            x_i_transform,
            x_j_transform,
            e_transform,
            bias,
        )
        dim_out_mpnn2 = x_i_transform.shape[1]

        # A.3: Two complete A.2 endpoints joined by a single N-N bond.
        substructure_a_3_detector_weights = (
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(dim_out_mpnn2, 0),
            one_hot_encode(4, 0),
            0,
        )

        params_zipped = zip(
            substructure_a_3_detector_weights,
        )
        x_i_transform, x_j_transform, e_transform, bias = list(params_zipped)

        x_i_transform = torch.tensor(x_i_transform).T
        x_j_transform = torch.tensor(x_j_transform).T
        e_transform = torch.tensor(e_transform).T
        bias = torch.tensor(bias)

        self.mpnn3 = MPNN(
            x_i_transform,
            x_j_transform,
            e_transform,
            bias,
        )

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
        return self.readout(x3), [x1, x2, x3]
