import torch


class BatchDimensionRemover(torch.nn.Module):
    """
    Stuff like Captum loves batch dimension. We don't. Hence, this.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args):
        args = [arg.squeeze(0) for arg in args]
        output = self.model(*args)
        return output.unsqueeze(0)
