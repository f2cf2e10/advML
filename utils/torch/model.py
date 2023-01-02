import torch


class Ours(torch.nn.Module):
    __constants__ = ['const']
    const: bool
    w: torch.Tensor

    def __init__(self, w: torch.Tensor, const: bool):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.w = torch.nn.Parameter(w)
        self.const = const

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = torch.hstack((torch.Tensor(x), torch.ones((len(x), 1)))) if self.const else x
        return torch.sign(torch.matmul(x, self.w))
