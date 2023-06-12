import torch


class Ours(torch.nn.Linear):
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
        if x.ndim == 2:
            x = torch.hstack((torch.Tensor(x), torch.ones((len(x), 1)))) if self.const else x
            return (torch.sign(torch.matmul(x, self.w)) + 1) / 2
        else:
            x = torch.hstack((torch.Tensor(x), torch.ones(1))) if self.const else x
            return (torch.sign(torch.matmul(x, self.w)) + 1) / 2


class CNN(torch.nn.Module):
    def __init__(self, xi: float = 0.0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.xi = xi

    def forward(self, x):
        x = self.pool(torch.nn.functionalF.relu(self.conv1(x) - 2 * self.xi))
        x = self.pool(torch.nn.functional.F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.F.relu(self.fc1(x))
        x = torch.nn.functional.F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
