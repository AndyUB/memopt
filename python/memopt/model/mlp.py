import torch

from memopt.model.transformer import Linear


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        self.fc2 = Linear(hidden_dim, output_dim, device=device, dtype=dtype)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
