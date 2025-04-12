import torch.nn as nn
class SimpleCameraPoseEncoder(nn.Module):
    def __init__(self, c_in, c_out, hidden_dim=128):
        super(SimpleCameraPoseEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(c_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_out)
        )
    def forward(self, x):
        return self.model(x)

