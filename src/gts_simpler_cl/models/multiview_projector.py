from torch import nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, hidden_dim, subspace_dim, len_subspace):  # 768, 128, 3
        super(Projector, self).__init__()
        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim
        self.len_subspace = len_subspace
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.subspace_dim * len_subspace),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.projection(x)  #x:(N,H=576)  out:(N,H=384)
        out = out.reshape(x.size(0), self.len_subspace, self.subspace_dim)
        out = F.normalize(out, dim=-1, p=2)
        return out  # [N, 3, 128]