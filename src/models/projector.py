"""
The projector module used on the contrastive loss in three different views to adaptively adjust
the contribution of different view on the model learning to achieve the optimal performance
"""

from torch import nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    A projection module that maps input features into multiple normalized subspaces.

    Args:
        hidden_dim (int): Dimension of the input features.
        subspace_dim (int): Dimension of each projected subspace.
        len_subspace (int): Number of subspaces to project into.

    The input is first linearly projected into (subspace_dim × len_subspace) dimensions,
    then reshaped into [batch_size, len_subspace, subspace_dim]
    and L2-normalized along the last dimension.
    """

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
        """
        Forward pass of the projector.

        Args:
            x (Tensor): Input tensor of shape (batch_size, hidden_dim).

        Returns:
            Tensor: Projected and normalized tensor of shape
            (batch_size, len_subspace, subspace_dim).
        """
        out = self.projection(x)  #x:(N,H=576)  out:(N,H=384)
        out = out.reshape(x.size(0), self.len_subspace, self.subspace_dim)
        out = F.normalize(out, dim=-1, p=2)
        return out  # [N, 3, 128]
