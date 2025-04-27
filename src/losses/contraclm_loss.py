"""
The Contrastive loss used in ContraCLM model used in and adapted from:
Nihal Jain, Dejiao Zhang, Wasi Uddin Ahmad, Zijian Wang, Feng Nan, Xiaopeng Li, Ming Tan,
Ramesh Nallapati, Baishakhi Ray, Parminder Bhatia, Xiaofei Ma, and Bing Xiang. 2023.
ContraCLM: Contrastive Learning For Causal Language Model.
In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp. 6436–6459.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ContraCLMSeqLoss(nn.Module):
    """
    Sequence-level Contrastive Loss for Language Models (CLM).

    Encourages similar sequences (positive pairs) to have similar representations,
    while pushing apart dissimilar sequences (negative pairs).

    Args:
        pad_token_id (int): ID of the padding token used in the input sequences.
        temperature (float, optional): Temperature scaling factor for the contrastive loss. Default is 0.05.

    Prints:
        Temperature value when initialized.
    """
    def __init__(self, pad_token_id, temperature=0.05):
        super(ContraCLMSeqLoss, self).__init__()
        self.pad_token_id = pad_token_id
        self.temperature = temperature
        print(f"Sequence-Level Contrastive Loss:\t temperature: {temperature}")

    def forward(self, last_hidden_states_1, last_hidden_states_2, token_mask):
        """
        Compute the contrastive loss between two sets of sequence representations.

        Args:
            last_hidden_states_1 (Tensor): Hidden states from encoder 1,
            shape (batch_size, seq_len, hidden_dim).
            last_hidden_states_2 (Tensor): Hidden states from encoder 2,
            shape (batch_size, seq_len, hidden_dim).
            token_mask (Tensor): Mask indicating non-pad tokens,
            shape (batch_size, seq_len), 1 for real tokens, 0 for padding.

        Returns:
            Tensor: A scalar loss value.

        Workflow:
            - Mean-pool the hidden states over valid (non-pad) tokens
            to get sequence-level embeddings.
            - Normalize embeddings.
            - Compute contrastive positive and negative scores.
            - Calculate contrastive loss encouraging positive pairs
            to have higher similarity than negatives.
        """
        device = last_hidden_states_1.device  # [N, L, H]
        batch_size = last_hidden_states_1.size(0)

        # get the sequence representation via mean pooling
        token_mask = token_mask.unsqueeze(-1)  # [N, L, 1]
        features_1 = torch.sum(last_hidden_states_1 * token_mask, dim=1) / torch.sum(token_mask, dim=1)  # [N, H]
        features_2 = torch.sum(last_hidden_states_2 * token_mask, dim=1) / torch.sum(token_mask, dim=1)  # [N, H]
        features_1, features_2 = F.normalize(features_1, dim=1), F.normalize(features_2, dim=1)  # [N, H]
        features = torch.cat([features_1, features_2], dim=0)  # [2N, H]

        # create block diagonal mask to avoid contrast within the neighborhood of each example
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1).to(torch.float32) / self.temperature)  # [N]
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()).to(torch.float32) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        negative_sum = neg.sum(dim=-1)
        loss = (- torch.log(pos / (negative_sum + pos))).mean()

        return loss
