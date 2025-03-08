from torch import nn


class ContraCLMSeqLoss(nn.Module):
    def __init__(self, pad_token_id, temperature=0.05):
        super(ContraCLMSeqLoss, self).__init__()
        self.pad_token_id = pad_token_id
        self.temperature = temperature
        print(f"Sequence-Level Contrastive Loss:\t temperature: {temperature}")

    def forward(self, last_hidden_states_1, last_hidden_states_2, token_mask):
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

        Ng = neg.sum(dim=-1)
        loss = (- torch.log(pos / (Ng + pos))).mean()

        return loss