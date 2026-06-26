import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device

        features = F.normalize(features, dim=1)

        batch_size = features.shape[0]

        similarity = torch.matmul(
            features,
            features.T
        ) / self.temperature

        labels = labels.contiguous().view(-1,1)

        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        mask = mask * logits_mask

        exp_sim = torch.exp(similarity) * logits_mask

        log_prob = similarity - torch.log(
            exp_sim.sum(1, keepdim=True)
        )

        mean_log_prob_pos = (
            mask * log_prob
        ).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()

        return loss