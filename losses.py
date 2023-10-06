import torch
import torch.nn as nn
import torch.nn.functional as F

class MATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        p_mask = labels + th_label
        n_mask = 1 - labels
        row_idx = []
        for i in range(len(labels.sum(1))):
            if labels.sum(1)[i] > 0:
                row_idx.append(i)
        loss_pos_all = torch.tensor(0.0).to(labels)
        count = 0.0
        for i in row_idx:
            for j in range(labels.shape[1]):
                if j == 0:
                    loss_pos_all += torch.log(1 - torch.sigmoid(logits[i, j]))
                    count += 1
                elif labels[i, j] == 1:
                    loss_pos_all += torch.log(torch.sigmoid(logits[i, j]))
                    count += 1
        loss1 = -loss_pos_all / count
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1, threshold=0.85):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit + torch.abs(th_logit)*threshold)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output