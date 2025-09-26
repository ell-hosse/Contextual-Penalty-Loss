import torch
import torch.nn as nn


class FastCPLoss(nn.Module):
    """
    Class-wise CPL:
      Build S[y,c] = sum_{i: y_i=y} p_i(c)  (accumulate by GT class),
      then loss = sum_{y,c} S[y,c] * D[y,c] / M_valid
    """
    def __init__(self, S: torch.Tensor, ignore_index: int = 255, reduction='mean', from_logits=True, eps=1e-8):
        super().__init__()
        assert S.dim() == 2 and S.size(0) == S.size(1)
        S = S.clamp(0,1)
        eye = torch.eye(S.size(0), dtype=S.dtype, device=S.device)
        self.register_buffer("D", (1.0 - torch.maximum(S, eye)))
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if self.from_logits:
            probs = F.softmax(logits.float(), dim=1)
        else:
            probs = (logits.float() + self.eps) / (logits.float().sum(dim=1, keepdim=True) + self.eps)

        N, C, H, W = probs.shape
        probs_flat  = probs.permute(0,2,3,1).reshape(-1, C)  # (M,C)
        target_flat = target.reshape(-1).to(logits.device)
        valid = (target_flat != self.ignore_index)

        if not valid.any():
            return logits.sum()*0.0

        t_idx = target_flat[valid] # (M,)
        p_val = probs_flat[valid] # (M,C)

        # Accumulate into a (C,C) table: rows=true class, cols=pred class prob mass
        Cc = C
        sum_by_true = torch.zeros(Cc, Cc, device=logits.device, dtype=probs.dtype)
        sum_by_true.index_add_(0, t_idx, p_val) # add rows of p_val into row t_idx

        total = (sum_by_true * self.D.to(logits.device)).sum()
        if self.reduction == 'mean':
            return total / float(p_val.size(0))
        elif self.reduction == 'sum':
            return total
        else:
            # No natural per-pixel map; return mean-equivalent
            return total / float(p_val.size(0))