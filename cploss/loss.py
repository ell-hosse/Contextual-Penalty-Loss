import torch
import torch.nn as nn
import torch.nn.functional as F

class CPLoss(nn.Module):
    """
    Contextual Penalty Loss (CPL)
    - Inputs:
        logits: (N, C, H, W)
        target: (N, H, W) with class ids in [0, C-1], or ignore_index
    - Params:
        S: (C, C) similarity matrix in [0,1], diag=1
        alpha: mix with CE (0 = only CPL, 1 = only CE)  -> loss = (1-alpha)*CPL + alpha*CE
        ignore_index: class id to ignore
        reduction: 'mean' | 'sum' | 'none'
        from_logits: if False, 'logits' is already probabilities
        eps: small clamp for stability
    """
    def __init__(self, S: torch.Tensor, alpha: float = 0.0,
                 ignore_index: int = 255, reduction: str = 'mean',
                 from_logits: bool = True, eps: float = 1e-8):
        super().__init__()
        assert S.dim() == 2 and S.size(0) == S.size(1), "S must be (C,C)"
        self.register_buffer("S", S.clamp(0, 1))
        C = S.size(0)
        eye = torch.eye(C, device=S.device, dtype=S.dtype)
        self.S = torch.maximum(self.S, eye) # ensure diag >= 1, typically exactly 1
        self.D = (1.0 - self.S) # distance matrix
        self.alpha = float(alpha)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        N, C, H, W = logits.shape
        device = logits.device

        if self.from_logits:
            # stable softmax
            probs = F.softmax(logits.float(), dim=1)
        else:
            probs = (logits.float() + self.eps) / (logits.float().sum(dim=1, keepdim=True) + self.eps)

        # flatten spatial dims for vectorized compute
        probs = probs.permute(0, 2, 3, 1).reshape(-1, C) # (N*H*W, C)

        target_flat = target.reshape(-1).to(device)

        # mask ignored pixels
        if self.ignore_index is not None:
            valid = (target_flat != self.ignore_index)
        else:
            valid = torch.ones_like(target_flat, dtype=torch.bool)

        if valid.any():
            t_idx = target_flat[valid] # (M,)
            # Select the row of D for each pixel’s true class:
            # D_y: (M, C)
            D_y = self.D.to(device)[t_idx]

            # CPL term (fully vectorized): sum_c p_c * D(true,c)
            p_valid = probs[valid] # (M, C)
            cpl_per_pix = (p_valid * D_y).sum(dim=1) # (M,)

            if self.reduction == 'mean':
                cpl_loss = cpl_per_pix.mean()
            elif self.reduction == 'sum':
                cpl_loss = cpl_per_pix.sum()
            else:
                # return per-pixel map with ignored set to 0
                full = torch.zeros_like(target_flat, dtype=probs.dtype, device=device)
                full[valid] = cpl_per_pix
                cpl_loss = full.reshape(target.shape)
        else:
            cpl_loss = torch.tensor(0.0, device=device, dtype=probs.dtype)

        #  CE mix-in (alpha)
        if self.alpha > 0:
            ce = F.cross_entropy(
                logits.float(), target.long(),
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )
            return (1.0 - self.alpha) * cpl_loss + self.alpha * ce
        else:
            return cpl_loss
