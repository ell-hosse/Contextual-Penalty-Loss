# cploss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _one_hot(target: torch.Tensor, num_classes: int):
    # target: (N,H,W) -> (N,H,W,C) one-hot (float32)
    return F.one_hot(target.clamp(min=0), num_classes=num_classes).to(torch.float32)

def _soft_dice_multiclass(
    probs: torch.Tensor,          # (N,C,H,W), already softmaxed
    target: torch.Tensor, # (N,H,W) class ids
    ignore_index: int or None,
    eps: float = 1e-6,
    class_weights: torch.Tensor | None = None,  # (C,) optional
    reduction: str = "mean",
):
    N, C, H, W = probs.shape
    device = probs.device
    # mask out ignored pixels
    if ignore_index is not None:
        valid_mask = (target != ignore_index).unsqueeze(1)  # (N,1,H,W)
        probs = probs * valid_mask
        tgt = target.clone()
        tgt[target == ignore_index] = 0  # dummy index for one_hot
    else:
        valid_mask = None
        tgt = target

    tgt_oh = _one_hot(tgt, C).permute(0, 3, 1, 2)  # (N,C,H,W)
    if valid_mask is not None:
        tgt_oh = tgt_oh * valid_mask

    # per-class soft dice
    intersection = (probs * tgt_oh).sum(dim=(0, 2, 3))  # (C,)
    sums = probs.sum(dim=(0, 2, 3)) + tgt_oh.sum(dim=(0, 2, 3))  # (C,)
    dice_c = (2.0 * intersection + eps) / (sums + eps)  # (C,)
    dice_loss_c = 1.0 - dice_c  # (C,)

    if class_weights is not None:
        # normalize weights to sum to C so scale is stable
        w = class_weights.to(device).clamp(min=0)
        w = w * (C / (w.sum() + eps))
        dice_loss_c = dice_loss_c * w

    if reduction == "mean":
        return dice_loss_c.mean()
    elif reduction == "sum":
        return dice_loss_c.sum()
    else:
        return dice_loss_c  # (C,)

class CPLoss(nn.Module):
    """
    Contextual Penalty Loss (CPL) with optional CE and Dice blending.

    Total loss = w_cpl * CPL + w_ce * CE + w_dice * Dice

    Args:
        S: (C,C) similarity matrix in [0,1], diag=1
        w_cpl, w_ce, w_dice: nonnegative weights
        dice_class_weights: optional (C,) tensor to rebalance classes in Dice
        ignore_index: index to ignore
        reduction: 'mean' | 'sum' | 'none'
        from_logits: if True, apply softmax; else 'logits' are probs
        eps: numerical stability
    """
    def __init__(self, S: torch.Tensor,
                 w_cpl: float = 1.0,
                 w_ce: float = 0.0,
                 w_dice: float = 0.0,
                 dice_class_weights: torch.Tensor | None = None,
                 ignore_index: int = 255,
                 reduction: str = 'mean',
                 from_logits: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        assert S.dim() == 2 and S.size(0) == S.size(1), "S must be (C,C)"
        self.register_buffer("S", S.clamp(0, 1))
        C = S.size(0)
        eye = torch.eye(C, device=S.device, dtype=S.dtype)
        self.S = torch.maximum(self.S, eye)
        self.D = (1.0 - self.S)

        self.w_cpl = float(w_cpl)
        self.w_ce = float(w_ce)
        self.w_dice = float(w_dice)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.from_logits = from_logits
        self.eps = eps

        if dice_class_weights is not None:
            self.register_buffer("dice_class_weights", dice_class_weights.to(S.dtype))
        else:
            self.dice_class_weights = None

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        N, C, H, W = logits.shape
        device = logits.device

        if self.from_logits:
            probs = F.softmax(logits.float(), dim=1)
        else:
            probs = (logits.float() + self.eps) / (logits.float().sum(dim=1, keepdim=True) + self.eps)

        # ---- CPL (vectorized) ----
        target_flat = target.reshape(-1).to(device)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)

        if self.ignore_index is not None:
            valid = (target_flat != self.ignore_index)
        else:
            valid = torch.ones_like(target_flat, dtype=torch.bool)

        if valid.any():
            t_idx = target_flat[valid]                       # (M,)
            D_y = self.D.to(device)[t_idx]                  # (M,C)
            p_valid = probs_flat[valid]                     # (M,C)
            cpl_per_pix = (p_valid * D_y).sum(dim=1)        # (M,)
            if self.reduction == 'mean':
                cpl_loss = cpl_per_pix.mean()
            elif self.reduction == 'sum':
                cpl_loss = cpl_per_pix.sum()
            else:
                full = torch.zeros_like(target_flat, dtype=probs.dtype, device=device)
                full[valid] = cpl_per_pix
                cpl_loss = full.reshape(target.shape)
        else:
            cpl_loss = torch.tensor(0.0, device=device, dtype=probs.dtype)

        # ---- CE (optional) ----
        if self.w_ce > 0:
            ce_loss = F.cross_entropy(
                logits.float(), target.long(),
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )
        else:
            ce_loss = 0.0

        # ---- Dice (optional) ----
        if self.w_dice > 0:
            dice_loss = _soft_dice_multiclass(
                probs=probs,
                target=target,
                ignore_index=self.ignore_index,
                eps=self.eps,
                class_weights=self.dice_class_weights,
                reduction=self.reduction
            )
        else:
            dice_loss = 0.0

        # ---- Weighted blend ----
        total = self.w_cpl * cpl_loss + self.w_ce * ce_loss + self.w_dice * dice_loss
        return total
