import torch

def to_device(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.to(device=like.device, dtype=like.dtype)

def check_shapes(logits, target, S):
    C = logits.size(1)
    assert S.size(0) == S.size(1) == C, f"S must be (C,C) but got {S.shape} for C={C}"
