from cploss import CPLoss, build_similarity_from_llm
import torch

classes = ["road", "sidewalk", "car"]
prompt = f"Semantic similarity for penalties in urban segmentation. Classes: {classes}"

def provider(prompt, classes):
    return [
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]

S = build_similarity_from_llm(prompt, classes, provider_callback=provider)

criterion = CPLoss(
    S=S,
    w_cpl=0.7,     # CPL weight
    w_ce=0.2,      # Cross-entropy weight
    w_dice=0.1,    # Dice weight
    ignore_index=255,
    reduction='mean',
    from_logits=True,
)

# forward
N, C, H, W = 2, len(classes), 256, 256
logits = torch.randn(N, C, H, W, device='cuda')
target = torch.randint(0, C, (N, H, W), device='cuda')
loss = criterion(logits, target)
loss.backward()
