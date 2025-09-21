from typing import List, Optional, Dict
import torch
import hashlib
import json
import os

_DEFAULT_CACHE = os.path.expanduser("~/.cache/cploss")

def _hash_key(classes: List[str], prompt: str) -> str:
    key = json.dumps({"classes": classes, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()

def _ensure_cache():
    os.makedirs(_DEFAULT_CACHE, exist_ok=True)

def save_matrix(S: torch.Tensor, classes: List[str], prompt: str):
    _ensure_cache()
    fname = os.path.join(_DEFAULT_CACHE, f"{_hash_key(classes, prompt)}.pt")
    torch.save({"S": S.cpu(), "classes": classes, "prompt": prompt}, fname)
    return fname

def load_cached_matrix(classes: List[str], prompt: str) -> Optional[torch.Tensor]:
    fname = os.path.join(_DEFAULT_CACHE, f"{_hash_key(classes, prompt)}.pt")
    if os.path.isfile(fname):
        payload = torch.load(fname, map_location="cpu")
        if payload.get("classes") == classes:
            return payload.get("S")
    return None

def normalize_similarity(S: torch.Tensor) -> torch.Tensor:
    S = S.clamp(0, 1)
    # enforce symmetry + diag=1
    S = 0.5 * (S + S.t())
    C = S.size(0)
    S[torch.arange(C), torch.arange(C)] = 1.0
    return S

def build_similarity_from_llm(prompt: str, classes: List[str],
                              provider_callback=None,
                              temperature: float = 0.0) -> torch.Tensor:
    """
    provider_callback: a function(prompt, classes)-> List[List[float]]
      Implement this in your project to actually call an LLM (OpenAI, etc.)
      Return a CxC matrix with entries in [0,1].
    If provider_callback is None, we fallback to a heuristic (coarse).
    """

    cached = load_cached_matrix(classes, prompt)
    if cached is not None:
        return cached

    C = len(classes)
    if provider_callback is None:
        # Heuristic fallback: higher similarity for lexically close names
        import itertools

        S = torch.zeros(C, C, dtype=torch.float32)
        for i, j in itertools.product(range(C), range(C)):
            if i == j:
                S[i, j] = 1.0
            else:
                a, b = classes[i].lower(), classes[j].lower()
                # simple token overlap as a weak prior
                overlap = len(set(a.split()) & set(b.split()))
                S[i, j] = 0.2 + 0.2 * overlap  # 0.2..0.6
        S = normalize_similarity(S)
    else:
        matrix = provider_callback(prompt, classes)  # user-supplied LLM call
        S = torch.tensor(matrix, dtype=torch.float32)
        S = normalize_similarity(S)

    save_matrix(S, classes, prompt)
    return S
