"""CPU-only reproduction of no-grad autocast cache contamination."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


class OffsetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
        return (y * (1 + self.weight)).to(x.dtype)


def run(mode):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), OffsetNorm())
    x = torch.tensor([[1., 2., 3., 4.]])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        if mode == 'historical_shared_cache':
            with torch.no_grad():
                model(x)
        elif mode == 'fixed_isolated_cache':
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
                model(x)
        output = model(x)
        loss = output[0, 0].float()
    loss.backward()
    return {name: {'has_grad': p.grad is not None,
                  'grad_norm': None if p.grad is None else float(p.grad.float().norm())}
            for name, p in model.named_parameters()}


if __name__ == '__main__':
    root = Path('/opt/experiment/src/jspace_plasticity')
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in ['intervention.py', 'readout.py', 'synthetic_recovery_sft.py', 'modeling.py']
              if (root/name).exists()}
    print(json.dumps({'torch': torch.__version__, 'device': 'cpu',
        'cases': {mode: run(mode) for mode in ['no_clean_forward', 'historical_shared_cache', 'fixed_isolated_cache']},
        'image_source_sha256': hashes}, indent=2))
