import contextlib

import torch


def weak_sigreg_loss(x, sketch_dim=64, eps=1e-6):
    """Weak-SIGReg loss on a matrix of representations [N, C]."""
    if x.dim() != 2:
        raise ValueError(f'weak_sigreg_loss expects [N, C], got {tuple(x.shape)}')

    n, c = x.shape
    if n < 2 or c < 1:
        return x.sum() * 0.0

    x = x.float()
    k = min(int(sketch_dim), c)
    if k <= 0:
        raise ValueError(f'sketch_dim must be positive, got {sketch_dim}')

    if c > k:
        sketch = torch.randn(k, c, device=x.device, dtype=x.dtype) / (c ** 0.5)
        x = x @ sketch.t()

    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / (n - 1 + eps)
    target = torch.eye(k, device=x.device, dtype=x.dtype)
    return torch.norm(cov - target, p='fro')


class SigRegHookCollector:
    """Collects ViT block activations and computes averaged Weak-SIGReg."""

    def __init__(self, model, sketch_dim=64, representation='mean_tokens',
                 layer_ids=None):
        self.model = model.module if hasattr(model, 'module') else model
        self.sketch_dim = int(sketch_dim)
        self.representation = representation
        self.activations = []
        self.handles = []
        self.enabled = False

        blocks = getattr(self.model, 'blocks', None)
        if blocks is None:
            raise ValueError('SigRegHookCollector requires a model with .blocks')

        if layer_ids is None:
            selected = range(len(blocks))
        else:
            selected = [int(i) for i in layer_ids]

        for idx in selected:
            self.handles.append(blocks[idx].register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return
        if isinstance(output, tuple):
            output = output[0]
        if not torch.is_tensor(output):
            return
        self.activations.append(output)

    def clear(self):
        self.activations.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    @contextlib.contextmanager
    def capture(self):
        self.clear()
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def _to_matrix(self, x):
        if x.dim() == 3:
            if self.representation == 'mean_tokens':
                return x.mean(dim=1)
            if self.representation == 'tokens':
                return x.reshape(-1, x.shape[-1])
            raise ValueError(
                f'Invalid sigreg representation={self.representation!r}. '
                "Expected 'mean_tokens' or 'tokens'.")
        if x.dim() == 2:
            return x
        raise ValueError(f'Unsupported activation shape for SIGReg: {tuple(x.shape)}')

    def loss(self):
        if not self.activations:
            return None
        losses = [
            weak_sigreg_loss(self._to_matrix(x), sketch_dim=self.sketch_dim)
            for x in self.activations
        ]
        return torch.stack(losses).mean()
