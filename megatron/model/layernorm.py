import torch

class SPLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, sequence_parallel=False):
        super(SPLayerNorm, self).__init__(normalized_shape, eps)
        self.sequence_parallel = sequence_parallel
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def forward(self, x):
        return super(SPLayerNorm, self).forward(x)
