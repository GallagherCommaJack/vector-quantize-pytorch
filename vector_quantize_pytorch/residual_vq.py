import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
        quantized_out = x + (quantized_out - x).detach()

        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, all_indices, all_losses
