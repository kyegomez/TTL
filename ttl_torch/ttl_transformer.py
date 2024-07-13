import torch
from torch import nn, Tensor
from ttl_torch.ttl_linear import TTLinear
from zeta import SwiGLU, RMSNorm


class TTLMLP(nn.Module):
    """
    TTLMLP is a multi-layer perceptron (MLP) module that applies Tensor Train (TT) linear layers
    with activation and normalization to the input tensor.

    Args:
        dim (int): The input dimension of the tensor.
        expanse_factor (int): The expansion factor for the TT linear layers.

    Attributes:
        dim (int): The input dimension of the tensor.
        expanse_factor (int): The expansion factor for the TT linear layers.
        fc1 (TTLinear): The first TT linear layer.
        fc2 (TTLinear): The second TT linear layer.
        norm (RMSNorm): The RMS normalization layer.
        act (SwiGLU): The activation layer.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs the forward pass of the TTLMLP module.

    """

    def __init__(
        self,
        dim: int,
        expanse_factor: int,
    ):
        super(TTLMLP, self).__init__()
        self.dim = dim
        self.expanse_factor = expanse_factor
        self.fc1 = TTLinear(dim, expanse_factor * dim)
        self.fc2 = TTLinear(expanse_factor * dim, dim)
        self.norm = RMSNorm(dim)
        self.act = SwiGLU(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x


class TTLTransformerBlock(nn.Module):
    """
    A single block of the TTL Transformer.

    Args:
        input_dim (int): The input dimension of the block.
        output_dim (int): The output dimension of the block.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super(TTLTransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ttl_linear = TTLinear(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.SiLU()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Forward pass of the TTLTransformerBlock.

        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.
            v (Tensor): The value tensor.

        Returns:
            Tensor: The output tensor after applying the TTLTransformerBlock.
        """
        out = self.ttl_linear(in_seq=[q, k, v])

        # Concatenate the output of the TTL layer
        # TODO: Check if this is the correct way to concatenate the output
        out = torch.cat(out, dim=0)

        print(f"out: {out.shape}")

        # Norm
        normed = self.norm(out)

        return self.act(normed)


# # Model
# model = TTLTransformerBlock(input_dim=512, output_dim=512)

# # Input
# q = torch.randn(512, 512)

# k = torch.randn(512, 512)

# v = torch.randn(512, 512)

# # Forward pass
# output = model(q, k, v)
# print(f"Output shape: {output.shape}")
