import torch
from ttl_torch.ttl_linear import TTLinear


input_dim, output_dim = 10, 10  # Dimensions for the linear model
ttt_layer = TTLinear(input_dim, output_dim)

# Generate some example data
example_data = [
    torch.randn(1, input_dim, output_dim) for _ in range(5)
]

# Forward pass through the TTT layer
output_data = ttt_layer(example_data)

for i, output in enumerate(output_data):
    print(f"Output at step {i}: {output}")
