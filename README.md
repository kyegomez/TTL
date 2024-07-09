[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

## TTL
Pytorch Implementation of the paper: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"




## Install
```bash
$ pip install ttl-torch

```

## Usage
```python

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

```


# License
MIT


## Citation
```bibtex
@misc{sun2024learninglearntesttime,
    title={Learning to (Learn at Test Time): RNNs with Expressive Hidden States}, 
    author={Yu Sun and Xinhao Li and Karan Dalal and Jiarui Xu and Arjun Vikram and Genghan Zhang and Yann Dubois and Xinlei Chen and Xiaolong Wang and Sanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin},
    year={2024},
    eprint={2407.04620},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2407.04620}, 
}

```