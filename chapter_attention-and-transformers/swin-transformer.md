# Swin Transformer
:label:`sec_swin-transformer`

```{.python .input  n=1}
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
X = torch.randn(128, 49, 96)
window_mask = torch.ones(64, 49, 49)  # (64 = (224/4/7)**2)

attention = d2l.MultiHeadAttention(num_hiddens=96, num_heads=3, dropout=0.1)
attention(X, X, X, None, window_mask).shape
```
