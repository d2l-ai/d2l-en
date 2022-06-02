# Swin Transformer
:label:`sec_swin-transformer`

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=1}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab mxnet
X = d2l.zeros((128, 49, 96))
window_mask = d2l.ones((64, 49, 49))  # (64 = (224/4/7)**2)
attention = d2l.MultiHeadAttention(num_hiddens=96, num_heads=3, dropout=0.1)
attention.initialize()
attention(X, X, X, None, window_mask).shape
```

```{.python .input}
%%tab pytorch
X = d2l.zeros((128, 49, 96))
window_mask = d2l.ones((64, 49, 49))  # (64 = (224/4/7)**2)

attention = d2l.MultiHeadAttention(num_hiddens=96, num_heads=3, dropout=0.1)
attention(X, X, X, None, window_mask).shape
```

```{.python .input}
%%tab tensorflow
X = d2l.zeros((128, 49, 96))
window_mask = d2l.ones((64, 49, 49))  # (64 = (224/4/7)**2)

attention = d2l.MultiHeadAttention(96, 96, 96, 96, num_heads=3, dropout=0.1)
attention(X, X, X, None, window_mask).shape
```
