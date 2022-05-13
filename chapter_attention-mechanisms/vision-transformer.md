# Vision Transformers
:label:`sec_vision-transformer`

```{.python .input  n=1}
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=2}
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512,
                 norm_layer=None):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input  n=9}
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.randn(batch_size, 32, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, 
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = self.ln1(X)
        return X + self.mlp(self.ln2(
            X + self.attention(X, X, X, valid_lens)))
```

```{.python .input}
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, [100, 24], 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
class ViT(d2l.Classifier):
    """Vision transformer."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False):
        super().__init__()
        self.patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
        
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", ViTBlock(num_hiddens,
                 norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias))
        # In vision transformer, positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, num_hiddens))

    def forward(self, X):
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X)
        return X
```

## Training

```{.python .input}
#model = VisionTransformer(lr=0.05)
#trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
#data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
#trainer.fit(model, data)
```
