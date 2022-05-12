# Vision Transformers
:label:`sec_vision-transformer`

```{.python .input  n=7}
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=8}
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
patch_emb = PatchEmbedding(96, 16, 512)
X = d2l.randn(4, 32, 96, 96)
d2l.check_shape(patch_emb(X), (4, 36, 512))  # 36 = (96/16)^2
```

```{.python .input}
class VisionTransformer(d2l.Classifier):
    """Vision transformer."""
    def __init__(self, vocab_size, num_hiddens, norm_shape,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000):
        super().__init__()
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(num_hiddens, \
                 norm_shape, ffn_num_hiddens, num_heads, dropout, True))
        # In Vision transformer, positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, X):
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, None)
        return X
```

## Training

```{.python .input}
model = VisionTransformer(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
#trainer.fit(model, data)
```
