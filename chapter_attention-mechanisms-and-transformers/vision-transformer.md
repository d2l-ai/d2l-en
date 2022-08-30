```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch'])
```

# Transformers for Vision
:label:`sec_vision-transformer`

The transformer architecture was initially proposed for sequence to sequence learning, such as for machine translation. 
With high effectiveness,
transformers subsequently became the model of choice in various natural language processing tasks :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`. 
However, 
in the field of computer vision
the dominant architecture
had been based on
CNNs (:numref:`chap_modern_cnn`).
*Can we adapt transformers
to model image data*?
This question has sparked immense interest
in the computer vision community.
:cite:`ramachandran2019stand` proposed to replace convolution with self-attention. 
However, its use of specialized patterns in attention makes it hard to scale up models on hardware accelerators.
:cite:`cordonnier2020relationship` theoretically proved that self-attention can learn to behave similarly to convolution. Empirically, $2 \times 2$ patches were taken from images as inputs, but the small patch size makes the model only applicable to image data with low resolutions.

Without specific constraints on patch size,
*vision transformers* (ViTs)
extract patches from images
and feed them into a transformer encoder
to obtain a global representation,
which will finally be transformed for classification :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Notably, transformers show better scalability than CNNs:
when training larger models on larger datasets,
vision transformers outperform ResNets by a significant margin. Similar to the landscape of network architecture design in natural language processing,
transformers also became a game-changer in computer vision.


## Model

:numref:`fig_vit` depicts
the model architecture of vision transformers.
This architecture consists of a stem
that patchifies images, 
a body based on the multi-layer transformer encoder,
and a head that transforms the global representation
into the output label.

![The vision transformer architecture. In this example, an image is split into 9 patches. A special “&lt;cls&gt;” token and the 9 flattened image patches are transformed via patch embedding and $n$ transformer encoder blocks into 10 representations, respectively. The “&lt;cls&gt;” representation is further transformed into the output label.](../img/vit.svg)
:label:`fig_vit`

Consider an input image with height $h$, width $w$,
and $c$ channels.
Specifying the patch height and width both as $p$,
the image is split into a sequence of $m = hw/p^2$ patches,
where each patch is flattened to a vector of length $cp^2$.
In this way, image patches can be treated similarly to tokens in text sequences by transformer encoders.
A special “&lt;cls&gt;” (class) token and
the $m$ flattened image patches are linearly projected
into a sequence of $m+1$ vectors,
summed with learnable positional embeddings.
The multi-layer transformer encoder
transforms $m+1$ input vectors
into the same amount of output vector representations of the same length.
It works exactly the same way as the original transformer encoder in :numref:`fig_transformer`,
only differing in the position of normalization.
Since the “&lt;cls&gt;” token attends to all the image patches via self-attention (see :numref:`fig_cnn-rnn-self-attention`),
its representation from the transformer encoder output
will be further transformed into the output label.

```{.python .input  n=1}
from d2l import torch as d2l
import torch
from torch import nn
```

## Patch Embedding

To implement a vision transformer, let's start with patch embedding in :numref:`fig_vit`. Splitting an image into patches and linearly projecting these flattened patches can be simplified as a single convolution operation, where both the kernel size and the stride size are set to the patch size.

```{.python .input  n=2}
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
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

In the following example, taking images with height and width of `img_size` as inputs, the patch embedding outputs `(img_size//patch_size)**2` patches that are linearly projected to vectors of length `num_hiddens`.

```{.python .input  n=9}
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.randn(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Vision Transformer Encoder
:label:`subsec_vit-encoder`

The MLP of the vision transformer encoder
is slightly different from the position-wise FFN of the original transformer encoder (see :numref:`subsec_positionwise-ffn`). First, here the activation function uses the Gaussian error linear unit (GELU), which can be considered as a smoother version of the ReLU :cite:`hendrycks2016gaussian`.
Second, dropout is applied to the output of each fully connected layer in the MLP for regularization.

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

The vision transformer encoder block implementation
just follows the pre-normalization design in :numref:`fig_vit`,
where normalization is applied right *before* multi-head attention or the MLP.
In contrast to post-normalization ("add & norm" in :numref:`fig_transformer`), where normalization is placed right *after* residual connections,
pre-normalization leads to more effective or efficient training for transformers :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.

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
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

Same as in :numref:`subsec_transformer-encoder`,
any vision transformer encoder block does not change its input shape.

```{.python .input}
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

## Putting It All Together

The forward pass of vision transformers below is straightforward.
First, input images are fed into an `PatchEmbedding` instance,
whose output is concatenated with the “&lt;cls&gt;”  token embedding. They are summed with learnable positional embeddings before dropout.
Then the output is fed into the transformer encoder that stacks `num_blks` instances of the `ViTBlock` class.
Finally, the representation of the “&lt;cls&gt;”  token is projected by the network head.

```{.python .input}
class ViT(d2l.Classifier):
    """Vision transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

## Training

Training a vision transformer on the Fashion-MNIST dataset is just like how CNNs were trained in :numref:`chap_modern_cnn`.

```{.python .input}
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Summary and Discussion

You may notice that for small datasets like Fashion-MNIST, our implemented vision transformer does not outperform the ResNet in :numref:`sec_resnet`.
Similar observations can be made even on the ImageNet dataset (1.2 million images).
This is because transformers *lack* those useful principles in convolution, such as translation invariance and locality (:numref:`sec_why-conv`).
However, the picture changes when training larger models on larger datasets (e.g., 300 million images),
where vision transformers outperform ResNets by a large margin in image classification, demonstrating
intrinsic superiority of transformers in scalability :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
The introduction of vision transformers
has changed the landscape of network design for modeling image data.
They were soon shown effective on the ImageNet dataset with data-efficient training strategies of DeiT :cite:`touvron2021training`.
However,
quadratic complexity
of self-attention (:numref:`sec_self-attention-and-positional-encoding`)
makes the transformer architecture
less suitable for higher-resolution images.
Towards a general-purpose backbone network in computer vision,
Swin transformers addressed the quadratic computational complexity with respect to image size (:numref:`subsec_cnn-rnn-self-attention`)
and added back convolution-like priors,
extending the applicability of transformers to a range of computer vision tasks beyond image classification with state-of-the-art results :cite:`liu2021swin`.







## Exercises

1. How does the value of `img_size` affect training time?
1. Instead of projecting the “&lt;cls&gt;” token representation to the output, how to project the averaged patch representations? Implement this change and see how it affects the accuracy.
1. Can you modify hyperparameters to improve the accuracy of the vision transformer?


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8943)
:end_tab:
