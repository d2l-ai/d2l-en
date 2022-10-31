```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Keys, Values and Queries

:label:`sec_attention-basics`

So far all the networks we reviewed crucially relied on the input being of a well-defined size. For instance, the images in ImageNet are of size $224 \times 224$ pixels and CNNs are specifically tuned to this size. Even in natural language processing the input size for RNNs is well-defined and fixed. Variable size is addressed by sequentially processing one token at a time, or by specially designed convolution kernels :cite:`Kalchbrenner.Grefenstette.Blunsom.2014`. This approach can lead to significant problems when the input is truly of varying size with varying information content, such as in :ref:`sec_seq2seq` to transform text :cite:`Sutskever.Vinyals.Le.2014`. In particular, for long sequences it becomes quite difficult to keep track of everything that's already been generated or even viewed by the network. Even explicit tracking heuristics such as :cite:`yang2016neural` only offer limited benefit. 

Compare this to databases. In their simplest form they are collections of keys (k) and values (v). For instance, our database DB might consist of tuples \{(Zhang, Aston), (Lipton, Zachary), (Li, Mu), (Smola, Alex), (Hu, Rachel), (Werness, Brent)\} with the last name being the key and the first name being the value. We can operate on DB, for instance with the exact query (q) for 'Li' which would return the value 'Mu'. In case (Li, Mu) was not a record in the database, there would be no valid answer. If we also allowed for approximate matches, we would retrieve (Lipton, Zachary) instead. This quite simple and trivial example nonetheless teaches us a number of useful things:

* We can design queries q that operate on (k,v) pairs in such a manner as to be valid regardless the size of the database. 
* The same query can receive different answers, according to the contents of the database. 
* The 'code' being executed to operate on a large state space (the database) can be quite simple (e.g., exact match, approximate match, top-k). 
* There is no need to compress or otherwise simplify the database to make the operations effective. 

Clearly we wouldn't have introduced a simple database here if it wasn't for the purpose of explaining deep learning. Indeed, this leads to one of the most exciting concepts arguably introduced in deep learning in the past decade --- the attention mechanism :cite:`Bahdanau.Cho.Bengio.2014`. We will cover the specifics of its application to machine translation later. For now, simply consider the following: denote by $\mathrm{DB} := \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$ a database of $m$ tuples of keys and values. Moreover, denote by $\mathbf{q}$ a query. Then we can define the attention over $\mathrm{DB}$ as

$$\mathop{\mathrm{Attn}}(\mathrm{DB}, \mathbf{q}) := \sum_{(\mathbf{k}, \mathbf{v}) \in \mathrm{DB}} \alpha(\mathbf{q}, \mathbf{k}) \mathbf{v}$$

Here $\alpha(\mathbf{q}, \mathbf{k}) \in \mathbb{R}$ are scalar 'attention' weights. The operation itself is typically referred to as 'attention pooling'. The name attention derives from the fact that the operation pays particular attention to the terms for which the weight $\alpha$ is significant (i.e., large). As such, the attention over $\mathrm{DB}$ generates a linear combination of values contained in the database. In fact, this contains the above example as a special case where all but one weight is zero. We have a number of special cases:

* The weights $\alpha(\mathbf{q}, \mathbf{k})$ are nonnegative. In this case the answer of the Attention mechanism is contained in the convex cone spanned by the values $\mathbf{v}_i$. 
* The weights $\alpha(\mathbf{q}, \mathbf{k})$ form a convex combination, i.e., $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$ and $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$ for all $i$. This is the most common setting in deep learning. 
* Exactly one of the weights $\alpha(\mathbf{q}, \mathbf{k}_i)$ is $1$, all others are $0$. This is akin to a traditional database query. 
* All weights are equal, i.e.\ $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$ for all $i$. This amounts to averaging across the entire database, also called average pooling in deep learning. 

A common strategy to ensure that the weights sum up to $1$ is to normalize them via 

$$\alpha(\mathbf{q}, \mathbf{k}_i) \leftarrow \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}$$
:eqlabel:`eq_normalized_attention`

In particular, to ensure that the weights are also nonnegative, one can resort to exponentiation. This means that we can now pick any function  $a(\mathbf{q}, \mathbf{k})$ and then applying the softmax operation used for multinomial models to it via

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i)}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j)}. $$
:eqlabel:`eq_softmax_attention`

This operation is readily available in all deep learning frameworks, it is differentiable and its gradient never vanishes, all of which are desirable properties in a model. Note though, the attention mechanism introduced above is not the only option. For instance, we can design a non-differentiable attention model that can be trained using reinforcement learning methods :cite:`Mnih.Heess.Graves.ea.2014`. As one would expect, training such a model is quite complex. Consequently the bulk of modern attention research 
follows the framework outlined in :numref:`fig_qkv`. We thus focus our exposition on this family of differentiable mechanisms. 

![The attention mechanisms computes a linear combination over values via attention pooling,
where weights are derived according to the compatibility of a query and keys.](../img/qkv.svg)
:label:`fig_qkv`

What is quite remarkable is that the actual 'code' to execute on the set of keys and values, namely the query, can be quite concise, even though the space to operate on is significant. This is a desirable property for a network layer as it doesn't require too many parameters to learn. Just as convenient is the fact that attention can operate on arbitrarily large databases without the need to change the way the attention pooling operation is performed. 

## Visualization

One of the benefits of the attention mechanism is that it can be quite intuitive, particularly when the weights are nonnegative and sum to $1$. In this case we might *interpret* large weights as a way for the model to select components of relevance. While this is a good intuition, it is important to remember that it is just that, an *intuition*. Regardless, we may want to visualize its effect on the given set of keys, when applying a variety of different queries. This function will come in handy later.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=2}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

We thus define the `show_heatmaps` function. Note, though, that it doesn't take a matrix (of attention weights) as its input but rather a 4D tensor, allowing for an array of different queries and weights. Consequently the input `matrices` has the shape (number of rows for display, number of columns for display, number of queries, number of keys). This will come in handy later on when we want to visualize the workings of :ref:`sec_multihead-attention` which is used to design Transformers.

```{.python .input  n=17}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    (num_rows, num_cols, _, _) = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel(ylabel)
            if titles: ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

As a quick sanity check let's visualize the identity matrix, representing a case where the attention is only $1$ 
where the attention weight is one only when query and key are the same.

```{.python .input  n=20}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

## Summary

The attention mechanism allows us to aggregate data from many (key, value) pairs. So far our discussion was 
quite abstract, simply describing a way to pool data. We haven't explained yet where those mysterious keys, values and queries might arise from. Some intuition might help here: for instance, in a regression setting, the query might correspond to the location where the regression should be carried out. The keys are the locations where past data was observed and the values are the (regression) values themselves. This is the so-called Watson-Nadarya estimator :cite:`Watson.1964,Nadaraya.1964` that we'll be studying in the next section. Another example is the integration of 

By design, the attention mechanisms provides a *differentiable* means of control 
by which a neural network can select elements from a set and to construct an associated weighted sum over representations. 

## Exercises

1. Suppose that you wanted to re-implement approximate (key,query) matches as used in classical databases, which attention function $a(\mathbf{q}, \mathbf{k})$ would you pick? 
1. Suppose that the attention function is given by $a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$ and that $\mathbf{k} = \mathbf{v}$. Denote by $p(\mathbf{k}; \mathbf{q})$ the probabilitiy distribution over keys when using the softmax normalization in :numref:`eq_softmax_attention`. Prove that $\partial_{\mathbf{q}} \mathop{\mathrm{Attn}}(\mathrm{DB}, \mathbf{q}) = \mathrm{Var}_{p(\mathbf{k}; \mathbf{q})}[\mathbf{k}]$.
1. Design a differentiable search engine using the attention mechanism. 
1. Review the design of the Squeeze and Excitation networks :cite:`Hu.Shen.Sun.2018` and interpret them through the lens of the attention mechanism. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:

```{.python .input}

```
