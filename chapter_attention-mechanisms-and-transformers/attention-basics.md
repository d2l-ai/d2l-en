```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Attention Basics: Queries, Keys, and Values

:label:`sec_attention-basics`


The key idea behind attention is that the neural network has some mechanism
by which to *choose* which inputs to focus on.
Typically in order to control the attention, 
the relevant part of the model must produce 
a *query* vector.
This *query* vector is then typically compared with a number of 
*key* vectors (one per input). 
Based on some notion of compatibility between the query and the key,
a probability distribution over the available inputs (e.g., input tokens) is computed,
assigning high weights to especially compatible inputs 
and low weights to incompatible inputs. 
Finally, using these weights, 
we compute a weighted average 
of the representations (i.e., the *values*)
corresponding to each input.
This weighted summation is often called *attention pooling*.
As shown in :numref:`fig_qkv`,
in an attention mechanisms, 
a given *query* must somehow interact with each key
to produce a set of weights.

![Attention mechanisms conduct a weighted average over values via attention pooling,
where weights are derived according to the compatibility of a query and keys.](../img/qkv.svg)
:label:`fig_qkv`

Note that there are many variations on attention mechanisms.
For instance, we can design a non-differentiable attention model
that can be trained using reinforcement learning methods :cite:`Mnih.Heess.Graves.ea.2014`.
That said, given that the dominant attention models 
follow the framework outlined in :numref:`fig_qkv`,
we focus our exposition on this family of differentiable mechanisms.


## Visualization of Attention

Average pooling can be treated as a weighted average of inputs, 
where weights are uniform.
In practice, attention pooling aggregates values using weighted average,
where weights are computed between the given query and different keys.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

To visualize attention weights, we define the `show_heatmaps` function.
Its input `matrices` has the shape (number of rows for display, 
number of columns for display, number of queries, number of keys).

```{.python .input}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    num_rows, num_cols = len(matrices), len(matrices[0])
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

For demonstration, we consider a simple case
where the attention weight is one 
only when the query and the key are the same; 
otherwise it is zero.

```{.python .input}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

In the subsequent sections,
we will often invoke this function to visualize attention weights.

## Summary

Attention mechanisms provide a differentiable means of control, 
by which a neural network can "decide" which tokens 
to assign higher vs. lower weights when constructing 
a weighted sum over a sequence of representations. 
Attention mechanisms compute a weighted sum over *values*,
where the weights are determined by the given *query* and *keys*.
Practitioners often attempt to derive intuitions and insights
by visualizing attention weights between queries and keys.

## Exercises

1. How might you expect the decoder of a language translation model to direct its attention at each step?
1. Name three other plausible sequence to sequence tasks and speculate about how you might expect the decoder to direct its attention in these tasks.
1. Randomly generate a $10 \times 10$ matrix and use the softmax operation to ensure each row is a valid probability distribution. Visualize the output attention weights.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
