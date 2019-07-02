# Attention Mechanism

In :numref:`chapter_seq2seq`, we encode the source sequence input information in the recurrent unit state, and then pass it to the decoder to generate the target sequence. A token in the target sequence may closely relate to some tokens in the source sequence instead of the whole source sequence. For example, when translating "Hello world." to "Bonjour le monde.", "Bonjour" maps to "Hello" and "monde" maps to "world". In the seq2seq model, the decoder may implicitly select the corresponding information from the state passed by the decoder. The attention mechanism, however, makes this selection explicit.

Attention is a generalized pooling method with bias alignment over inputs. The core component in the attention mechanism is the attention layer, or called attention for simplicity. An input of the attention layer is called a query. For a query, the attention layer returns the output based on its memory, which is a set of key-value pairs. To be more specific, assume a query $\mathbf{q}\in\mathbb R^{d_q}$, and the memory contains $n$ key-value pairs, $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_n, \mathbf{v}_n)$, with $\mathbf{k}_i\in\mathbb R^{d_k}$, $\mathbf{v}_i\in\mathbb R^{d_v}$. The attention layer then returns an output $\mathbf o\in\mathbb R^{d_v}$ with the same shape has a value.

![The attention layer returns an output based on the input query and its memory.](../img/attention.svg)

To compute the output, we first assume there is a score function $\alpha$ which measure the similarity between the query and a key. Then we compute all $n$ scores $a_1, \ldots, a_n$ by

$$a_i = \alpha(\mathbf q, \mathbf k_i).$$

Next we use softmax to obtain the attention weights

$$b_1, \ldots, b_n = \textrm{softmax}(a_1, \ldots, a_n).$$

The output is then a weight sum of the values

$$\mathbf o = \sum_{i=1}^n b_i \mathbf v_i.$$

Different choices of the score function lead to different attention layers. We will discuss two commonly used attention layers in the rest of this section. Before diving into the implementation, we first introduce a masked version of the softmax operator and explain a specialized dot operator `nd.batched_dot`.

```{.python .input  n=1}
import math
from mxnet import nd
from mxnet.gluon import nn
```

The masked softmax takes a 3-dim input and allows to filter out some elements by
specifying valid lengths for the last dimension. (refer to
:numref:`chapter_machine_translation` for the
definition of a valid length.)

```{.python .input  n=6}
# Save to the d2l package.
def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    if valid_length is None:
        return X.softmax()
    else:
        shape = X.shape
        if valid_length.ndim == 1:
            valid_length = valid_length.repeat(shape[1], axis=0)
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = nd.SequenceMask(X.reshape((-1, shape[-1])), valid_length, True,
                            axis=1, value=-1e6)
        return X.softmax().reshape(shape)
```

Construct two examples, which each example is a 2-by-4 matrix, as the input. If specify the valid length for the first example to be 2, then only the first two columns of this example are used to compute softmax.

```{.python .input  n=5}
masked_softmax(nd.random.uniform(shape=(2,2,4)), nd.array([2,3]))
```

The operator `nd.batched_dot` takes two inputs $X$ and $Y$ with shapes $(b, n, m)$ and $(b, m, k)$, respectively. It computes $b$ dot products, with `Z[i,:,:]=dot(X[i,:,:], Y[i,:,:]` for $i=1,\ldots,n$.

```{.python .input  n=4}
nd.batch_dot(nd.ones((2,1,3)), nd.ones((2,3,2)))
```

## Dot Product Attention

The dot product assume the query has the same dimension with the keys, namely $\mathbf q, \mathbf k_i \in\mathbb R^d$ for all $i$. It computes the score by an inner product between the query and a key, and often then divided by $\sqrt{d}$ to make the scores less sensitive to the dimension $d$. In other words,

$$\alpha(\mathbf q, \mathbf k) = \langle \mathbf q, \mathbf k \rangle /\sqrt{d}.$$

Assume $\mathbf Q\in\mathbb R^{m\times d}$ contains $m$ queries and $\mathbf K\in\mathbb R^{n\times d}$ has all $n$ keys. We can compute all $mn$ scores by

$$\alpha(\mathbf Q, \mathbf K) = \mathbf Q \mathbf K^T /\sqrt{d}.$$

Now let's implement this layer that supports a batch of queries and key-value pairs. In addition, it supports to randomly drop some attention weights as a regularization.

```{.python .input  n=5}
# Save to the d2l package.
class DotProductAttention(nn.Block): 
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_length: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key
        scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return nd.batch_dot(attention_weights, value)
```

Now we create two batches, and each batch has one query and 10 key-value pairs.  We specify through `valid_length` that the first batch we will only pay attention to the first 2 key-value pairs, while the second batch will check the first 6 key-value pairs. Therefore, both batches have the same query, key-value pairs, we obtain different outputs.

```{.python .input  n=6}
atten = DotProductAttention(dropout=0.5)
atten.initialize()
keys = nd.ones((2,10,2))
values = nd.arange(40).reshape((1,10,4)).repeat(2,axis=0)
atten(nd.ones((2,1,2)), keys, values, nd.array([2, 6]))
```

## Multilayer Perception Attention

In multilayer perception attention, we first project both query and keys into

To be more specific, assume learnable parameters $\mathbf W_k\in\mathbb R^{h\times d_k}$, $\mathbf W_q\in\mathbb R^{h\times d_q}$, and $\mathbf v\in\mathbb R^{p}$, then the score function is defined by

$$\alpha(\mathbf k, \mathbf q) = \mathbf v^T \text{tanh}(\mathbf W_k \mathbf k + \mathbf W_q\mathbf q). $$

It equals to concatenate the key and value in the feature dimension, and the feed into a single hidden-layer perception with hidden size $h$ and output size $1$. The hidden layer activation function is tanh, and no bias is applied.

```{.python .input  n=7}
# Save to the d2l package.
class MLPAttention(nn.Block):  
    def __init__(self, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes.
        self.W_k = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.W_q = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length):
        query, key = self.W_k(query), self.W_q(key)
        # expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast.
        features = query.expand_dims(axis=2) + key.expand_dims(axis=1)
        scores = self.v(features).squeeze(axis=-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return nd.batch_dot(attention_weights, value)
```

Despite `MLPAttention` contains an additional MLP model in it, given the same inputs with identical keys, we obtain the same output as for `DotProductAttention`.

```{.python .input  n=8}
atten = MLPAttention(units=8, dropout=0.1)
atten.initialize()
atten(nd.ones((2,1,2)), keys, values, nd.array([2, 6]))
```

## Summary

* An attention layer explicitly selects related information.
* An attention layer's memory consists of key-value pairs, so its output is close to the values whose keys are similar to the query.
