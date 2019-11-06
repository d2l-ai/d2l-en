# Attention Mechanism

Recall from :numref:`sec_seq2seq`, we encode the source sequence input information in the recurrent unit state and then pass it to the decoder to generate the target sequence. A token in the target sequence may closely relate to one or more tokens in the source sequence, instead of the whole source sequence. For example, when translating "Hello world." to "Bonjour le monde.", "Bonjour" maps onto "Hello" and "monde" maps onto "world". In the seq2seq model, the decoder may implicitly select the corresponding information from the state passed by the encoder. The attention mechanism, however, makes this selection explicit.


*Attention* is a generalized pooling method with bias alignment over inputs. The core component in the attention mechanism is the attention layer, or called *attention* for simplicity. An input of the attention layer is referred to as a *query*. For a query, attention returns a output based on the memory - a set of key-value pairs encoded in the attention layer . To be more specific, assume that the memory contains $n$ key-value pairs, $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_n, \mathbf{v}_n)$, with $\mathbf{k}_i \in \mathbb R^{d_k}$, $\mathbf{v}_i \in \mathbb R^{d_v}$. Given a query $\mathbf{q} \in \mathbb R^{d_q}$, then the attention layer returns an output $\mathbf{o} \in \mathbb R^{d_v}$ with the same shape as a value.

![The attention layer returns an output based on the input query and its memory.](../img/attention.svg)
:label:`fig_attention`


More specifically, the full process of attention mechanism is expressed in :numref:`fig_attention_output`. To compute the output of attention, we first utilize a score function $\alpha$ which measures the similarity between the query and a key. Then for each key $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_n, \mathbf{v}_n)$, we compute the scores $a_1, \ldots, a_n$ by

$$a_i = \alpha(\mathbf q, \mathbf k_i).$$

Next we use softmax to obtain the attention weights, i.e.,

$$\mathbf{b} = \mathrm{softmax}(\mathbf{a})\quad \text{, where }\quad
{b}_i = \frac{\exp(a_i)}{\sum_j \exp(a_j)}, \mathbf{b} = [b_1, \ldots, b_n]^T .$$


Finally, the output is then a weighted sum of the values:

$$\mathbf o = \sum_{i=1}^n b_i \mathbf v_i.$$


![The attention output is a weighted sum of the values.](../img/attention_output.svg)
:label:`fig_attention_output`



Different choices of the score function lead to different attention layers. Below, we introduce two commonly used attention layers. Before diving into the implementation, we first express two operators to get you up and running: a masked version of the softmax operator `masked_softmax` and a specialized dot operator `batched_dot`.

```{.python .input  n=1}
import math
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

The masked softmax takes a 3 dimensional input and enables us to filter out some elements by specifying a valid length for the last dimension. (Refer to
:numref:`sec_machine_translation` for the definition of a valid length.) As a result, any value outside the valid length will be masked as $0$. Let us implement the `masked_softmax` realization first.

```{.python .input  n=2}
# Saved in the d2l package for later use
def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    if valid_length is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_length.ndim == 1:
            valid_length = valid_length.repeat(shape[1], axis=0)
        else:
            valid_length = valid_length.reshape(-1)
        # fill masked elements with a large negative, whose exp is 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_length, True,
                              axis=1, value=-1e6)
        return npx.softmax(X).reshape(shape)
```

To illustrate how does this function work, we construct two $2 \times 4$ matrixes as the input. In addition, we specify that the valid length equals 2 for the first example, and 3 for the second example. Then, as we can see from the following outputs, the value outside valid lengths are masked as zero.


```{.python .input  n=3}
masked_softmax(np.random.uniform(size=(2,2,4)), np.array([2,3]))
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "array([[[0.488994  , 0.511006  , 0.        , 0.        ],\n        [0.43654838, 0.56345165, 0.        , 0.        ]],\n\n       [[0.28817102, 0.3519408 , 0.3598882 , 0.        ],\n        [0.29034293, 0.25239873, 0.45725834, 0.        ]]])"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Moreover, the second operator `batched_dot` takes two inputs $X$ and $Y$ with shapes $(b, n, m)$ and $(b, m, k)$, respectively, and returns an output with shape $(b, n, k)$. To be specific, it computes $b$ dot products for $i= \{1,\ldots,b\}$, i.e., 

$$Z[i,:,:] = X[i,:,:]  Y[i,:,:] $$


Here, we will not dive into the detailed implementation of `batched_dot`. Rather, for convenience, we simply call `npx.batched_dot` as a built-in function in MXNet.

```{.python .input  n=4}
npx.batch_dot(np.ones((2,1,3)), np.ones((2,3,2)))
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "array([[[3., 3.]],\n\n       [[3., 3.]]])"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Dot Product Attention

Equipping with the above two operators: the `masked_softmax` and the `batched_dot`, let us dive into the details of two widely used attentions layers. The first one is the *dot product attention*, which assumes the query has the same dimension as the keys, namely $\mathbf q, \mathbf k_i \in\mathbb R^d$ for all $i$. The dot product attention computes the scores by an dot product between the query and a key, which is then divided by $\sqrt{d}$ to minimize the unrelated influence of the dimension $d$ on the scores. In other words,

$$\alpha(\mathbf q, \mathbf k) = \langle \mathbf q, \mathbf k \rangle /\sqrt{d}.$$

Beyond the single dimension queries and keys, we can always generalize to a multi-dimensional queries and keys. To be specific, assume that $\mathbf Q\in\mathbb R^{m\times d}$ contains $m$ queries and $\mathbf K\in\mathbb R^{n\times d}$ has all $n$ keys. We can compute all $mn$ scores by

$$\alpha(\mathbf Q, \mathbf K) = \mathbf Q \mathbf K^\top /\sqrt{d}.$$


With the above formula, let us implement the dot product attention layer `DotProductAttention` that supports a batch of queries and key-value pairs. In addition, for the purpose of regularization, we enable a randomly dropping functionality by specifying the degree of dropout within our implementation.



```{.python .input  n=5}
# Saved in the d2l package for later use
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
        scores = npx.batch_dot(query, key, transpose_b=True) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return npx.batch_dot(attention_weights, value)
```

Let us try the class `DotProductAttention` in a toy example. 

First, we create two batches, with each batch has one query and 10 key-value pairs.  We specify through `valid_length` with "$2$" for the first batch and "$6$" for the second batch, which means that we will check the first $2$ key-value pairs for the first batch and $6$ for the second one. Therefore, even though both batches have the same query and key-value pairs, we obtain different outputs.

```{.python .input  n=6}
atten = DotProductAttention(dropout=0.5)
atten.initialize()
keys = np.ones((2,10,2))
values = np.arange(40).reshape(1,10,4).repeat(2,axis=0)
atten(np.ones((2,1,2)), keys, values, np.array([2, 6]))
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "array([[[ 2.,  3.,  4.,  5.]],\n\n       [[10., 11., 12., 13.]]])"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```


As we can see above, dot product attention simply multiplies the query and key together, and hopes to derive their similarities from there. Whereas, the query and key may not be of the same dimension sometimes. Can we solve the problem by applying some neural network architecture on the attention layers? The answer is the multilayer perceptron attention!



## Multilayer Perceptron Attention

Another widely used attention is called *multilayer perceptron attention*, where we project both query and keys into $\mathbb R^{h}$ by learnable weights parameters. To be more clear, assume that the learnable weights are $\mathbf W_k\in\mathbb R^{h\times d_k}$, $\mathbf W_q\in\mathbb R^{h\times d_q}$, and $\mathbf v\in\mathbb R^{h}$. Then the score function is defined by

$$\alpha(\mathbf k, \mathbf q) = \mathbf v^\top \text{tanh}(\mathbf W_k \mathbf k + \mathbf W_q\mathbf q). $$

To provide you some intuition about the weights, you can imagine $\mathbf W_k \mathbf k + \mathbf W_q\mathbf q$ as concatenating the key and value in the feature dimension and feeding them into a single hidden layer perceptron with hidden layer size $h$ and output layer size $1$. In this hidden layer, the activation function is $\tanh$ and no bias is applied. Now let us implement the multilayer perceptron attention together!

```{.python .input  n=7}
# Saved in the d2l package for later use
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
        features = np.expand_dims(query, axis=2) + np.expand_dims(key, axis=1)
        scores = np.squeeze(self.v(features), axis=-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return npx.batch_dot(attention_weights, value)
```

To test the above class `MLPAttention`, we use the same inputs as in the provious toy example. As we can see below, despite `MLPAttention` containing an additional MLP model, we obtain the same output as for `DotProductAttention`.

```{.python .input  n=8}
atten = MLPAttention(units=8, dropout=0.1)
atten.initialize()
atten(np.ones((2,1,2)), keys, values, np.array([2, 6]))
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "array([[[ 2.,  3.,  4.,  5.]],\n\n       [[10., 11., 12., 13.]]])"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Summary

* An attention layer explicitly selects related information explicitly.
* An attention layer's memory consists of key-value pairs, so its output is close to the values whose keys are similar to the query.
* Two commonly used attentions are dot product attention and multilayer perceptron attention.


## Exercises

* What are the advantages and disadvantages for DotProductAttention and MLPAttention respectively?


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/attention/4343)

![](../img/qr_attention.svg)
