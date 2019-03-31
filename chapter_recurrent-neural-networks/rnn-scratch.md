# Implementation of Recurrent Neural Networks from Scratch

In this section we implement a language model from scratch. It is based on a character-level recurrent neural network trained on H. G. Wells' 'The Time Machine'. As before, we start by reading the dataset first.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_time_machine()
```

## One-hot Encoding

One-hot encoding vectors provide an easy way to express words as vectors in order to process them in a deep network. In a nutshell, we map each word to a different unit vector: assume that the number of different characters in the dictionary is $N$ (the `vocab_size`) and each character has a one-to-one correspondence with a single value in the index of successive integers from 0 to $N-1$. If the index of a character is the integer $i$, then we create a vector $\mathbf{e}_i$ of all 0s with a length of $N$ and set the element at position $i$ to 1. This vector is the one-hot vector of the original character. The one-hot vectors with indices 0 and 2 are shown below (the length of the vector is equal to the dictionary size).

```{.python .input  n=2}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

Note that one-hot encodings are just a convenient way of separating the encoding (e.g. mapping the character `a` to $(1,0,0, \ldots) vector)$ from the embedding (i.e. multiplying the encoded vectors by some weight matrix $\mathbf{W}). This simplifies the code greatly relative to storing an embedding matrix that the user needs to maintain. 

The shape of the mini-batch we sample each time is (batch size, time step). The following function transforms such mini-batches into a number of matrices with the shape of (batch size, dictionary size) that can be entered into the network. The total number of vectors is equal to the number of time steps. That is, the input of time step $t$ is $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$, where $n$ is the batch size and $d$ is the number of inputs. That is the one-hot vector length (the dictionary size).

```{.python .input  n=3}
# This function is saved in the d2l package for future use
def to_onehot(X, size):  
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
```

The code above generates 5 minibatches containing 2 vectors each. Since we have a total of 43 distinct symbols in "The Time Machine" we get 43-dimensional vectors.

