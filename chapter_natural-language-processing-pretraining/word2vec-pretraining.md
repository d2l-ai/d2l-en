# Pretraining word2vec
:label:`sec_word2vec_pretraining`


We go on to implement the skip-gram
model defined in
:numref:`sec_word2vec`.
Then
we will pretrain word2vec using negative sampling
on the PTB dataset.
First of all,
let us obtain the data iterator
and the vocabulary for this dataset
by calling the `d2l.load_data_ptb`
function, which was described in :numref:`sec_word2vec_data`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## The Skip-Gram Model

We implement the skip-gram model
by using embedding layers and batch matrix multiplications.
First, let us review
how embedding layers work.


### Embedding Layer

As described in :numref:`sec_seq2seq`,
an embedding layer
maps a token's index to its feature vector.
The weight of this layer
is a matrix whose number of rows equals to
the dictionary size (`input_dim`) and
number of columns equals to
the vector dimension for each token (`output_dim`).

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

The input of an embedding layer is the
index of a token (word).
For any token index $i$,
its vector representation
can be obtained from
the $i^\mathrm{th}$ row of the weight matrix
in the embedding layer.
Since the vector dimension (`output_dim`)
was set to 4,
the embedding layer
returns vectors with shape (2, 3, 4)
for a minibatch of token indices with shape
(2, 3).

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Defining the Forward Propagation

In the forward propagation,
the input of the skip-gram model
includes
the center word indices `center`
of shape (batch size, 1)
and
the concatenated context and noise word indices `contexts_and_negatives`
of shape (batch size, `max_len`),
where `max_len`
is defined
in :numref:`subsec_word2vec-minibatch-loading`.
These two variables are first transformed from the
token indices into vectors via the embedding layer,
then their batch matrix multiplication
(described in :numref:`subsec_batch_dot`)
returns
an output of shape (batch size, 1, `max_len`).
Each element in the output is the dot product of
a center word vector and a context or noise word vector.

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Let us print the output shape of this `skip_gram` function for some example inputs.

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Training

Before training the word embedding model, we need to define the loss function of the model.

### Binary Cross Entropy Loss Function

According to the definition of the loss function in negative sampling, we can directly use the binary cross-entropy loss function from high-level APIs.

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    "BCEWithLogitLoss with masking on call."
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

It is worth mentioning that we can use the mask variable to specify the partial predicted value and label that participate in loss function calculation in the minibatch: when the mask is 1, the predicted value and label of the corresponding position will participate in the calculation of the loss function; When the mask is 0, they do not participate. As we mentioned earlier, mask variables can be used to avoid the effect of padding on loss function calculations.

Given two identical examples, different masks lead to different loss values.

```{.python .input}
#@tab all
pred = d2l.tensor([[.5]*4]*2)
label = d2l.tensor([[1., 0., 1., 0.]]*2)
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask)
```

We can normalize the loss in each example due to various lengths in each example.

```{.python .input}
#@tab all
loss(pred, label, mask) / mask.sum(axis=1) * mask.shape[1]
```

### Initializing Model Parameters

We construct the embedding layers of the central and context words, respectively, and set the hyperparameter word vector dimension `embed_size` to 100.

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Training

The training function is defined below. Because of the existence of padding, the calculation of the loss function is slightly different compared to the previous training functions.

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    metric = d2l.Accumulator(2)  # Sum of losses, no. of tokens
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    metric = d2l.Accumulator(2)  # Sum of losses, no. of tokens
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Now, we can train a skip-gram model using negative sampling.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Applying the Word Embedding Model

After training the word embedding model, we can represent similarity in meaning between words based on the cosine similarity of two word vectors. As we can see, when using the trained word embedding model, the words closest in meaning to the word "chip" are mostly related to chips.

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Summary

* We can pretrain a skip-gram model through negative sampling.


## Exercises

1. Set `sparse_grad=True` when creating an instance of `nn.Embedding`. Does it accelerate training? Look up MXNet documentation to learn the meaning of this argument.
1. Try to find synonyms for other words.
1. Tune the hyperparameters and observe and analyze the experimental results.
1. When the dataset is large, we usually sample the context words and the noise words for the central target word in the current minibatch only when updating the model parameters. In other words, the same central target word may have different context words or noise words in different epochs. What are the benefits of this sort of training? Try to implement this training method.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
