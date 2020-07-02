# Pretraining fastText
:label:`sec_word2vec_gluon`

In this section, we will
train a skip-gram model defined in
:numref:`sec_word2vec`.

First, import the
packages and modules required for the experiment, and load the PTB dataset.

```{.python .input  n=1}
from collections import defaultdict
from d2l import mxnet as d2l
from functools import partial
from mxnet import autograd, gluon, init, np, npx, cpu
from mxnet.gluon import nn
import random

npx.set_np()
```

```{.python .input  n=2}
def compute_subword(token):
    if token[0] != '<' and token[-1] != '>':
        token = '<' + token + '>'
        subwords = {token}
        for i in range(len(token)-3):
            for j in range(i + 3, len(token)+1):
                if j - i <= 6:
                    subwords.add(token[i:j])
        return subwords
    else:
        return [token]
```

```{.python .input  n=3}
def get_subword_map(vocab):
    tokenid_to_subword, subword_to_idx = defaultdict(list), defaultdict(int)
    for token, tokenid in vocab.token_to_idx.items():
        subwords = compute_subword(token)
        for subword in subwords:
            if subword not in subword_to_idx:
                subword_to_idx[subword] = len(subword_to_idx)
            tokenid_to_subword[tokenid].append(subword_to_idx[subword])
    return tokenid_to_subword, subword_to_idx
```

```{.python .input  n=4}
def token_transform(tokens, vocab, subword_map):
    if not isinstance(tokens, (list, tuple)):
        return d2l.truncate_pad(subword_map[tokens],
                                 64, vocab['<pad>'])
    return [token_transform(token, vocab, subword_map) for token in tokens]
```

```{.python .input  n=5}
def batchify(data, vocab, subword_map):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [token_transform([center], vocab, subword_map)]
        contexts_negatives += [token_transform(context + negative + \
                               [1] * (max_len - cur_len), vocab, subword_map)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (np.array(centers), np.array(contexts_negatives),
            np.array(masks), np.array(labels))
```

```{.python .input  n=6}
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = d2l.get_dataloader_workers()
    sentences = d2l.read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10, reserved_tokens=['<pad>'])
    subsampled = d2l.subsampling(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = d2l.get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = d2l.get_negatives(all_contexts, corpus, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    subword_map, subword_to_idx = get_subword_map(vocab)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      batchify_fn=partial(batchify, vocab=vocab, subword_map=subword_map),
                                      num_workers=num_workers)
    return data_iter, vocab, subword_to_idx
```

```{.python .input  n=7}
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab, subword_to_idx = load_data_ptb(batch_size, max_window_size, num_noise_words)
```

```{.python .input  n=8}
names = ['centers', 'contexts_negatives', 'masks', 'labels']
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## The Skip-Gram Model

We will implement the skip-gram model by using embedding
layers and minibatch multiplication. These methods are also often used to
implement other natural language processing applications.

### Embedding Layer
The layer in which the obtained word is embedded is called the embedding layer,
which can be obtained by creating an `nn.Embedding` instance in Gluon. The
weight of the embedding layer is a matrix whose number of rows is the dictionary
size (`input_dim`) and whose number of columns is the dimension of each word
vector (`output_dim`). We set the dictionary size to $20$ and the word vector
dimension to $4$.

```{.python .input  n=9}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

### Skip-gram Model Forward Calculation

In forward calculation, the input of
the skip-gram model contains the central target word index `center` and the
concatenated context and noise word index `contexts_and_negatives`. In which,
the `center` variable has the shape (batch size, 1), while the
`contexts_and_negatives` variable has the shape (batch size, `max_len`). These
two variables are first transformed from word indexes to word vectors by the
word embedding layer, and then the output of shape (batch size, 1, `max_len`) is
obtained by minibatch multiplication. Each element in the output is the inner
product of the central target word vector and the context word vector or noise
word vector.

```{.python .input  n=10}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u, padding):
    v_embedding = embed_v(center)
    v_mask = (center!=padding).astype('float32')
    v = (v_embedding * np.expand_dims(v_mask, axis=-1)).sum(-1)/(np.expand_dims(v_mask.sum(-1), axis=-1)+1e-5)
    u_embedding = embed_u(contexts_and_negatives)
    u_mask = (contexts_and_negatives!=padding).astype('float32')
    u = (u_embedding * np.expand_dims(u_mask, axis=-1)).sum(-1)/(np.expand_dims(u_mask.sum(-1), axis=-1)+1e-5)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

Verify that the output shape should be (batch size, 1, `max_len`).

```{.python .input  n=12}
skip_gram(np.ones((2, 1, 64)), np.ones((2, 6, 64)), embed, embed, vocab['<pad>']).shape
```

## Training

Before training the word embedding model, we need to define the
loss function of the model.

### Binary Cross Entropy Loss Function

According
to the definition of the loss function in negative sampling, we can directly use
Gluon's binary cross-entropy loss function `SigmoidBinaryCrossEntropyLoss`.

```{.python .input  n=13}
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

It is worth mentioning that we can use the mask variable to specify the partial
predicted value and label that participate in loss function calculation in the
minibatch: when the mask is 1, the predicted value and label of the
corresponding position will participate in the calculation of the loss function;
When the mask is 0, the predicted value and label of the corresponding position
do not participate in the calculation of the loss function. As we mentioned
earlier, mask variables can be used to avoid the effect of padding on loss
function calculations.

Given two identical examples, different masks lead to
different loss values.

```{.python .input  n=14}
pred = np.array([[.5]*4]*2)
label = np.array([[1, 0, 1, 0]]*2)
mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask)
```

We can normalize the loss in each example due to various lengths in each
example.

```{.python .input  n=15}
loss(pred, label, mask) / mask.sum(axis=1) * mask.shape[1]
```

### Initializing Model Parameters

We construct the embedding layers of the
central and context words, respectively, and set the hyperparameter word vector
dimension `embed_size` to 100.

```{.python .input  n=16}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(subword_to_idx), output_dim=embed_size),
        nn.Embedding(input_dim=len(subword_to_idx), output_dim=embed_size))
```

### Training

The training function is defined below. Because of the existence
of padding, the calculation of the loss function is slightly different compared
to the previous training functions.

```{.python .input  n=17}
def train(net, data_iter, lr, num_epochs, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1], vocab['<pad>'])
                l = (loss(pred.reshape(label.shape), label, mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter),
                             (metric[0]/metric[1],))
            npx.waitall()
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

Now, we can train a skip-gram model using negative sampling.

```{.python .input  n=20}
lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs)
```

```{.python .input}
def get_similar_tokens(query_token, k, embed, vocab, subword_to_idx):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    all_v = []
    for token in vocab.idx_to_token:
        subword = compute_subword(token)
        w_v = W[[subword_to_idx[s] for s in subword]].sum(0)
        all_v.append(np.expand_dims(w_v, 0))
    all_v = np.concatenate(all_v, 0)
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(all_v, x) / np.sqrt(np.sum(all_v * all_v, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print('cosine sim=%.3f: %s' % (cos[i], (vocab.idx_to_token[i])))
```

```{.python .input}
get_similar_tokens('chip', 3, net[0], vocab, subword_to_idx)
```
