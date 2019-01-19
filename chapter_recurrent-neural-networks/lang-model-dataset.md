# Text Preprocessing

In the previous section we discussed some properties that make language unique. The key is that the number of tokens (aka words) is large and very unevenly distributed. Hence, a naive multiclass classification approach to predict the next symbol doesn't always work very well. Moreover, we need to turn text into a format that we can optimize over, i.e. we need to map it to vectors. At its extreme we have two alternatives. One is to treat each word as a unique entity, as proposed e.g. by [Salton, Wong and Yang, 1975](https://dl.acm.org/citation.cfm?id=361220). The problem with this strategy is that we might well have to deal with 100,000 to 1,000,000 vectors for very large and diverse corpora. 

At the other extreme lies the strategy to predict one character at a time, as suggested e.g. by [Ling et al., 2015](https://arxiv.org/pdf/1508.02096.pdf). A good balance in between both strategies is [byte-pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding), as described by [Sennrich, Haddow and Birch, 2015](https://arxiv.org/abs/1508.07909) for the purpose of neural machine translation. It decomposes text into syllable-like fragments that occur frequently. This allows for models that are able to generate words like `heteroscedastic` or `pentagram` based on previously viewed words, e.g. `heterogeneous`, `homoscedastic`, `diagram`, and `pentagon`. Going into details of these models is beyond the scope of the current chapter. We will address this later when discussing [Natural Language Processing](../chapter_natural-language-processing/) in much more detail. Suffice it to say that it can contribute significantly to the accuracy of natural language processing models. 

For the sake of simplicity we will limit ourselves to pure character sequences. We use H.G. Wells' *The Timemachine* as before. We begin by filtering the text and convert it into a a sequence of character IDs. 

## Data Loading

We begin, as before, by loading the data and by mapping it into a sequence of whitespaces, punctuation signs and regular characters. Preprocessing is minimal and we limit ourselves to removing multiple whitespaces.

```{.python .input  n=10}
import sys
sys.path.insert(0, '..')

from mxnet import nd
import random

with open('../data/timemachine.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = ' '.join(' '.join(lines).lower().split())

print('number of characters: ', len(raw_dataset))
print(raw_dataset[0:70])
```

## Character Index

This data set has more than 178,000 characters. Next we need to identify all the characters occurring in the document. That is, we want to map each character to continuous integers starting from 0, also known as an index, to facilitate subsequent data processing. To get the index, we extract all the different characters in the data set and then map them to the index one by one to construct the dictionary. Then, print `vocab_size`, which is the number of different characters in the dictionary, i.e. the dictionary size.

```{.python .input  n=13}
idx_to_char = list(set(raw_dataset))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(char_to_idx)
```

After that, each character in the training data set is converted into an index ID. To illustrate things we print the first 20 characters and their corresponding indices.

```{.python .input  n=15}
corpus_indices = [char_to_idx[char] for char in raw_dataset]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

We packaged the above code in the `(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_timemachine()` function of the `d2l` package to make it easier to call it in later chapters. 

## Training Data Preparation

During training, we need to read mini-batches of examples and labels at random. Since sequence data is by its very nature sequential, we need to address the issue of processing it. We did so in a rather ad-hoc manner when we introduced [Sequence Models](sequence.md). Let's formalize this a bit. Consider the beginning of the book we just processed. If we want to split it up into sequences of 5 symbols each, we have quite some freedom since we could pick an arbitrary offset. 

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)

In fact, any one of these offsets is fine. Hence, which one should we pick? In fact, all of them are equally good. But if we pick all offsets we end up with rather redundant data due to overlap, particularly if the sequences are long. Picking just a random set of initial positions is no good either since it does not guarantee uniform coverage of the array. For instance, if we pick $n$ elements at random out of a set of $n$ with random replacement, the probability for a particular element not being picked is $(1-1/n)^n \to e^{-1}$. This means that we cannot expect uniform coverage this way. Even randomly permuting a set of all offsets does not offer good guarantees. Instead we can use a simple trick to get both *coverage* and *randomness*: use a random offset, after which one uses the terms sequentially. We describe how to accomplish this for both random sampling and sequential partitioning strategies below.

### Random sampling

The following code randomly generates a minibatch from the data each time. Here, the batch size `batch_size` indicates to the number of examples in each mini-batch and `num_steps` is the length of the sequence (or time steps if we have a time series) included in each example.
In random sampling, each example is a sequence arbitrarily captured on the original sequence. The positions of two adjacent random mini-batches on the original sequence are not necessarily adjacent. The target is to predict the next character based on what we've seen so far, hence the labels are the original sequence, shifted by one character. Note that this is not recommended for latent variable models, since we do not have access to the hidden state *prior* to seeing the sequence. We packaged the above code in the `load_data_timemachine` function of the `d2l` package to make it easier to call it in later chapters. It returns four variables: `corpus_indices`, `char_to_idx`, `idx_to_char`, and `vocab_size`.

```{.python .input  n=5}
# This function is saved in the d2l package for future use.
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0,num_steps))
    corpus_indices = corpus_indices[offset:]
    # subtract 1 extra since we need to account for the sequence length
    num_examples = ((len(corpus_indices) - 1) // num_steps) - 1
    # discard half empty batches
    num_batches = num_examples // batch_size
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)
    
    # This returns a sequence of the length num_steps starting from pos.
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(0, batch_size * num_batches, batch_size):
        # batch_size indicates the random examples read each time.
        batch_indices = example_indices[i:(i+batch_size)]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]       
        yield nd.array(X, ctx), nd.array(Y, ctx)
```

Let us generate an artificial sequence from 0 to 30. We assume that
the batch size and numbers of time steps are 2 and 5
respectively. This means that depending on the offset we can generate between 4 and 5 $(x,y)$ pairs. With a minibatch size of 2 we only get 2 minibatches.

```{.python .input  n=6}
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### Sequential partitioning

In addition to random sampling of the original sequence, we can also make the positions of two adjacent random mini-batches adjacent in the original sequence. Now, we can use a hidden state of the last time step of a mini-batch to initialize the hidden state of the next mini-batch, so that the output of the next mini-batch is also dependent on the input of the mini-batch, with this pattern continuing in subsequent mini-batches. This has two effects on the implementation of a recurrent neural network. On the one hand,
when training the model, we only need to initialize the hidden state at the beginning of each epoch.
On the other hand, when multiple adjacent mini-batches are concatenated by passing hidden states, the gradient calculation of the model parameters will depend on all the mini-batch sequences that are concatenated. In the same epoch as the number of iterations increases, the costs of gradient calculation rise.
So that the model parameter gradient calculations only depend on the mini-batch sequence read by one iteration, we can separate the hidden state from the computational graph before reading the mini-batch (this can be done by detaching the graph). We will gain a deeper understand this approach in the following sections.

```{.python .input  n=7}
# This function is saved in the d2l package for future use.
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    # offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0,num_steps))
    # slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus_indices) - offset) // batch_size) * batch_size
    indices = nd.array(corpus_indices[offset:(offset + num_indices)], ctx=ctx)
    indices = indices.reshape((batch_size,-1))
    # need to leave one last token since targets are shifted by 1
    num_epochs = ((num_indices // batch_size) - 1) // num_steps

    for i in range(0, num_epochs * num_steps, num_steps):
        X = indices[:,i:(i+num_steps)]
        Y = indices[:,(i+1):(i+1+num_steps)]
        yield X, Y
```

Using the same settings, print input `X` and label `Y` for each mini-batch of examples read by random sampling. The positions of two adjacent random mini-batches on the original sequence are adjacent.

```{.python .input  n=8}
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y)
```

Sequential partitioning decomposes the sequence into `batch_size` many strips of data which are traversed as we iterate over minibatches. Note that the $i$-th element in a minibatch matches with the $i$-th element of the next minibatch rather than within a minibatch. 

## Summary

* Documents are preprocessed by tokenizing the words and mapping them into IDs. There are multiple methods:
    * Character encoding which uses individual characters (good e.g. for Chinese)
    * Word encoding (good e.g. for English)
    * Byte-pair encoding (good for languages that have lots of morphology, e.g. German)
* The main choices for sequence partitioning are whether we pick consecutive or random sequences. In particular for recurrent networks the former is critical. 
* Given the overall document length, it is usually acceptable to be slightly wasteful with the documents and discard half-empty minibatches. 

## Problems

1. Which other other mini-batch data sampling methods can you think of?
1. Why is it a good idea to have a random offset? 
    * Does it really lead to a perfectly uniform distribution over the sequences on the document?
    * What would you have to do to make things even more uniform?
1. If we want a sequence example to be a complete sentence, what kinds of problems does this introduce in mini-batch sampling? Why would we want to do this anyway?

## Discuss on our Forum

<div id="discuss" topic_id="2363"></div>
