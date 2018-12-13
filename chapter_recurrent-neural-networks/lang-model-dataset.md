# Language Model Data Sets (Time Machine)

@TODO(smolix/astonzhang): the data set was just changed from lyrics to time machine, so descriptions/hyperparameters have to change.

This section describes how to preprocess a language model data set and convert it to the input format required for a character-level recurrent neural network. To this end, we collected Jay Chou's lyrics from his first album "Jay" to his tenth album "The Era". In subsequent chapters, we will a recurrent neural network to train a language model on this data set. Once the model is trained, we can use it to write lyrics.

## Read the Data Sets

First, read this data set and see what the first 40 characters look like.

```{.python .input  n=1}
from mxnet import nd
import random
import zipfile

with open('../data/timemachine.txt') as f:
    corpus_chars = f.read()
corpus_chars[:40]
```

This data set has more than 50,000 characters. For ease of printing, we replaced line breaks with spaces and then used only the first 10,000 characters to train the model.

```{.python .input  n=2}
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ').lower()
corpus_chars = corpus_chars[0:10000]
```

## Establish a Character Index

We map each character to continuous integers starting from 0, also known as an index, to facilitate subsequent data processing. To get the index, we extract all the different characters in the data set and then map them to the index one by one to construct the dictionary. Then, print `vocab_size`, which is the number of different characters in the dictionary, i.e. the dictionary size.

```{.python .input  n=3}
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
vocab_size
```

After that, each character in the training data set is converted into an index, and the first 20 characters and their corresponding indexes are printed.

```{.python .input  n=4}
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

We packaged the above code in the `load_data_jay_lyrics` function of the `gluonbook` package for to facilitate calling in later chapters. After calling this function, we will get four variables in turn, `corpus_indices`, `char_to_idx`, `idx_to_char`, and `vocab_size`.

## Time Series Data Sampling

During training, we need to randomly read mini-batches of examples and labels each time. Unlike the experimental data from the previous chapter, a timing data instance usually contains consecutive characters. Assume that there are 5 time steps and the example sequence is 5 characters: "I", "want", "to", "have", "a". The label sequence of the example is the character that follows these characters in the training set: "want", "to", "have", "a", "helicopter". We have two ways to sample timing data, random sampling and adjacent sampling.

### Random sampling

The following code randomly samples a mini--batch from the data each time. Here, the batch size `batch_size` indicates to the number of examples in each mini-batch and `num_steps` is the number of time steps included in each example.
In random sampling, each example is a sequence arbitrarily captured on the original sequence. The positions of two adjacent random mini-batches on the original sequence are not necessarily adjacent. Therefore, we cannot initialize the hidden state of the next mini-batch with the hidden state of final time step of the previous mini-batch. When training the model, the hidden state needs to be reinitialized before each random sampling.

```{.python .input  n=5}
# This function is saved in the gluonbook package for future use.
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # We subtract one because the index of the output is the index of the corresponding input plus one.
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # This returns a sequence of the length num_steps starting from pos.
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # batch_size indicates the random examples read each time.
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)
```

Let us input an artificial sequence from 0 to 29. We assume the batch size and numbers of time steps are 2 and 6 respectively. Then we print input `X` and label `Y` for each mini-batch of examples read by random sampling. As we can see, the positions of two adjacent random mini-batches on the original sequence are not necessarily adjacent.

```{.python .input  n=6}
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

### Adjacent sampling

In addition to random sampling of the original sequence, we can also make the positions of two adjacent random mini-batches adjacent on the original sequence. Now, we can use a hidden state of the last time step of a mini-batch to initialize the hidden state of the next mini-batch, so that the output of the next mini-batch is also dependent on the input of the mini-batch, with this pattern continuing in subsequent mini-batches. This has two effects on the implementation of recurrent neural network. On the one hand,
when training the model, we only need to initialize the hidden state at the beginning of each epoch.
On the other hand, when multiple adjacent mini-batches are concatenated by passing hidden states, the gradient calculation of the model parameters will depend on all the mini-batch sequences that are concatenated. In the same epoch as the number of iterations increases, the costs of gradient calculation rise.
So that the model parameter gradient calculations only depend on the mini-batch sequence read by one iteration, we can separate the hidden state from the computational graph before reading the mini-batch. We will gain a deeper understand this approach in the following sections.

```{.python .input  n=7}
# This function is saved in the gluonbook package for future use.
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```

Using the same settings, print input `X` and label `Y` for each mini-batch of examples read by random sampling. The positions of two adjacent random mini-batches on the original sequence are adjacent.

```{.python .input  n=8}
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

## Summary

* Timing data sampling methods include random sampling and adjacent sampling. These two methods are implemented slightly differently in recurrent neural network model training.

## Problems

* What other mini-batch data sampling methods can you think of?
* If we want a sequence example to be a complete sentence, what kinds of problems does this introduce in mini-batch sampling?

## Discuss on our Forum

<div id="discuss" topic_id="2363"></div>
