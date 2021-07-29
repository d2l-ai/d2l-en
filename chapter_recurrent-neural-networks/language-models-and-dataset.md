# The Language Model Dataset
:label:`sec_language_model`

In :numref:`sec_text_preprocessing`, we see how to map text data into tokens, where these tokens can be viewed as a sequence of discrete observations, such as words or characters. Assume that the tokens in a text sequence of length $T$ are in turn $x_1, x_2, \ldots, x_T$.
The goal of *language models*
is to estimate the joint probability of the whole sequence:

$$P(x_1, x_2, \ldots, x_T),$$

where statistical tools
in :numref:`sec_sequence`
can be applied.

Language models are incredibly useful. For instance, an ideal language model would be able to generate natural text just on its own, simply by drawing one token at a time $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Quite unlike the monkey using a typewriter, all text emerging from such a model would pass as natural language, e.g., English text. Furthermore, it would be sufficient for generating a meaningful dialog, simply by conditioning the text on previous dialog fragments.
Clearly we are still very far from designing such a system, since it would need to *understand* the text rather than just generate grammatically sensible content.

Nonetheless, language models are of great service even in their limited form.
For instance, the phrases "to recognize speech" and "to wreck a nice beach" sound very similar.
This can cause ambiguity in speech recognition,
which is easily resolved through a language model that rejects the second translation as outlandish.
Likewise, in a document summarization algorithm
it is worthwhile knowing that "dog bites man" is much more frequent than "man bites dog", or that "I want to eat grandma" is a rather disturbing statement, whereas "I want to eat, grandma" is much more benign.

In the rest of the chapter, we focus on  using neural networks for language modeling
based on *The Time Machine* dataset.
Before introducing the model,
let us assume that it
processes a minibatch of sequences with predefined length
at a time.
Now the question is how to [**read minibatches of input sequences and label sequences at random.**]

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=2}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input  n=3}
#@tab all
corpus, vocab = d2l.load_corpus_time_machine()
```

## Partitioning Sequences

Now the dataset takes the form of a sequence of $T$ token indices in `corpus`.
We will
partition it
into subsequences, where each subsequence has $n$ tokens (time steps).
To iterate over 
(almost) all the tokens of the entire dataset 
for each epoch
and obtain all possible length-$n$ subsequences,
we can introduce randomness.
More concretely,
at the beginning of each epoch,
discard the first $d$ tokens,
where $d\in [0,n)$ is uniformly sampled at random.
The rest of the sequence
is then partitioned
into $m=\lfloor (T-d)/n \rfloor$ subsequences.
Denote by $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$ the length-$n$ subsequence starting from token $x_t$ at time step $t$. 
The resulting $m$ partitioned subsequences
are 
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}.$
Each subsequence will be used as an input sequence into the language model.


For language modeling,
the target is to predict the next token based on what tokens we have seen so far, hence the labels are the original sequence, shifted by one token.
The label sequence for any input sequence $\mathbf x_t$
is $\mathbf x_{t+1}$ with length $n$.

![Obtaining 5 pairs of input sequences and label sequences from partitioned length-5 subsequences.](../img/lang-model-data.svg) 
:label:`fig_lang_model_data`

:numref:`fig_lang_model_data` shows an example of obtaining 5 pairs of input sequences and label sequences with $n=5$ and $d=2$. 


## [**Random Sampling**]


To train language models,
we will randomly sample 
pairs of input sequences and label sequences
in minibatches.
The following data loader randomly generates a minibatch from the dataset each time.
The argument `batch_size` specifies the number of subsequence examples (`self.b`) in each minibatch
and `num_steps` is the subsequence length in tokens (`self.n`).

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """The sequence data iterator generating minibatches of subsequences."""
    def __init__(self, corpus, batch_size, num_steps):
        self.corpus, self.b, self.n = corpus, batch_size, num_steps

    def __iter__(self):
        # Randomly drop the first d tokens.
        corpus = self.corpus[random.randint(0, self.n - 1):]
        # No. of subsequences. Subtract 1 to account for labels.
        m = (len(corpus)-1) // self.n
        # The starting indices for input sequences.
        initial_indices = list(range(0, m*self.n, self.n))
        random.shuffle(initial_indices)
        for i in range(0, m // self.b):
            # The randomized starting indices for this minibatch.
            batch_indicies = initial_indices[i*self.b : (i+1) * self.b]
            X = [corpus[j : j+self.n] for j in batch_indicies]
            Y = [corpus[j+1 : j+1+self.n] for j in batch_indicies]
            yield d2l.tensor(X), d2l.tensor(Y)
```

Let us [**manually generate a sequence from 0 to 34.**]
We assume that
the batch size and numbers of time steps are 3 and 5,
respectively.
This means that we can generate $\lfloor (35 - 1) / 5 \rfloor= 6$ feature-label subsequence pairs. With a minibatch size of 3, we only get 2 minibatches.

```{.python .input}
#@tab all
for X, Y in SeqDataLoader(list(range(35)), batch_size=3, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

[**Last, we define a function `load_data_time_machine` that returns both the data iterator and the vocabulary**], so we can use it similarly as other other functions with the `load_data` prefix, such as `d2l.load_data_fashion_mnist` defined in :numref:`sec_fashion_mnist`.

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps, max_tokens=10000):  #@save
    """Return the iterator and the vocabulary of the time machine dataset."""
    corpus, vocab = d2l.load_corpus_time_machine(max_tokens)
    data_iter = SeqDataLoader(corpus, batch_size, num_steps)
    return data_iter, vocab
```

## Summary

* Language models estimate the joint probability of a text sequence.
* To train language models, we can randomly sample pairs of input sequences and label sequences in minibatches.


## Exercises

1. How would you model a dialogue?
1. What other methods can you think of for reading long sequence data?
1. Consider our method for discarding a uniformly random number of the first few tokens at the beginning of each epoch.
    1. Does it really lead to a perfectly uniform distribution over the sequences on the document?
    1. What would you have to do to make things even more uniform? 
1. If we want a sequence example to be a complete sentence, what kind of problem does this introduce in minibatch sampling? How can we fix the problem?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
