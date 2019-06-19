# Language Models and Data Sets

:label:`chapter_language_model`


In :numref:`chapter_text_preprocessing`, we see how to map text data into tokens, and these tokens can be viewed as a time series of discrete observations. Assuming the tokens in a text of length $T$ are in turn $x_1, x_2, \ldots, x_T$, then, in the discrete time series, $x_t$($1 \leq t \leq T$) can be considered as the output or label of time step $t$. Given such a sequence, the goal of a language model is to estimate the probability

$$p(x_1,x_2, \ldots, x_T).$$

Language models are incredibly useful. For instance, an ideal language model would be able to generate natural text just on its own, simply by drawing one word at a time $w_t \sim p(w_t|w_{t-1}, \ldots w_1)$. Quite unlike the monkey using a typewriter, all text emerging from such a model would pass as natural language, e.g. English text. Furthermore, it would be sufficient for generating a meaningful dialog, simply by conditioning the text on previous dialog fragments. Clearly we are still very far from designing such a system, since it would need to *understand* the text rather than just generate grammatically sensible content.

Nonetheless language models are of great service even in their limited form. For instance, the phrases *'to recognize speech'* and *'to wreck a nice beach'* sound very similar. This can cause ambiguity in speech recognition, ambiguity that is easily resolved through a language model which rejects the second translation as outlandish. Likewise, in a document summarization algorithm it's worth while knowing that *'dog bites man'* is much more frequent than *'man bites dog'*, or that *'let's eat grandma'* is a rather disturbing statement, whereas *'let's eat, grandma'* is much more benign.

## Estimating a language model

The obvious question is how we should model a document, or even a sequence of words. We can take recourse to the analysis we applied to sequence models in the previous section. Let's start by applying basic probability rules:

$$p(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T p(w_t | w_1, \ldots, w_{t-1}).$$

For example, the probability of a text sequence containing four tokens consisting of words and punctuation would be given as:

$$p(\mathrm{Statistics}, \mathrm{is},  \mathrm{fun}, \mathrm{.}) =  p(\mathrm{Statistics}) p(\mathrm{is} | \mathrm{Statistics}) p(\mathrm{fun} | \mathrm{Statistics}, \mathrm{is}) p(\mathrm{.} | \mathrm{Statistics}, \mathrm{is}, \mathrm{fun}).$$

In order to compute the language model, we need to calculate the
probability of words and the conditional probability of a word given
the previous few words, i.e. language model parameters. Here, we
assume that the training data set is a large text corpus, such as all
Wikipedia entries, Project Gutenberg, or all text posted online on the
web. The probability of words can be calculated from the relative word
frequency of a given word in the training data set.

For example, $p(\mathrm{Statistics})​$ can be calculated as the
probability of any sentence starting with the word 'statistics'. A
slightly less accurate approach would be to count all occurrences of
the word 'statistics' and divide it by the total number of words in
the corpus. This works fairly well, particularly for frequent
words. Moving on, we could attempt to estimate

$$\hat{p}(\mathrm{is}|\mathrm{Statistics}) = \frac{n(\mathrm{Statistics~is})}{n(\mathrm{Statistics})}.$$

Here $n(w)$ and $n(w, w')$ are the number of occurrences of singletons
and pairs of words respectively. Unfortunately, estimating the
probability of a word pair is somewhat more difficult, since the
occurrences of *'Statistics is'* are a lot less frequent. In
particular, for some unusual word combinations it may be tricky to
find enough occurrences to get accurate estimates. Things take a turn for the worse for 3 word combinations and beyond. There will be many plausible 3-word combinations that we likely won't see in our dataset. Unless we provide some solution to give such word combinations nonzero weight we will not be able to use these as a language model. If the dataset is small or if the words are very rare, we might not find even a single one of them.

A common strategy is to perform some form of Laplace smoothing. We already
encountered this in our discussion of
naive bayes in :numref:`chapter_naive_bayes` where the solution was to
add a small constant to all counts. This helps with singletons, e.g. via

$$\begin{aligned}
	\hat{p}(w) & = \frac{n(w) + \epsilon_1/m}{n + \epsilon_1} \\
	\hat{p}(w'|w) & = \frac{n(w,w') + \epsilon_2 \hat{p}(w')}{n(w) + \epsilon_2} \\
	\hat{p}(w''|w',w) & = \frac{n(w,w',w'') + \epsilon_3 \hat{p}(w',w'')}{n(w,w') + \epsilon_3}
\end{aligned}$$

Here the coefficients $\epsilon_i > 0$ determine how much we use the
estimate for a shorter sequence as a fill-in for longer
ones. Moreover, $m$ is the total number of words we encounter. The
above is a rather primitive variant of what is Kneser-Ney smoothing
and Bayesian Nonparametrics can accomplish. See e.g. the Sequence
Memoizer of Wood et al., 2012 for more details of how to accomplish
this. Unfortunately, models like this get unwieldy rather quickly:
first off, we need to store all counts and secondly, this entirely
ignores the meaning of the words. For instance, *'cat'* and *'feline'*
should occur in related contexts. Deep learning based language models
are well suited to take this into account. This, it is quite difficult
to adjust such models to additional context. Lastly, long word
sequences are almost certain to be novel, hence a model that simply
counts the frequency of previously seen word sequences is bound to
perform poorly there.


## Markov Models and $n$-grams

Before we discuss solutions involving deep learning we need some more terminology and concepts. Recall our discussion of Markov Models in the previous section. Let's apply this to language modeling. A distribution over sequences satisfies the Markov property of first order if $p(w_{t+1}|w_t, \ldots w_1) = p(w_{t+1}|w_t)$. Higher orders correspond to longer dependencies. This leads to a number of approximations that we could apply to model a sequence:

$$
\begin{aligned}
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2) p(w_3) p(w_4)\\
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2 | w_1) p(w_3 | w_2) p(w_4 | w_3)\\
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2 | w_1) p(w_3 | w_1, w_2) p(w_4 | w_2, w_3)
\end{aligned}
$$

Since they involve one, two or three terms, these are typically referred to as unigram, bigram and trigram models. In the following we will learn how to design better models.

## Natural Language Statistics

Let's see how this works on real data. We construct a vocabulary based on the time machine data similar to :numref:`chapter_text_preprocessing` and print the top words

```{.python .input  n=1}
import d2l
from mxnet import np, npx
import random
npx.set_np()

tokens = d2l.tokenize(d2l.read_time_machine())
vocab = d2l.Vocab(tokens)
print(vocab.token_freqs[:10])
```

As we can see, the most popular words are actually quite boring to look at. They are often referred to as [stop words](https://en.wikipedia.org/wiki/Stop_words) and thus filtered out. That said, they still carry meaning and we will use them nonetheless. However, one thing that is quite clear is that the word frequency decays rather rapidly. The 10th word is less than $1/5$ as common as the most popular one. To get a better idea we plot the graph of word frequencies.

```{.python .input  n=2}
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token', ylabel='frequency',
         xscale='log', yscale='log')
```

We're on to something quite fundamental here - the word frequencies decay rapidly in a well defined way. After dealing with the first four words as exceptions ('the', 'i', 'and', 'of'), all remaining words follow a straight line on a log-log plot. This means that words satisfy [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) which states that the item frequency is given by

$$n(x) \propto (x + c)^{-\alpha} \text{ and hence }
\log n(x) = -\alpha \log (x+c) + \mathrm{const.}​$$

This should already give us pause if we want to model words by count statistics and smoothing. After all, we will significantly overestimate the frequency of the tail, aka the infrequent words. But what about word pairs (and trigrams and beyond)? Let's see.

```{.python .input  n=3}
bigram_tokens = [[pair for pair in zip(line[:-1], line[1:])] for line in tokens]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])
```

Two things are notable. Out of the 10 most frequent word pairs, 9 are composed of stop words and only one is relevant to the actual book - 'the time'. Let's see whether the bigram frequencies behave in the same manner as the unigram frequencies.

```{.python .input  n=4}
trigram_tokens = [[triple for triple in zip(line[:-2], line[1:-1], line[2:])]
                  for line in tokens]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
```

Last, let's visualize the token frequencies among these three gram models.

```{.python .input  n=5}
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token',
         ylabel='frequency', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

The graph is quite exciting for a number of reasons. Firstly, beyond words, also sequences of words appear to be following Zipf's law, albeit with a lower exponent, depending on sequence length. Secondly, the number of distinct n-grams is not that large. This gives us hope that there is quite a lot of structure in language. Third, *many* n-grams occur very rarely, which makes Laplace smoothing rather unsuitable for language modeling. Instead, we will use deep learning based models.

## Training Data Preparation

Before introducing the model, let's assume we will use a neural network to train a language model. Now the question is how to read mini-batches of examples and labels at
random. Since sequence data is by its very nature sequential, we need to address
the issue of processing it. We did so in a rather ad-hoc manner when we
introduced in :numref:`chapter_sequence`. Let's formalize this a bit. 

In :numref:`fig_timemachine_5gram`, we visualized several possible ways to obtain 5-grams in a sentence, here a token is a character. Note that we have quite some freedom since we could pick an arbitrary offset.

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

In fact, any one of these offsets is fine. Hence, which one should we pick? In fact, all of them are equally good. But if we pick all offsets we end up with rather redundant data due to overlap, particularly if the sequences are long. Picking just a random set of initial positions is no good either since it does not guarantee uniform coverage of the array. For instance, if we pick $n​$ elements at random out of a set of $n​$ with random replacement, the probability for a particular element not being picked is $(1-1/n)^n \to e^{-1}​$. This means that we cannot expect uniform coverage this way. Even randomly permuting a set of all offsets does not offer good guarantees. Instead we can use a simple trick to get both *coverage* and *randomness*: use a random offset, after which one uses the terms sequentially. We describe how to accomplish this for both random sampling and sequential partitioning strategies below.

### Random Sampling

The following code randomly generates a minibatch from the data each time. Here, the batch size `batch_size` indicates to the number of examples in each mini-batch and `num_steps` is the length of the sequence (or time steps if we have a time series) included in each example.
In random sampling, each example is a sequence arbitrarily captured on the original sequence. The positions of two adjacent random mini-batches on the original sequence are not necessarily adjacent. The target is to predict the next character based on what we've seen so far, hence the labels are the original sequence, shifted by one character.

```{.python .input  n=5}
# Save to the d2l package.
def seq_data_iter_random(corpus, batch_size, num_steps):
    # Offset the iterator over the data for uniform starts
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract 1 extra since we need to account for label
    num_examples = ((len(corpus) - 1) // num_steps)
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)
    # This returns a sequence of the length num_steps starting from pos
    data = lambda pos: corpus[pos: pos + num_steps]
    # Discard half empty batches
    num_batches = num_examples // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Batch_size indicates the random examples read each time
        batch_indices = example_indices[i:(i+batch_size)]
        X = [data(j) for j in batch_indices]
        Y = [data(j + 1) for j in batch_indices]
        yield np.array(X), np.array(Y)
```

Let us generate an artificial sequence from 0 to 30. We assume that
the batch size and numbers of time steps are 2 and 5
respectively. This means that depending on the offset we can generate between 4 and 5 $(x,y)$ pairs. With a minibatch size of 2 we only get 2 minibatches.

```{.python .input  n=6}
my_seq = list(range(30))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y)
```

### Sequential partitioning

In addition to random sampling of the original sequence, we can also make the positions of two adjacent random mini-batches adjacent in the original sequence.

```{.python .input  n=7}
# Save to the d2l package.
def seq_data_iter_consecutive(corpus, batch_size, num_steps):
    # Offset for the iterator over the data for uniform starts
    offset = random.randint(0, num_steps)
    # Slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset+num_indices])
    Ys = np.array(corpus[offset+1:offset+1+num_indices])
    Xs, Ys = Xs.reshape((batch_size, -1)), Ys.reshape((batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:,i:(i+num_steps)]
        Y = Ys[:,i:(i+num_steps)]
        yield X, Y
```

Using the same settings, print input `X` and label `Y` for each mini-batch of examples read by random sampling. The positions of two adjacent random mini-batches on the original sequence are adjacent.

```{.python .input  n=8}
for X, Y in seq_data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y)
```

Now we wrap the above two sampling functions to a class so that we can use it as a normal Gluon data iterator later.

```{.python .input}
# Save to the d2l package.
class SeqDataLoader(object):
    """A iterator to load sequence data"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            data_iter_fn = d2l.seq_data_iter_random
        else:
            data_iter_fn = d2l.seq_data_iter_consecutive
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.get_iter = lambda: data_iter_fn(self.corpus, batch_size, num_steps)

    def __iter__(self):
        return self.get_iter()
```

Lastly, we define a function `load_data_time_machine` that returns both the data iterator and the vocabulary, so we can use it similarly as other functions with `load_data` prefix.

```{.python .input}
# Save to the d2l package.
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, 
                           max_tokens=10000):
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab    
```

## Summary

* Language models are an important technology for natural language processing.
* $n$-grams provide a convenient model for dealing with long sequences by truncating the dependence.
* Long sequences suffer from the problem that they occur very rarely or never. 
* Zipf's law governs the word distribution for both unigrams and n-grams.
* There's a lot of structure but not enough frequency to deal with infrequent word combinations efficiently via smoothing.
* The main choices for sequence partitioning are whether we pick consecutive or random sequences. 
* Given the overall document length, it is usually acceptable to be slightly wasteful with the documents and discard half-empty minibatches.

## Exercises

1. Suppose there are 100,000 words in the training data set. How many word frequencies and multi-word adjacent frequencies does a four-gram need to store?
1. Review the smoothed probability estimates. Why are they not accurate? Hint - we are dealing with a contiguous sequence rather than singletons.
1. How would you model a dialogue?
1. Estimate the exponent of Zipf's law for unigrams, bigrams and trigrams.
1. Which other other mini-batch data sampling methods can you think of?
1. Why is it a good idea to have a random offset?
    * Does it really lead to a perfectly uniform distribution over the sequences on the document?
    * What would you have to do to make things even more uniform?
1. If we want a sequence example to be a complete sentence, what kinds of problems does this introduce in mini-batch sampling? Why would we want to do this anyway?


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2361)

![](../img/qr_lang-model.svg)
