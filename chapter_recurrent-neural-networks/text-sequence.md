# Text Sequences
:label:`sec_text-sequence`

We have reviewed and evaluated
statistical tools
and prediction challenges
for sequence data.
Such data can take many forms.
Specifically,
as we will focus on
in many chapters of the book,
text is one of the most popular examples of sequence data.
For example,
an article can be simply viewed as a sequence of words, or even a sequence of characters.
To facilitate our future experiments
with sequence data,
we will dedicate this section
to explain common preprocessing steps for text.
Usually, these steps are:

1. Load text as strings into memory.
1. Split strings into tokens (e.g., words and characters).
1. Build a table of vocabulary to map the split tokens to numerical indices.
1. Convert text into sequences of numerical indices so they can be manipulated by models easily.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Reading the Dataset

To get started, we load text 
from H. G. Wells' [*The Time Machine*](http://www.gutenberg.org/ebooks/35).
This is a fairly small corpus of just over 30000 words, but for the purpose of what we want to illustrate this is just fine. More realistic document collections contain many billions of words.
The following function 
(**reads the raw text into a string**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root, 
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

## Preprocessing

For simplicity, we ignore punctuation and capitalization when preprocessing the raw text.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenization


*Tokens* are the atomic (indivisible) units of text
and what constitutes a token 
(e.g., characters or words)
is a design choice.
Although originated from natural language processing,
the concept of tokens is also getting popular in
computer vision, 
such as for referring to image patches :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Below, we tokenize our preprocessed text into characters.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Vocabulary

While these tokens are still strings,
our models require numerical inputs.
[**Thus, we will need a class
to construct a *vocabulary*
that assigns a unique index 
to each distinct token.**]
To this end,
we first count the unique tokens in all the documents from the training set, namely a *corpus*,
and then assign a numerical index to each unique token.
Rarely appeared tokens are often removed to reduce the complexity. Any token that does not exist in the corpus or has been removed is mapped into a special unknown token "&lt;unk&gt;". 
In the future,
we may supplement the vocabulary
with a list of reserved tokens.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)        
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]        
    
    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

We can now [**construct the vocabulary**] for our dataset, 
using it to convert a text sequence
into a list of numerical indices.
Note that we have not lost any information
and can easily convert our dataset 
back to its original (string) representation.

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indicies = vocab[tokens[:10]]
print('indices:', indicies)
print('words:', vocab.to_tokens(indicies))
```

## Putting All Things Together

Using the above classes and methods, we [**package everything into the following `build` method of the `TimeMachine` class**], which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of *The Time Machine* corpus.
The modifications we did here are:
(i) we tokenize text into characters, not words, to simplify the training in later sections;
(ii) `corpus` is a single list, not a list of token lists, since each text line in *The Time Machine* dataset is not necessarily a sentence or paragraph.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):    
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Natural Language Statistics
:label:`subsec_natural-lang-stat`


Using the real corpus and the `Vocab` class
defined above,
let's also investigate
word token statistics.
We construct a vocabulary based on *The Time Machine* corpus and print the top 10 most frequent words.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

As we can see, (**the most popular words are**) actually quite boring to look at.
They are often referred to as (***stop words***) and thus filtered out.
Nonetheless, they still carry meaning and we will still use them.
Besides, it is quite clear that the word frequency decays rather rapidly. The $10^{\mathrm{th}}$ most frequent word is less than $1/5$ as common as the most popular one. To get a better idea, we [**plot the figure of the word frequency**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

We are on to something quite fundamental here: the word frequency decays rapidly in a well-defined way.
After dealing with the first few words as exceptions, all the remaining words roughly follow a straight line on a log-log plot. This means that words satisfy *Zipf's law*,
which states that the frequency $n_i$ of the $i^\mathrm{th}$ most frequent word
is:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

which is equivalent to

$$\log n_i = -\alpha \log i + c,$$

where $\alpha$ is the exponent that characterizes the distribution and $c$ is a constant.
This should already give us pause if we want to model words by counting statistics.
After all, we will significantly overestimate the frequency of the tail, also known as the infrequent words. But [**what about the other word combinations, such as two consecutive words (bigrams), three consecutive words (trigrams)**], and beyond?
Let's see whether the bigram frequency behaves in the same manner as the single word (unigram) frequency.

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

One thing is notable here. Out of the ten most frequent word pairs, nine are composed of both stop words and only one is relevant to the actual book---"the time". Furthermore, let's see whether the trigram frequency behaves in the same manner.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Last, let's [**visualize the token frequency**] among these three models: unigrams, bigrams, and trigrams.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

This figure is quite exciting.
First, beyond unigram words, sequences of words also appear to be following Zipf's law, albeit with a smaller exponent $\alpha$ in :eqref:`eq_zipf_law`, depending on the sequence length.
Second, the number of distinct $n$-grams is not that large. This gives us hope that there is quite a lot of structure in language.
Third, many $n$-grams occur very rarely.
This makes certain methods unsuitable for language modeling and motivates the use of deep learning models.
We will discuss this in the next section.


## Summary

* Text is an important form of sequence data.
* To preprocess text, we usually split text into tokens, build a vocabulary to map token strings into numerical indices, and convert text data into token indices for  models to manipulate.
* Zipf's law governs the word distribution for not only unigrams but also the other $n$-grams.


## Exercises

1. Tokenization is a key preprocessing step. It varies for different languages. Try to find another three commonly used methods to tokenize text.
1. In the experiment of this section, tokenize text into words and vary the `min_freq` argument value of the `Vocab` instance. How does this affect the vocabulary size?
1. Estimate the exponent of Zipf’s law for unigrams, bigrams, and trigrams.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
