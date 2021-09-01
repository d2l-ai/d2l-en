```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Text Preprocessing
:label:`sec_text_preprocessing`

Sequence data can take many forms.
For example, text documents
can be viewed as sequences of words.
Alternatively, they can be viewed
as sequences of characters.
Throughout this book we will work extensively
with sequential representations of text.
To lay the groundwork for what follows,
we briefly explain some common preprocessing steps
for converting raw text into 
sequences of numerical values
that our models can ingest.

```{.python .input  n=2}
%%tab mxnet
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input  n=3}
%%tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input  n=4}
%%tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## Reading the Dataset

To get started, we load text 
from H. G. Wells' [*The Time Machine*](http://www.gutenberg.org/ebooks/35).
This book contains just over 30000 words,
so we can load them into memory.
The following function 
(**reads the lines of text into a list**),
where each line is represented as a string.
For simplicity, we ignore punctuation and capitalization.

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    def _download(self):
        fname = d2l.download(d2l.DATA_URL+'timemachine.txt', self.root, 
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

## Preprocessing

```{.python .input}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenization

The following `tokenize` function
takes a list (`lines`) as input,
where each element is a line of text.
[**We then split each line into a list of tokens**].
*Tokens* are the atomic (indivisible) units of text
and what constitutes a token 
(e.g., characters or words)
is a design choice.
Below, we tokenize our lines into words.

```{.python .input  n=6}
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
[**To this end, we will need a class
to construct a *vocabulary*
that assigns a unique index 
to each distinct token value.**]
First, we count the occurrences 
of each element of the vocabulary,
lumping the rarest ones all together
into a special value "&lt;unk&gt;" (unknown token).
In the future,
we may supplement the vocabulary
with a list of reserved tokens.

```{.python .input  n=7}
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
using it to convert each text line 
from a list of tokens into a list of indices.
Note that we have not lost any information
and can easily convert our dataset 
back to its original (string) representation.

```{.python .input  n=8}
%%tab all
vocab = Vocab(tokens)
indicies = vocab[tokens[0]]
print('indices:', indicies)
print('words:', vocab.to_tokens(indicies))
```

## Putting All Things Together

Using the above functions, we [**package everything into the `load_corpus_time_machine` function**], which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of *The Time Machine* corpus.
The modifications we did here are:
(i) we tokenize text into characters, not words, to simplify the training in later sections;
(ii) `corpus` is a single list, not a list of token lists, since each text line in *The Time Machine* dataset is not necessarily a sentence or a paragraph.

```{.python .input  n=9}
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

## Summary

* Text is an important form of sequence data.
* To preprocess text, we usually split text into tokens, build a vocabulary to map token strings into numerical indices, and convert text data into token indices for  models to manipulate.


## Exercises

1. Tokenization is a key preprocessing step. It varies for different languages. Try to find another three commonly used methods to tokenize text.
1. In the experiment of this section, tokenize text into words and vary the `min_freq` arguments of the `Vocab` instance. How does this affect the vocabulary size?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/115)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3857)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3858)
:end_tab:
