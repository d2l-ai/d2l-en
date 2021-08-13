```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "89dc8081d5be4cd199eaab3df1eff094",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "interactive(children=(Dropdown(description='tab', options=('mxnet', 'pytorch', 'tensorflow'), value='mxnet'), \u2026"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
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

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"mxnet\" cell."
 }
]
```

```{.python .input  n=4}
%%tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"mxnet\" cell."
 }
]
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
    def load(self):
        fname = d2l.download(d2l.DATA_URL+'timemachine.txt', self.root, 
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            lines = f.readlines()
            return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() 
                    for line in lines]

data = TimeMachine()
lines = data.load()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "# text lines: 3221\nthe time machine by h g wells\ntwinkled and his usually pale face was flushed and animated the\n"
 }
]
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
def tokenize(self, lines):
    return [list(line) for line in lines]

tokens = data.tokenize(lines)
for i in range(7, 10):
    print(f'line {i+1}: {tokens[i][:12]}')
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "line 8: []\nline 9: ['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 't', 'r', 'a']\nline 10: ['w', 'a', 's', ' ', 'e', 'x', 'p', 'o', 'u', 'n', 'd', 'i']\n"
 }
]
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
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
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

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "indices: [21, 9, 6, 0, 21, 10, 14, 6, 0, 14, 2, 4, 9, 10, 15, 6, 0, 3, 26, 0, 9, 0, 8, 0, 24, 6, 13, 13, 20]\nwords: ['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 'm', 'a', 'c', 'h', 'i', 'n', 'e', ' ', 'b', 'y', ' ', 'h', ' ', 'g', ' ', 'w', 'e', 'l', 'l', 's']\n"
 }
]
```

## Putting All Things Together

Using the above functions, we [**package everything into the `load_corpus_time_machine` function**], which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of *The Time Machine* corpus.
The modifications we did here are:
(i) we tokenize text into characters, not words, to simplify the training in later sections;
(ii) `corpus` is a single list, not a list of token lists, since each text line in *The Time Machine* dataset is not necessarily a sentence or a paragraph.

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def prepare_data(self):
    tokens = self.tokenize(self.load())
    self.vocab = Vocab(tokens)
    self.corpus = [self.vocab[token] for line in tokens for token in line]

data.prepare_data()
len(data.corpus), len(data.vocab)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "(170580, 28)"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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
