# Text Preprocessing
:label:`sec_text_preprocessing`

We have reviewed and evaluated
statistical tools
and prediction challenges
for sequence data.
Such data can take many forms.
We already saw how to deal with a sequence with numbers. 
Another popular example of sequence data is text. 
For example,
an article can be simply viewed as a sequence of words, or even a sequence of characters.
We will use text in many chapters of the book.
To facilitate our future experiments with text data, 
we will dedicate this section
to explain common preprocessing steps to convert text into sequences of numerical indices so they can be manipulated by models easily.

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## Reading the Dataset

To get started we load text from H. G. Wells' [*The Time Machine*](http://www.gutenberg.org/ebooks/35).
This is a fairly small corpus of just over 30000 words, but for the purpose of what we want to illustrate this is just fine.
More realistic document collections contain many billions of words.
The following function (**reads the dataset into a list of text lines**), where each line is a string.
For simplicity, here we ignore punctuation and capitalization.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## Tokenization

The following `tokenize` function
takes a list (`lines`) as the input,
where each element is a text sequence (e.g., a text line).
[**Each text sequence is split into a list of tokens**].
A *token* is the basic unit in text.
In the end,
a list of token lists are returned,
where each token is a string.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    assert token in ('word', 'char'), 'unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]

tokens = tokenize(lines)
for i in range(7, 10):
    print(f'line {i+1}: {tokens[i]}')
```

## Vocabulary

The string type of the token is inconvenient to be used by models, which take numerical inputs.
Now let us [**build a dictionary, often called *vocabulary* as well, to map string tokens into numerical indices starting from 0**].
To do so, we first count the unique tokens in all the documents from the training set,
namely a *corpus*,
and then assign a numerical index to each unique token.
Rarely appeared tokens are often removed to reduce the complexity.
Any token that does not exist in the corpus or has been removed is mapped into a special unknown token “&lt;unk&gt;”.
We optionally add a list of reserved tokens for future uses. 

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Expand a list of list into a list if needed.
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies.
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens. Sorting it for a deterministic mapping.
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
```

We [**construct a vocabulary**] using the time machine dataset as the corpus.
Then we can convert a text line into a list of numerical indices, and also convert them back into tokens. 

```{.python .input}
#@tab all
vocab = Vocab(tokens)
indicies = vocab[tokens[0]]
print('indices:', indicies)
print('words:', vocab.to_tokens(indicies))
```

## Putting All Things Together

Using the above functions, we [**package everything into the `load_corpus_time_machine` function**], which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of the time machine corpus.
The modifications we did here are:
(i) we tokenize text into characters, not words, to simplify the training in later sections;
(ii) `corpus` is a single list, not a list of token lists, since each text line in the time machine dataset is not necessarily a sentence or a paragraph.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=None):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens and max_tokens > 0: corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Summary

* Text is an important form of sequence data.
* To preprocess text, we usually split text into tokens, build a vocabulary to map token strings into numerical indices, and convert text data into token indices for  models to manipulate.


## Exercises

1. Tokenization is a key preprocessing step. It varies for different languages. Try to find another three commonly used methods to tokenize text.
1. In the experiment of this section, tokenize text into words and vary the `min_freq` arguments of the `Vocab` instance. How does this affect the vocabulary size?

[Discussions](https://discuss.d2l.ai/t/115)
