# Text Preprocessing
:label:`chapter_text_preprocessing`

Text is an important example of sequence data. An article can be simply viewed as a sequence of words or a sequence of characters. Given text data is a major data format besides images we are using in this book, this section will dedicate to explain the common preprocessing steps for text data. Such preprocessing often consists of four steps:

1. Loads texts as strings into memory.
1. Splits strings into tokens, a token could be a word or a character. 
1. Builds a vocabulary for these tokens to map them into numerical indices. 
1. Maps all tokens in the data into indices to facilitate to feed into models. 

## Data Loading

To get started we load text from H. G. Wells' [Time Machine](http://www.gutenberg.org/ebooks/35). This is a fairly small corpus of just over 30,000 words but for the purpose of what we want to illustrate this is just fine. More realistic document collections contain many billions of words. The following function read the dataset into a list of sentences, each sentence is a string. Here we ignore punctuation and capitalization.

```{.python .input}
import collections
import re

# Save to the d2l package. 
def read_time_machine():
    """Load the time machine book into a list of sentences."""
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower()) 
            for line in lines]

lines = read_time_machine()
'# sentences %d' % len(lines)
```

## Tokenization

For each sentence, we split it into a list of tokens. A token is a data point the model will train and predict. The following function supports splitting a sentence into words or characters, and return a list of split sentences.

```{.python .input}
# Save to the d2l package.
def tokenize(lines, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
tokens[0:2]
```

## Vocabulary

The string type of the tokens is inconvenient to be used by the models, which take numerical inputs. Now let's build a dictionary, often called *vocabulary* as well, to map string tokens into numerical indices starting from 0. To so do, we first count the unique tokens in all documents, called *corpus*, and then assign a numerical index to each unique token according to its frequency. Rarely appeared tokens are often removed to reduce the complexity. A token that doesn't exist in the corpus or has been removed is mapped into a special unknown (“&lt;unk&gt;”) token. We optionally add another three special tokens: “&lt;pad&gt;” a token for padding, “&lt;bos&gt;” to present the beginning for a sentence, and “&lt;eos&gt;” for the ending of a sentence.

```{.python .input  n=9}
# Save to the d2l package. 
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            uniq_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk, uniq_tokens = 0, ['<unk>']
        uniq_tokens +=  [token for token, freq in self.token_freqs 
                         if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
            
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

# Save to the d2l package.
def count_corpus(sentences):
    # Flatten a list of token lists into a list of tokens
    tokens = [tk for line in sentences for tk in line]
    return collections.Counter(tokens)
```

We construct a vocabulary with the time machine dataset as the corpus, and then print the map between a few tokens to indices.

```{.python .input  n=23}
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
```

After that, we can convert each sentence into a list of numerical indices. To illustrate things we print two sentences with their corresponding indices.

```{.python .input  n=25}
for i in range(8, 10):
    print('words:', tokens[i]) 
    print('indices:', vocab[tokens[i]])
```

## Put All Things Together

We packaged the above code in the `load_corpus_time_machine` function, which returns `corpus`, a list of token indices, and `vocab`, the vocabulary. The modification we did here is that `corpus` is a single list, not a list of token lists, since we do not the sequence information in the following models. Besides, we use character tokens to simplify the training in later sections.

```{.python .input}
# Save to the d2l package.
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[tk] for line in tokens for tk in line]
    if max_tokens > 0: corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Summary

* Documents are preprocessed by tokenizing the words or characters and mapping them into indices.

## Exercises

* Tokenization is a key preprocessing step. It varies for different languages. Try to find another 3 commonly used methods to tokenize sentences. 

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2363)

![](../img/qr_lang-model-dataset.svg)
