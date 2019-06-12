# Text Preprocessing
:label:`chapter_text_preprocessing`

Text is an important example of sequence data. An article can be simply viewed as a sequence of words, or even a sequence of chars in a more simplified viewpoint. Text data is the second major input data format besides images we will consider in this book. In this section, we will introduce the common preprocessing steps for text data. It often consists of four steps:

1. Loads the dataset as strings into memory.
1. Splits strings into tokens, a token could be a word or a char. 
1. Builds a vocabulary for these tokens to map them into numerical indices. 
1. Maps all tokens in the data into indices to facilitate to feed into models. 

## Data Loading

To get started we load text from H.G. Wells' [Time Machine](http://www.gutenberg.org/ebooks/35). This is a fairly small corpus of just over 30,000 words but for the purpose of what we want to illustrate this is just fine. More realistic document collections contain many billions of words. The following function read the dataset into a list of sentences, each sentence is a string. Here we ignore punctuation and capitalization.

```{.python .input}
import collections
import re

# Save to the d2l package. 
def read_time_machine():
    """Load the time machine book into a list of sentences."""
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()
    
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower()) for line in lines]

lines = read_time_machine()
'# sentences %d' % len(lines)
```

## Tokenization

For each sentence, we split it into a list of tokens. A token, often a word or a char, a data point the model will train and predict.

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

Then we need to map tokens into numerical indices. We often call it a vocabulary. Its input is a list of token lists,  called a corpus. Then it counts the frequency of each token in this corpus, and then assigns an numerical index to each token according to its frequency. Rarely appeared tokens are often removed to reduce the complexity. A token doesn't exist in corpus or has been removed is mapped into a special unknown (“&lt;unk&gt;”) token. We optionally add another three special tokens: “&lt;pad&gt;” a token for padding, “&lt;bos&gt;” to present the beginning for a sentence, and “&lt;eos&gt;” for the ending of a sentence.

```{.python .input  n=9}
# Save to the d2l package. 
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        # Flatten a list of token lists into a list of tokens
        tokens = [tk for line in tokens for tk in line]
        # sort by frequency and token
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk = 0
            tokens = ['<unk>']
        tokens +=  [token for token, freq in self.token_freqs 
                    if freq >= min_freq]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        else:
            return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        else:
            return [self.idx_to_token[index] for index in indices]
```

We construct a vocabulary with the time machine dataset as the corpus, and then print the map between a few tokens to indices.

```{.python .input  n=23}
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
```

After that, each character in the training data set is converted into an index ID. To illustrate things we print the first 20 characters and their corresponding indices.

```{.python .input  n=25}
for i in range(8, 10):
    print('words:', tokens[i]) 
    print('indices:', vocab[tokens[i]])
```

## Put all things together.


We packaged the above code in the `(corpus_indices, vocab) = load_data_timemachine()` function of the `d2l` package to make it easier to call it in later chapters.



```{.python .input}
def xx():
    pass
```

Sequential partitioning decomposes the sequence into `batch_size` many strips of data which are traversed as we iterate over minibatches. Note that the $i$-th element in a minibatch matches with the $i$-th element of the next minibatch rather than within a minibatch.

## Summary

* Documents are preprocessed by tokenizing the words and mapping them into IDs. There are multiple methods:
    * Character encoding which uses individual characters (good e.g. for Chinese)
    * Word encoding (good e.g. for English)
    * Byte-pair encoding (good for languages that have lots of morphology, e.g. German)
* The main choices for sequence partitioning are whether we pick consecutive or random sequences. In particular for recurrent networks the former is critical.
* Given the overall document length, it is usually acceptable to be slightly wasteful with the documents and discard half-empty minibatches.

## Exercises

1. Which other other mini-batch data sampling methods can you think of?
1. Why is it a good idea to have a random offset?
    * Does it really lead to a perfectly uniform distribution over the sequences on the document?
    * What would you have to do to make things even more uniform?
1. If we want a sequence example to be a complete sentence, what kinds of problems does this introduce in mini-batch sampling? Why would we want to do this anyway?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2363)

![](../img/qr_lang-model-dataset.svg)
