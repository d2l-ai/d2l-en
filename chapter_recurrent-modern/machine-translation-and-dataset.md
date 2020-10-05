# Machine Translation and the Dataset
:label:`sec_machine_translation`

We have used RNNs to design language models,
which are key to natural language processing.
Another flagship benchmark is *machine translation*,
a central problem domain for *sequence transduction* models
that transform input sequences into output sequences.
Playing a crucial role in various modern artificial intelligence applications,
sequence transduction models will form the focus of the remainder of this chapter
and :numref:`chap_attention`.
To this end,
this section introduces the machine translation problem
and its dataset that will be used later.


*Machine translation* refers to the
automatic translation of a sequence
from one language to another.
In fact, this field
may date back to 1940s
soon after digital computers were invented,
especially by considering the use of computers
for cracking language codes in World War II.
For decades,
statistical approaches
had been dominant in this field :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`
before the rise 
of 
end-to-end learning using
neural networks. 
The latter 
is often called
*neural machine translation*
to distinguish itself from
*statistical machine translation*
that involves statistical analysis
in components such as 
the translation model and the language model. 


Emphasizing end-to-end learning,
this book will focus on neural machine translation methods.
Different from our language model problem
in :numref:`sec_language_model`
whose corpus is in one single language,
machine translation datasets
are composed of pairs of text sequences
that are in 
the source language and the target language, respectively.
Thus,
instead of reusing the preprocessing routine
for language modeling,
we need a different way to preprocess
machine translation datasets.
In the following,
we show how to 
load the preprocessed data
into minibatches for training.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import os
```

## Downloading and Preprocessing the Dataset

To begin with,
we download an English-French dataset
that consists of [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/).
Each line in the dataset
is a tab-delimited pair
of an English sentence
and the translated French sentence.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

After downloading the dataset,
we proceed with several preprocessing steps
for the raw text data.
For instance,
we replace non-breaking space with space,
convert uppercase letters to lowercase ones,
and insert space between words and punctuation marks.

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## Tokenization

Different to using character tokens in :numref:`sec_language_model`, here a token is either a word or a punctuation mark. The following function tokenizes the text data to return `source` and `target`. Each one is a list of token list, with `source[i]` is the $i^\mathrm{th}$ sentence in the source language and `target[i]` is the $i^\mathrm{th}$ sentence in the target language. To make the latter training faster, we sample the first `num_examples` sentences pairs.

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

We visualize the histogram of the number of tokens per sentence in the following figure. As can be seen, a sentence in average contains 5 tokens, and most of the sentences have less than 10 tokens.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.hist([[len(l) for l in source], [len(l) for l in target]],
             label=['source', 'target'])
d2l.plt.legend(loc='upper right');
```

## Vocabulary

Since the tokens in the source language could be different to the ones in the target language, we need to build a vocabulary for each of them. Since we are using words instead of characters  as tokens, it makes the vocabulary size significantly large. Here we map every token that appears less than 3 times into the &lt;unk&gt; token :numref:`sec_text_preprocessing`. In addition, we need other special tokens such as padding and sentence beginnings.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=3, 
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## Loading the Dataset

In language models, each example is a `num_steps` length sequence from the corpus, which may be a segment of a sentence, or span over multiple sentences. In machine translation, an example should contain a pair of source sentence and target sentence. These sentences might have different lengths, while we need same length examples to form a minibatch. 

One way to solve this problem is that if a sentence is longer than `num_steps`, we trim its length, otherwise pad with a special &lt;pad&gt; token to meet the length. Therefore we could transform any sentence to a fixed length.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Trim
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

Now we can convert a list of sentences into an `(num_example, num_steps)` index array. We also record the length of each sentence without the padding tokens, called *valid length*, which might be used by some models. In addition, we add the special “&lt;bos&gt;” and “&lt;eos&gt;” tokens to the target sentences so that our model will know the signals for starting and ending predicting.

```{.python .input}
#@save
def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]
    if not is_source:
        lines = [[vocab['<bos>']] + l + [vocab['<eos>']] for l in lines]
    array = np.array([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).sum(axis=1)
    return array, valid_len
```

```{.python .input}
#@tab pytorch
#@save
def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]
    if not is_source:
        lines = [[vocab['<bos>']] + l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).sum(dim=1)
    return array, valid_len
```

Then we can construct minibatches based on these arrays. 

## Putting All Things Together

Finally, we define the function `load_data_nmt` to return the data iterator with the vocabularies for source language and target language.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=1000):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=3, 
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=3, 
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array(
        source, src_vocab, num_steps, True)
    tgt_array, tgt_valid_len = build_array(
        target, tgt_vocab, num_steps, False)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return src_vocab, tgt_vocab, data_iter
```

Let us read the first batch.

```{.python .input}
src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size=2, num_steps=8)
for X, X_vlen, Y, Y_vlen in train_iter:
    print('X:', X.astype('int32'))
    print('valid lengths for X:', X_vlen)
    print('Y:', Y.astype('int32'))
    print('valid lengths for Y:', Y_vlen)
    break
```

```{.python .input}
#@tab pytorch
src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size=2, num_steps=8)
for X, X_vlen, Y, Y_vlen in train_iter:
    print('X:', X.type(torch.int32))
    print('valid lengths for X:', X_vlen)
    print('Y:', Y.type(torch.int32))
    print('valid lengths for Y:', Y_vlen)
    break
```

## Summary

* Machine translation (MT) refers to the automatic translation of a segment of text from one language to another. 
* We read, preprocess, and tokenize the datasets from both source language and target language.


## Exercises

1. Find a machine translation dataset online and process it.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
