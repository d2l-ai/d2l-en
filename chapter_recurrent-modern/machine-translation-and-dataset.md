# Machine Translation and the Dataset
:label:`sec_machine_translation`

We have used RNNs to design language models,
which are key to natural language processing.
Another flagship benchmark is *machine translation*,
a central problem domain for *sequence transduction* models
that transform input sequences into output sequences.
Playing a crucial role in various modern AI applications,
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

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## [**Downloading and Preprocessing the Dataset**]

To begin with,
we download an English-French dataset
that consists of [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/).
Each line in the dataset
is a tab-delimited pair
of an English text sequence
and the translated French text sequence.
Note that each text sequence
can be just one sentence or a paragraph of multiple sentences.
In this machine translation problem
where English is translated into French,
English is the *source language*
and French is the *target language*.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
            
data = MTFraEng() 
raw_text = data._download()
raw_text[:60]
```

After downloading the dataset,
we [**proceed with several preprocessing steps**]
for the raw text data.
For instance,
we replace non-breaking space with space,
convert uppercase letters to lowercase ones,
and insert space between words and punctuation marks.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)

text = data._preprocess(raw_text)
text[:60]
```

## [**Tokenization**]

Different from character-level tokenization
in :numref:`sec_language_model`,
for machine translation
we prefer word-level tokenization here
(state-of-the-art models may use more advanced tokenization techniques).
The following `tokenize_nmt` function
tokenizes the the first `num_examples` text sequence pairs,
where
each token is either a word or a punctuation mark.
This function returns
two lists of token lists: `source` and `target`.
Specifically,
`source[i]` is a list of tokens from the
$i^\mathrm{th}$ text sequence in the source language (English here) and `target[i]` is that in the target language (French here).

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'<bos> {parts[1]} <eos>'.split(' ') if t])
    return src, tgt

src, tgt = data._tokenize(text)
src[:4], tgt[:4]
```

Let's [**plot the histogram of the number of tokens per text sequence.**]
In this simple English-French dataset,
most of the text sequences have fewer than 20 tokens.

```{.python .input  n=8}
%%tab all
d2l.set_figsize((4.5, 2.5))
_, _, patches = d2l.plt.hist([[len(l) for l in src], 
                              [len(l) for l in tgt]])
d2l.plt.xlabel('# tokens per sequence'), d2l.plt.ylabel('count')
d2l.plt.xlim([0, 20])
for patch in patches[1].patches: patch.set_hatch('-')
_ = d2l.plt.legend(['source', 'target'])
```

## Fixed Length Examples

Recall that in language modeling
[**each sequence example**],
either a segment of one sentence
or a span over multiple sentences,
(**has a fixed length.**)
This was specified by the `num_steps`
(number of time steps or tokens) argument in :numref:`sec_language_model`.
In machine translation, each example is
a pair of source and target text sequences,
where each text sequence may have different lengths.

For computational efficiency,
we can still process a minibatch of text sequences
at one time by *truncation* and *padding*.
Suppose that every sequence in the same minibatch
should have the same length `num_steps`.
If a text sequence has fewer than `num_steps` tokens,
we will keep appending the special "&lt;pad&gt;" token
to its end until its length reaches `num_steps`.
Otherwise,
we will truncate the text sequence
by only taking its first `num_steps` tokens
and discarding the remaining.
In this way,
every text sequence
will have the same length
to be loaded in minibatches of the same shape.

Now we define a function to [**transform
text sequences into minibatches for training.**]
We append the special “&lt;eos&gt;” token
to the end of every sequence to indicate the
end of the sequence.
When a model is predicting
by
generating a sequence token after token,
the generation
of the “&lt;eos&gt;” token
can suggest that
the output sequence is complete.
Besides,
we also record the length
of each text sequence excluding the padding tokens.
This information will be needed by
some models that
we will cover later.

## [**Vocabulary**]

Since the machine translation dataset
consists of pairs of languages,
we can build two vocabularies for
both the source language and
the target language separately.
With word-level tokenization,
the vocabulary size will be significantly larger
than that using character-level tokenization.
To alleviate this,
here we treat infrequent tokens
that appear less than 2 times
as the same unknown ("&lt;unk&gt;") token.
Besides that,
we specify additional special tokens
such as for padding ("&lt;pad&gt;") sequences to the same length in minibatches,
and for marking the beginning ("&lt;bos&gt;") or end ("&lt;eos&gt;") of sequences.
Such special tokens are commonly used in
natural language processing tasks.

## [**Putting All Things Together**]

Finally, we define the `load_data_nmt` function
to return the data iterator, together with
the vocabularies for both the source language and the target language.

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps, num_train=1000, num_val=1000):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
    
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_one(sentences, vocab):
        pad_or_trim = lambda s, n: (
            s[:n] if len(s) > n else s + ['<pad>'] * (n - len(s)))
        sentences = [pad_or_trim(seq, self.num_steps) for seq in sentences]
        if vocab is None: vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[sent] for sent in sentences])
        return array, vocab
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab = _build_one(src, src_vocab)
    tgt_array, tgt_vocab = _build_one(tgt, tgt_vocab)
    return (src_array, tgt_array[:,:-1], tgt_array[:,1:]), src_vocab, tgt_vocab

```

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

Let's [**read the first minibatch from the English-French dataset.**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3, num_steps=6)
src, tgt, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('target:', d2l.astype(tgt, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src+'\t'+tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=14}
%%tab all
src, tgt, _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Summary

* Machine translation refers to the automatic translation of a sequence from one language to another.
* Using word-level tokenization, the vocabulary size will be significantly larger than that using character-level tokenization. To alleviate this, we can treat infrequent tokens as the same unknown token.
* We can truncate and pad text sequences so that all of them will have the same length to be loaded in minibatches.


## Exercises

1. Try different values of the `num_examples` argument in the `load_data_nmt` function. How does this affect the vocabulary sizes of the source language and the target language?
1. Text in some languages such as Chinese and Japanese does not have word boundary indicators (e.g., space). Is word-level tokenization still a good idea for such cases? Why or why not?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3863)
:end_tab:
