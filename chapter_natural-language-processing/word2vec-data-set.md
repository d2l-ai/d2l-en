# Data Sets for Word2vec
:label:`chapter_word2vec_data`

In this section, we will introduce how to preprocess a data set with
negative sampling :numref:`chapter_approx_train` and load into mini-batches for
word2vec training. The data set we use is [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42), which is a small but commonly-used corpus. It takes samples from Wall Street Journal articles and includes training sets, validation sets, and test sets. 

First, import the packages and modules required for the experiment.

```{.python .input  n=1}
import collections
import d2l
import math
from mxnet import np, gluon
import random
import zipfile
```

## Read and Preprocessing

This data set has already been preprocessed. Each line of the data set acts as a sentence. All the words in a sentence are separated by spaces. In the word embedding task, each word is a token.

```{.python .input  n=2}
# Save to the d2l package.
def read_ptb():
    with zipfile.ZipFile('../data/ptb.zip', 'r') as f:
        raw_text = f.read('ptb/ptb.train.txt').decode("utf-8")
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
'# sentences: %d' % len(sentences)
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "'# sentences: 42069'"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Next we build a vocabulary with words appeared not greater than 10 times mapped into a "&lt;unk&gt;" token. Note that the preprocessed PTB data also contains "&lt;unk&gt;" tokens presenting rare words.

```{.python .input  n=3}
vocab = d2l.Vocab(sentences, min_freq=10)
'vocab size: %d' % len(vocab)
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "'vocab size: 6719'"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Subsampling

In text data, there are generally some words that appear at high frequencies, such "the", "a", and "in" in English. Generally speaking, in a context window, it is better to train the word embedding model when a word (such as "chip") and a lower-frequency word (such as "microprocessor") appear at the same time, rather than when a word appears with a higher-frequency word (such as "the"). Therefore, when training the word embedding model, we can perform subsampling[2] on the words. Specifically, each indexed word $w_i$ in the data set will drop out at a certain probability. The dropout probability is given as:

$$ \mathbb{P}(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

Here, $f(w_i)$ is the ratio of the instances of word $w_i$ to the total number of words in the data set, and the constant $t$ is a hyper-parameter (set to $10^{-4}$ in this experiment). As we can see, it is only possible to drop out the word $w_i$ in subsampling when $f(w_i) > t$. The higher the word's frequency, the higher its dropout probability.

```{.python .input  n=4}
# Save to the d2l package.
def subsampling(sentences, vocab):
    # Map low frequency words into <unk>
    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line]
                 for line in sentences]
    # Count the frequency for each word
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())
    # Return True if to keep this token during subsampling
    keep = lambda token: (
        random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))
    # Now do the subsampling.
    return [[tk for tk in line if keep(tk)] for line in sentences]

subsampled = subsampling(sentences, vocab)
```

Compare the sequence lengths before and after sampling, we can see subsampling significantly reduced the sequence length.

```{.python .input  n=5}
d2l.set_figsize((3.5, 2.5))
d2l.plt.hist([[len(line) for line in sentences],
              [len(line) for line in subsampled]] )
d2l.plt.xlabel('# tokens per sentence')
d2l.plt.ylabel('count')
d2l.plt.legend(['origin', 'subsampled']);
```

```{.json .output n=5}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"184.15625pt\" version=\"1.1\" viewBox=\"0 0 265.690625 184.15625\" width=\"265.690625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 184.15625 \nL 265.690625 184.15625 \nL 265.690625 -0 \nL 0 -0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 59.690625 146.6 \nL 254.990625 146.6 \nL 254.990625 10.7 \nL 59.690625 10.7 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 68.567898 146.6 \nL 75.814651 146.6 \nL 75.814651 124.453259 \nL 68.567898 124.453259 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 86.684781 146.6 \nL 93.931534 146.6 \nL 93.931534 85.197951 \nL 86.684781 85.197951 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 104.801664 146.6 \nL 112.048417 146.6 \nL 112.048417 75.663402 \nL 104.801664 75.663402 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 122.918547 146.6 \nL 130.1653 146.6 \nL 130.1653 96.385825 \nL 122.918547 96.385825 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_7\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 141.03543 146.6 \nL 148.282183 146.6 \nL 148.282183 125.665325 \nL 141.03543 125.665325 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_8\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 159.152313 146.6 \nL 166.399067 146.6 \nL 166.399067 139.489583 \nL 159.152313 139.489583 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_9\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 177.269196 146.6 \nL 184.51595 146.6 \nL 184.51595 144.991359 \nL 177.269196 144.991359 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_10\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 195.38608 146.6 \nL 202.632833 146.6 \nL 202.632833 146.192254 \nL 195.38608 146.192254 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_11\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 213.502963 146.6 \nL 220.749716 146.6 \nL 220.749716 146.460361 \nL 213.502963 146.460361 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_12\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 231.619846 146.6 \nL 238.866599 146.6 \nL 238.866599 146.521802 \nL 231.619846 146.521802 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_13\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 75.814651 146.6 \nL 83.061404 146.6 \nL 83.061404 17.171429 \nL 75.814651 17.171429 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_14\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 93.931534 146.6 \nL 101.178287 146.6 \nL 101.178287 53.377022 \nL 93.931534 53.377022 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_15\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 112.048417 146.6 \nL 119.29517 146.6 \nL 119.29517 134.959695 \nL 112.048417 134.959695 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_16\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 130.1653 146.6 \nL 137.412054 146.6 \nL 137.412054 145.963246 \nL 130.1653 145.963246 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_17\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 148.282183 146.6 \nL 155.528937 146.6 \nL 155.528937 146.54973 \nL 148.282183 146.54973 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_18\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 166.399067 146.6 \nL 173.64582 146.6 \nL 173.64582 146.6 \nL 166.399067 146.6 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_19\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 184.51595 146.6 \nL 191.762703 146.6 \nL 191.762703 146.6 \nL 184.51595 146.6 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_20\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 202.632833 146.6 \nL 209.879586 146.6 \nL 209.879586 146.6 \nL 202.632833 146.6 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_21\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 220.749716 146.6 \nL 227.996469 146.6 \nL 227.996469 146.6 \nL 220.749716 146.6 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"patch_22\">\n    <path clip-path=\"url(#p12fa3636ca)\" d=\"M 238.866599 146.6 \nL 246.113352 146.6 \nL 246.113352 146.6 \nL 238.866599 146.6 \nz\n\" style=\"fill:#ff7f0e;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m8fb074a16d\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.756209\" xlink:href=\"#m8fb074a16d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(63.574959 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"110.943729\" xlink:href=\"#m8fb074a16d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 20 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(104.581229 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"155.131249\" xlink:href=\"#m8fb074a16d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 40 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(148.768749 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"199.318769\" xlink:href=\"#m8fb074a16d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 60 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(192.956269 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"243.506289\" xlink:href=\"#m8fb074a16d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 80 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(237.143789 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- # tokens per sentence -->\n     <defs>\n      <path d=\"M 51.125 44 \nL 36.921875 44 \nL 32.8125 27.6875 \nL 47.125 27.6875 \nz\nM 43.796875 71.78125 \nL 38.71875 51.515625 \nL 52.984375 51.515625 \nL 58.109375 71.78125 \nL 65.921875 71.78125 \nL 60.890625 51.515625 \nL 76.125 51.515625 \nL 76.125 44 \nL 58.984375 44 \nL 54.984375 27.6875 \nL 70.515625 27.6875 \nL 70.515625 20.21875 \nL 53.078125 20.21875 \nL 48 0 \nL 40.1875 0 \nL 45.21875 20.21875 \nL 30.90625 20.21875 \nL 25.875 0 \nL 18.015625 0 \nL 23.09375 20.21875 \nL 7.71875 20.21875 \nL 7.71875 27.6875 \nL 24.90625 27.6875 \nL 29 44 \nL 13.28125 44 \nL 13.28125 51.515625 \nL 30.90625 51.515625 \nL 35.890625 71.78125 \nz\n\" id=\"DejaVuSans-35\"/>\n      <path id=\"DejaVuSans-32\"/>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 31.109375 \nL 44.921875 54.6875 \nL 56.390625 54.6875 \nL 27.390625 29.109375 \nL 57.625 0 \nL 45.90625 0 \nL 18.109375 26.703125 \nL 18.109375 0 \nL 9.078125 0 \nz\n\" id=\"DejaVuSans-107\"/>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n     </defs>\n     <g transform=\"translate(100.433594 174.876562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-35\"/>\n      <use x=\"83.789062\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"115.576172\" xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"154.785156\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"215.966797\" xlink:href=\"#DejaVuSans-107\"/>\n      <use x=\"273.830078\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"335.353516\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"398.732422\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"450.832031\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"482.619141\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"546.095703\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"607.619141\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"648.732422\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"680.519531\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"732.619141\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"794.142578\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"857.521484\" xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"896.730469\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"958.253906\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"1021.632812\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"1076.613281\" xlink:href=\"#DejaVuSans-101\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_6\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"mb0d70ec968\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"59.690625\" xlink:href=\"#mb0d70ec968\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0 -->\n      <g transform=\"translate(46.328125 150.399219)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_7\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"59.690625\" xlink:href=\"#mb0d70ec968\" y=\"118.672205\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 5000 -->\n      <defs>\n       <path d=\"M 10.796875 72.90625 \nL 49.515625 72.90625 \nL 49.515625 64.59375 \nL 19.828125 64.59375 \nL 19.828125 46.734375 \nQ 21.96875 47.46875 24.109375 47.828125 \nQ 26.265625 48.1875 28.421875 48.1875 \nQ 40.625 48.1875 47.75 41.5 \nQ 54.890625 34.8125 54.890625 23.390625 \nQ 54.890625 11.625 47.5625 5.09375 \nQ 40.234375 -1.421875 26.90625 -1.421875 \nQ 22.3125 -1.421875 17.546875 -0.640625 \nQ 12.796875 0.140625 7.71875 1.703125 \nL 7.71875 11.625 \nQ 12.109375 9.234375 16.796875 8.0625 \nQ 21.484375 6.890625 26.703125 6.890625 \nQ 35.15625 6.890625 40.078125 11.328125 \nQ 45.015625 15.765625 45.015625 23.390625 \nQ 45.015625 31 40.078125 35.4375 \nQ 35.15625 39.890625 26.703125 39.890625 \nQ 22.75 39.890625 18.8125 39.015625 \nQ 14.890625 38.140625 10.796875 36.28125 \nz\n\" id=\"DejaVuSans-53\"/>\n      </defs>\n      <g transform=\"translate(27.240625 122.471424)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-53\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"59.690625\" xlink:href=\"#mb0d70ec968\" y=\"90.744411\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 10000 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n      </defs>\n      <g transform=\"translate(20.878125 94.543629)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"59.690625\" xlink:href=\"#mb0d70ec968\" y=\"62.816616\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 15000 -->\n      <g transform=\"translate(20.878125 66.615835)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-53\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"59.690625\" xlink:href=\"#mb0d70ec968\" y=\"34.888821\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 20000 -->\n      <g transform=\"translate(20.878125 38.68804)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_12\">\n     <!-- count -->\n     <defs>\n      <path d=\"M 8.5 21.578125 \nL 8.5 54.6875 \nL 17.484375 54.6875 \nL 17.484375 21.921875 \nQ 17.484375 14.15625 20.5 10.265625 \nQ 23.53125 6.390625 29.59375 6.390625 \nQ 36.859375 6.390625 41.078125 11.03125 \nQ 45.3125 15.671875 45.3125 23.6875 \nL 45.3125 54.6875 \nL 54.296875 54.6875 \nL 54.296875 0 \nL 45.3125 0 \nL 45.3125 8.40625 \nQ 42.046875 3.421875 37.71875 1 \nQ 33.40625 -1.421875 27.6875 -1.421875 \nQ 18.265625 -1.421875 13.375 4.4375 \nQ 8.5 10.296875 8.5 21.578125 \nz\nM 31.109375 56 \nz\n\" id=\"DejaVuSans-117\"/>\n     </defs>\n     <g transform=\"translate(14.798438 92.75625)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"54.980469\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"116.162109\" xlink:href=\"#DejaVuSans-117\"/>\n      <use x=\"179.541016\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"242.919922\" xlink:href=\"#DejaVuSans-116\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_23\">\n    <path d=\"M 59.690625 146.6 \nL 59.690625 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_24\">\n    <path d=\"M 254.990625 146.6 \nL 254.990625 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_25\">\n    <path d=\"M 59.690625 146.6 \nL 254.990625 146.6 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_26\">\n    <path d=\"M 59.690625 10.7 \nL 254.990625 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_27\">\n     <path d=\"M 155.389063 48.05625 \nL 247.990625 48.05625 \nQ 249.990625 48.05625 249.990625 46.05625 \nL 249.990625 17.7 \nQ 249.990625 15.7 247.990625 15.7 \nL 155.389063 15.7 \nQ 153.389063 15.7 153.389063 17.7 \nL 153.389063 46.05625 \nQ 153.389063 48.05625 155.389063 48.05625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"patch_28\">\n     <path d=\"M 157.389063 27.298437 \nL 177.389063 27.298437 \nL 177.389063 20.298437 \nL 157.389063 20.298437 \nz\n\" style=\"fill:#1f77b4;\"/>\n    </g>\n    <g id=\"text_13\">\n     <!-- origin -->\n     <defs>\n      <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n      <path d=\"M 45.40625 27.984375 \nQ 45.40625 37.75 41.375 43.109375 \nQ 37.359375 48.484375 30.078125 48.484375 \nQ 22.859375 48.484375 18.828125 43.109375 \nQ 14.796875 37.75 14.796875 27.984375 \nQ 14.796875 18.265625 18.828125 12.890625 \nQ 22.859375 7.515625 30.078125 7.515625 \nQ 37.359375 7.515625 41.375 12.890625 \nQ 45.40625 18.265625 45.40625 27.984375 \nz\nM 54.390625 6.78125 \nQ 54.390625 -7.171875 48.1875 -13.984375 \nQ 42 -20.796875 29.203125 -20.796875 \nQ 24.46875 -20.796875 20.265625 -20.09375 \nQ 16.0625 -19.390625 12.109375 -17.921875 \nL 12.109375 -9.1875 \nQ 16.0625 -11.328125 19.921875 -12.34375 \nQ 23.78125 -13.375 27.78125 -13.375 \nQ 36.625 -13.375 41.015625 -8.765625 \nQ 45.40625 -4.15625 45.40625 5.171875 \nL 45.40625 9.625 \nQ 42.625 4.78125 38.28125 2.390625 \nQ 33.9375 0 27.875 0 \nQ 17.828125 0 11.671875 7.65625 \nQ 5.515625 15.328125 5.515625 27.984375 \nQ 5.515625 40.671875 11.671875 48.328125 \nQ 17.828125 56 27.875 56 \nQ 33.9375 56 38.28125 53.609375 \nQ 42.625 51.21875 45.40625 46.390625 \nL 45.40625 54.6875 \nL 54.390625 54.6875 \nz\n\" id=\"DejaVuSans-103\"/>\n     </defs>\n     <g transform=\"translate(185.389063 27.298437)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"61.181641\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"102.294922\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"130.078125\" xlink:href=\"#DejaVuSans-103\"/>\n      <use x=\"193.554688\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"221.337891\" xlink:href=\"#DejaVuSans-110\"/>\n     </g>\n    </g>\n    <g id=\"patch_29\">\n     <path d=\"M 157.389063 41.976562 \nL 177.389063 41.976562 \nL 177.389063 34.976562 \nL 157.389063 34.976562 \nz\n\" style=\"fill:#ff7f0e;\"/>\n    </g>\n    <g id=\"text_14\">\n     <!-- subsampled -->\n     <defs>\n      <path d=\"M 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\nM 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nz\n\" id=\"DejaVuSans-98\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-109\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n      <path d=\"M 45.40625 46.390625 \nL 45.40625 75.984375 \nL 54.390625 75.984375 \nL 54.390625 0 \nL 45.40625 0 \nL 45.40625 8.203125 \nQ 42.578125 3.328125 38.25 0.953125 \nQ 33.9375 -1.421875 27.875 -1.421875 \nQ 17.96875 -1.421875 11.734375 6.484375 \nQ 5.515625 14.40625 5.515625 27.296875 \nQ 5.515625 40.1875 11.734375 48.09375 \nQ 17.96875 56 27.875 56 \nQ 33.9375 56 38.25 53.625 \nQ 42.578125 51.265625 45.40625 46.390625 \nz\nM 14.796875 27.296875 \nQ 14.796875 17.390625 18.875 11.75 \nQ 22.953125 6.109375 30.078125 6.109375 \nQ 37.203125 6.109375 41.296875 11.75 \nQ 45.40625 17.390625 45.40625 27.296875 \nQ 45.40625 37.203125 41.296875 42.84375 \nQ 37.203125 48.484375 30.078125 48.484375 \nQ 22.953125 48.484375 18.875 42.84375 \nQ 14.796875 37.203125 14.796875 27.296875 \nz\n\" id=\"DejaVuSans-100\"/>\n     </defs>\n     <g transform=\"translate(185.389063 41.976562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"52.099609\" xlink:href=\"#DejaVuSans-117\"/>\n      <use x=\"115.478516\" xlink:href=\"#DejaVuSans-98\"/>\n      <use x=\"178.955078\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"231.054688\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"292.333984\" xlink:href=\"#DejaVuSans-109\"/>\n      <use x=\"389.746094\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"453.222656\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"481.005859\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"542.529297\" xlink:href=\"#DejaVuSans-100\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p12fa3636ca\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"59.690625\" y=\"10.7\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

For individual tokens, the sampling rate of the high-frequency word "the" is less than 1/20.

```{.python .input  n=6}
def compare_counts(token):
    return '# of "%s": before=%d, after=%d' % (token, sum(
        [line.count(token) for line in sentences]), sum(
        [line.count(token) for line in subsampled]))

compare_counts('the')
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "'# of \"the\": before=50770, after=2146'"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

But the low-frequency word "join" is completely preserved.

```{.python .input  n=7}
compare_counts('join')
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "'# of \"join\": before=45, after=45'"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Lastly, we map each token into an index to construct the corpus.

```{.python .input  n=8}
corpus = [vocab[line] for line in subsampled]
corpus[0:3]
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "[[], [392, 2132, 145, 275, 406], [5464, 3080, 1595, 95]]"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Load the Data Set

Next we read the corpus with token indicies into data batches for training.

### Extract Central Target Words and Context Words

We use words with a distance from the central target word not exceeding the context window size as the context words of the given center target word. The following definition function extracts all the central target words and their context words. It uniformly and randomly samples an integer to be used as the context window size between integer 1 and the `max_window_size` (maximum context window).

```{.python .input  n=9}
# Save to the d2l package.
def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        # Each sentence needs at least 2 words to form a
        # "central target word - context word" pair
        if len(line) < 2: continue
        centers += line
        for i in range(len(line)):  # Context window centered at i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the central target word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Next, we create an artificial data set containing two sentences of 7 and 3 words, respectively. Assume the maximum context window is 2 and print all the central target words and their context words.

```{.python .input  n=10}
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "dataset [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]\ncenter 0 has contexts [1]\ncenter 1 has contexts [0, 2, 3]\ncenter 2 has contexts [0, 1, 3, 4]\ncenter 3 has contexts [2, 4]\ncenter 4 has contexts [3, 5]\ncenter 5 has contexts [3, 4, 6]\ncenter 6 has contexts [5]\ncenter 7 has contexts [8, 9]\ncenter 8 has contexts [7, 9]\ncenter 9 has contexts [7, 8]\n"
 }
]
```

We set the maximum context window size to 5. The following extracts all the central target words and their context words in the data set.

```{.python .input  n=11}
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
'# center-context pairs: %d' % len(all_centers)
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "'# center-context pairs: 353083'"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Negative Sampling

We use negative sampling for approximate training. For a central and context word pair, we randomly sample $K$ noise words ($K=5$ in the experiment). According to the suggestion in the Word2vec paper, the noise word sampling probability $\mathbb{P}(w)$ is the ratio of the word frequency of $w$ to the total word frequency raised to the power of 0.75 [2].

We first define a class to draw a candidate according to the sampling weights. It caches a 10000 size random number bank instead of calling `random.choices` every time.

```{.python .input  n=12}
# Save to the d2l package.
class RandomGenerator(object):
    """Draw a random int in [0, n] according to n sampling weights"""
    def __init__(self, sampling_weights):
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]

generator = RandomGenerator([2,3,4])
[generator.draw() for _ in range(10)]
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "[1, 2, 1, 0, 1, 1, 2, 2, 0, 1]"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=13}
# Save to the d2l package.
def get_negatives(all_contexts, corpus, K):
    counter = d2l.count_corpus(corpus)
    sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, corpus, 5)
```

### Read into Batches

We extract all central target words `all_centers`, and the context words `all_contexts` and noise words `all_negatives` of each central target word from the data set. We will read them in random mini-batches.

In a mini-batch of data, the $i$-th example includes a central word and its corresponding $n_i$ context words and $m_i$ noise words. Since the context window size of each example may be different, the sum of context words and noise words, $n_i+m_i$, will be different. When constructing a mini-batch, we concatenate the context words and noise words of each example, and add 0s for padding until the length of the concatenations are the same, that is, the length of all concatenations is $\max_i n_i+m_i$(`max_len`). In order to avoid the effect of padding on the loss function calculation, we construct the mask variable `masks`, each element of which corresponds to an element in the concatenation of context and noise words, `contexts_negatives`. When an element in the variable `contexts_negatives` is a padding, the element in the mask variable `masks` at the same position will be 0. Otherwise, it takes the value 1. In order to distinguish between positive and negative examples, we also need to distinguish the context words from the noise words in the `contexts_negatives` variable. Based on the construction of the mask variable, we only need to create a label variable `labels` with the same shape as the `contexts_negatives` variable and set the elements corresponding to context words (positive examples) to 1, and the rest to 0.

Next, we will implement the mini-batch reading function `batchify`. Its mini-batch input `data` is a list whose length is the batch size, each element of which contains central target words `center`, context words `context`, and noise words `negative`. The mini-batch data returned by this function conforms to the format we need, for example, it includes the mask variable.

```{.python .input  n=14}
# Save to the d2l package.
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (np.array(centers).reshape((-1, 1)), np.array(contexts_negatives),
            np.array(masks), np.array(labels))
```

Construct two simple examples:

```{.python .input  n=15}
x_1 = (1, [2,2], [3,3,3,3])
x_2 = (1, [2,2,2], [3,3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "centers = [[1.]\n [1.]]\n<ndarray shape=(2, 1)>\ncontexts_negatives = [[2. 2. 3. 3. 3. 3.]\n [2. 2. 2. 3. 3. 0.]]\n<ndarray shape=(2, 6)>\nmasks = [[1. 1. 1. 1. 1. 1.]\n [1. 1. 1. 1. 1. 0.]]\n<ndarray shape=(2, 6)>\nlabels = [[1. 1. 0. 0. 0. 0.]\n [1. 1. 1. 0. 0. 0.]]\n<ndarray shape=(2, 6)>\n"
 }
]
```

We use the `batchify` function just defined to specify the mini-batch reading method in the `DataLoader` instance. 

## Put All Things Together

Lastly, we define the `load_data_ptb` function that read the PTB data set and return the data loader.

```{.python .input  n=16}
# Save to the d2l package.
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled = subsampling(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, corpus, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      batchify_fn=batchify)
    return data_iter, vocab
```

Let's print the first mini-batch of the data iterator.

```{.python .input  n=17}
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "centers shape: (512, 1)\ncontexts_negatives shape: (512, 60)\nmasks shape: (512, 60)\nlabels shape: (512, 60)\n"
 }
]
```

## Summary

* Subsampling attempts to minimize the impact of high-frequency words on the training of a word embedding model.
* We can pad examples of different lengths to create mini-batches with examples of all the same length and use mask variables to distinguish between padding and non-padding elements, so that only non-padding elements participate in the calculation of the loss function.

## Exercises

* We use the `batchify` function to specify the mini-batch reading method in the `DataLoader` instance and print the shape of each variable in the first batch read. How should these shapes be calculated?
