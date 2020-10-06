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
In this machine translation problem
where English is translated into French,
English is the *source language*
and French is the *target language*.

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

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Go.\tVa !\nHi.\tSalut !\nRun!\tCours\u202f!\nRun!\tCourez\u202f!\nWho?\tQui ?\nWow!\t\u00c7a alors\u202f!\n\n"
 }
]
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

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "go .\tva !\nhi .\tsalut !\nrun !\tcours !\nrun !\tcourez !\nwho?\tqui ?\nwow !\t\u00e7a alors !\n\n"
 }
]
```

## Tokenization

Different from character-level tokenization
in :numref:`sec_language_model`, 
for machine translation
we prefer word-level tokenization here
(state-of-the-art models may use more advanced tokenization techniques).
The following `tokenize_nmt` function
tokenizes the the first `num_examples` sentence pairs,
where
each token is either a word or a punctuation mark. 
This function returns
two lists of token lists: `source` and `target`.
Specifically,
`source[i]` is a list of tokens from the
$i^\mathrm{th}$ sentence in the source language (English here) and `target[i]` is that in the target language (French here).

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

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "([['go', '.'],\n  ['hi', '.'],\n  ['run', '!'],\n  ['run', '!'],\n  ['who?'],\n  ['wow', '!']],\n [['va', '!'],\n  ['salut', '!'],\n  ['cours', '!'],\n  ['courez', '!'],\n  ['qui', '?'],\n  ['\u00e7a', 'alors', '!']])"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us plot the histogram of the number of tokens per sentence. 
In this simple English-French dataset,
most of the sentences have fewer than 20 tokens.

```{.python .input}
#@tab all
d2l.set_figsize()
_, _, patches = d2l.plt.hist(
    [[len(l) for l in source], [len(l) for l in target]],
    label=['source', 'target'])
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right');
```

```{.json .output n=20}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"166.978125pt\" version=\"1.1\" viewBox=\"0 0 254.166217 166.978125\" width=\"254.166217pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 166.978125 \nL 254.166217 166.978125 \nL 254.166217 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 46.0125 143.1 \nL 241.3125 143.1 \nL 241.3125 7.2 \nL 46.0125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 54.889773 143.1 \nL 62.136526 143.1 \nL 62.136526 28.855425 \nL 54.889773 28.855425 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 73.006656 143.1 \nL 80.253409 143.1 \nL 80.253409 26.419498 \nL 73.006656 26.419498 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 91.123539 143.1 \nL 98.370292 143.1 \nL 98.370292 134.708466 \nL 91.123539 134.708466 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 109.240422 143.1 \nL 116.487175 143.1 \nL 116.487175 142.639226 \nL 109.240422 142.639226 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_7\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 127.357305 143.1 \nL 134.604058 143.1 \nL 134.604058 143.005262 \nL 127.357305 143.005262 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_8\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 145.474188 143.1 \nL 152.720942 143.1 \nL 152.720942 143.081339 \nL 145.474188 143.081339 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_9\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 163.591071 143.1 \nL 170.837825 143.1 \nL 170.837825 143.091387 \nL 163.591071 143.091387 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_10\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 181.707955 143.1 \nL 188.954708 143.1 \nL 188.954708 143.098565 \nL 181.707955 143.098565 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_11\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 199.824838 143.1 \nL 207.071591 143.1 \nL 207.071591 143.097129 \nL 199.824838 143.097129 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_12\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 217.941721 143.1 \nL 225.188474 143.1 \nL 225.188474 143.1 \nL 217.941721 143.1 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_13\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 62.136526 143.1 \nL 69.383279 143.1 \nL 69.383279 46.735163 \nL 62.136526 46.735163 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_14\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 80.253409 143.1 \nL 87.500162 143.1 \nL 87.500162 13.671429 \nL 80.253409 13.671429 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_15\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 98.370292 143.1 \nL 105.617045 143.1 \nL 105.617045 130.234226 \nL 98.370292 130.234226 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_16\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 116.487175 143.1 \nL 123.733929 143.1 \nL 123.733929 142.092327 \nL 116.487175 142.092327 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_17\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 134.604058 143.1 \nL 141.850812 143.1 \nL 141.850812 142.923442 \nL 134.604058 142.923442 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_18\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 152.720942 143.1 \nL 159.967695 143.1 \nL 159.967695 143.052631 \nL 152.720942 143.052631 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_19\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 170.837825 143.1 \nL 178.084578 143.1 \nL 178.084578 143.097129 \nL 170.837825 143.097129 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_20\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 188.954708 143.1 \nL 196.201461 143.1 \nL 196.201461 143.097129 \nL 188.954708 143.097129 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_21\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 207.071591 143.1 \nL 214.318344 143.1 \nL 214.318344 143.097129 \nL 207.071591 143.097129 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"patch_22\">\n    <path clip-path=\"url(#p163268efb5)\" d=\"M 225.188474 143.1 \nL 232.435227 143.1 \nL 232.435227 143.095694 \nL 225.188474 143.095694 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m47469bb143\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"49.899684\" xlink:href=\"#m47469bb143\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(46.718434 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"113.467695\" xlink:href=\"#m47469bb143\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 20 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(107.105195 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"177.035706\" xlink:href=\"#m47469bb143\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 40 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(170.673206 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"240.603717\" xlink:href=\"#m47469bb143\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 60 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(234.241217 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_5\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m1de98b9016\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"46.0125\" xlink:href=\"#m1de98b9016\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 0 -->\n      <g transform=\"translate(32.65 146.899219)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"46.0125\" xlink:href=\"#m1de98b9016\" y=\"114.391366\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 20000 -->\n      <g transform=\"translate(7.2 118.190584)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_7\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"46.0125\" xlink:href=\"#m1de98b9016\" y=\"85.682731\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 40000 -->\n      <g transform=\"translate(7.2 89.48195)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"46.0125\" xlink:href=\"#m1de98b9016\" y=\"56.974097\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 60000 -->\n      <g transform=\"translate(7.2 60.773316)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"46.0125\" xlink:href=\"#m1de98b9016\" y=\"28.265463\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 80000 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(7.2 32.064682)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_23\">\n    <path d=\"M 46.0125 143.1 \nL 46.0125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_24\">\n    <path d=\"M 241.3125 143.1 \nL 241.3125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_25\">\n    <path d=\"M 46.0125 143.1 \nL 241.3125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_26\">\n    <path d=\"M 46.0125 7.2 \nL 241.3125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_27\">\n     <path d=\"M 169.109375 44.55625 \nL 234.3125 44.55625 \nQ 236.3125 44.55625 236.3125 42.55625 \nL 236.3125 14.2 \nQ 236.3125 12.2 234.3125 12.2 \nL 169.109375 12.2 \nQ 167.109375 12.2 167.109375 14.2 \nL 167.109375 42.55625 \nQ 167.109375 44.55625 169.109375 44.55625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"patch_28\">\n     <path d=\"M 171.109375 23.798438 \nL 191.109375 23.798438 \nL 191.109375 16.798438 \nL 171.109375 16.798438 \nz\n\" style=\"fill:#1f77b4;\"/>\n    </g>\n    <g id=\"text_10\">\n     <!-- source -->\n     <defs>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 8.5 21.578125 \nL 8.5 54.6875 \nL 17.484375 54.6875 \nL 17.484375 21.921875 \nQ 17.484375 14.15625 20.5 10.265625 \nQ 23.53125 6.390625 29.59375 6.390625 \nQ 36.859375 6.390625 41.078125 11.03125 \nQ 45.3125 15.671875 45.3125 23.6875 \nL 45.3125 54.6875 \nL 54.296875 54.6875 \nL 54.296875 0 \nL 45.3125 0 \nL 45.3125 8.40625 \nQ 42.046875 3.421875 37.71875 1 \nQ 33.40625 -1.421875 27.6875 -1.421875 \nQ 18.265625 -1.421875 13.375 4.4375 \nQ 8.5 10.296875 8.5 21.578125 \nz\nM 31.109375 56 \nz\n\" id=\"DejaVuSans-117\"/>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n     </defs>\n     <g transform=\"translate(199.109375 23.798438)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"52.099609\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"113.28125\" xlink:href=\"#DejaVuSans-117\"/>\n      <use x=\"176.660156\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"215.523438\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"270.503906\" xlink:href=\"#DejaVuSans-101\"/>\n     </g>\n    </g>\n    <g id=\"patch_29\">\n     <path d=\"M 171.109375 38.476562 \nL 191.109375 38.476562 \nL 191.109375 31.476562 \nL 171.109375 31.476562 \nz\n\" style=\"fill:url(#hd7e6d4ca53);\"/>\n    </g>\n    <g id=\"text_11\">\n     <!-- target -->\n     <defs>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 45.40625 27.984375 \nQ 45.40625 37.75 41.375 43.109375 \nQ 37.359375 48.484375 30.078125 48.484375 \nQ 22.859375 48.484375 18.828125 43.109375 \nQ 14.796875 37.75 14.796875 27.984375 \nQ 14.796875 18.265625 18.828125 12.890625 \nQ 22.859375 7.515625 30.078125 7.515625 \nQ 37.359375 7.515625 41.375 12.890625 \nQ 45.40625 18.265625 45.40625 27.984375 \nz\nM 54.390625 6.78125 \nQ 54.390625 -7.171875 48.1875 -13.984375 \nQ 42 -20.796875 29.203125 -20.796875 \nQ 24.46875 -20.796875 20.265625 -20.09375 \nQ 16.0625 -19.390625 12.109375 -17.921875 \nL 12.109375 -9.1875 \nQ 16.0625 -11.328125 19.921875 -12.34375 \nQ 23.78125 -13.375 27.78125 -13.375 \nQ 36.625 -13.375 41.015625 -8.765625 \nQ 45.40625 -4.15625 45.40625 5.171875 \nL 45.40625 9.625 \nQ 42.625 4.78125 38.28125 2.390625 \nQ 33.9375 0 27.875 0 \nQ 17.828125 0 11.671875 7.65625 \nQ 5.515625 15.328125 5.515625 27.984375 \nQ 5.515625 40.671875 11.671875 48.328125 \nQ 17.828125 56 27.875 56 \nQ 33.9375 56 38.28125 53.609375 \nQ 42.625 51.21875 45.40625 46.390625 \nL 45.40625 54.6875 \nL 54.390625 54.6875 \nz\n\" id=\"DejaVuSans-103\"/>\n     </defs>\n     <g transform=\"translate(199.109375 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"100.488281\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"139.851562\" xlink:href=\"#DejaVuSans-103\"/>\n      <use x=\"203.328125\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"264.851562\" xlink:href=\"#DejaVuSans-116\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p163268efb5\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"46.0125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n <defs>\n  <pattern height=\"72\" id=\"hd7e6d4ca53\" patternUnits=\"userSpaceOnUse\" width=\"72\" x=\"0\" y=\"0\">\n   <rect fill=\"#ff7f0e\" height=\"73\" width=\"73\" x=\"0\" y=\"0\"/>\n   <path d=\"M -36 36 \nL 36 -36 \nM -24 48 \nL 48 -24 \nM -12 60 \nL 60 -12 \nM 0 72 \nL 72 0 \nM 12 84 \nL 84 12 \nM 24 96 \nL 96 24 \nM 36 108 \nL 108 36 \n\" style=\"fill:#000000;stroke:#000000;stroke-linecap:butt;stroke-linejoin:miter;stroke-width:1.0;\"/>\n  </pattern>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## Vocabulary

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
that appear less than 3 times
as the same unknown ("&lt;unk&gt;") token.
Besides that,
we specify additional special tokens
such as for padding ("&lt;pad&gt;") sequences to the same length in minibatches,
and for marking the beginning ("&lt;bos&gt;") or end ("&lt;eos&gt;") of sequences.
Such special tokens are commonly used in
natural language processing tasks.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=3, 
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "9140"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Loading the Dataset

Recall that in language modeling
each sequence example,
either a segment of one sentence
or a span over multiple sentences,
has a fixed length.
This was specified by the `num_steps`
(number of time steps) argument in :numref:`sec_language_model`.
In machine translation, an example should contain a pair of source sentence and target sentence. These sentences might have different lengths, while we need same length examples to form a minibatch. 

One way to solve this problem is that if a sentence is longer than `num_steps`, we trim its length, otherwise pad with a special &lt;pad&gt; token to meet the length. Therefore we could transform any sentence to a fixed length.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "[47, 4, 1, 1, 1, 1, 1, 1, 1, 1]"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "X: [[ 30  37   4   1   1   1   1   1]\n [  0 156   4   1   1   1   1   1]]\nvalid lengths for X: [3 3]\nY: [[  2  61  59   0   4   3   1   1]\n [  2   0 152   4   3   1   1   1]]\nvalid lengths for Y: [6 5]\n"
 }
]
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
