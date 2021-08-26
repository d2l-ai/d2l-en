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

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "5b0ff805f64c406b9fdb4a02167e2c12",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "interactive(children=(Dropdown(description='tab', index=1, options=('mxnet', 'pytorch', 'tensorflow'), value='\u2026"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
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

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
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

```{.python .input  n=21}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    def download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            self.raw_text = f.read()
            
data = MTFraEng() 
data.download()
data.raw_text[:60]
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "'Go.\\tVa !\\nHi.\\tSalut !\\nRun!\\tCours\\u202f!\\nRun!\\tCourez\\u202f!\\nWho?\\tQui ?\\nW'"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

```

After downloading the dataset,
we [**proceed with several preprocessing steps**]
for the raw text data.
For instance,
we replace non-breaking space with space,
convert uppercase letters to lowercase ones,
and insert space between words and punctuation marks.

```{.python .input  n=23}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def preprocess(self):
    # Replace non-breaking space with space
    text = self.raw_text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    self.text = ''.join(out)

data.preprocess()
data.text[:60]
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "'go .\\tva !\\nhi .\\tsalut !\\nrun !\\tcours !\\nrun !\\tcourez !\\nwho ?\\tqu'"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.python .input  n=26}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def tokenize(self):
    self.source, self.target = [], []
    for i, line in enumerate(self.text.split('\n')):
        parts = line.split('\t')
        if len(parts) == 2:
            self.source.append(parts[0].split(' '))
            self.target.append(parts[1].split(' '))

data.tokenize()        
data.source[:4], data.target[:4]
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "([['go', '.'], ['hi', '.'], ['run', '!'], ['run', '!']],\n [['va', '!'], ['salut', '!'], ['cours', '!'], ['courez', '!']])"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let's [**plot the histogram of the number of tokens per text sequence.**]
In this simple English-French dataset,
most of the text sequences have fewer than 20 tokens.

```{.python .input  n=42}
%%tab all
d2l.set_figsize((4.5, 2.5))
_, _, patches = d2l.plt.hist([[len(l) for l in data.source], 
                              [len(l) for l in data.target]])
d2l.plt.xlabel('# tokens per sequence'), d2l.plt.ylabel('count')
d2l.plt.xlim([0, 20])
for patch in patches[1].patches: patch.set_hatch('-')
_ = d2l.plt.legend(['source', 'target'])
```

```{.json .output n=42}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 335.485938 180.65625\" width=\"335.485938pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2021-08-26T19:12:44.467028</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.4.2, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 335.485937 180.65625 \nL 335.485937 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 66.053125 143.1 \nL 317.153125 143.1 \nL 317.153125 7.2 \nL 66.053125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 98.193925 143.1 \nL 126.317125 143.1 \nL 126.317125 13.671429 \nL 98.193925 13.671429 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 168.501925 143.1 \nL 196.625125 143.1 \nL 196.625125 69.112838 \nL 168.501925 69.112838 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 238.809925 143.1 \nL 266.933125 143.1 \nL 266.933125 138.560551 \nL 238.809925 138.560551 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 309.117925 143.1 \nL 337.241125 143.1 \nL 337.241125 142.652168 \nL 309.117925 142.652168 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_7\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 379.425925 143.1 \nL 407.549125 143.1 \nL 407.549125 143.045112 \nL 379.425925 143.045112 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_8\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 449.733925 143.1 \nL 477.857125 143.1 \nL 477.857125 143.083783 \nL 449.733925 143.083783 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_9\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 520.041925 143.1 \nL 548.165125 143.1 \nL 548.165125 143.092515 \nL 520.041925 143.092515 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_10\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 590.349925 143.1 \nL 618.473125 143.1 \nL 618.473125 143.098753 \nL 590.349925 143.098753 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_11\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 660.657925 143.1 \nL 688.781125 143.1 \nL 688.781125 143.097505 \nL 660.657925 143.097505 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_12\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 730.965925 143.1 \nL 759.089125 143.1 \nL 759.089125 143.1 \nL 730.965925 143.1 \nz\n\" style=\"fill:#1f77b4;\"/>\n   </g>\n   <g id=\"patch_13\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 126.317125 143.1 \nL 154.440325 143.1 \nL 154.440325 26.397854 \nL 126.317125 26.397854 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_14\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 196.625125 143.1 \nL 224.748325 143.1 \nL 224.748325 59.453878 \nL 196.625125 59.453878 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_15\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 266.933125 143.1 \nL 295.056325 143.1 \nL 295.056325 136.044456 \nL 266.933125 136.044456 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_16\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 337.241125 143.1 \nL 365.364325 143.1 \nL 365.364325 142.176891 \nL 337.241125 142.176891 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_17\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 407.549125 143.1 \nL 435.672325 143.1 \nL 435.672325 142.993967 \nL 407.549125 142.993967 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_18\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 477.857125 143.1 \nL 505.980325 143.1 \nL 505.980325 143.058834 \nL 477.857125 143.058834 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_19\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 548.165125 143.1 \nL 576.288325 143.1 \nL 576.288325 143.097505 \nL 548.165125 143.097505 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_20\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 618.473125 143.1 \nL 646.596325 143.1 \nL 646.596325 143.097505 \nL 618.473125 143.097505 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_21\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 688.781125 143.1 \nL 716.904325 143.1 \nL 716.904325 143.097505 \nL 688.781125 143.097505 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"patch_22\">\n    <path clip-path=\"url(#pb9a023735e)\" d=\"M 759.089125 143.1 \nL 787.212325 143.1 \nL 787.212325 143.096258 \nL 759.089125 143.096258 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m601ba7f53d\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0.0 -->\n      <g transform=\"translate(58.101563 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" id=\"DejaVuSans-30\" transform=\"scale(0.015625)\"/>\n        <path d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" id=\"DejaVuSans-2e\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"97.440625\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 2.5 -->\n      <g transform=\"translate(89.489063 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" id=\"DejaVuSans-32\" transform=\"scale(0.015625)\"/>\n        <path d=\"M 691 4666 \nL 3169 4666 \nL 3169 4134 \nL 1269 4134 \nL 1269 2991 \nQ 1406 3038 1543 3061 \nQ 1681 3084 1819 3084 \nQ 2600 3084 3056 2656 \nQ 3513 2228 3513 1497 \nQ 3513 744 3044 326 \nQ 2575 -91 1722 -91 \nQ 1428 -91 1123 -41 \nQ 819 9 494 109 \nL 494 744 \nQ 775 591 1075 516 \nQ 1375 441 1709 441 \nQ 2250 441 2565 725 \nQ 2881 1009 2881 1497 \nQ 2881 1984 2565 2268 \nQ 2250 2553 1709 2553 \nQ 1456 2553 1204 2497 \nQ 953 2441 691 2322 \nL 691 4666 \nz\n\" id=\"DejaVuSans-35\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-35\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"128.828125\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 5.0 -->\n      <g transform=\"translate(120.876563 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-35\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"160.215625\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 7.5 -->\n      <g transform=\"translate(152.264063 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 525 4666 \nL 3525 4666 \nL 3525 4397 \nL 1831 0 \nL 1172 0 \nL 2766 4134 \nL 525 4134 \nL 525 4666 \nz\n\" id=\"DejaVuSans-37\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-37\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-35\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"191.603125\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10.0 -->\n      <g transform=\"translate(180.470313 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" id=\"DejaVuSans-31\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"222.990625\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 12.5 -->\n      <g transform=\"translate(211.857813 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-35\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_7\">\n     <g id=\"line2d_7\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"254.378125\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 15.0 -->\n      <g transform=\"translate(243.245313 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-35\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_8\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"285.765625\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 17.5 -->\n      <g transform=\"translate(274.632813 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-37\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-35\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_9\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"317.153125\" xlink:href=\"#m601ba7f53d\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 20.0 -->\n      <g transform=\"translate(306.020313 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"159.033203\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_10\">\n     <!-- # tokens per sequence -->\n     <g transform=\"translate(133.660938 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 3272 2816 \nL 2363 2816 \nL 2100 1772 \nL 3016 1772 \nL 3272 2816 \nz\nM 2803 4594 \nL 2478 3297 \nL 3391 3297 \nL 3719 4594 \nL 4219 4594 \nL 3897 3297 \nL 4872 3297 \nL 4872 2816 \nL 3775 2816 \nL 3519 1772 \nL 4513 1772 \nL 4513 1294 \nL 3397 1294 \nL 3072 0 \nL 2572 0 \nL 2894 1294 \nL 1978 1294 \nL 1656 0 \nL 1153 0 \nL 1478 1294 \nL 494 1294 \nL 494 1772 \nL 1594 1772 \nL 1856 2816 \nL 850 2816 \nL 850 3297 \nL 1978 3297 \nL 2297 4594 \nL 2803 4594 \nz\n\" id=\"DejaVuSans-23\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" id=\"DejaVuSans-74\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" id=\"DejaVuSans-6f\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 581 4863 \nL 1159 4863 \nL 1159 1991 \nL 2875 3500 \nL 3609 3500 \nL 1753 1863 \nL 3688 0 \nL 2938 0 \nL 1159 1709 \nL 1159 0 \nL 581 0 \nL 581 4863 \nz\n\" id=\"DejaVuSans-6b\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" id=\"DejaVuSans-65\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" id=\"DejaVuSans-6e\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" id=\"DejaVuSans-73\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1159 525 \nL 1159 -1331 \nL 581 -1331 \nL 581 3500 \nL 1159 3500 \nL 1159 2969 \nQ 1341 3281 1617 3432 \nQ 1894 3584 2278 3584 \nQ 2916 3584 3314 3078 \nQ 3713 2572 3713 1747 \nQ 3713 922 3314 415 \nQ 2916 -91 2278 -91 \nQ 1894 -91 1617 61 \nQ 1341 213 1159 525 \nz\nM 3116 1747 \nQ 3116 2381 2855 2742 \nQ 2594 3103 2138 3103 \nQ 1681 3103 1420 2742 \nQ 1159 2381 1159 1747 \nQ 1159 1113 1420 752 \nQ 1681 391 2138 391 \nQ 2594 391 2855 752 \nQ 3116 1113 3116 1747 \nz\n\" id=\"DejaVuSans-70\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" id=\"DejaVuSans-72\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 947 1747 \nQ 947 1113 1208 752 \nQ 1469 391 1925 391 \nQ 2381 391 2643 752 \nQ 2906 1113 2906 1747 \nQ 2906 2381 2643 2742 \nQ 2381 3103 1925 3103 \nQ 1469 3103 1208 2742 \nQ 947 2381 947 1747 \nz\nM 2906 525 \nQ 2725 213 2448 61 \nQ 2172 -91 1784 -91 \nQ 1150 -91 751 415 \nQ 353 922 353 1747 \nQ 353 2572 751 3078 \nQ 1150 3584 1784 3584 \nQ 2172 3584 2448 3432 \nQ 2725 3281 2906 2969 \nL 2906 3500 \nL 3481 3500 \nL 3481 -1331 \nL 2906 -1331 \nL 2906 525 \nz\n\" id=\"DejaVuSans-71\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 544 1381 \nL 544 3500 \nL 1119 3500 \nL 1119 1403 \nQ 1119 906 1312 657 \nQ 1506 409 1894 409 \nQ 2359 409 2629 706 \nQ 2900 1003 2900 1516 \nL 2900 3500 \nL 3475 3500 \nL 3475 0 \nL 2900 0 \nL 2900 538 \nQ 2691 219 2414 64 \nQ 2138 -91 1772 -91 \nQ 1169 -91 856 284 \nQ 544 659 544 1381 \nz\nM 1991 3584 \nL 1991 3584 \nz\n\" id=\"DejaVuSans-75\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" id=\"DejaVuSans-63\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-23\"/>\n      <use x=\"83.789062\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"115.576172\" xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"154.785156\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"215.966797\" xlink:href=\"#DejaVuSans-6b\"/>\n      <use x=\"270.251953\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"331.775391\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"395.154297\" xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"447.253906\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"479.041016\" xlink:href=\"#DejaVuSans-70\"/>\n      <use x=\"542.517578\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"604.041016\" xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"645.154297\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"676.941406\" xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"729.041016\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"790.564453\" xlink:href=\"#DejaVuSans-71\"/>\n      <use x=\"854.041016\" xlink:href=\"#DejaVuSans-75\"/>\n      <use x=\"917.419922\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"978.943359\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"1042.322266\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"1097.302734\" xlink:href=\"#DejaVuSans-65\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_10\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m66cf204fbe\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m66cf204fbe\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0 -->\n      <g transform=\"translate(52.690625 146.899219)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m66cf204fbe\" y=\"118.151116\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 20000 -->\n      <g transform=\"translate(27.240625 121.950335)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m66cf204fbe\" y=\"93.202233\"/>\n      </g>\n     </g>\n     <g id=\"text_13\">\n      <!-- 40000 -->\n      <g transform=\"translate(27.240625 97.001451)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" id=\"DejaVuSans-34\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_13\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m66cf204fbe\" y=\"68.253349\"/>\n      </g>\n     </g>\n     <g id=\"text_14\">\n      <!-- 60000 -->\n      <g transform=\"translate(27.240625 72.052568)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" id=\"DejaVuSans-36\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-36\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m66cf204fbe\" y=\"43.304465\"/>\n      </g>\n     </g>\n     <g id=\"text_15\">\n      <!-- 80000 -->\n      <g transform=\"translate(27.240625 47.103684)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" id=\"DejaVuSans-38\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-38\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_15\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"66.053125\" xlink:href=\"#m66cf204fbe\" y=\"18.355581\"/>\n      </g>\n     </g>\n     <g id=\"text_16\">\n      <!-- 100000 -->\n      <g transform=\"translate(20.878125 22.1548)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"190.869141\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"254.492188\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"318.115234\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_17\">\n     <!-- count -->\n     <g transform=\"translate(14.798438 89.25625)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"54.980469\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"116.162109\" xlink:href=\"#DejaVuSans-75\"/>\n      <use x=\"179.541016\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"242.919922\" xlink:href=\"#DejaVuSans-74\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_23\">\n    <path d=\"M 66.053125 143.1 \nL 66.053125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_24\">\n    <path d=\"M 317.153125 143.1 \nL 317.153125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_25\">\n    <path d=\"M 66.053125 143.1 \nL 317.153125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_26\">\n    <path d=\"M 66.053125 7.2 \nL 317.153125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_27\">\n     <path d=\"M 244.95 44.55625 \nL 310.153125 44.55625 \nQ 312.153125 44.55625 312.153125 42.55625 \nL 312.153125 14.2 \nQ 312.153125 12.2 310.153125 12.2 \nL 244.95 12.2 \nQ 242.95 12.2 242.95 14.2 \nL 242.95 42.55625 \nQ 242.95 44.55625 244.95 44.55625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"patch_28\">\n     <path d=\"M 246.95 23.798437 \nL 266.95 23.798437 \nL 266.95 16.798437 \nL 246.95 16.798437 \nz\n\" style=\"fill:#1f77b4;\"/>\n    </g>\n    <g id=\"text_18\">\n     <!-- source -->\n     <g transform=\"translate(274.95 23.798437)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"52.099609\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"113.28125\" xlink:href=\"#DejaVuSans-75\"/>\n      <use x=\"176.660156\" xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"215.523438\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"270.503906\" xlink:href=\"#DejaVuSans-65\"/>\n     </g>\n    </g>\n    <g id=\"patch_29\">\n     <path d=\"M 246.95 38.476562 \nL 266.95 38.476562 \nL 266.95 31.476562 \nL 246.95 31.476562 \nz\n\" style=\"fill:url(#hc71f152476);\"/>\n    </g>\n    <g id=\"text_19\">\n     <!-- target -->\n     <g transform=\"translate(274.95 38.476562)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" id=\"DejaVuSans-61\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2906 1791 \nQ 2906 2416 2648 2759 \nQ 2391 3103 1925 3103 \nQ 1463 3103 1205 2759 \nQ 947 2416 947 1791 \nQ 947 1169 1205 825 \nQ 1463 481 1925 481 \nQ 2391 481 2648 825 \nQ 2906 1169 2906 1791 \nz\nM 3481 434 \nQ 3481 -459 3084 -895 \nQ 2688 -1331 1869 -1331 \nQ 1566 -1331 1297 -1286 \nQ 1028 -1241 775 -1147 \nL 775 -588 \nQ 1028 -725 1275 -790 \nQ 1522 -856 1778 -856 \nQ 2344 -856 2625 -561 \nQ 2906 -266 2906 331 \nL 2906 616 \nQ 2728 306 2450 153 \nQ 2172 0 1784 0 \nQ 1141 0 747 490 \nQ 353 981 353 1791 \nQ 353 2603 747 3093 \nQ 1141 3584 1784 3584 \nQ 2172 3584 2450 3431 \nQ 2728 3278 2906 2969 \nL 2906 3500 \nL 3481 3500 \nL 3481 434 \nz\n\" id=\"DejaVuSans-67\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"100.488281\" xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"139.851562\" xlink:href=\"#DejaVuSans-67\"/>\n      <use x=\"203.328125\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"264.851562\" xlink:href=\"#DejaVuSans-74\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pb9a023735e\">\n   <rect height=\"135.9\" width=\"251.1\" x=\"66.053125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n <defs>\n  <pattern height=\"72\" id=\"hc71f152476\" patternUnits=\"userSpaceOnUse\" width=\"72\" x=\"0\" y=\"0\">\n   <rect fill=\"#ff7f0e\" height=\"73\" width=\"73\" x=\"0\" y=\"0\"/>\n   <path d=\"M 0 66 \nL 72 66 \nM 0 54 \nL 72 54 \nM 0 42 \nL 72 42 \nM 0 30 \nL 72 30 \nM 0 18 \nL 72 18 \nM 0 6 \nL 72 6 \n\" style=\"fill:#000000;stroke:#000000;stroke-linecap:butt;stroke-linejoin:miter;stroke-width:1.0;\"/>\n  </pattern>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 324x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

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

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## Reading the Dataset
:label:`subsec_mt_data_loading`

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

The following `truncate_pad` function
(**truncates or pads text sequences**) as described before.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

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

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## [**Putting All Things Together**]

Finally, we define the `load_data_nmt` function
to return the data iterator, together with
the vocabularies for both the source language and the target language.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

Let's [**read the first minibatch from the English-French dataset.**]

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
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
