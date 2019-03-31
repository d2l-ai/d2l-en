# Implementation of Recurrent Neural Networks from Scratch

In this section we implement a language model from scratch. It is based on a character-level recurrent neural network trained on H. G. Wells' 'The Time Machine'. As before, we start by reading the dataset first.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_time_machine()
```
