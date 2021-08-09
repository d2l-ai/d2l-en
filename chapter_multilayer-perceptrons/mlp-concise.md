```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Concise Implementation of Multilayer Perceptrons
:label:`sec_mlp_concise`

As you might expect, by (**relying on the high-level APIs,
we can implement MLPs even more concisely.**)

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Model

As compared with our concise implementation
of softmax regression implementation
(:numref:`sec_softmax_concise`),
the only difference is that we add
*two* fully connected layers
(previously, we added *one*).
The first is [**our hidden layer**],
the second is our output layer.

```{.python .input}
%%tab mxnet
class MLP(d2l.Classification):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classification):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.Linear(num_hiddens, num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classification):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])  
```

[**The training loop**] is exactly the same
as when we implemented softmax regression.
This modularity enables us to separate
matters concerning the model architecture
from orthogonal considerations.

```{.python .input}
%%tab all
if tab.selected(['mxnet', 'tensorflow']):
    model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
if tab.selected(['pytorch']):
    model = MLP(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Summary

* Using high-level APIs, we can implement MLPs much more concisely.
* For the same classification problem, the implementation of an MLP is the same as that of softmax regression except for additional hidden layers with activation functions.

## Exercises

1. Try adding different numbers of hidden layers (you may also modify the learning rate). What setting works best?
1. Try out different activation functions. Which one works best?
1. Try different schemes for initializing the weights. What method works best?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
