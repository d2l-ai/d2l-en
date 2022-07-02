# Mise en œuvre concise des réseaux neuronaux récurrents
:label:`sec_rnn-concise` 

 Bien que :numref:`sec_rnn-scratch` ait été instructif de voir comment les RNN sont mis en œuvre,
ce n'est ni pratique ni rapide.
Cette section montre comment implémenter le même modèle de langage plus efficacement
en utilisant des fonctions fournies par les API de haut niveau
d'un cadre d'apprentissage profond.
Nous commençons comme précédemment par lire le jeu de données *The Time Machine*.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## [**Définir le modèle**]

Nous définissons la classe suivante
en utilisant le RNN implémenté
par des API de haut niveau.

:begin_tab:`mxnet`
Plus précisément, pour initialiser l'état caché,
, nous invoquons la méthode membre `begin_state`.
Celle-ci renvoie une liste
qui contient
un état caché initial
pour chaque exemple du minilot,
dont la forme est
(nombre de couches cachées, taille du lot, nombre d'unités cachées).
Pour certains modèles
qui seront introduits ultérieurement
(par exemple, la mémoire à long terme),
une telle liste contient également
d'autres informations.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

Héritant de la classe `RNNLMScratch` dans :numref:`sec_rnn-scratch` , 
la classe suivante `RNNLM` définit un modèle de langage complet basé sur un RNN.
Notez que nous devons créer une couche de sortie séparée entièrement connectée.

```{.python .input}
%%tab all
class RNNLM(d2l.RNNLMScratch):  #@save
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('pytorch'):
            self.linear = nn.LazyLinear(self.vocab_size)
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet', 'pytorch'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

## Entraînement et prédiction

Avant d'entraîner le modèle, faisons [**une prédiction avec un modèle dont les poids sont aléatoires.**]
Étant donné que nous n'avons pas entraîné le réseau, il générera des prédictions absurdes.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Comme il est évident, ce modèle ne fonctionne pas du tout. Ensuite, nous [**entraînons notre modèle avec des API de haut niveau**].

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

Comparé à :numref:`sec_rnn-scratch` ,
ce modèle atteint une perplexité comparable,
bien que dans un délai plus court, en raison de l'optimisation du code par
les API de haut niveau du cadre d'apprentissage profond.
Nous pouvons également générer des tokens prédits suivant la chaîne de préfixe spécifiée.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Résumé

* Les API de haut niveau du cadre d'apprentissage profond fournissent une implémentation des RNN.
* L'utilisation des API de haut niveau permet un apprentissage plus rapide des RNN que l'utilisation de son implémentation à partir de zéro.

## Exercices

1. Pouvez-vous rendre le modèle RNN surajusté en utilisant les API de haut niveau ?
1. Implémentez le modèle autorégressif de :numref:`sec_sequence` en utilisant un RNN.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:
