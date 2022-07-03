# Gated Recurrent Units (GRU)
:label:`sec_gru` 

 Dans :numref:`sec_bptt` ,
nous avons examiné comment les gradients sont calculés
dans les RNN.
En particulier, nous avons découvert que les produits longs des matrices peuvent conduire
à des gradients qui disparaissent ou explosent.
Réfléchissons brièvement à ce que de telles anomalies de gradient
signifient en pratique :

* Nous pouvons rencontrer une situation dans laquelle une observation précoce est hautement
 significative pour prédire toutes les observations futures. Considérons le cas quelque peu contourné
 où la première observation contient une somme de contrôle et où le but est
 de discerner si la somme de contrôle est correcte à la fin de la séquence. Dans ce cas
, l'influence du premier jeton est vitale. Nous aimerions disposer de certains mécanismes
 pour stocker les premières informations vitales dans une *cellule mémoire*. Sans un tel mécanisme
, nous devrons attribuer un gradient très important à cette observation,
 puisqu'il affecte toutes les observations suivantes.
* Nous pouvons rencontrer des situations où certains tokens ne portent aucune observation pertinente
. Par exemple, lors de l'analyse d'une page Web, il peut y avoir du code HTML auxiliaire
 qui n'est pas pertinent pour évaluer le sentiment
 véhiculé par la page. Nous aimerions disposer d'un mécanisme permettant de *sauter* de tels tokens
 dans la représentation de l'état latent.
* Nous pouvons rencontrer des situations où il existe une rupture logique entre les parties d'une séquence
. Par exemple, il peut y avoir une transition entre les chapitres d'un livre
, ou une transition entre un marché baissier et un marché haussier pour les titres. Dans ce cas,
 il serait agréable de disposer d'un moyen de *réinitialiser* notre représentation de l'état interne
.

Un certain nombre de méthodes ont été proposées pour résoudre ce problème. L'une des plus anciennes est la mémoire à long terme :cite:`Hochreiter.Schmidhuber.1997` dont nous parlerons dans
 :numref:`sec_lstm` . L'unité récurrente gated (GRU)
:cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` est une variante légèrement plus rationalisée
qui offre souvent des performances comparables et qui est nettement plus rapide à calculer
 :cite:`Chung.Gulcehre.Cho.ea.2014` .
En raison de sa simplicité, commençons par la GRU.

## Gated Hidden State

La principale distinction entre les RNNs classiques et les GRUs
est que ces derniers supportent le gating de l'état caché.
Cela signifie que nous avons des mécanismes dédiés pour
quand un état caché doit être *mis à jour* et
aussi quand il doit être *réinitialisé*.
Ces mécanismes sont appris et ils répondent aux préoccupations énumérées ci-dessus.
Par exemple, si le premier jeton est d'une grande importance
, nous apprendrons à ne pas mettre à jour l'état caché après la première observation.
De même, nous apprendrons à sauter les observations temporaires non pertinentes.
Enfin, nous apprendrons à réinitialiser l'état latent chaque fois que nécessaire.
Nous en parlons en détail ci-dessous.


#### Porte de réinitialisation et porte de mise à jour

La première chose que nous devons introduire est
la *porte de réinitialisation* et la *porte de mise à jour*.
Nous les concevons comme des vecteurs avec des entrées dans $(0, 1)$
 de sorte que nous puissions effectuer des combinaisons convexes.
Par exemple,
une porte de réinitialisation nous permettrait de contrôler la quantité de l'état précédent que nous souhaitons encore mémoriser.
De même, une porte de mise à jour nous permettrait de contrôler la part du nouvel état qui n'est qu'une copie de l'ancien.

Nous commençons par concevoir ces portes.
:numref:`fig_gru_1` illustre les entrées pour
les portes de réinitialisation et de mise à jour dans un GRU, étant donné l'entrée
du pas de temps actuel
et l'état caché du pas de temps précédent.
Les sorties de deux portes
sont données par deux couches entièrement connectées
avec une fonction d'activation sigmoïde.

![Computing the reset gate and the update gate in a GRU model.](../img/gru-1.svg)
:label:`fig_gru_1`

Mathématiquement,
pour un pas de temps donné $t$,
supposons que l'entrée est
un minibatch
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (nombre d'exemples : $n$, nombre d'entrées : $d$) et l'état caché du pas de temps précédent est $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (nombre d'unités cachées : $h$). Ensuite, la porte de réinitialisation $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ et la porte de mise à jour $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ sont calculées comme suit :

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

où $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ et
$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ sont des paramètres de poids
et $\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$ sont des biais
.
Notez que la diffusion (voir :numref:`subsec_broadcasting` ) est déclenchée pendant la sommation.
Nous utilisons des fonctions sigmoïdes (introduites dans :numref:`sec_mlp` ) pour transformer les valeurs d'entrée dans l'intervalle $(0, 1)$.

### État caché candidat

Ensuite, intégrons
la porte de réinitialisation $\mathbf{R}_t$ avec
le mécanisme régulier de mise à jour de l'état latent
dans :eqref:`rnn_h_with_state` .
Cela conduit à l'état caché candidat
*suivant
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ au pas de temps $t$:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$ 
 :eqlabel:`gru_tilde_H` 

 où $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ et $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$
 sont des paramètres de poids,
$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ 
 est le biais,
et le symbole $\odot$ est l'opérateur produit (par éléments) de Hadamard.
Nous utilisons ici une non-linéarité sous la forme de tanh pour nous assurer que les valeurs de l'état caché candidat restent dans l'intervalle $(-1, 1)$.

Le résultat est un *candidat* puisque nous devons encore incorporer l'action de la porte de mise à jour.

Par rapport à :eqref:`rnn_h_with_state` ,
l'influence des états précédents
peut maintenant être réduite grâce à la multiplication par éléments de
$\mathbf{R}_t$ et $\mathbf{H}_{t-1}$
 dans :eqref:`gru_tilde_H` .
Lorsque les entrées de la porte de réinitialisation $\mathbf{R}_t$ sont proches de 1, nous récupérons un RNN vanille comme dans :eqref:`rnn_h_with_state` .
Pour toutes les entrées de la porte de réinitialisation $\mathbf{R}_t$ qui sont proches de 0, l'état caché candidat est le résultat d'un MLP avec $\mathbf{X}_t$ comme entrée. Tout état caché préexistant est donc *réinitialisé* aux valeurs par défaut.

:numref:`fig_gru_2` illustre le flux de calcul après l'application de la porte de réinitialisation.

![Computing the candidate hidden state in a GRU model.](../img/gru-2.svg)
:label:`fig_gru_2`


### État caché

Enfin, nous devons intégrer l'effet de la porte de mise à jour $\mathbf{Z}_t$. Celle-ci détermine dans quelle mesure le nouvel état caché $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ est simplement l'ancien état $\mathbf{H}_{t-1}$ et dans quelle mesure le nouvel état candidat $\tilde{\mathbf{H}}_t$ est utilisé.
La porte de mise à jour $\mathbf{Z}_t$ peut être utilisée à cette fin, simplement en prenant des combinaisons convexes par éléments entre $\mathbf{H}_{t-1}$ et $\tilde{\mathbf{H}}_t$.
Cela conduit à l'équation de mise à jour finale pour le GRU :

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$ 

 
 Chaque fois que la porte de mise à jour $\mathbf{Z}_t$ est proche de 1, nous conservons simplement l'ancien état. Dans ce cas, les informations provenant de $\mathbf{X}_t$ sont essentiellement ignorées, ce qui permet de sauter l'étape $t$ dans la chaîne de dépendance. En revanche, lorsque $\mathbf{Z}_t$ est proche de 0, le nouvel état latent $\mathbf{H}_t$ se rapproche de l'état latent candidat $\tilde{\mathbf{H}}_t$. Ces conceptions peuvent nous aider à faire face au problème du gradient évanescent dans les RNN et à mieux capturer les dépendances pour les séquences avec de grandes distances de pas de temps.
Par exemple,
si la porte de mise à jour a été proche de 1
pour tous les pas de temps d'une sous-séquence entière,
l'ancien état caché au pas de temps de son début
sera facilement conservé et transmis
à sa fin,
quelle que soit la longueur de la sous-séquence.



 :numref:`fig_gru_3` illustre le flux de calcul après que la porte de mise à jour soit en action.

![Computing the hidden state in a GRU model.](../img/gru-3.svg)
:label:`fig_gru_3`


En résumé, les GRU ont les deux caractéristiques suivantes :

* Les portes de réinitialisation aident à capturer les dépendances à court terme dans les séquences.
* Les portes de mise à jour aident à capturer les dépendances à long terme dans les séquences.

## Implémentation à partir de zéro

Pour mieux comprendre le modèle GRU, nous allons l'implémenter à partir de zéro.

```{.python .input  n=5}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input  n=6}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=8}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

### (**Initialisation des paramètres du modèle**)

La première étape consiste à initialiser les paramètres du modèle.
Nous tirons les poids d'une distribution gaussienne
dont l'écart-type est `sigma` et nous fixons le biais à 0. L'hyperparamètre `num_hiddens` définit le nombre d'unités cachées.
Nous instancions tous les poids et biais relatifs à la porte de mise à jour, à la porte de réinitialisation et à l'état caché candidat.

```{.python .input}
%%tab all
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        if tab.selected('mxnet'):
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))            
        if tab.selected('pytorch'):
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        if tab.selected('tensorflow'):
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))            
            
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state        
```

### Définition du modèle

Nous sommes maintenant prêts à [**définir le calcul GRU forward**].
Sa structure est la même que celle de la cellule RNN de base, sauf que les équations de mise à jour sont plus complexes.

```{.python .input}
%%tab all
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    matmul_H = lambda A, B: d2l.matmul(A, B) if H is not None else 0
    outputs = []
    for X in inputs:
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) + (
            d2l.matmul(H, self.W_hz) if H is not None else 0) + self.b_z)
        if H is None: H = d2l.zeros_like(Z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) + 
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilda = d2l.tanh(d2l.matmul(X, self.W_xh) + 
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilda
        outputs.append(H)
    return outputs, (H, )
```

### Training

[**Training**] un modèle de langage sur le jeu de données *The Time Machine*
fonctionne exactement de la même manière que dans :numref:`sec_rnn-scratch` .

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch'):
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Concise Implementation**]

Dans les API de haut niveau,
nous pouvons directement
instancier un modèle GPU.
Cela encapsule tous les détails de configuration que nous avons explicités ci-dessus.

```{.python .input}
%%tab all
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens)
        if tab.selected('tensorflow'):
            self.rnn = tf.keras.layers.GRU(num_hiddens, return_sequences=True, 
                                           return_state=True)
```

Le code est nettement plus rapide lors de l'apprentissage car il utilise des opérateurs compilés plutôt que Python pour de nombreux détails que nous avons explicités précédemment.

```{.python .input}
%%tab all
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

Après l'apprentissage,
nous imprimons la perplexité sur l'ensemble d'apprentissage
et la séquence prédite suivant
le préfixe fourni.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Résumé

* Les RNN à porte peuvent mieux capturer les dépendances pour les séquences avec de grandes distances de pas de temps.
* Les portes de réinitialisation aident à capturer les dépendances à court terme dans les séquences.
* Les portes de mise à jour aident à capturer les dépendances à long terme dans les séquences.
* Les GRUs contiennent des RNNs de base comme cas extrême lorsque la porte de réinitialisation est activée. Ils peuvent également sauter des sous-séquences en activant la porte de mise à jour.


## Exercices

1. Supposons que nous voulons uniquement utiliser l'entrée au pas de temps $t'$ pour prédire la sortie au pas de temps $t > t'$. Quelles sont les meilleures valeurs pour les portes de réinitialisation et de mise à jour pour chaque pas de temps ?
1. Ajustez les hyperparamètres et analysez leur influence sur le temps d'exécution, la perplexité et la séquence de sortie.
1. Comparez le temps d'exécution, la perplexité et les chaînes de sortie pour les implémentations `rnn.RNN` et `rnn.GRU` entre elles.
1. Que se passe-t-il si vous n'implémentez qu'une partie d'un GRU, par exemple, avec seulement une porte de réinitialisation ou seulement une porte de mise à jour ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1056)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3860)
:end_tab:
