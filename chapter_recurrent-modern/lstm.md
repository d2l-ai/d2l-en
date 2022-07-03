# Mémoire à long terme et à court terme (LSTM)
:label:`sec_lstm` 

 Le défi de traiter la préservation des informations à long terme et le saut d'entrée à court terme
dans les modèles de variables latentes existe depuis longtemps. L'une des premières approches
à relever ce défi a été la mémoire à long terme (LSTM)
 :cite:`Hochreiter.Schmidhuber.1997` . Elle partage de nombreuses propriétés de l'URG
.
Il est intéressant de noter que les LSTM ont une conception légèrement plus complexe
que les GRU, mais qu'elles sont antérieures aux GRU de près de deux décennies.



## Cellule de mémoire à grille

On peut dire que la conception des LSTM s'inspire
des portes logiques d'un ordinateur.
Le LSTM introduit une *cellule mémoire* (ou *cellule* pour faire court)
qui a la même forme que l'état caché
(certaines littératures considèrent la cellule mémoire
comme un type spécial d'état caché),
conçue pour enregistrer des informations supplémentaires.
Pour contrôler la cellule mémoire
, nous avons besoin d'un certain nombre de portes.
Une porte est nécessaire pour lire les entrées de la cellule
.
Nous l'appellerons la porte de sortie * de
.
Une deuxième porte est nécessaire pour décider du moment où il faut lire les données dans la cellule
.
Nous l'appellerons la *porte d'entrée*.
Enfin, nous avons besoin d'un mécanisme pour réinitialiser
le contenu de la cellule, régi par une *porte d'oubli*.
La motivation d'une telle conception
est la même que celle des GRU,
à savoir être capable de décider quand se souvenir et
quand ignorer les entrées dans l'état caché via un mécanisme dédié. Voyons
comment cela fonctionne en pratique.


### Porte d'entrée, porte d'oubli et porte de sortie

Tout comme dans les GRU,
les données qui alimentent les portes LSTM sont
l'entrée au pas de temps actuel et
l'état caché du pas de temps précédent,
comme illustré dans :numref:`fig_lstm_0` .
Elles sont traitées par
trois couches entièrement connectées avec une fonction d'activation sigmoïde pour calculer les valeurs de
les portes d'entrée, d'oubli et de sortie.
Par conséquent, les valeurs des trois portes
sont comprises entre $(0, 1)$.

![Computing the input gate, the forget gate, and the output gate in an LSTM model.](../img/lstm-0.svg) 
:label:`fig_lstm_0` 

 Mathématiquement,
supposons qu'il y ait $h$ unités cachées, que la taille du lot soit $n$, et que le nombre d'entrées soit $d$.
Ainsi, l'entrée est $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ et l'état caché du pas de temps précédent est $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$. En conséquence, les portes au pas de temps $t$
 sont définies comme suit : la porte d'entrée est $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, la porte d'oubli est $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, et la porte de sortie est $\mathbf{O}_t \in \mathbb{R}^{n \times h}$. Elles sont calculées comme suit :

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

où $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ et $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$ sont des paramètres de poids et $\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$ sont des paramètres de biais.

### Cellule de mémoire candidate

Nous concevons ensuite la cellule de mémoire. Comme nous n'avons pas encore spécifié l'action des différentes portes, nous présentons d'abord la cellule mémoire *candidate* $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$. Son calcul est similaire à celui des trois portes décrites ci-dessus, mais en utilisant une fonction $\tanh$ avec une plage de valeurs pour $(-1, 1)$ comme fonction d'activation. Cela conduit à l'équation suivante au pas de temps $t$:

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$ 

 où $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$ et $\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$ sont des paramètres de poids et $\mathbf{b}_c \in \mathbb{R}^{1 \times h}$ est un paramètre de biais.

Une illustration rapide de la cellule mémoire candidate est présentée dans :numref:`fig_lstm_1` .

![Computing the candidate memory cell in an LSTM model.](../img/lstm-1.svg)
:label:`fig_lstm_1`

### Cellule de mémoire

Dans les GRU, nous disposons d'un mécanisme pour régir l'entrée et l'oubli (ou le saut).
De même,
dans les LSTM, nous disposons de deux portes dédiées à ces fins : la porte d'entrée $\mathbf{I}_t$ régit le degré de prise en compte des nouvelles données via $\tilde{\mathbf{C}}_t$ et la porte d'oubli $\mathbf{F}_t$ traite du degré de conservation du contenu de l'ancienne cellule mémoire $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$. En utilisant la même astuce de multiplication ponctuelle que précédemment, nous obtenons l'équation de mise à jour suivante :

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$ 

 Si la porte d'oubli est toujours approximativement à 1 et la porte d'entrée toujours approximativement à 0, les cellules de mémoire passées $\mathbf{C}_{t-1}$ seront sauvegardées au fil du temps et transmises au pas de temps actuel.
Cette conception est introduite pour atténuer le problème du gradient évanescent et pour mieux capturer
les dépendances à long terme dans les séquences.

Nous arrivons ainsi au diagramme de flux dans :numref:`fig_lstm_2` .

![Computing the memory cell in an LSTM model.](../img/lstm-2.svg)

:label:`fig_lstm_2`


### État caché

Enfin, nous devons définir comment calculer l'état caché $\mathbf{H}_t \in \mathbb{R}^{n \times h}$. C'est là que la porte de sortie entre en jeu. Dans le LSTM, il s'agit simplement d'une version à porte de $\tanh$ de la cellule mémoire.
Cela garantit que les valeurs de $\mathbf{H}_t$ sont toujours dans l'intervalle $(-1, 1)$.

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$


Chaque fois que la porte de sortie s'approche de 1, nous transmettons effectivement toutes les informations de la mémoire au prédicteur, tandis que pour la porte de sortie proche de 0, nous conservons toutes les informations uniquement dans la cellule de mémoire et n'effectuons aucun traitement supplémentaire.



 :numref:`fig_lstm_3` a une illustration graphique du flux de données.

![Computing the hidden state in an LSTM model.](../img/lstm-3.svg)
:label:`fig_lstm_3`



## Implémentation à partir de zéro

Implémentons maintenant un LSTM à partir de zéro.
Comme pour les expériences de :numref:`sec_rnn-scratch` ,
nous chargeons d'abord *The Time Machine* dataset.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
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

### [**Initialisation des paramètres du modèle**]

Ensuite, nous devons définir et initialiser les paramètres du modèle. Comme précédemment, l'hyperparamètre `num_hiddens` définit le nombre d'unités cachées. Nous initialisons les poids suivant une distribution gaussienne avec un écart type de 0,01, et nous fixons les biais à 0.

```{.python .input}
%%tab all
class LSTMScratch(d2l.Module):  #@save
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

        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate 
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate 
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate 
        self.W_xc, self.W_hc, self.b_c = triple()  # Candidate memory cell 
```

le modèle réel [**Le modèle réel**] est défini comme ce que nous avons discuté précédemment : fournir trois portes et une cellule mémoire auxiliaire. Notez que seul l'état caché est transmis à la couche de sortie. La cellule mémoire $\mathbf{C}_t$ ne participe pas directement au calcul de la sortie.

```{.python .input}
%%tab all
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    H, C = None, None if H_C is None else H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) + (
            d2l.matmul(H, self.W_hi) if H is not None else 0) + self.b_i)
        if H is None:
            H, C = d2l.zeros_like(I), d2l.zeros_like(I)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) + 
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) + 
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilda = d2l.tanh(d2l.matmul(X, self.W_xc) + 
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilda
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

### [**Training**] et Prediction

Entraînons un LSTM de la même manière que nous l'avons fait dans :numref:`sec_gru` , en instanciant la classe `RNNLMScratch` comme introduit dans :numref:`sec_rnn-scratch` .

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch'):
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Concise Implementation**]

En utilisant des API de haut niveau,
nous pouvons directement instancier un modèle `LSTM`.
Celui-ci encapsule tous les détails de configuration que nous avons rendus explicites ci-dessus. Le code est nettement plus rapide car il utilise des opérateurs compilés plutôt que Python pour de nombreux détails que nous avons explicités auparavant.

```{.python .input}
%%tab mxnet
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()        
        self.rnn = rnn.LSTM(num_hiddens)    
            
    def forward(self, inputs, H_C=None):
        if H_C is None: H_C = self.rnn.begin_state(
            inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)    
```

```{.python .input}
%%tab pytorch
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()        
        self.rnn = nn.LSTM(num_inputs, num_hiddens)        
            
    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)        
```

```{.python .input}
%%tab tensorflow
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()        
        self.rnn = tf.keras.layers.LSTM(
                num_hiddens, return_sequences=True, 
                return_state=True, time_major=True)
            
    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(inputs, H_C)
        return outputs, H_C
```

```{.python .input}
%%tab all
if tab.selected('pytorch'):
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'tensorflow'):
    lstm = LSTM(num_hiddens=32)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

Les LSTM sont le modèle autorégressif à variables latentes prototypique avec un contrôle d'état non trivial.
De nombreuses variantes de ce modèle ont été proposées au fil des ans, par exemple, des couches multiples, des connexions résiduelles, différents types de régularisation. Cependant, l'entraînement des LSTM et d'autres modèles de séquence (tels que les GRU) est assez coûteux en raison de la dépendance à long terme de la séquence.
Plus tard, nous rencontrerons des modèles alternatifs tels que les transformateurs qui peuvent être utilisés dans certains cas.


## Résumé

* Les LSTMs ont trois types de portes : les portes d'entrée, les portes d'oubli et les portes de sortie qui contrôlent le flux d'informations.
* La sortie de la couche cachée du LSTM comprend l'état caché et la cellule mémoire. Seul l'état caché est transmis à la couche de sortie. La cellule de mémoire est entièrement interne.
* Les LSTM peuvent atténuer les gradients évanescents et explosifs.


## Exercices

1. Ajustez les hyperparamètres et analysez leur influence sur le temps d'exécution, la perplexité et la séquence de sortie.
1. Comment devriez-vous modifier le modèle pour générer des mots propres par opposition à des séquences de caractères ?
1. Comparez le coût de calcul des GRU, des LSTM et des RNN ordinaires pour une dimension cachée donnée. Portez une attention particulière aux coûts de entrainement et d'inférence.
1. Puisque la cellule mémoire candidate garantit que la plage de valeurs est comprise entre $-1$ et $1$ en utilisant la fonction $\tanh$, pourquoi l'état caché doit-il utiliser à nouveau la fonction $\tanh$ pour garantir que la plage de valeurs de sortie est comprise entre $-1$ et $1$?
1. Implémentez un modèle LSTM pour la prédiction de séries temporelles plutôt que pour la prédiction de séquences de caractères.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1057)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3861)
:end_tab:
