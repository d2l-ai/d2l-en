# Implémentation d'un réseau neuronal récurrent à partir de zéro
:label:`sec_rnn-scratch` 

 Dans cette section, nous allons implémenter un RNN
à partir de zéro
pour un modèle de langage au niveau des caractères,
conformément à nos descriptions
dans :numref:`sec_rnn` .
Un tel modèle
sera entraîné sur le livre de H. G. Wells *The Time Machine*.
Comme précédemment, nous commençons par lire l'ensemble de données, qui est présenté dans :numref:`sec_text-sequence` .

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## Modèle RNN

En suivant les descriptions de
:numref:`subsec_rnn_w_hidden_states` ,
nous commençons par définir la classe pour le modèle RNN
avec ses paramètres de modèle uniquement.
Le nombre d'unités cachées `num_hiddens` est un hyperparamètre accordable.

```{.python .input}
%%tab all
class RNNScratch(d2l.Module):  #@save
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

[**La méthode suivante `forward` définit comment calculer la sortie et l'état caché à un pas de temps.**]
Notez que
le modèle RNN
boucle à travers la dimension la plus externe de `inputs`
 de sorte qu'il met à jour l'état caché d'un minilot,
pas de temps par pas de temps.
En outre,
la fonction d'activation utilise ici la fonction $\tanh$.
Comme
décrit dans :numref:`sec_mlp` , la valeur moyenne
de la fonction $\tanh$ est 0, lorsque les éléments sont uniformément
distribués sur les nombres réels.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is not None:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.W_hh.shape[0]))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

Par exemple, nous pouvons introduire un minilot de séquences d'entrée
dans un modèle RNN comme suit.

```{.python .input}
%%tab all
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

Vérifions si le modèle RNN
produit des résultats de forme correcte,
par exemple, pour nous assurer que la dimensionnalité de l'état caché reste inchangée.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    assert len(a) == n, f'list\'s len {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

d2l.check_len(outputs, num_steps)
d2l.check_shape(outputs[0], (batch_size, num_hiddens))
d2l.check_shape(state, (batch_size, num_hiddens))
```

## Modèle de langage basé sur RNN

La classe `RNNLMScratch` suivante définit 
un modèle de langage basé sur RNN,
où un RNN implémenté à partir de zéro
est transmis via l'argument `rnn`
 de la méthode `__init__`.
Lors de l'apprentissage des modèles de langage, les entrées et les sorties proviennent du même vocabulaire. Par conséquent, ils ont la même dimension, qui est égale à la taille du vocabulaire.
Notez que nous utilisons la perplexité pour évaluer le modèle. Comme nous l'avons vu dans :numref:`subsec_perplexity` , cela garantit que les séquences de différentes longueurs sont comparables.

```{.python .input}
%%tab all
class RNNLMScratch(d2l.Classifier):  #@save
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('pytorch'):
            self.W_hq = nn.Parameter(
                d2l.randn(
                    self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
            self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**One-Hot Encoding**]

Rappelons que chaque token est représenté par un indice numérique dans le vocabulaire.
L'alimentation directe de ces indices à un réseau neuronal pourrait rendre difficile l'apprentissage de
.
Nous représentons souvent chaque token sous la forme d'un vecteur de caractéristiques plus expressif.
La représentation la plus simple est appelée codage *à un coup*,
qui est présenté
dans :numref:`subsec_classification-problem` .

En bref, nous faisons correspondre chaque indice à un vecteur unitaire différent : supposons que le nombre de tokens différents dans le vocabulaire est $N$ (`len(vocab)`) et que les indices des tokens vont de $0$ à $N-1$.
Si l'indice d'un token est le nombre entier $i$, nous créons un vecteur de tous les 0 d'une longueur de $N$ et définissons l'élément à la position $i$ à 1.
Ce vecteur est le vecteur à un coup du token original. Les vecteurs à un coup avec les indices 0 et 2 sont présentés ci-dessous.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

(**La forme du minilot**) que nous échantillonnons à chaque fois (**est (taille du lot, nombre de pas de temps).
La méthode `one_hot` transforme un tel mini-batch en un tenseur tridimensionnel dont la dernière dimension est égale à la taille du vocabulaire (`len(vocab)`).**))
Nous transposons souvent l'entrée de manière à obtenir une sortie
de forme
(nombre de pas de temps, taille du lot, taille du vocabulaire).
Cela nous permettra
de boucler plus facilement
à travers la dimension la plus externe
pour mettre à jour les états cachés d'un mini-batch,
pas de temps par pas de temps
(par exemple, dans la méthode `forward` ci-dessus).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
```

### Transformation des sorties RNN

Le modèle de langage
utilise une couche de sortie entièrement connectée
pour transformer les sorties RNN 
en prédictions de jetons à chaque pas de temps.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Vérifions [**si les sorties du calcul prédictif ont la forme correcte.**]

```{.python .input}
%%tab all
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
d2l.check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradient Clipping**]

Pour une séquence de longueur $T$,
nous calculons les gradients sur ces $T$ pas de temps en une itération, ce qui donne une chaîne de produits matriciels de longueur $\mathcal{O}(T)$ pendant la rétropropagation.
Comme mentionné dans :numref:`sec_numerical_stability` , cela peut entraîner une instabilité numérique, par exemple, les gradients peuvent exploser ou disparaître, lorsque $T$ est grand. Par conséquent, les modèles RNN ont souvent besoin d'une aide supplémentaire pour stabiliser la formation.

D'une manière générale,
lors de la résolution d'un problème d'optimisation,
nous prenons des mesures de mise à jour pour le paramètre du modèle,
disons sous la forme vectorielle
$\mathbf{x}$ ,
dans la direction du gradient négatif $\mathbf{g}$ sur un minibatch.
Par exemple,
avec $\eta > 0$ comme taux d'apprentissage,
dans une itération nous mettons à jour
$\mathbf{x}$ 
 comme $\mathbf{x} - \eta \mathbf{g}$.
Supposons en outre que la fonction objectif $f$
 se comporte bien, c'est-à-dire qu'elle est *Lipschitz continue* avec une constante $L$.
Autrement dit,
pour tout $\mathbf{x}$ et $\mathbf{y}$, nous avons

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$ 

 Dans ce cas, nous pouvons supposer sans risque que si nous actualisons le vecteur de paramètres par $\eta \mathbf{g}$, alors

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$ 

 ce qui signifie que
nous n'observerons pas de changement de plus de $L \eta \|\mathbf{g}\|$. C'est à la fois une malédiction et une bénédiction.
Du côté de la malédiction,
elle limite la vitesse de progression ;
tandis que du côté de la bénédiction,
elle limite la mesure dans laquelle les choses peuvent mal tourner si nous allons dans la mauvaise direction.

Parfois, les gradients peuvent être très importants et l'algorithme d'optimisation peut ne pas converger. Nous pourrions remédier à ce problème en réduisant le taux d'apprentissage $\eta$. Mais que se passe-t-il si nous n'obtenons que *rarement* de grands gradients ? Dans ce cas, une telle approche peut sembler totalement injustifiée. Une alternative populaire consiste à écrêter le gradient $\mathbf{g}$ en le projetant dans une boule de rayon donné, disons $\theta$ via

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

En procédant ainsi, nous savons que la norme du gradient ne dépasse jamais $\theta$ et que le gradient mis à jour
est entièrement aligné sur la direction originale de $\mathbf{g}$.
Cette méthode a également l'effet secondaire souhaitable de limiter l'influence qu'un minibatch
donné (et à l'intérieur de celui-ci, un échantillon donné) peut exercer sur le vecteur de paramètres. Cela
confère un certain degré de robustesse au modèle. L'écrêtage du gradient fournit
une solution rapide à l'explosion du gradient. Bien qu'il ne résolve pas entièrement le problème, c'est l'une des nombreuses techniques permettant de l'atténuer.

Nous définissons ci-dessous une fonction permettant d'écrêter les gradients d'un modèle
.
Notez également que nous calculons la norme du gradient sur tous les paramètres du modèle.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

## Formation

En utilisant le jeu de données *The Time Machine* (`data`),
nous formons un modèle de langage au niveau des caractères (`model`)
basé sur le RNN (`rnn`) implémenté à partir de zéro.
Notez que nous coupons les gradients avant de mettre à jour les paramètres du modèle. Cela garantit que le modèle ne diverge pas, même si les gradients explosent à un moment donné au cours du processus d'apprentissage.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Prédiction

Nous devons [**définir la fonction de prédiction pour générer de nouveaux caractères à la suite de le `prefix` fourni par l'utilisateur**],
qui est une chaîne contenant plusieurs caractères.
En bouclant sur ces premiers caractères dans `prefix`,
nous continuons à transmettre l'état caché
au pas de temps suivant sans que
ne génère aucune sortie.
C'est ce qu'on appelle la période de *réchauffement*,
pendant laquelle le modèle se met à jour
(par exemple, il met à jour l'état caché)
mais ne fait pas de prédictions.
Après la période de réchauffement,
l'état caché est généralement meilleur que
sa valeur initialisée au début.
On génère donc les caractères prédits et on les émet.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict `num_preds` steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Dans ce qui suit, nous spécifions le préfixe 
et lui faisons générer 20 caractères supplémentaires.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

Bien que l'implémentation du modèle RNN ci-dessus à partir de zéro soit instructive, elle n'est pas pratique.
Dans la section suivante, nous verrons comment améliorer le modèle RNN,
notamment en facilitant l'implémentation de
et en le faisant fonctionner plus rapidement.


## Résumé

* Nous pouvons entraîner un modèle de langage RNN au niveau des caractères pour générer du texte suivant le préfixe fourni par l'utilisateur.
* Un modèle de langage RNN simple consiste en un codage d'entrée, une modélisation RNN et une génération de sortie.
* L'écrêtage du gradient empêche l'explosion du gradient, mais ne peut pas corriger les gradients disparus.
* Une période d'échauffement permet à un modèle de se mettre à jour (par exemple, obtenir un meilleur état caché que sa valeur initialisée) avant de faire une prédiction.


## Exercices

1. Le modèle de langage implémenté prédit-il le prochain token en se basant sur tous les tokens passés jusqu'au tout premier token dans *The Time Machine* ? Quel hyperparamètre contrôle la longueur de l'historique utilisé pour la prédiction ?
1. Montrez que l'encodage à un coup est équivalent à choisir un encastrement différent pour chaque objet.
1. Ajustez les hyperparamètres (par exemple, le nombre d'époques, le nombre d'unités cachées, le nombre d'étapes temporelles dans un minibatch et le taux d'apprentissage) pour améliorer la perplexité.
   * Jusqu'où pouvez-vous descendre ?
* Remplacer le codage à un coup par des incorporations apprenables. Cela conduit-il à de meilleures performances ?
   * Comment cela fonctionnera-t-il sur d'autres livres de H. G. Wells, par exemple, [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)?
1. Modifiez la fonction de prédiction de façon à utiliser l'échantillonnage plutôt que de choisir le prochain caractère le plus probable.
   * Que se passe-t-il ?
* Orientez le modèle vers des sorties plus probables, par exemple en échantillonnant à partir de $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ pour $\alpha > 1$.
1. Exécutez le code de cette section sans couper le gradient. Que se passe-t-il ?
1. Remplacez la fonction d'activation utilisée dans cette section par ReLU et répétez les expériences de cette section. Avons-nous encore besoin d'écrêter le gradient ? Pourquoi ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
