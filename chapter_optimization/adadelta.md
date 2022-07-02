# Adadelta
:label:`sec_adadelta` 

 Adadelta est encore une autre variante d'AdaGrad (:numref:`sec_adagrad` ). La principale différence réside dans le fait qu'il diminue la quantité par laquelle le taux d'apprentissage est adaptatif aux coordonnées. De plus, il est traditionnellement considéré comme n'ayant pas de taux d'apprentissage puisqu'il utilise la quantité de changement elle-même comme calibration pour le changement futur. L'algorithme a été proposé dans :cite:`Zeiler.2012` . Il est assez simple, étant donné la discussion des algorithmes précédents jusqu'à présent. 

## L'algorithme

En résumé, Adadelta utilise deux variables d'état, $\mathbf{s}_t$ pour stocker une moyenne de fuite du second moment du gradient et $\Delta\mathbf{x}_t$ pour stocker une moyenne de fuite du second moment du changement des paramètres du modèle lui-même. Notez que nous utilisons la notation et la dénomination originales des auteurs pour des raisons de compatibilité avec d'autres publications et implémentations (il n'y a pas d'autre raison réelle pour laquelle on devrait utiliser des variables grecques différentes pour indiquer un paramètre servant le même but dans momentum, Adagrad, RMSProp, et Adadelta). 

Voici les détails techniques d'Adadelta. Étant donné que le paramètre du jour est $\rho$, nous obtenons les mises à jour fuyantes suivantes de manière similaire à :numref:`sec_rmsprop` :

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

La différence avec :numref:`sec_rmsprop` est que nous effectuons les mises à jour avec le gradient redimensionné $\mathbf{g}_t'$, c'est-à-dire,

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Alors, qu'est-ce que le gradient redimensionné $\mathbf{g}_t'$? Nous pouvons le calculer comme suit :

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

où $\Delta \mathbf{x}_{t-1}$ est la moyenne futile des gradients redimensionnés au carré $\mathbf{g}_t'$. Nous initialisons $\Delta \mathbf{x}_{0}$ par $0$ et le mettons à jour à chaque étape avec $\mathbf{g}_t'$, c'est-à-dire,

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

et $\epsilon$ (une petite valeur telle que $10^{-5}$) est ajoutée pour maintenir la stabilité numérique.



## Implémentation

Adadelta doit maintenir deux variables d'état pour chaque variable, $\mathbf{s}_t$ et $\Delta\mathbf{x}_t$. Cela donne l'implémentation suivante.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Le choix de $\rho = 0.9$ équivaut à un temps de demi-vie de 10 pour chaque mise à jour des paramètres. Cela a tendance à fonctionner assez bien. Nous obtenons le comportement suivant.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Pour une implémentation concise, nous utilisons simplement l'algorithme `adadelta` de la classe `Trainer`. Cela donne la ligne unique suivante pour une invocation beaucoup plus compacte.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Résumé

* Adadelta n'a pas de paramètre de taux d'apprentissage. Au lieu de cela, il utilise le taux de changement dans les paramètres eux-mêmes pour adapter le taux d'apprentissage. 
* Adadelta nécessite deux variables d'état pour stocker les seconds moments du gradient et le changement des paramètres. 
* Adadelta utilise des moyennes de fuite pour garder une estimation courante des statistiques appropriées. 

## Exercices

1. Ajustez la valeur de $\rho$. Que se passe-t-il ?
1. Montrez comment implémenter l'algorithme sans utiliser $\mathbf{g}_t'$. Pourquoi est-ce une bonne idée ?
1. Adadelta est-il vraiment sans taux d'apprentissage ? Pourriez-vous trouver des problèmes d'optimisation qui cassent Adadelta ?
1. Comparez Adadelta à Adagrad et RMS prop pour discuter de leur comportement de convergence.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
