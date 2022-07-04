# RMSProp
:label:`sec_rmsprop` 

 
L'un des problèmes clés de :numref:`sec_adagrad` est que le taux d'apprentissage diminue selon un calendrier prédéfini de manière efficace $\mathcal{O}(t^{-\frac{1}{2}})$. Bien que cela soit généralement approprié pour les problèmes convexes, ce n'est peut-être pas idéal pour les problèmes non convexes, tels que ceux rencontrés dans l'apprentissage profond. Pourtant, l'adaptabilité coordonnée d'Adagrad est hautement souhaitable en tant que préconditionneur.

:cite:`Tieleman.Hinton.2012` a proposé l'algorithme RMSProp comme solution simple pour découpler l'ordonnancement des taux d'apprentissage adaptatifs coordonnés. Le problème est qu'Adagrad accumule les carrés du gradient $\mathbf{g}_t$ dans un vecteur d'état $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. En conséquence, $\mathbf{s}_t$ continue de croître sans limite en raison de l'absence de normalisation, essentiellement de manière linéaire lorsque l'algorithme converge.

Une façon de résoudre ce problème serait d'utiliser $\mathbf{s}_t / t$. Pour des distributions raisonnables de $\mathbf{g}_t$, cela convergera. Malheureusement, cela peut prendre beaucoup de temps jusqu'à ce que le comportement limite commence à avoir de l'importance puisque la procédure se souvient de la trajectoire complète des valeurs. Une alternative est d'utiliser une  leaky average de la même manière que nous l'avons utilisée dans la méthode du momentum, c'est-à-dire $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ pour un certain paramètre $\gamma > 0$. En gardant toutes les autres parties inchangées, on obtient RMSProp.

## L'algorithme

Écrivons les équations en détail.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

La constante $\epsilon > 0$ est généralement fixée à $10^{-6}$ pour s'assurer que nous ne souffrons pas de division par zéro ou de tailles de pas trop importantes. Compte tenu de cette expansion, nous sommes maintenant libres de contrôler le taux d'apprentissage $\eta$ indépendamment de la mise à l'échelle qui est appliquée sur une base par coordonnée. En termes de moyennes fuyantes, nous pouvons appliquer le même raisonnement que celui appliqué précédemment dans le cas de la méthode des moments. En élargissant la définition de $\mathbf{s}_t$, on obtient

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Comme précédemment dans :numref:`sec_momentum` nous utilisons $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. La somme des poids est donc normalisée à $1$ avec un temps de demi-vie d'une observation de $\gamma^{-1}$. Visualisons les poids pour les 40 derniers pas de temps pour différents choix de $\gamma$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Mise en œuvre à partir de zéro

Comme précédemment, nous utilisons la fonction quadratique $f(\mathbf{x})=0.1x_1^2+2x_2^2$ pour observer la trajectoire de RMSProp. Rappelez-vous que dans :numref:`sec_adagrad`, lorsque nous avons utilisé Adagrad avec un taux d'apprentissage de 0,4, les variables n'ont bougé que très lentement dans les derniers stades de l'algorithme car le taux d'apprentissage a diminué trop rapidement. Comme $\eta$ est contrôlé séparément, cela ne se produit pas avec RMSProp.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Ensuite, nous implémentons RMSProp pour être utilisé dans un réseau profond. C'est tout aussi simple.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Nous fixons le taux d'apprentissage initial à 0,01 et le terme de pondération $\gamma$ à 0,9. C'est-à-dire que $\mathbf{s}$ agrège en moyenne sur les observations passées $1/(1-\gamma) = 10$ du gradient carré.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Implémentation concise

Comme RMSProp est un algorithme plutôt populaire, il est également disponible dans l'instance `Trainer`. Tout ce que nous avons à faire est de l'instancier en utilisant un algorithme nommé `rmsprop`, en assignant $\gamma$ au paramètre `gamma1`.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Résumé

* RMSProp est très similaire à Adagrad dans la mesure où tous deux utilisent le carré du gradient pour mettre les coefficients à l'échelle.
* RMSProp partage avec momentum le leaky averaging. Cependant, RMSProp utilise cette technique pour ajuster le préconditionneur coefficient par coefficient.
* Le taux d'apprentissage doit être programmé par l'expérimentateur dans la pratique.
* Le coefficient $\gamma$ détermine la longueur de l'historique lors de l'ajustement de l'échelle par coordonnée.

## Exercices

1. Que se passe-t-il expérimentalement si nous réglons $\gamma = 1$? Pourquoi ?
1. Faites pivoter le problème d'optimisation pour minimiser $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Que se passe-t-il au niveau de la convergence ?
1. Essayez de voir ce qui arrive à RMSProp sur un vrai problème d'apprentissage automatique, comme l'entraînement sur Fashion-MNIST. Expérimentez avec différents choix pour ajuster le taux d'apprentissage.
1. Souhaitez-vous ajuster $\gamma$ au fur et à mesure de l'optimisation ? Dans quelle mesure RMSProp est-il sensible à cela ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
