# Momentum
:label:`sec_momentum` 

Dans :numref:`sec_sgd`, nous avons examiné ce qui se passe lors de la descente de gradient stochastique, c'est-à-dire lors de l'optimisation où seule une variante bruyante du gradient est disponible. En particulier, nous avons remarqué que pour les gradients bruyants, nous devons être très prudents lorsqu'il s'agit de choisir le taux d'apprentissage face au bruit. Si nous le diminuons trop rapidement, la convergence s'arrête. Si nous sommes trop indulgents, nous ne parvenons pas à converger vers une solution suffisamment bonne puisque le bruit continue à nous éloigner de l'optimalité.

### Notions de base

Dans cette section, nous allons explorer des algorithmes d'optimisation plus efficaces, notamment pour certains types de problèmes d'optimisation courants dans la pratique.


### Moyennes fuyantes

Dans la section précédente, nous avons abordé la SGD par minilots comme moyen d'accélérer les calculs. Cette méthode a également eu pour effet secondaire de réduire la variance en calculant la moyenne des gradients. La descente de gradient stochastique par minilots peut être calculée par :

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Pour garder une notation simple, nous utilisons ici $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ comme gradient de descente stochastique pour l'échantillon $i$ en utilisant les poids mis à jour au moment $t-1$.
Il serait bien de pouvoir bénéficier de l'effet de la réduction de la variance même au-delà du calcul de la moyenne des gradients sur un minilot. Une option pour accomplir cette tâche consiste à remplacer le calcul du gradient par une "moyenne fuyante" :

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$ 

pour un certain $\beta \in (0, 1)$. Cela remplace effectivement le gradient instantané par un gradient qui a été moyenné sur plusieurs gradients *passés*. $\mathbf{v}$ est appelé *momentum*. Il accumule les gradients passés de la même manière qu'une boule lourde qui roule sur le terrain de la fonction objectif intègre les forces passées. Pour voir ce qui se passe plus en détail, développons récursivement $\mathbf{v}_t$ en

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

Le grand $\beta$ équivaut à une moyenne à long terme, tandis que le petit $\beta$ ne représente qu'une légère correction par rapport à une méthode de gradient. Le nouveau remplacement du gradient ne pointe plus dans la direction de la descente la plus abrupte sur une instance particulière mais plutôt dans la direction d'une moyenne pondérée des gradients passés. Cela nous permet de réaliser la plupart des avantages du calcul de la moyenne sur un lot sans le coût du calcul des gradients sur celui-ci. Nous reviendrons plus en détail sur cette procédure de calcul de la moyenne plus tard.

Le raisonnement ci-dessus a constitué la base de ce que l'on appelle aujourd'hui les méthodes de gradient *accélérées*, telles que les gradients avec momentum. Elles présentent l'avantage supplémentaire d'être beaucoup plus efficaces dans les cas où le problème d'optimisation est mal conditionné (c'est-à-dire lorsqu'il y a certaines directions où la progression est beaucoup plus lente que dans d'autres, comme dans un canyon étroit). En outre, ils nous permettent de faire une moyenne des gradients ultérieurs pour obtenir des directions de descente plus stables. En effet, l'aspect de l'accélération même pour les problèmes convexes sans bruit est l'une des principales raisons pour lesquelles le momentum fonctionne et pourquoi il fonctionne si bien.

Comme on peut s'y attendre, en raison de son efficacité, le momentum est un sujet bien étudié en optimisation pour l'apprentissage profond et au-delà. Voir, par exemple, le magnifique [expository article](https://distill.pub/2017/momentum/) de :cite:`Goh.2017` pour une analyse approfondie et une animation interactive. Il a été proposé par :cite:`Polyak.1964`. :cite:`Nesterov.2018` présente une discussion théorique détaillée dans le contexte de l'optimisation convexe. On sait depuis longtemps que le momentum dans l'apprentissage profond est bénéfique. Voir, par exemple, la discussion de :cite:`Sutskever.Martens.Dahl.ea.2013` pour plus de détails.

### Un problème mal conditionné

Pour mieux comprendre les propriétés géométriques de la méthode de l'élan, nous revisitons la descente de gradient, mais avec une fonction objective nettement moins agréable. Rappelez-vous que dans :numref:`sec_gd` nous avons utilisé $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, c'est-à-dire un objectif ellipsoïde modérément déformé. Nous déformons davantage cette fonction en l'étirant dans la direction de $x_1$ via

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$ 

Comme précédemment, $f$ a son minimum à $(0, 0)$. Cette fonction est *très* plate dans la direction de $x_1$. Voyons ce qui se passe lorsque nous exécutons la descente du gradient comme précédemment sur cette nouvelle fonction. Nous choisissons un taux d'apprentissage de $0.4$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Par construction, le gradient dans la direction $x_2$ est *beaucoup* plus élevé et change beaucoup plus rapidement que dans la direction horizontale $x_1$. Nous sommes donc coincés entre deux choix indésirables : si nous choisissons un petit taux d'apprentissage, nous nous assurons que la solution ne diverge pas dans la direction $x_2$, mais nous nous retrouvons avec une convergence lente dans la direction $x_1$. Inversement, avec un taux d'apprentissage élevé, nous progressons rapidement dans la direction $x_1$ mais nous divergeons dans la direction $x_2$. L'exemple ci-dessous illustre ce qui se passe même après une légère augmentation du taux d'apprentissage de $0.4$ à $0.6$. La convergence dans la direction $x_1$ s'améliore mais la qualité globale de la solution est bien pire.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### La méthode des moments forts

La méthode des moments forts nous permet de résoudre le problème de descente de gradient décrit ci-dessus.
En regardant la trace d'optimisation ci-dessus, on pourrait penser que la moyenne des gradients sur le passé fonctionnerait bien. Après tout, dans la direction $x_1$, cela va agréger des gradients bien alignés, augmentant ainsi la distance que nous parcourons à chaque pas. À l'inverse, dans la direction $x_2$ où les gradients oscillent, un gradient agrégé réduira la taille des pas en raison des oscillations qui s'annulent.
En utilisant $\mathbf{v}_t$ au lieu du gradient $\mathbf{g}_t$, on obtient les équations de mise à jour suivantes :

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Notez que pour $\beta = 0$ nous récupérons la descente de gradient régulière. Avant d'approfondir les propriétés mathématiques, voyons rapidement comment l'algorithme se comporte en pratique.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Comme nous pouvons le constater, même avec le même taux d'apprentissage que celui utilisé précédemment, le momentum converge toujours bien. Voyons ce qui se passe lorsque nous diminuons le paramètre de momentum. Le réduire de moitié à $\beta = 0.25$ conduit à une trajectoire qui converge à peine. Néanmoins, c'est beaucoup mieux que sans momentum (lorsque la solution diverge).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Notez que nous pouvons combiner le momentum avec la descente de gradient stochastique et, en particulier, la descente de gradient stochastique en minibatchs. Le seul changement est que, dans ce cas, nous remplaçons les gradients $\mathbf{g}_{t, t-1}$ par $\mathbf{g}_t$. Enfin, par commodité, nous initialisons $\mathbf{v}_0 = 0$ au moment $t=0$. Voyons ce que le leaky averaging fait réellement aux mises à jour.

### Poids effectif de l'échantillon

Rappelons que $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Dans la limite, les termes s'ajoutent à $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. En d'autres termes, plutôt que de faire un pas de taille $\eta$ dans la descente par gradient ou la descente par gradient stochastique, nous faisons un pas de taille $\frac{\eta}{1-\beta}$ tout en traitant une direction de descente potentiellement bien meilleure. Ce sont deux avantages en un. Pour illustrer le comportement de la pondération pour différents choix de $\beta$, considérez le diagramme ci-dessous.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Expériences pratiques

Voyons comment le momentum fonctionne en pratique, c'est-à-dire lorsqu'il est utilisé dans le contexte d'un optimiseur approprié. Pour cela, nous avons besoin d'une mise en œuvre un peu plus évolutive.

### Mise en œuvre à partir de zéro

Par rapport à la descente de gradient stochastique (minibatch), la méthode de l'élan doit maintenir un ensemble de variables auxiliaires, à savoir la vitesse. Elle a la même forme que les gradients (et les variables du problème d'optimisation). Dans l'implémentation ci-dessous, nous appelons ces variables `states`.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Voyons comment cela fonctionne en pratique.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Lorsque nous augmentons l'hyperparamètre de momentum `momentum` à 0,9, cela équivaut à une taille d'échantillon effective nettement plus importante de $\frac{1}{1 - 0.9} = 10$. Nous réduisons légèrement le taux d'apprentissage à $0.01$ pour garder le contrôle.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

La réduction du taux d'apprentissage permet de résoudre les problèmes d'optimisation non lisse. En le fixant à $0.005$, on obtient de bonnes propriétés de convergence.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Mise en œuvre concise

Il y a très peu à faire dans Gluon puisque le solveur standard `sgd` a déjà intégré le momentum. Le réglage des paramètres correspondants donne une trajectoire très similaire.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

### Analyse théorique

Jusqu'à présent, l'exemple 2D de $f(x) = 0.1 x_1^2 + 2 x_2^2$ semblait plutôt artificiel. Nous allons maintenant voir qu'il est en fait assez représentatif des types de problèmes que l'on peut rencontrer, du moins dans le cas de la minimisation de fonctions objectives quadratiques convexes.

### Fonctions convexes quadratiques

Considérons la fonction

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$ 

Il s'agit d'une fonction quadratique générale. Pour les matrices définies positives $\mathbf{Q} \succ 0$, c'est-à-dire pour les matrices avec des valeurs propres positives, cette fonction a un minimiseur à $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ avec une valeur minimale $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Nous pouvons donc réécrire $h$ comme suit :

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$ 

Le gradient est donné par $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. C'est-à-dire qu'il est donné par la distance entre $\mathbf{x}$ et le minimiseur, multipliée par $\mathbf{Q}$. Par conséquent, l'élan est également une combinaison linéaire de termes $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

Puisque $\mathbf{Q}$ est définie positive, elle peut être décomposée en son système propre via $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ pour une matrice orthogonale (rotation) $\mathbf{O}$ et une matrice diagonale $\boldsymbol{\Lambda}$ de valeurs propres positives. Cela nous permet d'effectuer un changement de variables de $\mathbf{x}$ à $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ pour obtenir une expression beaucoup plus simple :

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$ 

Ici $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Comme $\mathbf{O}$ n'est qu'une matrice orthogonale, cela ne perturbe pas les gradients de manière significative. Exprimée en termes de $\mathbf{z}$ la descente de gradient devient

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$ 

Le fait important dans cette expression est que la descente de gradient *ne mélange pas* entre différents espaces propres. En d'autres termes, lorsqu'elle est exprimée en termes de système propre de $\mathbf{Q}$, le problème d'optimisation se déroule dans le sens des coordonnées. Ceci est également valable pour la quantité de mouvement.

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Ce faisant, nous venons de prouver le théorème suivant : La descente de gradient avec et sans momentum pour une fonction quadratique convexe se décompose en une optimisation par coordonnées dans la direction des vecteurs propres de la matrice quadratique.

### Fonctions scalaires

Compte tenu du résultat ci-dessus, voyons ce qui se passe lorsque nous minimisons la fonction $f(x) = \frac{\lambda}{2} x^2$. Pour la descente du gradient, nous avons

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$ 

Chaque fois que $|1 - \eta \lambda| < 1$ cette optimisation converge à un taux exponentiel puisqu'après $t$ étapes nous avons $x_t = (1 - \eta \lambda)^t x_0$. Cela montre comment le taux de convergence s'améliore initialement à mesure que nous augmentons le taux d'apprentissage $\eta$ jusqu'à $\eta \lambda = 1$. Au-delà, les choses divergent et pour $\eta \lambda > 2$ le problème d'optimisation diverge.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Pour analyser la convergence dans le cas du momentum, nous commençons par réécrire les équations de mise à jour en termes de deux scalaires : un pour $x$ et un pour le momentum $v$. Cela donne :

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Nous avons utilisé $\mathbf{R}$ pour désigner le $2 \times 2$ qui régit le comportement de convergence. Après $t$ étapes, le choix initial $[v_0, x_0]$ devient $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Par conséquent, ce sont les valeurs propres de $\mathbf{R}$ qui déterminent la vitesse de convergence. Voir les sites [Distill post](https://distill.pub/2017/momentum/) de :cite:`Goh.2017` pour une belle animation et :cite:`Flammarion.Bach.2015` pour une analyse détaillée. On peut montrer que l'élan de $0 < \eta \lambda < 2 + 2 \beta$ converge. Il s'agit d'une plus grande plage de paramètres réalisables par rapport à $0 < \eta \lambda < 2$ pour la descente de gradient. Cela suggère également qu'en général, de grandes valeurs de $\beta$ sont souhaitables. D'autres détails nécessitent une quantité assez importante de détails techniques et nous suggérons au lecteur intéressé de consulter les publications originales.

## Résumé

* Momentum remplace les gradients par une moyenne fuyante des gradients passés. Cela accélère la convergence de manière significative.
* Il est souhaitable à la fois pour la descente de gradient sans bruit et la descente de gradient stochastique (bruyante).
* Le momentum empêche le blocage du processus d'optimisation qui est beaucoup plus susceptible de se produire pour la descente par gradient stochastique.
* Le nombre effectif de gradients est donné par $\frac{1}{1-\beta}$ en raison de la repondération exponentielle des données passées.
* Dans le cas de problèmes convexes quadratiques, ceci peut être analysé explicitement en détail.
* L'implémentation est assez simple, mais elle nécessite de stocker un vecteur d'état supplémentaire (momentum $\mathbf{v}$).

## Exercices

1. Utilisez d'autres combinaisons d'hyperparamètres de momentum et de taux d'apprentissage et observez et analysez les différents résultats expérimentaux.
1. Essayez GD et le momentum pour un problème quadratique où vous avez plusieurs valeurs propres, c'est-à-dire $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, par exemple $\lambda_i = 2^{-i}$. Tracez comment les valeurs de $x$ diminuent pour l'initialisation $x_i = 1$.
1. Déterminez la valeur minimale et le minimiseur pour $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
1. Qu'est-ce qui change lorsque nous effectuons une descente de gradient stochastique avec momentum ? Que se passe-t-il lorsque nous utilisons la descente de gradient stochastique en minibatch avec momentum ? Expérimentez avec les paramètres ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
