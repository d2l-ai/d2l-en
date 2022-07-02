## Adagrad
:label:`sec_adagrad` 

 Commençons par examiner les problèmes d'apprentissage avec des caractéristiques qui apparaissent peu fréquemment.


## Caractéristiques éparses et taux d'apprentissage

Imaginez que nous formons un modèle de langage. Pour obtenir une bonne précision, nous souhaitons généralement diminuer le taux d'apprentissage au fur et à mesure de la formation, généralement à un taux de $\mathcal{O}(t^{-\frac{1}{2}})$ ou plus lent. Considérons maintenant un modèle qui s'entraîne sur des caractéristiques éparses, c'est-à-dire des caractéristiques qui n'apparaissent que rarement. Cette situation est courante dans le langage naturel, par exemple, il est beaucoup moins probable que nous voyions le mot *préconditionnement* que *apprentissage*. Cependant, elle est également courante dans d'autres domaines tels que la publicité informatique et le filtrage collaboratif personnalisé. Après tout, il y a beaucoup de choses qui ne sont intéressantes que pour un petit nombre de personnes.

Les paramètres associés aux caractéristiques peu fréquentes ne reçoivent des mises à jour significatives que lorsque ces caractéristiques apparaissent. Avec un taux d'apprentissage décroissant, nous pourrions nous retrouver dans une situation où les paramètres des caractéristiques communes convergent assez rapidement vers leurs valeurs optimales, alors que pour les caractéristiques peu fréquentes, nous sommes encore loin de les observer suffisamment fréquemment pour pouvoir déterminer leurs valeurs optimales. En d'autres termes, le taux d'apprentissage diminue soit trop lentement pour les caractéristiques fréquentes, soit trop rapidement pour les caractéristiques peu fréquentes.

Une solution possible pour remédier à ce problème serait de compter le nombre de fois que nous voyons une caractéristique particulière et de l'utiliser comme une horloge pour ajuster les taux d'apprentissage. Autrement dit, plutôt que de choisir un taux d'apprentissage de la forme $\eta = \frac{\eta_0}{\sqrt{t + c}}$, nous pourrions utiliser $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$. Ici, $s(i, t)$ compte le nombre de non-zéros pour la caractéristique $i$ que nous avons observés jusqu'au moment $t$. Cette méthode est en fait assez facile à mettre en œuvre, sans surcharge significative. Cependant, elle échoue lorsque nous n'avons pas tout à fait de sparsité mais plutôt des données où les gradients sont souvent très petits et seulement rarement grands. Après tout, on ne sait pas très bien où tracer la limite entre quelque chose qui peut être considéré comme une caractéristique observée ou non.

Adagrad par :cite:`Duchi.Hazan.Singer.2011` aborde ce problème en remplaçant le compteur plutôt grossier $s(i, t)$ par un agrégat des carrés des gradients précédemment observés. En particulier, il utilise $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ comme moyen d'ajuster le taux d'apprentissage. Cela présente deux avantages : premièrement, nous n'avons plus besoin de décider quand un gradient est suffisamment important. Deuxièmement, il s'adapte automatiquement à l'ampleur des gradients. Les coordonnées qui correspondent habituellement à de grands gradients sont réduites de manière significative, tandis que d'autres avec de petits gradients reçoivent un traitement beaucoup plus doux. En pratique, cela conduit à une procédure d'optimisation très efficace pour la publicité informatique et les problèmes connexes. Mais cela cache certains des avantages supplémentaires inhérents à Adagrad qui sont mieux compris dans le contexte du préconditionnement.


## Préconditionnement

Les problèmes d'optimisation convexe sont propices à l'analyse des caractéristiques des algorithmes. Après tout, pour la plupart des problèmes non convexes, il est difficile de dériver des garanties théoriques significatives, mais *l'intuition* et *la perspicacité* font souvent mouche.  Examinons le problème de la minimisation de $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

Comme nous l'avons vu dans :numref:`sec_momentum` , il est possible de réécrire ce problème en termes de sa décomposition en eigences $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ pour arriver à un problème beaucoup plus simple où chaque coordonnée peut être résolue individuellement :

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$ 

 Nous avons utilisé ici $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$ et par conséquent $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$. Le problème modifié a pour minimiseur $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ et pour valeur minimale $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Cette dernière est beaucoup plus facile à calculer puisque $\boldsymbol{\Lambda}$ est une matrice diagonale contenant les valeurs propres de $\mathbf{Q}$.

Si nous perturbons légèrement $\mathbf{c}$, nous pourrions espérer ne trouver que de légers changements dans le minimiseur de $f$. Malheureusement, ce n'est pas le cas. Si de légères modifications de $\mathbf{c}$ entraînent des modifications tout aussi légères de $\bar{\mathbf{c}}$, ce n'est pas le cas pour le minimiseur de $f$ (et de $\bar{f}$ respectivement). Lorsque les valeurs propres de $\boldsymbol{\Lambda}_i$ sont grandes, nous ne verrons que de petits changements dans $\bar{x}_i$ et dans le minimum de $\bar{f}$. Inversement, pour de petites valeurs de $\boldsymbol{\Lambda}_i$, les changements dans $\bar{x}_i$ peuvent être spectaculaires. Le rapport entre la plus grande et la plus petite valeur propre est appelé le nombre de conditions d'un problème d'optimisation.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Si le nombre de condition $\kappa$ est grand, il est difficile de résoudre le problème d'optimisation avec précision. Nous devons nous assurer que nous sommes attentifs à obtenir une large gamme dynamique de valeurs correctes. Notre analyse conduit à une question évidente, bien que quelque peu naïve : ne pourrions-nous pas simplement "résoudre" le problème en déformant l'espace de sorte que toutes les valeurs propres soient $1$. En théorie, c'est assez facile : nous n'avons besoin que des valeurs propres et des vecteurs propres de $\mathbf{Q}$ pour redimensionner le problème de $\mathbf{x}$ à $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. Dans le nouveau système de coordonnées, $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ pourrait être simplifié en $\|\mathbf{z}\|^2$. Hélas, cette suggestion n'est pas très pratique. Le calcul des valeurs propres et des vecteurs propres est en général *beaucoup plus coûteux* que la résolution du problème réel.

Si le calcul exact des valeurs propres peut s'avérer coûteux, les deviner et les calculer, même de manière approximative, peut déjà s'avérer bien meilleur que de ne rien faire du tout. En particulier, nous pourrions utiliser les entrées diagonales de $\mathbf{Q}$ et le rééchelonner en conséquence. C'est *beaucoup* moins cher que de calculer les valeurs propres.

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Dans ce cas, nous avons $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ et spécifiquement $\tilde{\mathbf{Q}}_{ii} = 1$ pour tout $i$. Dans la plupart des cas, cela simplifie considérablement le nombre de conditions. Par exemple, dans les cas que nous avons discutés précédemment, cela éliminerait entièrement le problème en question puisque le problème est aligné sur l'axe.

Malheureusement, nous sommes confrontés à un autre problème : dans l'apprentissage profond, nous n'avons généralement pas accès à la dérivée seconde de la fonction objective : pour $\mathbf{x} \in \mathbb{R}^d$, la dérivée seconde, même sur un minibatch, peut nécessiter $\mathcal{O}(d^2)$ de l'espace et du travail pour la calculer, ce qui la rend pratiquement irréalisable. L'idée ingénieuse d'Adagrad est d'utiliser un proxy pour cette diagonale insaisissable du Hessien qui est à la fois relativement bon marché à calculer et efficace - la magnitude du gradient lui-même.

Afin de comprendre pourquoi cela fonctionne, examinons $\bar{f}(\bar{\mathbf{x}})$. Nous avons que

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$ 

 où $\bar{\mathbf{x}}_0$ est le minimiseur de $\bar{f}$. Par conséquent, la magnitude du gradient dépend à la fois de $\boldsymbol{\Lambda}$ et de la distance de l'optimalité. Si $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ ne changeait pas, cela suffirait. Après tout, dans ce cas, la magnitude du gradient $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ suffit. Comme AdaGrad est un algorithme de descente de gradient stochastique, nous verrons des gradients avec une variance non nulle même à l'optimalité. Par conséquent, nous pouvons utiliser sans risque la variance des gradients comme un indicateur bon marché de l'échelle du Hessian. Une analyse approfondie dépasse le cadre de cette section (elle ferait plusieurs pages). Nous renvoyons le lecteur à :cite:`Duchi.Hazan.Singer.2011` pour plus de détails.

## L'algorithme

Formalisons la discussion ci-dessus. Nous utilisons la variable $\mathbf{s}_t$ pour accumuler la variance du gradient passé comme suit.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Ici, les opérations sont appliquées par coordonnées. C'est-à-dire que $\mathbf{v}^2$ a des entrées $v_i^2$. De même, $\frac{1}{\sqrt{v}}$ a des entrées $\frac{1}{\sqrt{v_i}}$ et $\mathbf{u} \cdot \mathbf{v}$ a des entrées $u_i v_i$. Comme précédemment, $\eta$ est le taux d'apprentissage et $\epsilon$ est une constante additive qui garantit que nous ne divisons pas par $0$. Enfin, nous initialisons $\mathbf{s}_0 = \mathbf{0}$.

Comme dans le cas du momentum, nous devons garder la trace d'une variable auxiliaire, dans ce cas pour permettre un taux d'apprentissage individuel par coordonnée. Cela n'augmente pas le coût d'Adagrad de manière significative par rapport à SGD, simplement parce que le coût principal est généralement de calculer $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ et sa dérivée.

Notez que l'accumulation des gradients au carré dans $\mathbf{s}_t$ signifie que $\mathbf{s}_t$ croît essentiellement à un taux linéaire (un peu plus lentement que linéairement en pratique, puisque les gradients diminuent initialement). Cela conduit à un taux d'apprentissage de $\mathcal{O}(t^{-\frac{1}{2}})$, bien qu'ajusté sur une base par coordonnée. Pour les problèmes convexes, cela est parfaitement adéquat. Dans l'apprentissage profond, cependant, nous pourrions vouloir diminuer le taux d'apprentissage plus lentement. Cela a conduit à un certain nombre de variantes d'Adagrad que nous aborderons dans les chapitres suivants. Pour l'instant, voyons comment il se comporte dans un problème convexe quadratique. Nous utilisons le même problème que précédemment :

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$ 

 Nous allons mettre en œuvre Adagrad en utilisant le même taux d'apprentissage que précédemment, c'est-à-dire $\eta = 0.4$. Comme nous pouvons le constater, la trajectoire itérative de la variable indépendante est plus lisse. Cependant, en raison de l'effet cumulatif de $\boldsymbol{s}_t$, le taux d'apprentissage décroît continuellement, de sorte que la variable indépendante ne bouge pas autant au cours des étapes ultérieures de l'itération.

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
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Lorsque nous augmentons le taux d'apprentissage à $2$, nous constatons un bien meilleur comportement. Cela indique déjà que la diminution du taux d'apprentissage pourrait être plutôt agressive, même dans le cas sans bruit et nous devons nous assurer que les paramètres convergent de manière appropriée.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Implémentation from Scratch

Tout comme la méthode momentum, Adagrad doit maintenir une variable d'état de la même forme que les paramètres.

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Par rapport à l'expérience de :numref:`sec_minibatch_sgd` , nous utilisons un taux d'apprentissage
plus élevé pour entraîner le modèle.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Mise en œuvre concise

En utilisant l'instance `Trainer` de l'algorithme `adagrad`, nous pouvons invoquer l'algorithme Adagrad dans Gluon.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Résumé

* Adagrad diminue dynamiquement le taux d'apprentissage sur une base par coordonnée.
* Il utilise la magnitude du gradient comme moyen d'ajuster la vitesse de progression - les coordonnées avec de grands gradients sont compensées par un taux d'apprentissage plus faible.
* Le calcul de la dérivée seconde exacte est généralement irréalisable dans les problèmes d'apprentissage profond en raison des contraintes de mémoire et de calcul. Le gradient peut être un proxy utile.
* Si le problème d'optimisation a une structure plutôt inégale, Adagrad peut aider à atténuer la distorsion.
* Adagrad est particulièrement efficace pour les caractéristiques éparses où le taux d'apprentissage doit diminuer plus lentement pour les termes peu fréquents.
* Sur les problèmes d'apprentissage profond, Adagrad peut parfois être trop agressif dans la réduction des taux d'apprentissage. Nous discuterons des stratégies pour atténuer ce problème dans le contexte de :numref:`sec_adam` .

## Exercices

1. Prouvez que pour une matrice orthogonale $\mathbf{U}$ et un vecteur $\mathbf{c}$, les conditions suivantes sont remplies : $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Pourquoi cela signifie-t-il que la magnitude des perturbations ne change pas après un changement orthogonal de variables ?
1. Essayez Adagrad pour $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ et aussi pour la fonction objectif a été tournée de 45 degrés, c'est-à-dire $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Se comporte-t-il différemment ?
1. Prouvez [Gerschgorin's circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) que les valeurs propres $\lambda_i$ d'une matrice $\mathbf{M}$ satisfont $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ pour au moins un choix de $j$.
1. Que nous apprend le théorème de Gerschgorin sur les valeurs propres de la matrice préconditionnée diagonalement $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
1. Essayez Adagrad pour un réseau profond approprié, tel que :numref:`sec_lenet` lorsqu'il est appliqué à Fashion-MNIST.
1. Comment devriez-vous modifier Adagrad pour obtenir une décroissance moins agressive du taux d'apprentissage ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
