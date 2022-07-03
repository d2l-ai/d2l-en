# Optimisation et apprentissage profond

Dans cette section, nous allons discuter de la relation entre l'optimisation et l'apprentissage profond ainsi que des défis liés à l'utilisation de l'optimisation dans l'apprentissage profond.
Pour un problème d'apprentissage profond, nous commençons généralement par définir une *fonction de perte*. Une fois que nous avons la fonction de perte, nous pouvons utiliser un algorithme d'optimisation pour tenter de minimiser la perte.
En optimisation, une fonction de perte est souvent appelée la *fonction objectif* du problème d'optimisation. Par tradition et convention, la plupart des algorithmes d'optimisation sont concernés par la *minimisation*. Si nous devons maximiser un objectif, il existe une solution simple : il suffit de changer le signe de l'objectif.

## But de l'optimisation

Bien que l'optimisation fournisse un moyen de minimiser la fonction de perte pour l'apprentissage profond,
les buts de l'optimisation et de l'apprentissage profond sont fondamentalement différents
.
Le premier vise principalement à minimiser un objectif
alors que le second vise à trouver un modèle approprié, compte tenu d'une quantité finie de données
.
Dans :numref:`sec_model_selection`,
nous avons discuté en détail de la différence entre ces deux objectifs.
Par exemple, l'erreur d'apprentissage
et l'erreur de généralisation diffèrent généralement : puisque la fonction objective
de l'algorithme d'optimisation est généralement une fonction de perte basée sur l'ensemble de données d'apprentissage,
le but de l'optimisation est de réduire l'erreur d'apprentissage.
Cependant, l'objectif de l'apprentissage profond (ou plus largement de l'inférence statistique) est de
réduire l'erreur de généralisation.
Pour atteindre ce dernier objectif, nous devons prêter
attention au surajustement en plus d'utiliser l'algorithme d'optimisation pour
réduire l'erreur d'apprentissage.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Pour illustrer les différents objectifs susmentionnés,
considérons 
le risque empirique et le risque. 
Comme décrit dans
dans :numref:`subsec_empirical-risk-and-risk`,
le risque empirique
est une perte moyenne
sur l'ensemble de données d'apprentissage
tandis que le risque est la perte attendue 
sur la population entière de données.
Nous définissons ci-dessous deux fonctions :
la fonction de risque `f`
et la fonction de risque empirique `g`.
Supposons que nous ne disposions que d'une quantité finie de données d'apprentissage.
Par conséquent, ici, `g` est moins lisse que `f`.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Le graphique ci-dessous illustre que le minimum du risque empirique sur un ensemble de données d'entraînement peut se trouver à un endroit différent du minimum du risque (erreur de généralisation).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Défis de l'optimisation dans l'apprentissage profond

Dans ce chapitre, nous allons nous concentrer spécifiquement sur la performance des algorithmes d'optimisation dans la minimisation de la fonction objectif, plutôt que sur l'erreur de généralisation d'un modèle.

Dans :numref:`sec_linear_regression` 
nous avons fait la distinction entre les solutions analytiques et les solutions numériques dans les problèmes d'optimisation.

Dans l'apprentissage profond, la plupart des fonctions objectives sont
compliquées et n'ont pas de solutions analytiques. Nous devons donc utiliser des algorithmes d'optimisation numériques.

Les algorithmes d'optimisation de ce chapitre
entrent tous dans cette catégorie
.

L'optimisation de l'apprentissage profond présente de nombreux défis. Les minima locaux, les points de selle et les gradients évanescents sont parmi les plus contrariants. 
Examinons-les.


### Minima locaux

Pour toute fonction objective $f(x)$,
si la valeur de $f(x)$ à $x$ est inférieure aux valeurs de $f(x)$ à tout autre point à proximité de $x$, alors $f(x)$ pourrait être un minimum local.
Si la valeur de $f(x)$ à $x$ est le minimum de la fonction objectif sur l'ensemble du domaine,
alors $f(x)$ est le minimum global.

Par exemple, étant donné la fonction

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$ 

 , nous pouvons approximer le minimum local et le minimum global de cette fonction.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

La fonction objectif des modèles d'apprentissage profond comporte généralement de nombreux optima locaux. 
Lorsque la solution numérique d'un problème d'optimisation est proche de l'optimum local, la solution numérique obtenue par l'itération finale peut ne minimiser la fonction objectif que *localement*, plutôt que *globalement*, car le gradient des solutions de la fonction objectif s'approche ou devient nul. 
Seul un certain degré de bruit peut faire sortir le paramètre du minimum local. En fait, c'est l'une des propriétés bénéfiques de la descente de gradient stochastique par minibatchs,
où la variation naturelle des gradients sur les minibatchs est capable de déloger les paramètres des minima locaux.


### Points de selle

Outre les minima locaux, les points de selle sont une autre raison pour laquelle les gradients disparaissent. Un *point de selle* est un endroit où tous les gradients d'une fonction disparaissent, mais qui n'est ni un minimum global ni un minimum local. 
Considérons la fonction $f(x) = x^3$. Sa dérivée première et sa dérivée seconde disparaissent pour $x=0$. L'optimisation peut s'arrêter à ce point, même s'il ne s'agit pas d'un minimum.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Les points de selle dans les dimensions supérieures sont encore plus insidieux, comme le montre l'exemple ci-dessous. Considérons la fonction $f(x, y) = x^2 - y^2$. Son point de selle se trouve à $(0, 0)$. Il s'agit d'un maximum par rapport à $y$ et d'un minimum par rapport à $x$. De plus, il *ressemble* à une selle, d'où le nom de cette propriété mathématique.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Nous supposons que l'entrée d'une fonction est un vecteur à $k$ dimensions et que sa sortie
est un scalaire, de sorte que sa matrice hessienne aura $k$ valeurs propres.
La solution de la fonction
peut être un minimum local, un maximum local ou un point de selle à une position
où le gradient de la fonction est nul :

* Lorsque les valeurs propres de la matrice hessienne de la fonction à la position du gradient nul sont toutes positives, nous avons un minimum local pour la fonction.
* Lorsque les valeurs propres de la matrice hessienne de la fonction à la position du gradient zéro sont toutes négatives, on a un maximum local pour la fonction.
* Lorsque les valeurs propres de la matrice hessienne de la fonction à la position du gradient zéro sont négatives et positives, nous avons un point de selle pour la fonction.

Pour les problèmes à haute dimension, la probabilité qu'au moins *certaines* des valeurs propres soient négatives est assez élevée. Cela rend les points selle plus probables que les minima locaux. Nous aborderons certaines exceptions à cette situation dans la section suivante, lorsque nous présenterons la convexité. En bref, les fonctions convexes sont celles dont les valeurs propres du hessien ne sont jamais négatives. Malheureusement, la plupart des problèmes d'apprentissage profond n'entrent pas dans cette catégorie. Néanmoins, c'est un excellent outil pour étudier les algorithmes d'optimisation.

### Gradients évanescents

Le problème le plus insidieux à rencontrer est probablement le gradient évanescent.
Rappelez-vous nos fonctions d'activation couramment utilisées et leurs dérivées dans :numref:`subsec_activation-functions`.
Par exemple, supposons que nous voulions minimiser la fonction $f(x) = \tanh(x)$ et que nous démarrions à $x = 4$. Comme nous pouvons le constater, le gradient de $f$ est proche de zéro.
Plus précisément, $f'(x) = 1 - \tanh^2(x)$ et donc $f'(4) = 0.0013$.
Par conséquent, l'optimisation sera bloquée pendant un long moment avant de progresser. C'est l'une des raisons pour lesquelles l'entrainement de modèles d'apprentissage profond était assez délicate avant l'introduction de la fonction d'activation ReLU.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Comme nous l'avons vu, l'optimisation pour l'apprentissage profond est pleine de défis. Heureusement, il existe une gamme robuste d'algorithmes performants et faciles à utiliser, même pour les débutants. En outre, il n'est pas vraiment nécessaire de trouver *la* meilleure solution. Les optima locaux ou même leurs solutions approximatives sont toujours très utiles.

## Résumé

* Minimiser l'erreur d'apprentissage ne garantit pas *que nous trouvions le meilleur ensemble de paramètres pour minimiser l'erreur de généralisation.
* Les problèmes d'optimisation peuvent avoir de nombreux minima locaux.
* Le problème peut avoir encore plus de points de selle, car généralement les problèmes ne sont pas convexes.
* Des gradients évanescents peuvent provoquer un blocage de l'optimisation. Une reparamétrisation du problème est souvent utile. Une bonne initialisation des paramètres peut également être bénéfique.


## Exercices

1. Considérez un MLP simple avec une seule couche cachée de, disons, $d$ dimensions dans la couche cachée et une seule sortie. Montrez que pour tout minimum local, il existe au moins $d!$ solutions équivalentes qui se comportent de manière identique.
1. Supposons que nous ayons une matrice aléatoire symétrique $\mathbf{M}$ où les entrées
$M_{ij} = M_{ji}$ sont chacune tirées d'une certaine distribution de probabilité
$p_{ij}$ . Supposons en outre que $p_{ij}(x) = p_{ij}(-x)$, c'est-à-dire que la distribution
est symétrique (voir par exemple :cite:`Wigner.1958` pour plus de détails).
   1. Prouvez que la distribution sur les valeurs propres est également symétrique. C'est-à-dire que, pour tout vecteur propre $\mathbf{v}$, la probabilité que la valeur propre associée $\lambda$ satisfasse à $P(\lambda > 0) = P(\lambda < 0)$.
   1. Pourquoi la formule *non* ci-dessus implique-t-elle $P(\lambda > 0) = 0.5$?
1. Quels autres défis liés à l'optimisation de l'apprentissage profond pouvez-vous imaginer ?
1. Supposons que vous vouliez équilibrer une balle (réelle) sur une selle (réelle).
   1. Pourquoi est-ce difficile ?
   1. Pouvez-vous exploiter cet effet également pour les algorithmes d'optimisation ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
