```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Calcul
:label:`sec_calculus` 

Pendant longtemps, la façon de calculer 
l'aire d'un cercle est restée un mystère.
C'est alors que l'ancien mathématicien grec Archimède
a eu l'idée astucieuse 
d'inscrire une série de polygones 
avec un nombre croissant de sommets
à l'intérieur d'un cercle
(:numref:`fig_circle_area` ). 
Pour un polygone avec $n$ sommets,
on obtient $n$ triangles.
La hauteur de chaque triangle se rapproche du rayon $r$ 
au fur et à mesure que l'on partitionne le cercle plus finement. 
En même temps, sa base se rapproche de $2 \pi r/n$, 
puisque le rapport entre l'arc et la sécante se rapproche de 1 
pour un grand nombre de sommets. 
Ainsi, l'aire du triangle se rapproche de
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$ . 

![Finding the area of a circle as a limit procedure.](../img/polygon-circle.svg)
:label:`fig_circle_area`

Cette procédure de limitation conduit à la fois au *calcul différentiel* 
et au *calcul intégral* 
(:numref:`sec_integral_calculus` ). 
Le premier peut nous indiquer comment augmenter
ou diminuer la valeur d'une fonction en manipulant ses arguments.

Cela s'avère pratique pour les problèmes d'*optimisation*
auxquels nous sommes confrontés en apprentissage profond,
où nous mettons à jour nos paramètres de manière répétée 
afin de diminuer la fonction de perte.
L'optimisation porte sur la manière d'ajuster nos modèles aux données d'apprentissage,
et le calcul en est la condition préalable essentielle.
Toutefois, n'oubliez pas que notre objectif ultime
est d'obtenir de bonnes performances sur des données *inédites*.
Ce problème est appelé *généralisation*
et fera l'objet d'autres chapitres.



## Dérivées et différentiation

En termes simples, une *dérivée* est le taux de changement
d'une fonction par rapport aux changements de ses arguments.
Les dérivées peuvent nous indiquer à quelle vitesse une fonction de perte
augmenterait ou diminuerait si nous 
*augmentions* ou *diminuions* chaque paramètre
d'une quantité infinitésimale.
Formellement, pour les fonctions $f: \mathbb{R} \rightarrow \mathbb{R}$,
qui correspondent à des scalaires vers des scalaires,
[**la *dérivée* de $f$ en un point $x$ est définie comme**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative` 

Ce terme à droite est appelé une *limite* 
et il nous indique ce qui arrive 
à la valeur d'une expression
lorsqu'une variable spécifiée 
approche une valeur particulière.
Cette limite nous indique ce que 
le rapport entre une perturbation $h$
et le changement de la valeur de la fonction 
$f(x + h) - f(x)$ converge vers 
lorsque nous réduisons sa taille à zéro.

Lorsque $f'(x)$ existe, on dit que $f$ 
est *différentiable* à $x$;
et lorsque $f'(x)$ existe pour tous $x$
sur un ensemble, par exemple l'intervalle $[a,b]$, 
on dit que $f$ est différentiable sur cet ensemble.
Toutes les fonctions ne sont pas différentiables,
y compris de nombreuses fonctions que nous souhaitons optimiser,
y compris la précision et l'aire sous la caractéristique opérationnelle de réception (AUC) de
.
Cependant, comme le calcul de la dérivée de la perte 
est une étape cruciale dans presque tous les 
algorithmes de entrainement de réseaux neuronaux profonds,
nous optimisons souvent un *substitut* différentiable à la place.


Nous pouvons interpréter la dérivée 
$f'(x)$ 
comme le taux de variation *instantané* 
de $f(x)$ par rapport à $x$.
Développons un peu d'intuition avec un exemple.
(**Définir $u = f(x) = 3x^2-4x$.**)

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

(**Définir $x=1$, $\frac{f(x+h) - f(x)}{h}$**)
(**s'approche de $2$ comme $h$ s'approche de $0$.**)
Bien que cette expérience n'ait pas 
la rigueur d'une preuve mathématique,
nous verrons bientôt qu'effectivement $f'(1) = 2$.

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

Il existe plusieurs conventions de notation équivalentes pour les dérivées.
Étant donné $y = f(x)$, les expressions suivantes sont équivalentes :

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$ 

où les symboles $\frac{d}{dx}$ et $D$ sont des opérateurs de *différenciation*.
Nous présentons ci-dessous les dérivées de quelques fonctions courantes :

$$\begin{aligned} \frac{d}{dx} C & = 0 && \text{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \text{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1} \end{aligned}$$

Les fonctions composées à partir de fonctions différentiables 
sont souvent elles-mêmes différentiables.
Les règles suivantes sont utiles 
pour travailler avec des compositions 
de fonctions différentiables quelconques 
$f$ et $g$, et de constantes $C$.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \text{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \text{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \text{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \text{Quotient rule} \end{aligned}$$

Grâce à cela, nous pouvons appliquer les règles 
pour trouver la dérivée de $3 x^2 - 4x$ via

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$ 

 . En introduisant $x = 1$, on constate qu'effectivement, 
la dérivée est $2$ à cet endroit. 
Notez que les dérivées nous indiquent 
la *pente* d'une fonction 
à un endroit particulier. 

## Utilitaires de visualisation

[**Nous pouvons visualiser les pentes des fonctions en utilisant la bibliothèque `matplotlib` **].
Nous devons définir quelques fonctions. 
Comme son nom l'indique, `use_svg_display` 
indique à `matplotlib` de sortir les graphiques 
au format SVG pour des images plus nettes. 
Le commentaire `#@save` est un modificateur spécial 
qui nous permet d'enregistrer n'importe quelle fonction, 
classe ou autre bloc de code dans le paquet `d2l` 
afin de pouvoir l'invoquer ultérieurement 
sans répéter le code, 
par exemple, via `d2l.use_svg_display()`.

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

De façon pratique, nous pouvons définir la taille des figures avec `set_figsize`. 
Comme l'instruction d'importation `from matplotlib import pyplot as plt` 
a été marquée via `#@save` dans le paquet `d2l`, nous pouvons appeler `d2l.plt`.

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

La fonction `set_axes` peut associer les axes
à des propriétés, notamment des étiquettes, des plages,
et des échelles.

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Avec ces trois fonctions, nous pouvons définir une fonction `plot` 
pour superposer plusieurs courbes. 
Une grande partie du code ici consiste simplement à s'assurer 
que les tailles et les formes des entrées correspondent.

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None: axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Nous pouvons maintenant [**tracer la fonction $u = f(x)$ et sa ligne tangente $y = 2x - 3$ à $x=1$**],
où le coefficient $2$ est la pente de la ligne tangente.

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Dérivées partielles et gradients
:label:`subsec_calculus-grad` 

Jusqu'à présent, nous avons différencié
des fonctions d'une seule variable.
En apprentissage profond, nous devons également travailler
avec des fonctions de *nombreuses* variables.
Nous présentons brièvement les notions de dérivée
qui s'appliquent à de telles fonctions *multivariées*.


Soit $y = f(x_1, x_2, \ldots, x_n)$ une fonction avec $n$ variables. 
La dérivée *partielle* de $y$ 
par rapport à son paramètre $i^\mathrm{th}$ $x_i$ est

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$ 

 
Pour calculer $\frac{\partial y}{\partial x_i}$, 
nous pouvons traiter $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ comme des constantes 
et calculer la dérivée de $y$ par rapport à $x_i$.
Les conventions de notation suivantes pour les dérivées partielles 
sont toutes communes et signifient toutes la même chose :

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$ 

Nous pouvons concaténer les dérivées partielles 
d'une fonction multivariable 
par rapport à toutes ses variables 
pour obtenir un vecteur appelé
le *gradient* de la fonction.
Supposons que l'entrée de la fonction 
$f: \mathbb{R}^n \rightarrow \mathbb{R}$ 
soit un vecteur $n$-dimensionnel 
$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ 
et que la sortie soit un scalaire. 
Le gradient de la fonction $f$ 
par rapport à $\mathbf{x}$ 
est un vecteur de dérivées partielles $n$:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

Lorsqu'il n'y a pas d'ambiguïté,
$\nabla_{\mathbf{x}} f(\mathbf{x})$ 
est généralement remplacé 
par $\nabla f(\mathbf{x})$.
Les règles suivantes sont utiles 
pour différencier des fonctions multivariées :

* Pour tout $\mathbf{A} \in \mathbb{R}^{m \times n}$ on a $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ et $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* Pour les matrices carrées $\mathbf{A} \in \mathbb{R}^{n \times n}$ on a que $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ et en particulier
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$ .

De même, pour toute matrice $\mathbf{X}$, 
on a $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$. 



## Chain Rule

En apprentissage profond, les gradients de la préoccupation
sont souvent difficiles à calculer
parce que nous travaillons avec 
des fonctions profondément imbriquées 
(de fonctions (de fonctions...)).
Heureusement, la *règle de la chaîne* s'en charge. 
Pour en revenir aux fonctions d'une seule variable,
supposons que $y = f(g(x))$
et que les fonctions sous-jacentes 
$y=f(u)$ et $u=g(x)$ 
sont toutes deux différentiables.
La règle de la chaîne indique que 


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$ 

 

Pour en revenir aux fonctions à plusieurs variables,
supposons que $y = f(\mathbf{u})$ a des variables
$u_1, u_2, \ldots, u_m$ , 
où chaque $u_i = g_i(\mathbf{x})$ 
a des variables $x_1, x_2, \ldots, x_n$,
c'est-à-dire $\mathbf{u} = g(\mathbf{x})$.
La règle de la chaîne indique alors que

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \text{ and thus } \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$ 

où $\mathbf{A} \in \mathbb{R}^{n \times m}$ est une *matrice*
qui contient la dérivée du vecteur $\mathbf{u}$
par rapport au vecteur $\mathbf{x}$.
Ainsi, l'évaluation du gradient nécessite le calcul d'un produit vecteur-matrice sur .

C'est l'une des principales raisons pour lesquelles l'algèbre linéaire 
est un bloc de construction intégral 
dans la construction de systèmes d'apprentissage profond. 



## Discussion

Bien que nous n'ayons fait qu'effleurer la surface d'un sujet profond,
un certain nombre de concepts sont déjà mis en évidence : 
tout d'abord, les règles de composition pour la différenciation
peuvent être appliquées sans réfléchir, ce qui nous permet
de calculer les gradients *automatiquement*.
Cette tâche ne nécessite aucune créativité et 
nous pouvons donc concentrer nos capacités cognitives ailleurs.
Deuxièmement, le calcul des dérivées de fonctions à valeurs vectorielles 
exige que nous multipliions des matrices lorsque nous traçons 
le graphe de dépendance des variables de la sortie à l'entrée. 
En particulier, ce graphe est traversé dans une direction *avant* 
lorsque nous évaluons une fonction 
et dans une direction *arrière* 
lorsque nous calculons les gradients. 
Les chapitres suivants présenteront formellement la rétropropagation,
une procédure de calcul pour appliquer la règle de la chaîne.

Du point de vue de l'optimisation, les gradients nous permettent 
de déterminer comment déplacer les paramètres d'un modèle
afin de réduire la perte,
et chaque étape des algorithmes d'optimisation utilisés 
tout au long de cet ouvrage nécessitera le calcul du gradient.

## Exercices

1. Jusqu'à présent, nous avons considéré que les règles relatives aux dérivées étaient acquises. 
   En utilisant la définition et les limites, prouvez les propriétés 
pour (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$ et (iv) $f(x) = \log x$.
1. Dans la même veine, prouvez la règle du produit, de la somme et du quotient à partir des premiers principes. 
1. Prouvez que la règle du multiple constant est un cas particulier de la règle du produit. 
1. Calculez la dérivée de $f(x) = x^x$. 
1. Que signifie $f'(x) = 0$ pour un certain $x$? 
   Donnez un exemple d'une fonction $f$ 
et d'un lieu $x$ pour lesquels cela peut être vrai. 
1. Tracez la fonction $y = f(x) = x^3 - \frac{1}{x}$ 
et tracez sa ligne tangente à $x = 1$.
1. Trouvez le gradient de la fonction 
$f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ .
1. Quel est le gradient de la fonction 
$f(\mathbf{x}) = \|\mathbf{x}\|_2$ ? Que se passe-t-il pour $\mathbf{x} = \mathbf{0}$?
1. Pouvez-vous écrire la règle de la chaîne pour le cas 
où $u = f(x, y, z)$ et $x = x(a, b)$, $y = y(a, b)$, et $z = z(a, b)$?
1. Étant donné une fonction $f(x)$ qui est inversible, 
calculez la dérivée de son inverse $f^{-1}(x)$. 
   Ici, nous avons que $f^{-1}(f(x)) = x$ et inversement $f(f^{-1}(y)) = y$. 
   Conseil : utilisez ces propriétés dans votre dérivation 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
