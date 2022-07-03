# Convexité
:label:`sec_convexity` 

La convexité joue un rôle essentiel dans la conception des algorithmes d'optimisation. 
Cela est dû en grande partie au fait qu'il est beaucoup plus facile d'analyser et de tester les algorithmes dans un tel contexte. 
En d'autres termes,
si l'algorithme est peu performant même dans le cadre convexe,
il ne faut généralement pas espérer obtenir de grands résultats dans le cas contraire. 
En outre, même si les problèmes d'optimisation de l'apprentissage profond sont généralement non convexes, ils présentent souvent certaines propriétés des problèmes convexes à proximité des minima locaux. Cela peut conduire à de nouvelles variantes d'optimisation intéressantes telles que :cite:`Izmailov.Podoprikhin.Garipov.ea.2018` .

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

## Définitions

Avant de procéder à une analyse convexe,
nous devons définir les *ensembles convexes* et les *fonctions convexes*.
Elles conduisent à des outils mathématiques qui sont couramment appliqués à l'apprentissage automatique.


### Ensembles convexes

Les ensembles sont la base de la convexité. En termes simples, un ensemble $\mathcal{X}$ dans un espace vectoriel est *convexe* si pour tout $a, b \ dans \mathcal{X}$ le segment de droite reliant $a$ et $b$ est également dans $\mathcal{X}$. En termes mathématiques, cela signifie que pour tout $\lambda \in [0, 1])$, nous avons

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

Cela semble un peu abstrait. Considérons :numref:`fig_pacman` . Le premier ensemble n'est pas convexe car il existe des segments de droite qui ne sont pas contenus dans cet ensemble.
Les deux autres ensembles ne souffrent pas de ce problème.

![Le premier ensemble est non convexe et les deux autres sont convexes](../img/pacman.svg)
:label:`fig_pacman` 

Les définitions en elles-mêmes ne sont pas particulièrement utiles, sauf si vous pouvez en faire quelque chose.
Dans ce cas, nous pouvons examiner les intersections, comme le montre :numref:`fig_convex_intersect` .
Supposons que $\mathcal{X}$ et $\mathcal{Y}$ soient des ensembles convexes. Alors $\mathcal{X} \cap \mathcal{Y}$ est également convexe. Pour s'en convaincre, on considère tout $a, b \in \mathcal{X} \cap \mathcal{Y}$. Puisque $\mathcal{X}$ et $\mathcal{Y}$ sont convexes, les segments de droite reliant $a$ et $b$ sont contenus à la fois dans $\mathcal{X}$ et $\mathcal{Y}$. Étant donné cela, ils doivent également être contenus dans $\mathcal{X} \cap \mathcal{Y}$, prouvant ainsi notre théorème.

![L'intersection entre deux ensembles convexes est convexe](../img/convex-intersect.svg)
:label:`fig_convex_intersect` 

Nous pouvons renforcer ce résultat avec peu d'effort : étant donné les ensembles convexes $\mathcal{X}_i$, leur intersection $\cap_{i} \mathcal{X}_i$ est convexe.
Pour voir que l'inverse n'est pas vrai, considérez deux ensembles disjoints : $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Choisissez maintenant $a \in \mathcal{X}$ et $b \in \mathcal{Y}$. Le segment de droite dans :numref:`fig_nonconvex` reliant $a$ et $b$ doit contenir une partie qui n'est ni dans $\mathcal{X}$ ni dans $\mathcal{Y}$, puisque nous avons supposé que $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Par conséquent, le segment de droite n'est pas non plus dans $\mathcal{X} \cup \mathcal{Y}$, ce qui prouve qu'en général, les unions d'ensembles convexes ne sont pas nécessairement convexes.

![L'union de deux ensembles convexes n'est pas nécessairement convexe](../img/nonconvex.svg)
:label:`fig_nonconvex` 

Généralement, les problèmes d'apprentissage profond sont définis sur des ensembles convexes. Par exemple, $\mathbb{R}^d$,
l'ensemble des vecteurs à $d$ dimensions des nombres réels,
est un ensemble convexe (après tout, la ligne entre deux points quelconques de $\mathbb{R}^d$ reste dans $\mathbb{R}^d$. Dans certains cas, nous travaillons avec des variables de longueur limitée, telles que des boules de rayon $r$ définies par $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\| \leq r\}$.

### Fonctions convexes

Maintenant que nous avons des ensembles convexes, nous pouvons introduire les *fonctions convexes* $f$.
Étant donné un ensemble convexe $\mathcal{X}$, une fonction $f: \mathcal{X} \to \mathbb{R}$ est *convexe* si pour tout $x, x' \in \mathcal{X}$ et pour tout $\lambda \in [0, 1]$ nous avons

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Pour illustrer cela, traçons quelques fonctions et vérifions celles qui satisfont à la condition.
Nous définissons ci-dessous quelques fonctions, à la fois convexes et non convexes.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Comme prévu, la fonction cosinus est *non convexe*, alors que la parabole et la fonction exponentielle le sont. Notez que l'exigence selon laquelle $\mathcal{X}$ est un ensemble convexe est nécessaire pour que la condition ait un sens. Sinon, le résultat de $f(\lambda x + (1-\lambda) x')$ pourrait ne pas être bien défini.


### Inégalité de Jensen

Étant donné une fonction convexe $f$,
l'un des outils mathématiques les plus utiles
est l'inégalité de *Jensen*.
Elle revient à une généralisation de la définition de la convexité :

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

où $\alpha_i$ sont des nombres réels non négatifs tels que $\sum_i \alpha_i = 1$ et $X$ est une variable aléatoire.
En d'autres termes, l'espérance d'une fonction convexe n'est pas inférieure à la fonction convexe d'une espérance, cette dernière étant généralement une expression plus simple. 
Pour prouver la première inégalité, nous appliquons de manière répétée la définition de la convexité à un terme de la somme à la fois.


L'une des applications courantes de l'inégalité de Jensen est
pour lier une expression plus compliquée par une expression plus simple.
Par exemple,
son application peut être
en ce qui concerne la log-vraisemblance de variables aléatoires partiellement observées. C'est-à-dire que nous utilisons

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

puisque $\int P(Y) P(X \mid Y) dY = P(X)$.
Ceci peut être utilisé dans les méthodes variationnelles. Ici, $Y$ est généralement la variable aléatoire non observée, $P(Y)$ est la meilleure estimation de la façon dont elle pourrait être distribuée, et $P(X)$ est la distribution avec $Y$ intégré. Par exemple, dans le clustering, $Y$ peut être les étiquettes de cluster et $P(X \mid Y)$ est le modèle génératif lors de l'application des étiquettes de cluster.



## Propriétés

Les fonctions convexes ont de nombreuses propriétés utiles. Nous décrivons ci-dessous quelques-unes de celles qui sont couramment utilisées.


### Les minima locaux sont des minima globaux

Tout d'abord, les minima locaux des fonctions convexes sont également les minima globaux. 
Nous pouvons le prouver par contradiction comme suit.

Considérons une fonction convexe $f$ définie sur un ensemble convexe $\mathcal{X}$.
Supposons que $x^{\ast} \in \mathcal{X}$  soit un minimum local :
il existe une petite valeur positive $p$ telle que pour $x \in \mathcal{X}$ qui satisfait $0 < |x - x^{\ast}| \leq p$ nous avons $f(x^{\ast}) < f(x)$.

Supposons que le minimum local $x^{\ast}$
ne soit pas le minimum global de $f$:
il existe $x' \in \mathcal{X}$ for which $f(x') < f(x^{\ast})$. 
Il existe également 
$\lambda \in [0, 1)$ such as $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$
de sorte que
$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$. 

Cependant,
selon la définition des fonctions convexes, on a

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

ce qui est en contradiction avec notre affirmation que $x^{\ast}$ est un minimum local.
Par conséquent, il n'existe pas $x' \in \mathcal{X}$ pour lequel $f(x') < f(x^{\ast})$. Le minimum local $x^{\ast}$ est aussi le minimum global.

Par exemple, la fonction convexe $f(x) = (x-1)^2$ a un minimum local en $x=1$, qui est aussi le minimum global.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Le fait que les minima locaux des fonctions convexes soient également les minima globaux est très pratique. 
Cela signifie que si nous minimisons des fonctions, nous ne pouvons pas "rester coincés". 
Notez cependant que cela ne signifie pas qu'il ne peut pas y avoir plus d'un minimum global ou qu'il peut même en exister un. Par exemple, la fonction $f(x) = \mathrm{max}(|x|-1, 0)$ atteint sa valeur minimale sur l'intervalle $[-1, 1]$. Inversement, la fonction $f(x) = \exp(x)$ n'atteint pas une valeur minimale sur \mathbb{R} : pour $x \to -\infty$ elle s'asymptote à $0$, mais il n'existe pas de $x$ pour lequel $f(x) = 0$.

### Les ensembles inférieurs de fonctions convexes sont convexes

Nous pouvons commodément 
définir les ensembles convexes 
via les *ensembles inférieurs* de fonctions convexes.
Concrètement,
étant donné une fonction convexe $f$ définie sur un ensemble convexe \mathcal{X},
tout ensemble inférieur

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ et } f(x) \leq b\}$$

est convexe. 

Prouvons cela rapidement. Rappelons que pour tout $x, x' \in \mathcal{S}_b$, nous devons montrer que $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ tant que $\lambda \in [0, 1]$. 
Puisque $f(x) \leq b$ et $f(x') \leq b$,
par la définition de la convexité nous avons 

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$


### Convexité et dérivées secondes

Quelle que soit la dérivée seconde d'une fonction $f: \mathbb{R}^n \rightarrow \mathbb{R}$ existe, il est très facile de vérifier si $f$ est convexe. 
Tout ce que nous avons à faire est de vérifier si le Hessien de $f$ est semi-défini positif : $\nabla^2f \succeq 0$, c'est-à-dire, 
en désignant la matrice hessienne $\nabla^2f$ par $\mathbf{H}$,
$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$
pour tout $\mathbf{x} \in \mathbb{R}^n$.
Par exemple, la fonction $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ est convexe puisque $\nabla^2 f = \mathbf{1}$, ie.e.,sa Hessienne est la matrice identité.


Formellement, une fonction unidimensionnelle deux fois différentiable $f: \mathbb{R} \rightarrow \mathbb{R}$ est convexe
si et seulement si sa dérivée seconde $f'' \geq 0$. Pour toute fonction multidimensionnelle deux fois différentiable $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$,
elle est convexe si et seulement si son hessien $\nabla^2f \succeq 0$.

Tout d'abord, nous devons prouver le cas unidimensionnel.
Pour voir que 
la convexité de $f$ implique 
$f'' \geq 0$, nous utilisons le fait que

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

Comme la dérivée seconde est donnée par la limite sur les différences finies, il s'ensuit que

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

Pour voir que 
$f'' \geq 0$ implique que $f$ est convexe
nous utilisons le fait que $f'' \geq 0$ implique que $f'$ est une fonction monotone non décroissante. Soit  $a < x < b$ trois points dans $\mathbb{R}$,
où $x = (1-\lambda)a + \lambda b$ et $\lambda \in (0, 1)$.
Selon le Théorème des accroissements finis
il existe $\alpha \in [a, x]$ et $\beta \in [x, b]$
tels que

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ et } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$


Par monotonicité, $f'(\beta) \geq f'(\alpha)$, donc

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Puisque $x = (1-\lambda)a + \lambda b$,
nous avons

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

prouvant ainsi la convexité.

Deuxièmement, nous avons besoin d'un lemme avant de 
prouver le cas multidimensionnel:
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
est convexe si et seulement si pour tout $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ avec } z \in [0,1]$$ 

 est convexe.

Pour prouver que la convexité de $f$ implique que $g$ est convexe,
nous pouvons montrer que pour tout $a, b, \lambda \in [0, 1]$ (donc
$0 \leq \lambda a + (1-\lambda) b \leq 1$ )

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

Pour prouver l'inverse,
nous pouvons montrer que pour 
tous $\lambda \in [0, 1]$ 

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$


Enfin,
en utilisant le lemme ci-dessus et le résultat du cas unidimensionnel,
le cas multidimensionnel
peut être prouvé comme suit.
Une fonction multidimensionnelle $f: \mathbb{R}^n \rightarrow \mathbb{R}$ est convexe
si et seulement si pour tout $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$ , où $z \in [0,1]$,
est convexe.
Selon le cas unidimensionnel,
ceci est vrai si et seulement si
$g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$)
pour tous les $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,
ce qui est équivalent à $\mathbf{H} \succeq 0$
 selon la définition des matrices semi-définies positives.


## Contraintes

L'une des propriétés intéressantes de l'optimisation convexe est qu'elle nous permet de gérer efficacement les contraintes. C'est-à-dire qu'elle nous permet de résoudre des problèmes d'optimisation *sous contraintes* de la forme :

$$\begin{aligned} \mathop{\mathrm{minimize~)}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

où $f$ est l'objectif et les fonctions $c_i$ sont des fonctions de contrainte. Pour voir ce que cela fait, considérons le cas où $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. Dans ce cas, les paramètres $\mathbf{x}$ sont contraints à la boule unitaire. Si une deuxième contrainte est $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, alors cela correspond à tous les $\mathbf{x}$ situés dans un demi-espace. Satisfaire les deux contraintes simultanément revient à sélectionner une tranche d'une boule.

### Lagrangien

En général, la résolution d'un problème d'optimisation sous contrainte est difficile. Une façon de l'aborder découle de la physique avec une intuition plutôt simple. Imaginez une balle à l'intérieur d'une boîte. La balle roulera vers l'endroit le plus bas et les forces de gravité seront équilibrées par les forces que les côtés de la boîte peuvent imposer à la balle. En bref, le gradient de la fonction objectif (c'est-à-dire la gravité) sera compensé par le gradient de la fonction contrainte (la balle doit rester à l'intérieur de la boîte en vertu des parois qui la "repoussent"). 
Notez que certaines contraintes peuvent ne pas être actives :
les murs qui ne sont pas touchés par la balle
ne pourront exercer aucune force sur la balle.


Sans passer par la dérivation du *Lagrangien* $L$,
le raisonnement ci-dessus
peut être exprimé par le problème d'optimisation à point de selle suivant :

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$ 

 Ici, les variables $\alpha_i$ ($i=1,\ldots,n$) sont les *multiplicateurs de Lagrange* qui garantissent que les contraintes sont correctement appliquées. Elles sont choisies juste assez grandes pour garantir que $c_i(\mathbf{x}) \leq 0$ pour tout $i$. Par exemple, pour tout $\mathbf{x}$ où $c_i(\mathbf{x}) < 0$ naturellement, nous finirions par choisir $\alpha_i = 0$. De plus, il s'agit d'un problème d'optimisation à point de selle où l'on veut *maximiser* $L$ par rapport à tous les $\alpha_i$ et simultanément *minimiser* par rapport à $\mathbf{x}$. Il existe une abondante littérature expliquant comment arriver à la fonction $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$. Pour nos besoins, il suffit de savoir que le point de selle de $L$ est l'endroit où le problème d'optimisation sous contrainte original est résolu de manière optimale.

### Pénalités

Une façon de satisfaire les problèmes d'optimisation sous contraintes au moins *approximativement* est d'adapter le Lagrangien $L$. 
Plutôt que de satisfaire $c_i(\mathbf{x}) \leq 0$, nous ajoutons simplement $\alpha_i c_i(\mathbf{x})$ à la fonction objectif $f(x)$. Cela garantit que les contraintes ne seront pas trop violées.

En fait, nous utilisons cette astuce depuis le début. Considérons la décroissance du poids dans :numref:`sec_weight_decay` . Dans ce cas, nous ajoutons $\frac{\lambda}{2} \|\mathbf{w}\|^2$ à la fonction objectif pour nous assurer que $\mathbf{w}$ ne devient pas trop grand. Du point de vue de l'optimisation sous contrainte, nous pouvons voir que cela garantira que $\|\mathbf{w}\|^2 - r^2 \leq 0$ pour un certain rayon $r$. L'ajustement de la valeur de $\lambda$ nous permet de faire varier la taille de $\mathbf{w}$.

En général, l'ajout de pénalités est un bon moyen de garantir la satisfaction approximative des contraintes. En pratique, cela s'avère être beaucoup plus robuste que la satisfaction exacte. En outre, pour les problèmes non convexes, de nombreuses propriétés qui rendent l'approche exacte si attrayante dans le cas convexe (par exemple, l'optimalité) ne tiennent plus.

### Projections

Les projections constituent une autre stratégie de satisfaction des contraintes. Là encore, nous les avons déjà rencontrées, par exemple, lors du traitement de l'écrêtage du gradient dans :numref:`sec_rnn-scratch` . Dans ce cas, nous nous sommes assurés qu'un gradient avait une longueur limitée par $\theta$ via

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$ 

 . Il s'agit d'une *projection* de $\mathbf{g}$ sur la boule de rayon $\theta$. Plus généralement, une projection sur un ensemble convexe $\mathcal{X}$ est définie comme

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$ 

 qui est le point de $\mathcal{X}$ le plus proche de $\mathbf{x}$. 

![Convex Projections.](../img/projections.svg) 
:label:`fig_projections` 

 La définition mathématique des projections peut sembler un peu abstraite. :numref:`fig_projections` l'explique un peu plus clairement. Dans ce document, nous avons deux ensembles convexes, un cercle et un diamant. 
Les points à l'intérieur des deux ensembles (jaune) restent inchangés pendant les projections. 
Les points à l'extérieur des deux ensembles (noirs) sont projetés sur 
les points à l'intérieur des ensembles (rouges) qui sont proches des points originaux (noirs).
Bien que pour $\ell_2$ boules, la direction reste inchangée, ce n'est pas forcément le cas en général, comme on peut le voir dans le cas du diamant.


L'une des utilisations des projections convexes consiste à calculer des vecteurs de poids épars. Dans ce cas, nous projetons les vecteurs de poids sur une boule $\ell_1$,
qui est une version généralisée du cas du diamant dans :numref:`fig_projections` .


## Résumé

Dans le contexte de l'apprentissage profond, le principal objectif des fonctions convexes est de motiver les algorithmes d'optimisation et de nous aider à les comprendre en détail. Dans la suite, nous verrons comment la descente de gradient et la descente de gradient stochastique peuvent être dérivées en conséquence.


* Les intersections d'ensembles convexes sont convexes. Les unions ne le sont pas.
* L'espérance d'une fonction convexe n'est pas inférieure à la fonction convexe d'une espérance (inégalité de Jensen).
* Une fonction deux fois différentiable est convexe si et seulement si sa Hessienne (une matrice de dérivées secondes) est semi-définie positive.
* Les contraintes convexes peuvent être ajoutées via le Lagrangien. En pratique, nous pouvons simplement les ajouter avec une pénalité à la fonction objectif.
* Les projections correspondent aux points de l'ensemble convexe les plus proches des points d'origine.

## Exercices

1. Supposons que nous voulions vérifier la convexité d'un ensemble en traçant toutes les lignes entre les points de l'ensemble et en vérifiant si les lignes sont contenues.
   1. Prouvez qu'il suffit de vérifier uniquement les points de la frontière.
   1. Prouvez qu'il est suffisant de vérifier uniquement les sommets de l'ensemble.
1. Dénotez par $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ la boule de rayon $r$ en utilisant la norme $p$. Prouvez que $\mathcal{B}_p[r]$ est convexe pour tout $p \geq 1$.
1. Étant donné les fonctions convexes $f$ et $g$, montrez que $\mathrm{max}(f, g)$ est également convexe. Prouvez que $\mathrm{min}(f, g)$ n'est pas convexe.
1. Prouvez que la normalisation de la fonction softmax est convexe. Plus précisément, prouvez la convexité de
 $f(x) = \log \sum_i \exp(x_i)$ .
1. Prouvez que les sous-espaces linéaires, c'est-à-dire $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$, sont des ensembles convexes.
1. Prouvez que dans le cas de sous-espaces linéaires avec $\mathbf{b} = \mathbf{0}$, la projection $\mathrm{Proj}_\mathcal{X}$ peut être écrite sous la forme $\mathbf{M} \mathbf{x}$ pour une certaine matrice $\mathbf{M}$.
1. Montrez que pour les fonctions convexes deux fois différentiables $f$, on peut écrire $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ pour une certaine $\xi \in [0, \epsilon]$.
1. Étant donné un vecteur $\mathbf{w} \in \mathbb{R}^d$ avec $\|\mathbf{w}\|_1 > 1$, calculez la projection sur la boule unitaire $\ell_1$.
   1. Comme étape intermédiaire, écrivez l'objectif pénalisé $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$ et calculez la solution pour un $\lambda > 0$ donné.
 1. Pouvez-vous trouver la "bonne" valeur de $\lambda$ sans beaucoup d'essais et d'erreurs ?
1. Étant donné un ensemble convexe $\mathcal{X}$ et deux vecteurs $\mathbf{x}$ et $\mathbf{y}$, prouvez que les projections n'augmentent jamais les distances, c'est-à-dire $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$.


[Discussions](https://discuss.d2l.ai/t/350)
