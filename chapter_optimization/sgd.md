# Descente de gradient stochastique
:label:`sec_sgd` 

 Dans les chapitres précédents, nous avons continué à utiliser la descente de gradient stochastique dans notre procédure de formation, sans toutefois expliquer pourquoi elle fonctionne.
Pour y voir plus clair,
nous venons de décrire les principes de base de la descente de gradient
dans :numref:`sec_gd` .
Dans cette section, nous abordons plus en détail
*la descente de gradient stochastique*.

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

## Mises à jour stochastiques du gradient

En apprentissage profond, la fonction objectif est généralement la moyenne des fonctions de perte pour chaque exemple de l'ensemble de données d'apprentissage.
Étant donné un ensemble de données d'apprentissage de $n$ exemples,
nous supposons que $f_i(\mathbf{x})$ est la fonction de perte
par rapport à l'exemple d'apprentissage d'indice $i$,
où $\mathbf{x}$ est le vecteur de paramètres.
Nous obtenons alors la fonction objectif

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$ 

 Le gradient de la fonction objectif à $\mathbf{x}$ est calculé comme suit :

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$ 

 Si la descente de gradient est utilisée, le coût de calcul pour chaque itération de variable indépendante est $\mathcal{O}(n)$, qui croît linéairement avec $n$. Par conséquent, lorsque l'ensemble de données d'apprentissage est plus grand, le coût de la descente de gradient pour chaque itération sera plus élevé.

La descente de gradient stochastique (SGD) réduit le coût de calcul à chaque itération. À chaque itération de la descente de gradient stochastique, nous échantillonnons uniformément un indice $i\in\{1,\ldots, n\}$ pour les exemples de données au hasard, et calculons le gradient $\nabla f_i(\mathbf{x})$ pour mettre à jour $\mathbf{x}$:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$ 

 où $\eta$ est le taux d'apprentissage. Nous pouvons constater que le coût de calcul pour chaque itération diminue de $\mathcal{O}(n)$ de la descente du gradient à la constante $\mathcal{O}(1)$. De plus, nous voulons souligner que le gradient stochastique $\nabla f_i(\mathbf{x})$ est une estimation sans biais du gradient complet $\nabla f(\mathbf{x})$ car

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$ 

 Cela signifie qu'en moyenne, le gradient stochastique est une bonne estimation du gradient.

Maintenant, nous allons le comparer à la descente de gradient en ajoutant un bruit aléatoire avec une moyenne de 0 et une variance de 1 au gradient pour simuler une descente de gradient stochastique.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Comme nous pouvons le voir, la trajectoire des variables dans la descente de gradient stochastique est beaucoup plus bruyante que celle que nous avons observée dans la descente de gradient dans :numref:`sec_gd` . Ceci est dû à la nature stochastique du gradient. C'est-à-dire que même lorsque nous arrivons près du minimum, nous sommes toujours soumis à l'incertitude injectée par le gradient instantané via $\eta \nabla f_i(\mathbf{x})$. Même après 50 étapes, la qualité n'est toujours pas très bonne. Pire encore, elle ne s'améliorera pas après des étapes supplémentaires (nous vous encourageons à expérimenter avec un plus grand nombre d'étapes pour confirmer cela). Il ne nous reste donc qu'une seule alternative : modifier le taux d'apprentissage $\eta$. Cependant, si nous le choisissons trop petit, nous ne ferons aucun progrès significatif au départ. D'autre part, si nous le choisissons trop grand, nous n'obtiendrons pas une bonne solution, comme nous l'avons vu plus haut. La seule façon de résoudre ces objectifs contradictoires est de réduire le taux d'apprentissage *dynamiquement* à mesure que l'optimisation progresse.

C'est également la raison pour laquelle on a ajouté une fonction de taux d'apprentissage `lr` dans la fonction d'étape `sgd`. Dans l'exemple ci-dessus, toute fonctionnalité de planification du taux d'apprentissage reste en sommeil, car nous avons défini la fonction `lr` associée comme étant constante.

## Taux d'apprentissage dynamique

Le remplacement de $\eta$ par un taux d'apprentissage dépendant du temps $\eta(t)$ ajoute à la complexité du contrôle de la convergence d'un algorithme d'optimisation. En particulier, nous devons déterminer à quelle vitesse $\eta$ doit décroître. S'il est trop rapide, nous arrêterons l'optimisation prématurément. Si nous la diminuons trop lentement, nous perdons trop de temps en optimisation. Voici quelques stratégies de base utilisées pour ajuster $\eta$ au fil du temps (nous aborderons plus tard des stratégies plus avancées) :

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{polynomial decay}
\end{aligned}
$$

Dans le premier scénario *constant par morceaux*, nous diminuons le taux d'apprentissage, par exemple, chaque fois que la progression de l'optimisation s'arrête. Il s'agit d'une stratégie courante pour l'entraînement des réseaux profonds. Alternativement, nous pourrions le diminuer de manière beaucoup plus agressive par une *décroissance exponentielle*. Malheureusement, cela conduit souvent à un arrêt prématuré avant que l'algorithme n'ait convergé. Un choix populaire est la décroissance *polynomiale* avec $\alpha = 0.5$. Dans le cas de l'optimisation convexe, il existe un certain nombre de preuves qui montrent que ce taux se comporte bien.

Voyons à quoi ressemble la décroissance exponentielle en pratique.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Comme prévu, la variance des paramètres est considérablement réduite. Cependant, cela se fait au prix de l'échec de la convergence vers la solution optimale $\mathbf{x} = (0, 0)$. Même après 1000 étapes d'itération, nous sommes encore très loin de la solution optimale. En fait, l'algorithme ne parvient pas à converger du tout. D'autre part, si nous utilisons une décroissance polynomiale où le taux d'apprentissage décroît avec la racine carrée inverse du nombre d'étapes, la convergence s'améliore après seulement 50 étapes.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Il existe de nombreuses autres possibilités pour définir le taux d'apprentissage. Par exemple, nous pouvons commencer avec un taux faible, puis l'augmenter rapidement et le diminuer à nouveau, mais plus lentement. Nous pourrions même alterner entre des taux d'apprentissage plus petits et plus grands. Il existe une grande variété de programmes de ce type. Pour l'instant, nous nous concentrons sur les programmes de taux d'apprentissage pour lesquels une analyse théorique complète est possible, c'est-à-dire sur les taux d'apprentissage dans un cadre convexe. Pour les problèmes non convexes généraux, il est très difficile d'obtenir des garanties de convergence significatives, car en général, la minimisation des problèmes non linéaires non convexes est NP difficile. Pour un aperçu, voir par exemple l'excellent [lecture notes](https://www.stat.cmu.edu/~)ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) de Tibshirani 2015.



 ## Analyse de convergence pour les objectifs convexes

L'analyse suivante de la convergence de la descente de gradient stochastique pour les fonctions objectifs convexes
est facultative et sert principalement à transmettre plus d'intuition sur le problème.
Nous nous limitons à l'une des preuves les plus simples :cite:`Nesterov.Vial.2000` .
Des techniques de preuve beaucoup plus avancées existent, par exemple, lorsque la fonction objectif se comporte particulièrement bien.


Supposons que la fonction objectif $f(\boldsymbol{\xi}, \mathbf{x})$ soit convexe dans $\mathbf{x}$
 pour tout $\boldsymbol{\xi}$.
Plus concrètement,
nous considérons la mise à jour par descente de gradient stochastique :

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$ 

 où $f(\boldsymbol{\xi}_t, \mathbf{x})$
 est la fonction objectif
par rapport à l'exemple d'apprentissage $\boldsymbol{\xi}_t$
 tiré d'une certaine distribution
à l'étape $t$ et $\mathbf{x}$ est le paramètre du modèle.
Désignons par

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$ 

 le risque attendu et par $R^*$ son minimum par rapport à $\mathbf{x}$. Enfin, considérons que $\mathbf{x}^*$ est le minimiseur (nous supposons qu'il existe dans le domaine où $\mathbf{x}$ est défini). Dans ce cas, nous pouvons suivre la distance entre le paramètre actuel $\mathbf{x}_t$ au moment $t$ et le minimiseur de risque $\mathbf{x}^*$ et voir si elle s'améliore avec le temps :

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Nous supposons que la norme $\ell_2$ du gradient stochastique $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ est limitée par une certaine constante $L$, d'où l'équation suivante :

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$ 
 :eqlabel:`eq_sgd-L` 

 
 Nous sommes surtout intéressés par la façon dont la distance entre $\mathbf{x}_t$ et $\mathbf{x}^*$ évolue * dans l'espérance*. En fait, pour toute séquence spécifique d'étapes, la distance pourrait bien augmenter, en fonction de l'adresse $\boldsymbol{\xi}_t$ que nous rencontrons. Nous devons donc lier le produit scalaire.
Puisque pour toute fonction convexe $f$, il est établi que
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$ 
 pour tous les $\mathbf{x}$ et $\mathbf{y}$,
par convexité, nous avons

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$ 
 :eqlabel:`eq_sgd-f-xi-xstar` 

 En plaçant les deux inégalités :eqref:`eq_sgd-L` et :eqref:`eq_sgd-f-xi-xstar` dans :eqref:`eq_sgd-xt+1-xstar` , nous obtenons une limite sur la distance entre les paramètres au moment $t+1$ comme suit :

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Cela signifie que nous progressons tant que la différence entre la perte actuelle et la perte optimale est supérieure à $\eta_t L^2/2$. Puisque cette différence doit converger vers zéro, il s'ensuit que le taux d'apprentissage $\eta_t$ doit également *disparaître*.

Ensuite, nous prenons les attentes sur :eqref:`eqref_sgd-xt-diff` . Cela donne

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$ 

 La dernière étape consiste à faire la somme des inégalités pour $t \in \{1, \ldots, T\}$. Puisque la somme se télescope et en laissant tomber le terme inférieur, nous obtenons

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$ 
 :eqlabel:`eq_sgd-x1-xstar` 

 Notez que nous avons exploité le fait que $\mathbf{x}_1$ est donné et donc que l'espérance peut être abandonnée. Définissez enfin

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$ 

 Puisque

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$ 

 par l'inégalité de Jensen (en plaçant $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ dans :eqref:`eq_jensens-inequality` ) et la convexité de $R$, il s'ensuit que $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, donc

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$ 

 En plaçant cela dans l'inégalité :eqref:`eq_sgd-x1-xstar` , on obtient la borne

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

où $r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ est une limite sur la distance entre le choix initial des paramètres et le résultat final. En bref, la vitesse de convergence dépend de la manière dont
la norme du gradient stochastique est bornée ($L$) et de la distance de l'optimalité de la valeur initiale du paramètre ($r$). Notez que la limite est exprimée en termes de $\bar{\mathbf{x}}$ plutôt que de $\mathbf{x}_T$, car $\bar{\mathbf{x}}$ est une version lissée du chemin d'optimisation.
Lorsque $r, L$ et $T$ sont connus, nous pouvons choisir le taux d'apprentissage $\eta = r/(L \sqrt{T})$. Cela donne comme limite supérieure $rL/\sqrt{T}$. C'est-à-dire que nous convergeons avec le taux $\mathcal{O}(1/\sqrt{T})$ vers la solution optimale.





## Gradients stochastiques et échantillons finis

Jusqu'à présent, nous avons joué un peu vite et mal à propos de la descente de gradient stochastique. Nous avons postulé que nous tirons des instances $x_i$, généralement avec des étiquettes $y_i$ à partir d'une certaine distribution $p(x, y)$ et que nous les utilisons pour mettre à jour les paramètres du modèle d'une certaine manière. En particulier, pour un échantillon de taille finie, nous avons simplement affirmé que la distribution discrète $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$
 pour certaines fonctions $\delta_{x_i}$ et $\delta_{y_i}$
 nous permet d'effectuer une descente de gradient stochastique sur celle-ci.

Cependant, ce n'est pas vraiment ce que nous avons fait. Dans les exemples jouets de la section actuelle, nous avons simplement ajouté du bruit à un gradient autrement non stochastique, c'est-à-dire que nous avons prétendu avoir des paires $(x_i, y_i)$. Il s'avère que cela est justifié ici (voir les exercices pour une discussion détaillée). Le plus troublant est que dans toutes les discussions précédentes, nous n'avons clairement pas fait cela. Au lieu de cela, nous avons itéré sur toutes les instances *exactement une fois*. Pour comprendre pourquoi cela est préférable, considérons l'inverse, à savoir que nous échantillonnons $n$ observations de la distribution discrète *avec remplacement*. La probabilité de choisir un élément $i$ au hasard est $1/n$. Ainsi, la probabilité de le choisir *au moins* une fois est

$$P(\mathrm{choose~)} i) = 1 - P(\mathrm{omit~)} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$ 

 Un raisonnement similaire montre que la probabilité de choisir un certain échantillon (c'est-à-dire un exemple d'entraînement) *exactement une fois* est donnée par

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$ 

 L'échantillonnage avec remplacement entraîne une augmentation de la variance et une diminution de l'efficacité des données par rapport à l'échantillonnage *sans remplacement*. Par conséquent, en pratique, nous effectuons ce dernier (et c'est le choix par défaut tout au long de ce livre). Notez enfin que les passages répétés à travers l'ensemble de données d'apprentissage le traversent dans un ordre aléatoire *différent*.


## Résumé

* Pour les problèmes convexes, nous pouvons prouver que pour un large choix de taux d'apprentissage, la descente de gradient stochastique convergera vers la solution optimale.
* Pour l'apprentissage profond, ce n'est généralement pas le cas. Cependant, l'analyse des problèmes convexes nous donne un aperçu utile sur la façon d'aborder l'optimisation, à savoir réduire le taux d'apprentissage progressivement, mais pas trop rapidement.
* Des problèmes surviennent lorsque le taux d'apprentissage est trop faible ou trop élevé. En pratique, un taux d'apprentissage approprié n'est souvent trouvé qu'après plusieurs expériences.
* Lorsqu'il y a plus d'exemples dans l'ensemble de données d'apprentissage, il est plus coûteux de calculer chaque itération de la descente de gradient, de sorte que la descente de gradient stochastique est préférable dans ces cas.
* Les garanties d'optimalité pour la descente de gradient stochastique ne sont en général pas disponibles dans les cas non convexes, car le nombre de minima locaux à vérifier peut être exponentiel.




## Exercices

1. Expérimentez avec différents programmes de taux d'apprentissage pour la descente de gradient stochastique et avec différents nombres d'itérations. En particulier, tracez la distance de la solution optimale $(0, 0)$ en fonction du nombre d'itérations.
1. Prouvez que pour la fonction $f(x_1, x_2) = x_1^2 + 2 x_2^2$, ajouter un bruit normal au gradient équivaut à minimiser une fonction de perte $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ où $\mathbf{x}$ est tiré d'une distribution normale.
1. Comparez la convergence de la descente de gradient stochastique lorsque vous échantillonnez dans $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ avec remplacement et lorsque vous échantillonnez sans remplacement.
1. Comment modifieriez-vous le solveur de descente de gradient stochastique si un gradient (ou plutôt une coordonnée qui lui est associée) était systématiquement plus grand que tous les autres gradients ?
1. Supposons que $f(x) = x^2 (1 + \sin x)$. Combien de minima locaux possède $f$? Pouvez-vous modifier $f$ de telle sorte que pour le minimiser, il faille évaluer tous les minima locaux ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
