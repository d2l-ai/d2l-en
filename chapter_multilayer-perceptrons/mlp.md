```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Perceptrons multicouches
:label:`sec_mlp` 

 Dans :numref:`chap_classification` , nous avons introduit
la régression softmax (:numref:`sec_softmax` ),
en mettant en œuvre l'algorithme à partir de zéro
(:numref:`sec_softmax_scratch` ) et en utilisant des API de haut niveau
(:numref:`sec_softmax_concise` ). Cela nous a permis d'entraîner 
des classificateurs capables de reconnaître
10 catégories de vêtements à partir d'images à faible résolution.
En cours de route, nous avons appris à manipuler les données,
à contraindre nos sorties à une distribution de probabilité valide,
à appliquer une fonction de perte appropriée,
et à la minimiser en fonction des paramètres de notre modèle.
Maintenant que nous avons maîtrisé ces mécanismes
dans le contexte de modèles linéaires simples,
nous pouvons nous lancer dans l'exploration des réseaux de neurones profonds,
la classe comparativement riche de modèles
à laquelle ce livre s'intéresse principalement.


## Couches cachées

Nous avons décrit les transformations affines dans
:numref:`subsec_linear_model` comme 
des transformations linéaires avec un biais supplémentaire.
Pour commencer, rappelons l'architecture du modèle
correspondant à notre exemple de régression softmax,
illustré dans :numref:`fig_softmaxreg` .
Ce modèle fait directement correspondre les entrées aux sorties
via une seule transformation affine,
suivie d'une opération softmax.
Si nos étiquettes étaient réellement liées
aux données d'entrée par une simple transformation affine,
, cette approche serait suffisante.
Cependant, la linéarité (dans les transformations affines) est une hypothèse *forte*.

### Limites des modèles linéaires

Par exemple, la linéarité implique l'hypothèse *plus faible*
de *monotonicité*, c'est-à-dire
que toute augmentation de notre caractéristique doit
soit toujours provoquer une augmentation de la sortie de notre modèle
(si le poids correspondant est positif),
soit toujours provoquer une diminution de la sortie de notre modèle
(si le poids correspondant est négatif).
Parfois, cela a du sens.
Par exemple, si nous essayions de prédire
si un individu remboursera un prêt,
nous pourrions raisonnablement supposer que, toutes choses égales par ailleurs, 
un demandeur ayant un revenu plus élevé
serait toujours plus susceptible de rembourser
qu'un demandeur ayant un revenu plus faible.
Bien que monotone, cette relation
n'est probablement pas associée de façon linéaire à la probabilité de remboursement
. Une augmentation du revenu de \N$0 to \\$50 000
correspond probablement à une augmentation plus importante
de la probabilité de remboursement
qu'une augmentation de \N$1 million to \\$1,05 million.
Une façon de gérer cela pourrait être de post-traiter notre résultat 
de sorte que la linéarité devienne plus plausible,
en utilisant la carte logistique (et donc le logarithme de la probabilité du résultat). 

Notez que nous pouvons facilement trouver des exemples
qui violent la monotonicité.
Disons par exemple que nous voulons prédire la santé en fonction 
de la température corporelle. 
Pour les personnes dont la température corporelle
est supérieure à 37°C (98,6°F),
une température plus élevée indique un risque plus important.
Cependant, pour les individus dont la température corporelle
est inférieure à 37°C, des températures plus basses indiquent un plus grand risque !
Là encore, nous pourrions résoudre le problème
en effectuant un prétraitement astucieux, par exemple en utilisant la distance par rapport à 37°C 
comme caractéristique.


Mais qu'en est-il de la classification des images de chats et de chiens ?
L'augmentation de l'intensité
du pixel situé à l'emplacement (13, 17)
doit-elle toujours augmenter (ou toujours diminuer)
la probabilité que l'image représente un chien ?
Le recours à un modèle linéaire correspond à l'hypothèse implicite
selon laquelle la seule condition
pour différencier les chats des chiens est d'évaluer
la luminosité des pixels individuels.
Cette approche est vouée à l'échec dans un monde
où l'inversion d'une image préserve la catégorie.

Et pourtant, malgré l'absurdité apparente de la linéarité ici,
par rapport à nos exemples précédents,
il est moins évident que nous puissions résoudre le problème
avec une simple correction de prétraitement.
En effet, la signification de tout pixel
dépend de manière complexe de son contexte
(les valeurs des pixels environnants).
Bien qu'il puisse exister une représentation de nos données
qui prendrait en compte
les interactions pertinentes entre nos caractéristiques,
sur laquelle un modèle linéaire serait adapté,
nous ne savons tout simplement pas comment le calculer à la main.
Avec les réseaux neuronaux profonds, nous avons utilisé des données d'observation
pour apprendre conjointement une représentation via des couches cachées
et un prédicteur linéaire qui agit sur cette représentation.

Ce problème de non-linéarité est étudié depuis au moins un 
siècle :cite:`Fisher.1928` . Par exemple, les arbres de décision
dans leur forme la plus élémentaire utilisent une séquence de décisions binaires pour 
décider de l'appartenance à une classe :cite:`quinlan2014c4` . De même, les méthodes à noyau 
sont utilisées depuis plusieurs décennies pour modéliser les dépendances non linéaires 
:cite:`Aronszajn.1950` . Ces méthodes ont été utilisées, par exemple, dans les modèles splines non paramétriques 
 :cite:`Wahba.1990` et les méthodes à noyau
:cite:`Scholkopf.Smola.2002` . C'est également une question que le cerveau résout 
tout naturellement. Après tout, les neurones alimentent d'autres neurones qui, 
à leur tour, alimentent d'autres neurones encore :cite:`Cajal.Azoulay.1894` . 
Nous avons donc une séquence de transformations relativement simples. 

### Incorporation de couches cachées

Nous pouvons surmonter les limites des modèles linéaires
en incorporant une ou plusieurs couches cachées.
La façon la plus simple de le faire est d'empiler
de nombreuses couches entièrement connectées les unes sur les autres.
Chaque couche alimente la couche qui la précède,
jusqu'à ce que nous générions des sorties.
Nous pouvons considérer les premières couches $L-1$
 comme notre représentation et la couche finale
comme notre prédicteur linéaire.
Cette architecture est communément appelée
un *perceptron multicouche*,
souvent abrégé en *MLP* (:numref:`fig_mlp` ).

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

Ce MLP a 4 entrées, 3 sorties,
et sa couche cachée contient 5 unités cachées.
Étant donné que la couche d'entrée n'implique aucun calcul,
la production de sorties avec ce réseau
nécessite la mise en œuvre des calculs
pour les couches cachées et de sortie ;
ainsi, le nombre de couches dans ce MLP est de 2.
Notez que les deux couches sont entièrement connectées.
Chaque entrée influence chaque neurone de la couche cachée,
et chacun d'eux influence à son tour
chaque neurone de la couche de sortie. Hélas, nous n'avons pas encore tout à fait terminé
. 

### Du linéaire au non linéaire

Comme précédemment, nous désignons par la matrice $\mathbf{X} \in \mathbb{R}^{n \times d}$
 un minibatch d'exemples $n$ où chaque exemple possède $d$ entrées (caractéristiques).
Pour un MLP à une couche cachée dont la couche cachée comporte $h$ unités cachées,
nous désignons par $\mathbf{H} \in \mathbb{R}^{n \times h}$
 les sorties de la couche cachée, qui sont
*représentations cachées*.
Comme les couches cachée et de sortie sont toutes deux entièrement connectées,
nous avons les poids de la couche cachée $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ et les biais $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
 et les poids de la couche de sortie $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ et les biais $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.
Cela nous permet de calculer les sorties $\mathbf{O} \in \mathbb{R}^{n \times q}$
 de la MLP à une couche cachée comme suit :

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Notez qu'après l'ajout de la couche cachée,
notre modèle nous oblige maintenant à suivre et à mettre à jour
des ensembles de paramètres supplémentaires.
Qu'avons-nous donc gagné en échange ?
Vous pourriez être surpris de découvrir
que - dans le modèle défini ci-dessus - *nous
ne gagnons rien pour nos problèmes* !
La raison en est simple.
Les unités cachées ci-dessus sont données par
une fonction affine des entrées,
et les sorties (présoftmax) sont juste
une fonction affine des unités cachées.
Une fonction affine d'une fonction affine
est elle-même une fonction affine.
De plus, notre modèle linéaire était déjà
capable de représenter toute fonction affine.

Pour le voir formellement, il suffit de supprimer la couche cachée dans la définition ci-dessus,
, ce qui donne un modèle équivalent à une seule couche avec les paramètres
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ et $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

Afin de réaliser le potentiel des architectures multicouches,
nous avons besoin d'un autre ingrédient clé :
une *fonction d'activation non linéaire* $\sigma$
 à appliquer à chaque unité cachée
après la transformation affine. Par exemple, un choix populaire
est la fonction d'activation ReLU (Rectified Linear Unit) :cite:`Nair.Hinton.2010` 
 $\sigma(x) = \mathrm{max}(0, x)$ opérant sur ses arguments élément par élément. 
Les sorties des fonctions d'activation $\sigma(\cdot)$
 sont appelées *activations*.
En général, avec les fonctions d'activation en place,
il n'est plus possible de réduire notre MLP à un modèle linéaire :

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Puisque chaque ligne de $\mathbf{X}$ correspond à un exemple dans le mini-batch,
avec un certain abus de notation, nous définissons la non-linéarité
$\sigma$ pour qu'elle s'applique à ses entrées en fonction des lignes,
c'est-à-dire un exemple à la fois.
Notez que nous avons utilisé la même notation pour softmax
lorsque nous avons indiqué une opération par rangée dans :numref:`subsec_softmax_vectorization` .
Très souvent, les fonctions d'activation que nous utilisons s'appliquent non seulement par rangée mais aussi par élément ( 
). Cela signifie qu'après avoir calculé la partie linéaire de la couche,
nous pouvons calculer chaque activation
sans regarder les valeurs prises par les autres unités cachées.

Pour construire des MLP plus généraux, nous pouvons continuer à empiler
de telles couches cachées,
par exemple, $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
 et $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$,
les unes sur les autres, pour obtenir des modèles toujours plus expressifs.

### Approximateurs universels

Nous savons que le cerveau est capable d'analyses statistiques très sophistiquées. À ce titre, 
il convient de se demander à quel point un réseau profond peut être puissant. Cette question
a reçu de multiples réponses, par exemple dans :cite:`Cybenko.1989` dans le contexte 
des MLP, et dans :cite:`micchelli1984interpolation` dans le contexte de la reproduction des espaces de Hilbert à noyau 
d'une manière qui pourrait être considérée comme des réseaux à fonction de base radiale (RBF) avec une seule couche cachée. 
Ces résultats (ainsi que d'autres résultats connexes) suggèrent que même avec un réseau à une seule couche cachée,
si l'on dispose d'un nombre suffisant de nœuds (peut-être absurdement élevé),
et du bon ensemble de poids,
on peut modéliser n'importe quelle fonction.
L'apprentissage de cette fonction est cependant la partie la plus difficile.
Vous pouvez considérer votre réseau neuronal
comme un peu comme le langage de programmation C. Le langage, comme n'importe quel autre langage moderne, est le langage de base.
Ce langage, comme tout autre langage moderne,
est capable d'exprimer n'importe quel programme calculable.
Mais la partie la plus difficile est de trouver un programme
qui réponde à vos spécifications.

De plus, ce n'est pas parce qu'un réseau à couche unique cachée
*peut* apprendre n'importe quelle fonction
que vous devez essayer
de résoudre tous vos problèmes
avec des réseaux à couche unique cachée. En fait, dans ce cas, les méthodes à noyau 
sont bien plus efficaces, car elles sont capables de résoudre le problème 
*exactement* même dans des espaces de dimension infinie :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001` . 
En fait, nous pouvons approximer de nombreuses fonctions
de manière beaucoup plus compacte en utilisant des réseaux plus profonds (par rapport à des réseaux plus larges) :cite:`Simonyan.Zisserman.2014` .
Nous aborderons des arguments plus rigoureux dans les chapitres suivants.


## Fonctions d'activation
:label:`subsec_activation-functions` 

 Les fonctions d'activation décident si un neurone doit être activé ou non en
calculant la somme pondérée et en y ajoutant un biais.
Ce sont des opérateurs différentiables qui transforment les signaux d'entrée en sorties,
alors que la plupart d'entre eux ajoutent de la non-linéarité.
Les fonctions d'activation étant fondamentales pour l'apprentissage profond,
(**passons brièvement en revue quelques fonctions d'activation courantes**).

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

### Fonction ReLU

Le choix le plus populaire,
en raison à la fois de sa simplicité de mise en œuvre et
de ses bonnes performances sur une variété de tâches prédictives,
est l'unité linéaire *rectifiée* (*ReLU*) :cite:`Nair.Hinton.2010` .
[**ReLU fournit une transformation non linéaire très simple**].
Étant donné un élément $x$, la fonction est définie
comme le maximum de cet élément et $0$:

$$\operatorname{ReLU}(x) = \max(x, 0).$$ 

 De manière informelle, la fonction ReLU ne retient que les éléments positifs
et élimine tous les éléments négatifs
en fixant les activations correspondantes à 0.
Pour gagner en intuition, nous pouvons tracer la fonction.
Comme vous pouvez le constater, la fonction d'activation est linéaire par morceaux.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

Lorsque l'entrée est négative,
la dérivée de la fonction ReLU est 0,
et lorsque l'entrée est positive,
la dérivée de la fonction ReLU est 1.
Notez que la fonction ReLU n'est pas différentiable
lorsque l'entrée prend une valeur précisément égale à 0.
Dans ce cas, nous utilisons par défaut la dérivée de gauche
et disons que la dérivée est égale à 0 lorsque l'entrée est égale à 0.
Nous pouvons nous en sortir car
l'entrée peut ne jamais être nulle (les mathématiciens diraient 
qu'elle est indifférenciable sur un ensemble de mesure zéro).
Un vieil adage dit que si des conditions limites subtiles sont importantes,
nous faisons probablement des mathématiques (*réelles*) et non de l'ingénierie.
Cette sagesse conventionnelle peut s'appliquer ici, ou du moins, le fait que 
nous n'effectuons pas une optimisation sous contrainte :cite:`Mangasarian.1965,Rockafellar.1970` . 
Nous traçons la dérivée de la fonction ReLU représentée ci-dessous.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

La raison pour laquelle nous utilisons ReLU est que
ses dérivées se comportent particulièrement bien :
soit elles disparaissent, soit elles laissent passer l'argument.
Cela rend l'optimisation plus efficace
et atténue le problème bien documenté
des gradients évanescents qui affectait
les versions précédentes des réseaux neuronaux (nous y reviendrons plus tard).

Notez qu'il existe de nombreuses variantes de la fonction ReLU,
dont la fonction *ReLU* paramétrée (*pReLU*) :cite:`He.Zhang.Ren.ea.2015` .
Cette variante ajoute un terme linéaire à ReLU,
de sorte que certaines informations sont toujours transmises,
même lorsque l'argument est négatif :

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$ 

 ### Fonction Sigmoïde

[**La *fonction sigmoïde* transforme ses entrées**],
pour lesquelles les valeurs se situent dans le domaine $\mathbb{R}$,
(**en sorties qui se situent sur l'intervalle (0, 1).**)
Pour cette raison, la sigmoïde est
souvent appelée une *fonction d'écrasement* :
elle écrase toute entrée dans l'intervalle (-inf, inf)
à une valeur dans l'intervalle (0, 1) :

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$ 

 Dans les premiers réseaux neuronaux, les scientifiques
étaient intéressés par la modélisation des neurones biologiques
qui soit *tirent* soit *ne tirent pas*.
Ainsi, les pionniers de ce domaine,
en remontant jusqu'à McCulloch et Pitts,
les inventeurs du neurone artificiel,
se sont concentrés sur les unités de seuillage :cite:`McCulloch.Pitts.1943` .
Une activation à seuil prend la valeur 0
lorsque son entrée est inférieure à un certain seuil
et la valeur 1 lorsque l'entrée dépasse le seuil.

Lorsque l'attention s'est portée sur l'apprentissage basé sur le gradient,
la fonction sigmoïde était un choix naturel
parce qu'elle est une approximation lisse et différentiable
d'une unité de seuillage.
Les sigmoïdes sont encore largement utilisées comme
fonctions d'activation sur les unités de sortie,
lorsque nous voulons interpréter les sorties comme des probabilités
pour les problèmes de classification binaire : vous pouvez considérer la sigmoïde comme un cas particulier de la softmax.
Cependant, la sigmoïde a généralement été remplacée
par la ReLU
, plus simple et plus facile à entraîner, pour la plupart des utilisations dans les couches cachées. Cela est dû en grande partie 
au fait que la sigmoïde pose des problèmes d'optimisation
:cite:`LeCun.Bottou.Orr.ea.1998` car son gradient disparaît pour les arguments positifs * et négatifs * importants. 
Cela peut conduire à des plateaux dont il est difficile de sortir. 
Néanmoins, les sigmoïdes sont importantes. Dans les chapitres ultérieurs (par exemple, :numref:`sec_lstm` ) sur les réseaux neuronaux récurrents,
nous décrirons des architectures qui exploitent les unités sigmoïdes
pour contrôler le flux d'informations dans le temps.

Ci-dessous, nous traçons la fonction sigmoïde.
Notez que lorsque l'entrée est proche de 0,
la fonction sigmoïde se rapproche de
une transformation linéaire.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

La dérivée de la fonction sigmoïde est donnée par l'équation suivante :

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$ 

 
 La dérivée de la fonction sigmoïde est tracée ci-dessous.
Notez que lorsque l'entrée est 0,
la dérivée de la fonction sigmoïde
atteint un maximum de 0,25.
Lorsque l'entrée diverge de 0 dans un sens ou dans l'autre,
la dérivée se rapproche de 0.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### Fonction tanh

Comme la fonction sigmoïde, [**la fonction tanh (tangente hyperbolique)
écrase également ses entrées**],
en les transformant en éléments sur l'intervalle (**entre -1 et 1**) :

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$ 

 Nous traçons la fonction tanh ci-dessous. Notez que lorsque l'entrée se rapproche de 0, la fonction tanh se rapproche d'une transformation linéaire. Bien que la forme de la fonction soit similaire à celle de la fonction sigmoïde, la fonction tanh présente une symétrie ponctuelle autour de l'origine du système de coordonnées :cite:`Kalman.Kwasny.1992` .

```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

La dérivée de la fonction tanh est :

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$ 

 Elle est représentée ci-dessous.
Lorsque l'entrée se rapproche de 0,
la dérivée de la fonction tanh s'approche d'un maximum de 1.
Et comme nous l'avons vu avec la fonction sigmoïde,
lorsque l'entrée s'éloigne de 0 dans une direction ou dans une autre,
la dérivée de la fonction tanh s'approche de 0.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

## Résumé

Nous savons maintenant comment incorporer les non-linéarités
pour construire des architectures de réseaux neuronaux multicouches expressives.
À titre d'information, vos connaissances
vous placent déjà aux commandes d'une boîte à outils
similaire à celle d'un praticien des années 1990.
D'une certaine manière, vous avez un avantage
sur toute personne travaillant dans les années 1990,
car vous pouvez exploiter les puissants cadres d'apprentissage profond open-source

 pour construire des modèles rapidement, en utilisant seulement quelques lignes de code.
Auparavant, l'entraînement de ces réseaux
nécessitait que les chercheurs codent les couches et les dérivés
explicitement en C, Fortran ou même Lisp (dans le cas de LeNet). 

Un avantage secondaire est que ReLU se prête beaucoup plus à l'optimisation
que la fonction sigmoïde ou tanh. On pourrait dire que 
est l'une des innovations clés qui a contribué à la résurgence
de l'apprentissage profond au cours de la dernière décennie. Notez toutefois que la recherche sur les fonctions d'activation 
ne s'est pas arrêtée. Par exemple, la fonction d'activation Swish 
 $\sigma(x) = x \operatorname{sigmoid}(\beta x)$ telle que proposée dans
:cite:`Ramachandran.Zoph.Le.2017` peut donner une meilleure précision 
dans de nombreux cas.

## Exercices

1. Montrez que l'ajout de couches à un réseau profond *linéaire*, c'est-à-dire un réseau sans 
 non-linéarité $\sigma$ ne peut jamais augmenter le pouvoir expressif du réseau. 
   Donnez un exemple où il le réduit activement. 
1. Calculez la dérivée de la fonction d'activation pReLU.
1. Calculez la dérivée de la fonction d'activation Swish $x \operatorname{sigmoid}(\beta x)$. 
1. Montrez qu'un MLP utilisant uniquement ReLU (ou pReLU) construit une fonction linéaire par morceaux continue 
.
1. Sigmoïde et tanh sont très similaires. 
    1. Montrez que $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
   1. Prouvez que les classes de fonctions paramétrées par les deux non-linéarités sont identiques. Indice : les couches affines ont également des termes de biais.
1. Supposons que nous ayons une non-linéarité qui s'applique à un seul mini-lot à la fois, comme la normalisation par lot :cite:`Ioffe.Szegedy.2015` . Quels types de problèmes pensez-vous que cela va causer ?
1. Donnez un exemple où les gradients disparaissent pour la fonction d'activation sigmoïde 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
