# Réseaux neuronaux récurrents
:label:`sec_rnn` 

 
Dans :numref:`sec_language-model` nous avons décrit des modèles de Markov et $n$-grammes pour la modélisation du langage, où la probabilité conditionnelle d'un token $x_t$ au pas de temps $t$ ne dépend que des $n-1$ tokens précédents.
Si nous voulons intégrer l'effet possible des jetons antérieurs à l'étape temporelle $t-(n-1)$ sur $x_t$,
nous devons augmenter $n$.
Cependant, le nombre de paramètres du modèle augmenterait également de manière exponentielle, car nous devons stocker les numéros de $|\mathcal{V}|^n$ pour un ensemble de vocabulaire $\mathcal{V}$.
Par conséquent, plutôt que de modéliser $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$, il est préférable d'utiliser un modèle à variables latentes :

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$ 

où $h_{t-1}$ est un *état caché* qui stocke les informations sur la séquence jusqu'au pas de temps $t-1$.
En général,
l'état caché à n'importe quel pas de temps $t$ pourrait être calculé sur la base de l'entrée actuelle $x_{t}$ et de l'état caché précédent $h_{t-1}$:

$$h_t = f(x_{t}, h_{t-1}).$$ 
:eqlabel:`eq_ht_xt` 

Pour une fonction suffisamment puissante $f$ dans :eqref:`eq_ht_xt`, le modèle de variable latente n'est pas une approximation. Après tout, $h_t$ peut simplement stocker toutes les données qu'il a observées jusqu'à présent.
Cependant, cela pourrait potentiellement rendre le calcul et le stockage coûteux.

Rappelons que nous avons discuté des couches cachées avec les unités cachées dans :numref:`chap_perceptrons`.
Il convient de noter que
couches cachées et états cachés font référence à deux concepts très différents.
Les couches cachées sont, comme expliqué, des couches qui sont cachées à la vue sur le chemin de l'entrée à la sortie.
Les états cachés sont, techniquement parlant, les *entrées* de ce que nous faisons à une étape donnée,
et ils ne peuvent être calculés qu'en examinant les données des étapes précédentes.

*Les réseaux neuronaux récurrents* (RNN) sont des réseaux neuronaux à états cachés. Avant de présenter le modèle RNN, nous allons d'abord revoir le modèle MLP présenté dans :numref:`sec_mlp`.

## Réseaux neuronaux sans états cachés

Examinons un MLP à une seule couche cachée.
La fonction d'activation de la couche cachée est $\phi$.
Étant donné un minilot d'exemples $\mathbf{X} \in \mathbb{R}^{n \times d}$ avec une taille de lot $n$ et des entrées $d$, la sortie de la couche cachée $\mathbf{H} \in \mathbb{R}^{n \times h}$ est calculée comme suit :

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$ 
:eqlabel:`rnn_h_without_state` 

Dans :eqref:`rnn_h_without_state`, nous avons le paramètre de poids $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$, le paramètre de biais $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$, et le nombre d'unités cachées $h$, pour la couche cachée.
Ainsi, la diffusion (voir :numref:`subsec_broadcasting` ) est appliquée pendant la sommation.
Ensuite, la sortie de la couche cachée $\mathbf{H}$ est utilisée comme entrée de la couche de sortie. La couche de sortie est donnée par

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$ 

où $\mathbf{O} \in \mathbb{R}^{n \times q}$ est la variable de sortie, $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ est le paramètre de poids, et $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ est le paramètre de biais de la couche de sortie.  S'il s'agit d'un problème de classification, nous pouvons utiliser $\text{softmax}(\mathbf{O})$ pour calculer la distribution de probabilité des catégories de sortie.

Ceci est tout à fait analogue au problème de régression que nous avons résolu précédemment dans :numref:`sec_sequence`, nous n'en donnons donc pas les détails.
Il suffit de dire que nous pouvons choisir des paires caractéristique-étiquette au hasard et apprendre les paramètres de notre réseau par différenciation automatique et descente de gradient stochastique.

## Réseaux neuronaux récurrents avec états cachés
:label:`subsec_rnn_w_hidden_states` 

Les choses sont totalement différentes lorsque nous avons des états cachés. Examinons la structure plus en détail.

Supposons que nous ayons
un minibatch d'entrées
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ 
au pas de temps $t$.
En d'autres termes,
pour un minilot d'exemples de séquence $n$,
chaque ligne de $\mathbf{X}_t$ correspond à un exemple au pas de temps $t$ de la séquence.
Ensuite,
désigne par $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ la sortie de la couche cachée au pas de temps $t$.
Contrairement au MLP, nous sauvegardons ici la sortie de la couche cachée $\mathbf{H}_{t-1}$ du pas de temps précédent et introduisons un nouveau paramètre de poids $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ pour décrire comment utiliser la sortie de la couche cachée du pas de temps précédent dans le pas de temps actuel. Plus précisément, le calcul de la sortie de la couche cachée du pas de temps actuel est déterminé par l'entrée du pas de temps actuel et la sortie de la couche cachée du pas de temps précédent :

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$ 
:eqlabel:`rnn_h_with_state` 

Par rapport à :eqref:`rnn_h_without_state`, :eqref:`rnn_h_with_state` ajoute un terme supplémentaire $\mathbf{H}_{t-1} \mathbf{W}_{hh}$ et donc
instancie :eqref:`eq_ht_xt`.
D'après la relation entre les sorties de couche cachée $\mathbf{H}_t$ et $\mathbf{H}_{t-1}$ des pas de temps adjacents,
nous savons que ces variables ont capturé et conservé les informations historiques de la séquence jusqu'à leur pas de temps actuel, tout comme l'état ou la mémoire du pas de temps actuel du réseau neuronal. Par conséquent, une telle sortie de couche cachée est appelée un *état caché*.
Comme l'état caché utilise la même définition du pas de temps précédent dans le pas de temps actuel, le calcul de :eqref:`rnn_h_with_state` est *récurrent*. Par conséquent, comme nous l'avons dit, les réseaux neuronaux à états cachés
basés sur le calcul récurrent sont appelés
*réseaux neuronaux récurrents*.
Les couches qui effectuent
le calcul de :eqref:`rnn_h_with_state` 
dans les RNN
sont appelées couches *récurrentes*.


Il existe de nombreuses façons différentes de construire des RNN.
Les RNN avec un état caché défini par :eqref:`rnn_h_with_state` sont très courants.
Pour le pas de temps $t$,
la sortie de la couche de sortie est similaire au calcul dans le MLP :

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$ 

Les paramètres du RNN
comprennent les poids $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$,
et le biais $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$
de la couche cachée,
ainsi que les poids $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$
et le biais $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$
de la couche de sortie.
Il convient de mentionner que
même à différents pas de temps,
RNNs utilisent toujours ces paramètres de modèle.
Par conséquent, le coût de paramétrage d'un RNN
n'augmente pas avec le nombre de pas de temps.

:numref:`fig_rnn` illustre la logique de calcul d'un RNN à trois pas de temps adjacents.
À tout pas de temps $t$,
le calcul de l'état caché peut être traité comme suit :
(i) concaténation de l'entrée $\mathbf{X}_t$ au pas de temps actuel $t$ et de l'état caché $\mathbf{H}_{t-1}$ au pas de temps précédent $t-1$;
(ii) introduction du résultat de la concaténation dans une couche entièrement connectée avec la fonction d'activation $\phi$.
La sortie d'une telle couche entièrement connectée est l'état caché $\mathbf{H}_t$ de l'étape temporelle actuelle $t$.
Dans ce cas,
les paramètres du modèle sont la concaténation de $\mathbf{W}_{xh}$ et $\mathbf{W}_{hh}$, et un biais de $\mathbf{b}_h$, tous issus de :eqref:`rnn_h_with_state`.
L'état caché du pas de temps actuel $t$, $\mathbf{H}_t$, participera au calcul de l'état caché $\mathbf{H}_{t+1}$ du pas de temps suivant $t+1$.
De plus, $\mathbf{H}_t$ sera aussi
alimenté dans la couche de sortie entièrement connectée
pour calculer la sortie
$\mathbf{O}_t$ du pas de temps actuel $t$.

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

Nous venons de mentionner que le calcul de $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ pour l'état caché est équivalent à
la multiplication matricielle de
la concaténation de $\mathbf{X}_t$ et $\mathbf{H}_{t-1}$
et
la concaténation de $\mathbf{W}_{xh}$ et $\mathbf{W}_{hh}$.
Bien que cela puisse être prouvé en mathématiques,
dans ce qui suit, nous utilisons simplement un extrait de code simple pour le montrer.
Pour commencer,
nous définissons les matrices `X`, `W_xh`, `H`, et `W_hh`, dont les formes sont (3, 1), (1, 4), (3, 4), et (4, 4), respectivement.
En multipliant `X` par `W_xh`, et `H` par `W_hh`, respectivement, puis en additionnant ces deux multiplications, on obtient,
une matrice de forme (3, 4).

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Nous concaténons maintenant les matrices `X` et `H`
le long des colonnes (axe 1),
et les matrices
`W_xh` et `W_hh` le long des lignes (axe 0).
Ces deux concaténations
donnent lieu à des matrices
de forme (3, 5)
et de forme (5, 4), respectivement.
En multipliant ces deux matrices concaténées,
nous obtenons la même matrice de sortie de forme (3, 4)
que ci-dessus.

```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## Modèles de langage au niveau des caractères basés sur les RNN

Rappelons que pour la modélisation du langage dans :numref:`sec_language-model`,
nous cherchons à prédire le prochain token en nous basant sur
les tokens actuels et passés,
; nous décalons donc la séquence originale d'un token
comme cibles (étiquettes).
Bengio et al. ont proposé pour la première fois
d'utiliser un réseau de neurones pour la modélisation du langage :cite:`Bengio.Ducharme.Vincent.ea.2003`.
Dans ce qui suit, nous illustrons comment les RNN peuvent être utilisés pour construire un modèle de langage.
Supposons que la taille des minibatchs soit de un et que la séquence du texte soit "machine".
Pour simplifier l'apprentissage dans les sections suivantes,
nous segmentons le texte en caractères plutôt qu'en mots
et considérons un modèle de langage au niveau des caractères *.
:numref:`fig_rnn_train` montre comment prédire le caractère suivant en fonction des caractères actuels et précédents via un RNN pour la modélisation du langage au niveau des caractères.

![A character-level language model based on the RNN. The input and target sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

Au cours du processus de formation,
nous exécutons une opération softmax sur la sortie de la couche de sortie pour chaque pas de temps, puis nous utilisons la perte d'entropie croisée pour calculer l'erreur entre la sortie du modèle et la cible.
En raison du calcul récurrent de l'état caché dans la couche cachée, la sortie du pas de temps 3 dans :numref:`fig_rnn_train`,
$\mathbf{O}_3$ , est déterminée par la séquence de texte "m", "a" et "c". Comme le caractère suivant de la séquence dans les données d'apprentissage est "h", la perte de l'étape temporelle 3 dépendra de la distribution de probabilité du caractère suivant généré sur la base de la séquence de caractéristiques "m", "a", "c" et de la cible "h" de cette étape temporelle.

En pratique, chaque jeton est représenté par un vecteur à $d$-dimensions, et nous utilisons une taille de lot $n>1$. Par conséquent, l'entrée $\mathbf X_t$ au pas de temps $t$ sera une matrice $n\times d$, ce qui est identique à ce que nous avons discuté dans :numref:`subsec_rnn_w_hidden_states`.



 
Dans les sections suivantes, nous implémenterons des RNN
pour les modèles de langage au niveau des caractères et utiliserons la perplexité
pour évaluer ces modèles.


## Résumé

* Un réseau neuronal qui utilise le calcul récurrent pour les états cachés est appelé un réseau neuronal récurrent (RNN).
* L'état caché d'un RNN peut capturer des informations historiques de la séquence jusqu'au pas de temps actuel.
* Le nombre de paramètres du modèle RNN n'augmente pas avec le nombre de pas de temps.
* Nous pouvons créer des modèles de langage au niveau des caractères en utilisant un RNN.
* Nous pouvons utiliser la perplexité pour évaluer la qualité des modèles de langage.

## Exercices

1. Si nous utilisons un RNN pour prédire le prochain caractère dans une séquence de texte, quelle est la dimension requise pour toute sortie ?
1. Pourquoi les RNN peuvent-ils exprimer la probabilité conditionnelle d'un jeton à un certain moment en fonction de tous les jetons précédents dans la séquence de texte ?
1. Qu'arrive-t-il au gradient si l'on effectue une rétropropagation à travers une longue séquence ?
1. Quels sont certains des problèmes associés au modèle de langage décrit dans cette section ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
