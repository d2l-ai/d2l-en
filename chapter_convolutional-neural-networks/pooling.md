```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Pooling
:label:`sec_pooling` 

Dans de nombreux cas, notre tâche ultime pose une question globale sur l'image,
par exemple, *contient-elle un chat ?* Par conséquent, les unités de notre couche finale 
doivent être sensibles à l'ensemble de l'entrée.
En agrégeant progressivement les informations, en produisant des cartes de plus en plus grossières,
nous atteignons cet objectif d'apprentissage final d'une représentation globale,
tout en conservant tous les avantages des couches convolutionnelles aux couches de traitement intermédiaires.
Plus nous descendons dans le réseau,
plus le champ réceptif (par rapport à l'entrée)
auquel chaque nœud caché est sensible est grand. La réduction de la résolution spatiale 
accélère ce processus, 
puisque les noyaux de convolution couvrent une plus grande surface effective. 

De plus, lors de la détection de caractéristiques de plus bas niveau, telles que les bords
(comme nous l'avons vu dans :numref:`sec_conv_layer` ),
nous voulons souvent que nos représentations soient quelque peu invariantes à la translation.
Par exemple, si nous prenons l'image `X`
avec une délimitation nette entre le noir et le blanc
et que nous déplaçons l'ensemble de l'image d'un pixel vers la droite,
c'est-à-dire `Z[i, j] = X[i, j + 1]`,
le résultat pour la nouvelle image `Z` pourrait être très différent.
Le bord se sera déplacé d'un pixel.
En réalité, les objets ne se trouvent presque jamais exactement au même endroit.)[
En fait, même avec un trépied et un objet immobile,
les vibrations de l'appareil photo dues au mouvement de l'obturateur
peuvent tout décaler d'un pixel ou plus
(les appareils photo haut de gamme sont dotés de fonctions spéciales pour résoudre ce problème).

Cette section présente les couches de *pooling*,
qui ont pour double objectif 
d'atténuer la sensibilité des couches convolutionnelles à l'emplacement
et de sous-échantillonner spatialement les représentations.

## Maximum Pooling et Average Pooling

Comme les couches convolutives, les opérateurs de *pooling*
consistent en une fenêtre de forme fixe qui est glissée sur
toutes les régions de l'entrée en fonction de son pas,
calculant une seule sortie pour chaque emplacement traversé
par la fenêtre de forme fixe (parfois appelée fenêtre de *pooling*).
Cependant, contrairement au calcul de corrélation croisée
des entrées et des noyaux dans la couche convolutionnelle,
la couche de pooling ne contient aucun paramètre (il n'y a pas de *noyau*).
Au lieu de cela, les opérateurs de pooling sont déterministes,
calculant généralement la valeur maximale ou moyenne
des éléments de la fenêtre de pooling.
Ces opérations sont appelées respectivement *pooling maximal* (*max-pooling* pour faire court)
et *average pooling*.

L'*Average pooling* est essentiellement aussi vieux que les CNN. L'idée est semblable à celle de ,
qui consiste à sous-échantillonner une image. Plutôt que de prendre simplement la valeur d'un pixel sur deux (ou sur trois) 
pour l'image à faible résolution, nous pouvons faire la moyenne des pixels adjacents pour obtenir 
une image avec un meilleur rapport signal/bruit puisque nous combinons les informations 
de plusieurs pixels adjacents. *Le max-pooling* a été introduit à l'adresse 
:cite:`Riesenhuber.Poggio.1999` dans le contexte des neurosciences cognitives pour décrire 
comment l'agrégation des informations pourrait être hiérarchisée dans le but 
de la reconnaissance des objets, et une version antérieure dans la reconnaissance de la parole :cite:$$`Yamaguchi.Sakamoto.Akabane.ea.1990`. Dans presque tous les cas, le max-pooling, comme on l'appelle aussi, 
est préférable. 

Dans les deux cas, comme pour l'opérateur de corrélation croisée,
on peut considérer que la fenêtre de pooling
commence en haut à gauche du tenseur d'entrée
et glisse sur le tenseur d'entrée de gauche à droite et de haut en bas.
À chaque emplacement que la fenêtre de pooling rencontre,
elle calcule la valeur maximale ou moyenne
du sous-tenseur d'entrée dans la fenêtre,
selon que le max pooling ou l'average pooling est employée.


![Max-pooling avec une forme de fenêtre de pooling de $2\times 2$. Les parties ombragées sont le premier élément de sortie ainsi que les éléments tenseurs d'entrée utilisés pour le calcul de la sortie : $\max(0, 1, 3, 4=4$)$.](../img/pooling.svg)
:label:`fig_pooling` 

Le tenseur de sortie dans :numref:`fig_pooling` a une hauteur de 2 et une largeur de 2.
Les quatre éléments sont dérivés de la valeur maximale dans chaque fenêtre de pooling :

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Plus généralement, nous pouvons définir une couche de pooling $p \times q$ en agrégeant sur 
une région de ladite taille. Pour en revenir au problème de la détection des bords, 
nous utilisons la sortie de la couche convolutionnelle
comme entrée pour $2\times 2$ max-pooling.
On désigne par `X` l'entrée de la couche convolutionnelle et par `Y` la sortie de la couche de pooling. 
Que les valeurs de `X[i, j]`, `X[i, j + 1]`, 
`X[i+1, j]` et `X[i+1, j + 1]` soient différentes ou non,
la couche de pooling produit toujours `Y[i, j] = 1`.
En d'autres termes, en utilisant la couche de max pooling $2\times 2$,
nous pouvons toujours détecter si le motif reconnu par la couche convolutive
ne se déplace pas de plus d'un élément en hauteur ou en largeur.

Dans le code ci-dessous, nous (**implémentons la propagation vers l'avant
de la couche de pooling**) dans la fonction `pool2d`.
Cette fonction est similaire à la fonction `corr2d`
dans :numref:`sec_conv_layer`.
Cependant, aucun noyau n'est nécessaire, et la sortie
est calculée comme le maximum ou la moyenne de chaque région de l'entrée.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

Nous pouvons construire le tenseur d'entrée `X` dans :numref:`fig_pooling` pour [**valider la sortie de la couche bidimensionnelle de max-pooling**].

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

De plus, nous expérimentons avec (**la couche d'average pooling**).

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**Padding et Stride**]

Comme pour les couches convolutionnelles, les couches de pooling
modifient la forme de la sortie.
Et comme précédemment, nous pouvons ajuster l'opération pour obtenir une forme de sortie souhaitée
en remplissant l'entrée et en ajustant le pas.
Nous pouvons démontrer l'utilisation du remplissage et des pas
dans les couches de pooling grâce à la couche de max pooling bidimensionnelle intégrée dans le cadre de l'apprentissage profond.
Nous construisons d'abord un tenseur d'entrée `X` dont la forme a quatre dimensions,
où le nombre d'exemples (taille du lot) et le nombre de canaux sont tous deux égaux à 1.

:begin_tab:`tensorflow` 
Notez que contrairement à d'autres frameworks, TensorFlow préfère et est optimisé pour l'entrée *channels-last*.

:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Étant donné que le pooling regroupe les informations d'une zone, les cadres d'apprentissage profond ** adaptent par défaut la taille des fenêtres de pooling et le stride.** Par exemple, si nous utilisons une fenêtre de pooling de la forme `(3, 3)`
 , nous obtenons par défaut une forme de stride de `(3, 3)`.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

Comme prévu, [**le stride et le padding peuvent être spécifiés manuellement**] pour remplacer les valeurs par défaut du framework si nécessaire.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

Bien sûr, nous pouvons spécifier une fenêtre de pooling rectangulaire arbitraire avec une hauteur et une largeur arbitraires respectivement, comme le montre l'exemple ci-dessous.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

## Canaux multiples

Lors du traitement de données d'entrée à canaux multiples,
[**la couche de pooling met en commun chaque canal d'entrée séparément**],
plutôt que d'additionner les entrées sur les canaux
comme dans une couche convolutive.
Cela signifie que le nombre de canaux de sortie pour la couche de pooling
est le même que le nombre de canaux d'entrée.
Ci-dessous, nous allons concaténer les tenseurs `X` et `X + 1`
sur la dimension des canaux pour construire une entrée avec 2 canaux.

:begin_tab:`tensorflow`
Notez que cela nécessitera une concaténation
le long de la dernière dimension pour TensorFlow en raison de la syntaxe channels-last.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

Comme nous pouvons le voir, le nombre de canaux de sortie est toujours de 2 après la pooling.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

:begin_tab:`tensorflow`
Notez que la sortie pour le pooling TensorFlow semble à première vue être différente, cependant
numériquement les mêmes résultats sont présentés que MXNet et PyTorch.
La différence réside dans la dimensionnalité, et la lecture verticale de la sortie de
donne les mêmes résultats que les autres implémentations.
:end_tab:

## Résumé

La pooling est une opération extrêmement simple. Elle fait exactement ce que son nom indique, agréger les résultats sur une fenêtre de valeurs. Toutes les sémantiques de convolution, comme les strides et le padding, s'appliquent de la même manière que précédemment. Notez que le pooling est indifférent aux canaux, c'est-à-dire qu'il laisse le nombre de canaux inchangé et s'applique à chaque canal séparément. Enfin, parmi les deux choix populaires de pooling, le max pooling est préférable a l'average pooling, car elle confère un certain degré d'invariance à la sortie. Un choix populaire consiste à choisir une taille de fenêtre de pooling de $2 \times 2$ pour diviser par quatre la résolution spatiale de la sortie. 

Notez qu'il existe de nombreuses autres façons de réduire la résolution au-delà du pooling. Par exemple, dans le pooling stochastique :cite:`Zeiler.Fergus.2013` et le max-pooling fractionnel :cite:`Graham.2014`, l'agrégation est combinée à la randomisation. Cela peut améliorer légèrement la précision dans certains cas. Enfin, comme nous le verrons plus tard avec le mécanisme d'attention, il existe des moyens plus raffinés d'agréger sur les sorties, par exemple en utilisant l'alignement entre une requête et les vecteurs de représentation. 


## Exercices

1. Implémenter l'average pooling par une convolution. 
1. Prouvez que le max-pooling ne peut pas être implémenté par une convolution seule. 
1. Le max-pooling peut être réalisé en utilisant des opérations ReLU, c'est-à-dire $\mathrm{ReLU}(x) = \max(0, x)$.
  1. Exprimez $\max (a, b)$ en utilisant uniquement des opérations ReLU.
  1. Utilisez-le pour mettre en œuvre le max-pooling au moyen de convolutions et de couches ReLU. 
    1. De combien de canaux et de couches avez-vous besoin pour une convolution $2 \times 2$? Combien pour une convolution $3 \times 3$. 
1. Quel est le coût de calcul de la couche de pooling ? Supposons que l'entrée de la couche de pooling soit de taille $c\times h\times w$, la fenêtre de pooling a une forme de $p_h\times p_w$ avec un padding de $(p_h, p_w)$ et un stride de $(s_h, s_w)$.
1. Pourquoi pensez-vous que le max pooling et l'average pooling fonctionnent différemment ?
1. Avons-nous besoin d'une couche distincte de pooling minimale ? Peut-on la remplacer par une autre opération ?
1. Nous pourrions utiliser l'opération softmax pour la pooling. Pourquoi ne serait-elle pas si populaire ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab: 

:begin_tab:`pytorch` 
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab: 

:begin_tab:`tensorflow` 
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
