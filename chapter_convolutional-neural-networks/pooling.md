```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Pooling
:label:`sec_pooling`

In many cases our ultimate task asks some global question about the image,
e.g., *does it contain a cat?* Consequently, the units of our final layer 
should be sensitive to the entire input.
By gradually aggregating information, yielding coarser and coarser maps,
we accomplish this goal of ultimately learning a global representation,
while keeping all of the advantages of convolutional layers at the intermediate layers of processing.
The deeper we go in the network,
the larger the receptive field (relative to the input)
to which each hidden node is sensitive. Reducing spatial resolution 
accelerates this process, 
since the convolution kernels cover a larger effective area. 

Moreover, when detecting lower-level features, such as edges
(as discussed in :numref:`sec_conv_layer`),
we often want our representations to be somewhat invariant to translation.
For instance, if we take the image `X`
with a sharp delineation between black and white
and shift the whole image by one pixel to the right,
i.e., `Z[i, j] = X[i, j + 1]`,
then the output for the new image `Z` might be vastly different.
The edge will have shifted by one pixel.
In reality, objects hardly ever occur exactly at the same place.
In fact, even with a tripod and a stationary object,
vibration of the camera due to the movement of the shutter
might shift everything by a pixel or so
(high-end cameras are loaded with special features to address this problem).

This section introduces *pooling layers*,
which serve the dual purposes of
mitigating the sensitivity of convolutional layers to location
and of spatially downsampling representations.

## Maximum Pooling and Average Pooling

Like convolutional layers, *pooling* operators
consist of a fixed-shape window that is slid over
all regions in the input according to its stride,
computing a single output for each location traversed
by the fixed-shape window (sometimes known as the *pooling window*).
However, unlike the cross-correlation computation
of the inputs and kernels in the convolutional layer,
the pooling layer contains no parameters (there is no *kernel*).
Instead, pooling operators are deterministic,
typically calculating either the maximum or the average value
of the elements in the pooling window.
These operations are called *maximum pooling* (*max-pooling* for short)
and *average pooling*, respectively.

*Average pooling* is essentially as old as CNNs. The idea is akin to 
downsampling an image. Rather than just taking the value of every second (or third) 
pixel for the lower resolution image, we can average over adjacent pixels to obtain 
an image with better signal to noise ratio since we are combining the information 
from multiple adjacent pixels. *Max-pooling* was introduced in 
:cite:`Riesenhuber.Poggio.1999` in the context of cognitive neuroscience to describe 
how information aggregation might be aggregated hierarchically for the purpose 
of object recognition, and an earlier version in speech recognition :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. In almost all cases, max-pooling, as it is also referred to, 
is preferable. 

In both cases, as with the cross-correlation operator,
we can think of the pooling window
as starting from the upper-left of the input tensor
and sliding across the input tensor from left to right and top to bottom.
At each location that the pooling window hits,
it computes the maximum or average
value of the input subtensor in the window,
depending on whether max or average pooling is employed.


![Max-pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
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
On désigne par `X` l'entrée de la couche convolutionnelle et par `Y` la sortie de la couche de mise en commun. 
Que les valeurs de `X[i, j]`, `X[i, j + 1]`, 
`X[i+1, j]` et `X[i+1, j + 1]` soient différentes ou non,
la couche de mise en commun produit toujours `Y[i, j] = 1`.
En d'autres termes, en utilisant la couche de mise en commun maximale $2\times 2$,
nous pouvons toujours détecter si le motif reconnu par la couche convolutive
ne se déplace pas de plus d'un élément en hauteur ou en largeur.

Dans le code ci-dessous, nous (**implémentons la propagation vers l'avant
de la couche de pooling**) dans la fonction `pool2d`.
Cette fonction est similaire à la fonction `corr2d`
 dans :numref:`sec_conv_layer` .
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

De plus, nous expérimentons avec (**la couche de mise en commun moyenne**).

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
dans les couches de mise en commun grâce à la couche de mise en commun maximale bidimensionnelle intégrée dans le cadre de l'apprentissage profond.
Nous construisons d'abord un tenseur d'entrée `X` dont la forme a quatre dimensions,
où le nombre d'exemples (taille du lot) et le nombre de canaux sont tous deux égaux à 1.

:begin_tab:`tensorflow` 
 Notez que contrairement à d'autres frameworks, TensorFlow
préfère et est optimisé pour l'entrée *channels-last*.
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

Étant donné que le pooling regroupe les informations d'une zone, les cadres d'apprentissage profond ** adaptent par défaut la taille des fenêtres de pooling et le stride.** Par exemple, si nous utilisons une fenêtre de mise en commun de la forme `(3, 3)`
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
[**la couche de mise en commun met en commun chaque canal d'entrée séparément**],
plutôt que d'additionner les entrées sur les canaux
comme dans une couche convolutive.
Cela signifie que le nombre de canaux de sortie pour la couche de mise en commun
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

Comme nous pouvons le voir, le nombre de canaux de sortie est toujours de 2 après la mise en commun.

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

La mise en commun est une opération extrêmement simple. Elle fait exactement ce que son nom indique, agréger les résultats sur une fenêtre de valeurs. Toutes les sémantiques de convolution, comme les strides et le padding, s'appliquent de la même manière que précédemment. Notez que le pooling est indifférent aux canaux, c'est-à-dire qu'il laisse le nombre de canaux inchangé et s'applique à chaque canal séparément. Enfin, parmi les deux choix populaires de mise en commun, la mise en commun maximale est préférable à la mise en commun moyenne, car elle confère un certain degré d'invariance à la sortie. Un choix populaire consiste à choisir une taille de fenêtre de pooling de $2 \times 2$ pour diviser par quatre la résolution spatiale de la sortie. 

Notez qu'il existe de nombreuses autres façons de réduire la résolution au-delà du pooling. Par exemple, dans le pooling stochastique :cite:`Zeiler.Fergus.2013` et le max-pooling fractionnel :cite:`Graham.2014` , l'agrégation est combinée à la randomisation. Cela peut améliorer légèrement la précision dans certains cas. Enfin, comme nous le verrons plus tard avec le mécanisme d'attention, il existe des moyens plus raffinés d'agréger sur les sorties, par exemple en utilisant l'alignement entre une requête et les vecteurs de représentation. 


## Exercices

1. Implémenter la mise en commun des moyennes par une convolution. 
1. Prouvez que le max-pooling ne peut pas être implémenté par une convolution seule. 
1. Le max-pooling peut être réalisé en utilisant des opérations ReLU, c'est-à-dire $\mathrm{ReLU}(x) = \max(0, x)$.
   1. Exprimez $\max (a, b)$ en utilisant uniquement des opérations ReLU.
   1. Utilisez-le pour mettre en œuvre le max-pooling au moyen de convolutions et de couches ReLU. 
    1. De combien de canaux et de couches avez-vous besoin pour une convolution $2 \times 2$? Combien pour une convolution $3 \times 3$. 
1. Quel est le coût de calcul de la couche de mise en commun ? Supposons que l'entrée de la couche de mise en commun soit de taille $c\times h\times w$, la fenêtre de mise en commun a une forme de $p_h\times p_w$ avec un padding de $(p_h, p_w)$ et un stride de $(s_h, s_w)$.
1. Pourquoi pensez-vous que la mise en commun maximale et la mise en commun moyenne fonctionnent différemment ?
1. Avons-nous besoin d'une couche distincte de mise en commun minimale ? Peut-on la remplacer par une autre opération ?
1. Nous pourrions utiliser l'opération softmax pour la mise en commun. Pourquoi ne serait-elle pas si populaire ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
