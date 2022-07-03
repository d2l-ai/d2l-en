```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Padding et Stride
:label:`sec_padding`

Rappelez-vous l'exemple d'une convolution dans :numref:`fig_correlation`. 
L'entrée avait à la fois une hauteur et une largeur de 3
et le noyau de convolution avait une hauteur et une largeur de 2,
ce qui donne une représentation de sortie de dimension $2\times2$.
Si l'on suppose que la forme de l'entrée est $n_h\times n_w$.
et que la forme du noyau de convolution est $k_h\times k_w$,
la forme de la sortie sera $(n_h-k_h+1) \times (n_w-k_w+1)$ : 
on ne peut que décaler le noyau de convolution jusqu'à ce qu'il manque
de pixels auxquels appliquer la convolution. 

Dans ce qui suit, nous allons explorer un certain nombre de techniques, 
y compris le remplissage et les convolutions stridentes,
qui offrent un meilleur contrôle sur la taille de la sortie. 
En guise de motivation, notez que puisque les noyaux ont généralement
ont une largeur et une hauteur supérieures à $1$,
après avoir appliqué de nombreuses convolutions successives,
nous avons tendance à nous retrouver avec des sorties qui sont
considérablement plus petites que notre entrée.
Si nous commençons avec une image de 240 \times 240$ pixels,
10$ couches de 5$ \times 5$ convolutions
réduisent l'image à 200 pixels,
coupant 30 % de l'image et, par la même occasion.
oblitérant toute information intéressante
sur les limites de l'image originale.
*Le calage est l'outil le plus populaire pour traiter ce problème.
Dans d'autres cas, nous pouvons vouloir réduire radicalement la dimensionnalité,
par exemple, si nous trouvons que la résolution de l'entrée originale est difficile à manier.
Les *convolutions stridentes* sont une technique populaire qui peut aider dans ces cas.

## Padding

Comme décrit ci-dessus, un problème délicat lors de l'application de couches convolutives
est que nous avons tendance à perdre des pixels sur le périmètre de notre image. Considérez :numref:`img_conv_reuse` qui décrit l'utilisation des pixels en fonction de la taille du noyau de convolution et de la position dans l'image. Les pixels dans les coins sont à peine utilisés. 

![Utilisation des pixels pour des convolutions de taille 1 \times 1$, 2 \times 2$, et 3 \times 3$ respectivement.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

Puisque nous utilisons typiquement de petits noyaux,
pour toute convolution donnée,
on peut ne perdre que quelques pixels,
mais cela peut s'additionner lorsque nous appliquons
de nombreuses couches convolutionnelles successives.
Une solution directe à ce problème
est d'ajouter des pixels supplémentaires de remplissage autour de la limite de notre image d'entrée,
augmentant ainsi la taille effective de l'image.
Typiquement, nous fixons les valeurs des pixels supplémentaires à zéro.
Dans :numref:`img_conv_pad`, nous remplissons une entrée de 3$ \times 3$,
ce qui porte sa taille à 5 $.
La sortie correspondante passe alors à une matrice de 4$ \times 4$.
Les parties ombrées sont le premier élément de sortie ainsi que les éléments tensoriels d'entrée et de noyau utilisés pour le calcul de la sortie : $0\times0+0\times1+0\times2+0\times3=0$.

![Corrélation croisée bidimensionnelle avec padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

En général, si nous ajoutons un total de $p_h$ rangées de padding
(à peu près la moitié en haut et la moitié en bas)
et un total de $p_w$ colonnes de remplissage
(environ la moitié à gauche et la moitié à droite),
la forme de sortie sera

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$

Cela signifie que la hauteur et la largeur de la sortie
augmenteront respectivement de $p_h$ et $p_w$.

Dans de nombreux cas, nous souhaitons définir $p_h=k_h-1$ et $p_w=k_w-1$.
pour que l'entrée et la sortie aient la même hauteur et la même largeur.
Il sera ainsi plus facile de prédire la forme de la sortie de chaque couche lors de la construction du réseau.
lors de la construction du réseau.
En supposant que $k_h$ est impair ici,
nous ajouterons $p_h/2$ lignes de part et d'autre de la hauteur.
Si $k_h$ est pair, une possibilité est de
de remplir $\lceil p_h/2\rceil$ lignes sur le haut de l'entrée
et $\lfloor p_h/2\rfloor$ lignes sur le bas.
Nous allons remplir les deux côtés de la largeur de la même manière.

Les CNN utilisent généralement des noyaux de convolution
avec des valeurs de hauteur et de largeur impaires, telles que 1, 3, 5 ou 7.
Le choix de tailles de noyau impaires présente l'avantage
que nous pouvons préserver la dimensionnalité
tout en ayant le même nombre de lignes en haut et en bas,
et le même nombre de colonnes à gauche et à droite.

De plus, cette pratique consistant à utiliser des noyaux impairs
et le remplissage pour préserver précisément la dimensionnalité
offre un avantage administratif.
Pour tout tenseur bidimensionnel `X`,
lorsque la taille du noyau est impaire
et que le nombre de lignes et de colonnes de remplissage
sur tous les côtés sont les mêmes,
produisant une sortie avec la même hauteur et largeur que l'entrée,
on sait que la sortie `Y[i, j]` est calculée
par corrélation croisée de l'entrée et du noyau de convolution
avec la fenêtre centrée sur `X[i, j]`.

Dans l'exemple suivant, nous créons une couche convolutionnelle bidimensionnelle
avec une hauteur et une largeur de 3
et (**appliquer 1 pixel de remplissage sur tous les côtés.**)
Étant donné une entrée avec une hauteur et une largeur de 8,
nous constatons que la hauteur et la largeur de la sortie sont également de 8.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# We define a helper function to calculate convolutions. It initializes 
# the convolutional layer weights and performs corresponding dimensionality 
# elevations and reductions on the input and output.
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn

# We define a helper function to calculate convolutions. It initializes 
# the convolutional layer weights and performs corresponding dimensionality 
# elevations and reductions on the input and output.
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])
# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

# We define a helper function to calculate convolutions. It initializes 
# the convolutional layer weights and performs corresponding dimensionality 
# elevations and reductions on the input and output.
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return tf.reshape(Y, Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

Lorsque la hauteur et la largeur du noyau de convolution sont différentes,
nous pouvons faire en sorte que la sortie et l'entrée aient la même hauteur et largeur
en [**définissant des nombres de remplissage différents pour la hauteur et la largeur.**].

```{.python .input}
%%tab mxnet
# We use a convolution kernel with height 5 and width 3. The padding on 
# either side of the height and width are 2 and 1, respectively.
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# We use a convolution kernel with height 5 and width 3. The padding on 
# either side of the height and width are 2 and 1, respectively.
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# We use a convolution kernel with height 5 and width 3. The padding on 
# either side of the height and width are 2 and 1, respectively.
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

## Stride

Lors du calcul de la corrélation croisée,
nous commençons par la fenêtre de convolution
au coin supérieur gauche du tenseur d'entrée,
puis nous la faisons glisser sur tous les emplacements vers le bas et vers la droite.
Dans les exemples précédents, nous avons choisi par défaut de faire glisser un élément à la fois.
Cependant, parfois, soit pour des raisons d'efficacité de calcul,
soit parce que nous souhaitons réduire l'échantillonnage,
nous déplaçons notre fenêtre de plus d'un élément à la fois,
en sautant les emplacements intermédiaires. Ceci est particulièrement utile si le noyau de convolution 
est grand, car il capture une grande partie de l'image sous-jacente.

Nous faisons référence au nombre de lignes et de colonnes traversées par diapositive comme *stride*.
Jusqu'à présent, nous avons utilisé des strides de 1, à la fois pour la hauteur et la largeur.
Parfois, nous pouvons vouloir utiliser un stride plus grand.
:numref:`img_conv_stride` montre une opération de corrélation croisée bidimensionnelle
avec un stride de 3 verticalement et 2 horizontalement.
Les parties ombragées sont les éléments de sortie ainsi que les éléments du tenseur d'entrée et du noyau utilisés pour le calcul de la sortie : $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
Nous pouvons voir que lorsque le deuxième élément de la première colonne est généré,
la fenêtre de convolution glisse de trois rangées vers le bas.
La fenêtre de convolution glisse de deux colonnes vers la droite
lorsque le deuxième élément de la première ligne est généré.
Lorsque la fenêtre de convolution continue à glisser de deux colonnes vers la droite sur l'entrée,
il n'y a pas de sortie car l'élément d'entrée ne peut pas remplir la fenêtre
(à moins que nous n'ajoutions une autre colonne de remplissage).

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

En général, lorsque le pas pour la hauteur est $s_h$
et le pas pour la largeur est $s_w$, la forme de sortie est

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$ 

Si nous définissons $p_h=k_h-1$ et $p_w=k_w-1$,
alors la forme de sortie peut être simplifiée en
$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$ .
En allant un peu plus loin, si la hauteur et la largeur d'entrée
sont divisibles par les strides de la hauteur et de la largeur,
la forme de sortie sera $(n_h/s_h) \times (n_w/s_w)$.

Ci-dessous, nous [**fixons les pas sur la hauteur et la largeur à 2**],
divisant ainsi par deux la hauteur et la largeur d'entrée.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

Examinons [**un exemple légèrement plus compliqué**].

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## Résumé et discussion

Le remplissage peut augmenter la hauteur et la largeur de la sortie. Il est souvent utilisé pour donner à la sortie la même hauteur et la même largeur que l'entrée afin d'éviter un rétrécissement indésirable de la sortie. De plus, il permet de s'assurer que tous les pixels sont utilisés avec la même fréquence. En général, nous choisissons un remplissage symétrique des deux côtés de la hauteur et de la largeur de l'entrée. Dans ce cas, nous faisons référence au remplissage de $(p_h, p_w)$. Le plus souvent, nous définissons $p_h = p_w$, auquel cas nous indiquons simplement que nous choisissons le remplissage $p$. 

Une convention similaire s'applique aux stride. Lorsque le stride horizontal $s_h$ et le stride vertical $s_w$ correspondent, nous parlons simplement de stride $s$. Le stride peut réduire la résolution de la sortie, par exemple en réduisant la hauteur et la largeur de la sortie à seulement $1/n$ de la hauteur et de la largeur de l'entrée pour $n > 1$. Par défaut, le padding est égal à 0 et le stride à 1. 

Jusqu'à présent, tous les padding dont nous avons parlé ont simplement prolongé les images par des zéros. Cela présente un avantage significatif en termes de calcul, car c'est trivial à réaliser. De plus, les opérateurs peuvent être conçus pour tirer profit de ce remplissage de manière implicite sans avoir besoin d'allouer de la mémoire supplémentaire. En même temps, cela permet aux CNN de coder des informations de position implicites dans une image, simplement en apprenant où se trouve l'"espace blanc". Il existe de nombreuses alternatives à l'espacement zéro. :cite:`Alsallakh.Kokhlikyan.Miglani.ea.2020` fournit une vue d'ensemble des alternatives (bien qu'il n'y ait pas d'argument clair en faveur de l'utilisation d'espacements non nuls, sauf si des artefacts se produisent). 


## Exercices

1. Étant donné le dernier exemple de code de cette section avec la taille du noyau $(3, 5)$, le padding $(0, 1)$, et le stride $(3, 4)$, 
calculez la forme de sortie pour vérifier si elle est cohérente avec le résultat expérimental.
1. Pour les signaux audio, à quoi correspond un stride de 2 ?
1. Implémentez le padding miroir, c'est-à-dire le padding où les valeurs de bordure sont simplement mises en miroir pour étendre les tenseurs. 
1. Quels sont les avantages informatiques d'un stride supérieur à 1 ?
1. Quels sont les avantages statistiques d'un stride supérieur à 1 ?
1. Comment implémenteriez-vous un stride de $\frac{1}{2}$? A quoi correspond-il ? Quand cela serait-il utile ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:
