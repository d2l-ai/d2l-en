```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Padding and Stride
:label:`sec_padding`

Recall the example of a convolution in :numref:`fig_correlation`. 
The input had both a height and width of 3
and the convolution kernel had both a height and width of 2,
yielding an output representation with dimension $2\times2$.
Assuming that the input shape is $n_h\times n_w$
and the convolution kernel shape is $k_h\times k_w$,
the output shape will be $(n_h-k_h+1) \times (n_w-k_w+1)$: 
we can only shift the convolution kernel so far until it runs out
of pixels to apply the convolution to. 

In the following we will explore a number of techniques, 
including padding and strided convolutions,
that offer more control over the size of the output. 
As motivation, note that since kernels generally
have width and height greater than $1$,
after applying many successive convolutions,
we tend to wind up with outputs that are
considerably smaller than our input.
If we start with a $240 \times 240$ pixel image,
$10$ layers of $5 \times 5$ convolutions
reduce the image to $200 \times 200$ pixels,
slicing off $30 \%$ of the image and with it
obliterating any interesting information
on the boundaries of the original image.
*Padding* is the most popular tool for handling this issue.
In other cases, we may want to reduce the dimensionality drastically,
e.g., if we find the original input resolution to be unwieldy.
*Strided convolutions* are a popular technique that can help in these instances.

## Padding

As described above, one tricky issue when applying convolutional layers
is that we tend to lose pixels on the perimeter of our image. Consider :numref:`img_conv_reuse` that depicts the pixel utilization as a function of the convolution kernel size and the position within the image. The pixels in the corners are hardly used at all. 

![Pixel utilization for convolutions of size $1 \times 1$, $2 \times 2$, and $3 \times 3$ respectively.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

Since we typically use small kernels,
for any given convolution,
we might only lose a few pixels,
but this can add up as we apply
many successive convolutional layers.
One straightforward solution to this problem
is to add extra pixels of filler around the boundary of our input image,
thus increasing the effective size of the image.
Typically, we set the values of the extra pixels to zero.
In :numref:`img_conv_pad`, we pad a $3 \times 3$ input,
increasing its size to $5 \times 5$.
The corresponding output then increases to a $4 \times 4$ matrix.
The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+0\times1+0\times2+0\times3=0$.

![Two-dimensional cross-correlation with padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

In general, if we add a total of $p_h$ rows of padding
(roughly half on top and half on bottom)
and a total of $p_w$ columns of padding
(roughly half on the left and half on the right),
the output shape will be

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$

This means that the height and width of the output
will increase by $p_h$ and $p_w$, respectively.

In many cases, we will want to set $p_h=k_h-1$ and $p_w=k_w-1$
to give the input and output the same height and width.
This will make it easier to predict the output shape of each layer
when constructing the network.
Assuming that $k_h$ is odd here,
we will pad $p_h/2$ rows on both sides of the height.
If $k_h$ is even, one possibility is to
pad $\lceil p_h/2\rceil$ rows on the top of the input
and $\lfloor p_h/2\rfloor$ rows on the bottom.
We will pad both sides of the width in the same way.

CNNs commonly use convolution kernels
with odd height and width values, such as 1, 3, 5, or 7.
Choosing odd kernel sizes has the benefit
that we can preserve the dimensionality
while padding with the same number of rows on top and bottom,
and the same number of columns on left and right.

Moreover, this practice of using odd kernels
and padding to precisely preserve dimensionality
offers a clerical benefit.
For any two-dimensional tensor `X`,
when the kernel's size is odd
and the number of padding rows and columns
on all sides are the same,
producing an output with the same height and width as the input,
we know that the output `Y[i, j]` is calculated
by cross-correlation of the input and convolution kernel
with the window centered on `X[i, j]`.

In the following example, we create a two-dimensional convolutional layer
with a height and width of 3
and (**apply 1 pixel of padding on all sides.**)
Given an input with a height and width of 8,
we find that the height and width of the output is also 8.

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

When the height and width of the convolution kernel are different,
we can make the output and input have the same height and width
by [**setting different padding numbers for height and width.**]

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
, nous commençons par la fenêtre de convolution
au coin supérieur gauche du tenseur d'entrée,
, puis nous la faisons glisser sur tous les emplacements vers le bas et vers la droite.
Dans les exemples précédents, nous avons choisi par défaut de faire glisser un élément à la fois.
Cependant, parfois, soit pour des raisons d'efficacité de calcul
, soit parce que nous souhaitons réduire l'échantillonnage,
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
, la forme de sortie sera $(n_h/s_h) \times (n_w/s_w)$.

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

Une convention similaire s'applique aux enjambements. Lorsque le stride horizontal $s_h$ et le stride vertical $s_w$ correspondent, nous parlons simplement de stride $s$. Le stride peut réduire la résolution de la sortie, par exemple en réduisant la hauteur et la largeur de la sortie à seulement $1/n$ de la hauteur et de la largeur de l'entrée pour $n > 1$. Par défaut, le padding est égal à 0 et le stride à 1. 

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
