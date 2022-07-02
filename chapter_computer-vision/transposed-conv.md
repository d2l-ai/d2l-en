# Transposed Convolution
:label:`sec_transposed_conv`

The CNN layers we have seen so far,
such as convolutional layers (:numref:`sec_conv_layer`) and pooling layers (:numref:`sec_pooling`),
typically reduce (downsample) the spatial dimensions (height and width) of the input,
or keep them unchanged.
In semantic segmentation
that classifies at pixel-level,
it will be convenient if
the spatial dimensions of the
input and output are the same.
For example,
the channel dimension at one output pixel 
can hold the classification results
for the input pixel at the same spatial position.


To achieve this, especially after 
the spatial dimensions are reduced by CNN layers,
we can use another type
of CNN layers
that can increase (upsample) the spatial dimensions
of intermediate feature maps.
In this section,
we will introduce 
*transposed convolution*, which is also called *fractionally-strided convolution* :cite:`Dumoulin.Visin.2016`, 
for reversing downsampling operations
by the convolution.

```{.python .input}
#@tab mxnet
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## Basic Operation

Ignoring channels for now,
let's begin with
the basic transposed convolution operation
with stride of 1 and no padding.
Suppose that
we are given a 
$n_h \times n_w$ input tensor
and a $k_h \times k_w$ kernel.
Sliding the kernel window with stride of 1
for $n_w$ times in each row
and $n_h$ times in each column
yields 
a total of $n_h n_w$ intermediate results.
Each intermediate result is
a $(n_h + k_h - 1) \times (n_w + k_w - 1)$
tensor that are initialized as zeros.
To compute each intermediate tensor,
each element in the input tensor
is multiplied by the kernel
so that the resulting $k_h \times k_w$ tensor
replaces a portion in
each intermediate tensor.
Note that
the position of the replaced portion in each
intermediate tensor corresponds to the position of the element
in the input tensor used for the computation.
In the end, all the intermediate results
are summed over to produce the output.

As an example,
:numref:`fig_trans_conv` illustrates
how transposed convolution with a $2\times 2$ kernel is computed for a $2\times 2$ input tensor.


![Transposed convolution with a $2\times 2$ kernel. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv.svg)
:label:`fig_trans_conv`


We can (**implement this basic transposed convolution operation**) `trans_conv` for a input matrix `X` and a kernel matrix `K`.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

In contrast to the regular convolution (in :numref:`sec_conv_layer`) that *reduces* input elements
via the kernel,
the transposed convolution
*broadcasts* input elements 
via the kernel, thereby
producing an output
that is larger than the input.
We can construct the input tensor `X` and the kernel tensor `K` from :numref:`fig_trans_conv` to [**validate the output of the above implementation**] of the basic two-dimensional transposed convolution operation.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

Alternatively,
when the input `X` and kernel `K` are both
four-dimensional tensors,
we can [**use high-level APIs to obtain the same results**].

```{.python .input}
#@tab mxnet
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**Padding, Strides, and Multiple Channels**]

Different from in the regular convolution
where padding is applied to input,
it is applied to output
in the transposed convolution.
For example,
when specifying the padding number
on either side of the height and width 
as 1,
the first and last rows and columns
will be removed from the transposed convolution output.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

Dans la convolution transposée, les strides
sont spécifiés pour les résultats intermédiaires (donc la sortie), et non pour l'entrée.
En utilisant les mêmes tenseurs d'entrée et de noyau
de :numref:`fig_trans_conv` ,
changer le stride de 1 à 2
augmente à la fois la hauteur et le poids
des tenseurs intermédiaires, donc le tenseur de sortie
dans :numref:`fig_trans_conv_stride2` .


![Transposed convolution with a $2\times 2$ kernel with stride of 2. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`



L'extrait de code suivant peut valider la sortie de la convolution transposée pour un stride de 2 dans :numref:`fig_trans_conv_stride2` .

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

Pour les canaux d'entrée et de sortie multiples,
la convolution transposée
fonctionne de la même manière que la convolution ordinaire.
Supposons que
l'entrée ait $c_i$ canaux,
et que la convolution transposée
assigne un tenseur de noyau $k_h\times k_w$
 à chaque canal d'entrée.
Lorsque plusieurs canaux de sortie 
sont spécifiés,
nous aurons un noyau $c_i\times k_h\times k_w$ pour chaque canal de sortie.


Comme dans tous les cas, si nous alimentons $\mathsf{X}$ dans une couche convolutive $f$ pour sortir $\mathsf{Y}=f(\mathsf{X})$ et créer une couche convolutive transposée $g$ avec les mêmes hyperparamètres que $f$ sauf 
pour le nombre de canaux de sortie 
étant le nombre de canaux dans $\mathsf{X}$,
puis $g(Y)$ auront la même forme que $\mathsf{X}$.
Ceci peut être illustré dans l'exemple suivant.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**Connexion à la transposition de matrice**]
:label:`subsec-connection-to-mat-transposition` 

 La convolution transposée porte le nom de
la transposition de matrice.
Pour expliquer,
voyons d'abord
comment implémenter les convolutions
en utilisant les multiplications matricielles.
Dans l'exemple ci-dessous, nous définissons une entrée $3\times 3$ `X` et un noyau de convolution $2\times 2$ `K` , puis nous utilisons la fonction `corr2d` pour calculer la sortie de la convolution `Y`.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

Ensuite, nous réécrivons le noyau de convolution `K` comme
une matrice de poids clairsemée `W`
 contenant beaucoup de zéros. 
La forme de la matrice de poids est ($4$, $9$),
où les éléments non nuls proviennent de
le noyau de convolution `K`.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

Concaténer l'entrée `X` ligne par ligne pour obtenir un vecteur de longueur 9. Ensuite, la multiplication matricielle de `W` et de `X` vectorisé donne un vecteur de longueur 4.
Après l'avoir remodelé, nous pouvons obtenir le même résultat `Y`
 à partir de l'opération de convolution originale ci-dessus :
nous avons juste implémenté des convolutions en utilisant des multiplications matricielles.

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

De même, nous pouvons implémenter des convolutions transposées en utilisant des multiplications de matrices
.
Dans l'exemple suivant,
nous prenons la sortie $2 \times 2$ `Y` de la convolution régulière
ci-dessus
comme entrée de la convolution transposée.
Pour mettre en œuvre cette opération en multipliant des matrices,
il nous suffit de transposer la matrice de poids `W`
 avec la nouvelle forme $(9, 4)$.

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

Considérons l'implémentation de la convolution
en multipliant des matrices.
Étant donné un vecteur d'entrée $\mathbf{x}$
 et une matrice de poids $\mathbf{W}$,
la fonction de propagation vers l'avant de la convolution
peut être implémentée
en multipliant son entrée avec la matrice de poids
et en produisant un vecteur 
$\mathbf{y}=\mathbf{W}\mathbf{x}$ .
Puisque la rétropropagation
suit la règle de la chaîne
et $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$,
la fonction de rétropropagation de la convolution
peut être mise en œuvre
en multipliant son entrée avec la matrice de poids transposée 
 $\mathbf{W}^\top$ .
Par conséquent, 
la couche convolutionnelle transposée
peut simplement échanger la fonction de propagation directe
et la fonction de rétropropagation de la couche convolutionnelle :
ses fonctions de propagation directe 
et de rétropropagation
multiplient leur vecteur d'entrée avec 
$\mathbf{W}^\top$ et $\mathbf{W}$, respectivement.


## Résumé

* Contrairement à la convolution régulière qui réduit les éléments d'entrée via le noyau, la convolution transposée diffuse les éléments d'entrée via le noyau, produisant ainsi une sortie qui est plus grande que l'entrée.
* Si nous alimentons $\mathsf{X}$ dans une couche convolutionnelle $f$ pour produire $\mathsf{Y}=f(\mathsf{X})$ et créons une couche convolutionnelle transposée $g$ avec les mêmes hyperparamètres que $f$, à l'exception du nombre de canaux de sortie qui est le nombre de canaux dans $\mathsf{X}$, alors $g(Y)$ aura la même forme que $\mathsf{X}$.
* Nous pouvons mettre en œuvre des convolutions en utilisant des multiplications de matrices. La couche convolutionnelle transposée peut simplement échanger la fonction de propagation vers l'avant et la fonction de rétropropagation de la couche convolutionnelle.


## Exercices

1. Dans :numref:`subsec-connection-to-mat-transposition` , l'entrée de convolution `X` et la sortie de convolution transposée `Z` ont la même forme. Ont-elles la même valeur ? Pourquoi ?
1. Est-il efficace d'utiliser des multiplications matricielles pour implémenter des convolutions ? Pourquoi ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:
