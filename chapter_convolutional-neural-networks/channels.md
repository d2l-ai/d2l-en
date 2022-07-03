```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Canaux d'entrée et de sortie multiples
:label:`sec_channels` 

Bien que nous ayons décrit les canaux multiples
qui composent chaque image (par exemple, les images couleur ont les canaux RVB standard
pour indiquer la quantité de rouge, vert et bleu) et les couches de convolution pour les canaux multiples dans :numref:`subsec_why-conv-channels`,
jusqu'à présent, nous avons simplifié tous nos exemples numériques
en travaillant avec un seul canal d'entrée et un seul canal de sortie.
Cela nous a permis de considérer nos entrées, nos noyaux de convolution,
et nos sorties comme des tenseurs à deux dimensions.

Lorsque nous ajoutons des canaux au mélange,
nos entrées et nos représentations cachées
deviennent toutes deux des tenseurs tridimensionnels.
Par exemple, chaque image d'entrée RVB a une forme $3\times h\times w$.
Nous faisons référence à cet axe, de taille 3, comme étant la dimension *canal*. La notion de canaux
est aussi ancienne que les CNN eux-mêmes. Par exemple, LeNet5 :cite:`LeCun.Jackel.Bottou.ea.1995` les utilise. 
Dans cette section, nous allons examiner plus en profondeur
les noyaux de convolution avec plusieurs canaux d'entrée et de sortie.

## Canaux d'entrée multiples

Lorsque les données d'entrée contiennent plusieurs canaux,
nous devons construire un noyau de convolution
avec le même nombre de canaux d'entrée que les données d'entrée,
afin qu'il puisse effectuer une corrélation croisée avec les données d'entrée.
En supposant que le nombre de canaux pour les données d'entrée est $c_i$,
le nombre de canaux d'entrée du noyau de convolution doit également être $c_i$. Si la forme de la fenêtre de notre noyau de convolution est $k_h\times k_w$,
alors, lorsque $c_i=1$, nous pouvons considérer notre noyau de convolution
comme un simple tenseur bidimensionnel de forme $k_h\times k_w$.

Cependant, lorsque $c_i>1$, nous avons besoin d'un noyau
qui contient un tenseur de forme $k_h\times k_w$ pour *chaque* canal d'entrée. En concaténant ces tenseurs $c_i$ ensemble,
on obtient un noyau de convolution de forme $c_i\times k_h\times k_w$.
Puisque l'entrée et le noyau de convolution ont chacun $c_i$ canaux,
nous pouvons effectuer une opération de corrélation croisée
sur le tenseur bidimensionnel de l'entrée
et le tenseur bidimensionnel du noyau de convolution
pour chaque canal, en ajoutant les résultats $c_i$ ensemble
(en faisant la somme des canaux)
pour obtenir un tenseur bidimensionnel.
Il s'agit du résultat d'une intercorrélation bidimensionnelle
entre une entrée multicanaux et
un noyau de convolution multicanaux d'entrée.

:numref:`fig_conv_multi_in` fournit un exemple 
d'une intercorrélation bidimensionnelle avec deux canaux d'entrée.
Les parties ombrées sont le premier élément de sortie
ainsi que les éléments tensoriels d'entrée et de noyau utilisés pour le calcul de la sortie :
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$ .

![Cross-correlation computation with 2 input channels.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


Pour être sûr de bien comprendre ce qui se passe ici,
nous pouvons (**mettre en œuvre nous-mêmes les opérations de corrélation croisée avec plusieurs canaux d'entrée**).
Remarquez que tout ce que nous faisons est d'effectuer une opération de corrélation croisée
par canal, puis d'additionner les résultats.

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
%%tab mxnet, pytorch
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

Nous pouvons construire le tenseur d'entrée `X` et le tenseur de noyau `K`
correspondant aux valeurs dans :numref:`fig_conv_multi_in` 
pour (**valider la sortie**) de l'opération de corrélation croisée.

```{.python .input}
%%tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Canaux de sortie multiples
:label:`subsec_multi-output-channels` 

Quel que soit le nombre de canaux d'entrée,
jusqu'à présent, nous nous sommes toujours retrouvés avec un seul canal de sortie.
Cependant, comme nous l'avons vu dans :numref:`subsec_why-conv-channels`,
il s'avère essentiel de disposer de plusieurs canaux à chaque couche.
Dans les architectures de réseaux neuronaux les plus populaires,
nous augmentons en fait la dimension des canaux
à mesure que nous pénétrons plus profondément dans le réseau neuronal,
en réduisant généralement l'échantillonnage pour échanger la résolution spatiale
contre une plus grande *profondeur de canal*.
Intuitivement, on pourrait penser que chaque canal
répond à un ensemble différent de caractéristiques.
La réalité est un peu plus compliquée que cela. Une interprétation naïve suggérerait 
que les représentations sont apprises indépendamment par pixel ou par canal. 
Au contraire, les canaux sont optimisés pour être utiles conjointement.
Cela signifie qu'au lieu de faire correspondre un seul canal à un détecteur de bords, cela peut simplement signifier 
qu'une certaine direction dans l'espace des canaux correspond à la détection des bords.

Dénotez par $c_i$ et $c_o$ le nombre
de canaux d'entrée et de sortie, respectivement,
et laissez $k_h$ et $k_w$ être la hauteur et la largeur du noyau.
Pour obtenir une sortie avec plusieurs canaux,
nous pouvons créer un tenseur de noyau
de forme $c_i\times k_h\times k_w$
pour *chaque* canal de sortie.
Nous les concaténons sur la dimension du canal de sortie,
de sorte que la forme du noyau de convolution
est $c_o\times c_i\times k_h\times k_w$.
Dans les opérations de corrélation croisée,
le résultat sur chaque canal de sortie est calculé
à partir du noyau de convolution correspondant à ce canal de sortie
et prend en entrée tous les canaux du tenseur d'entrée.

Nous implémentons une fonction de corrélation croisée
pour [**calculer la sortie de plusieurs canaux**] comme indiqué ci-dessous.

```{.python .input}
%%tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Nous construisons un noyau de convolution trivial avec 3 canaux de sortie
en concaténant le tenseur du noyau pour `K` avec `K+1` et `K+2`.

```{.python .input}
%%tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Ci-dessous, nous effectuons des opérations de corrélation croisée
sur le tenseur d'entrée `X` avec le tenseur du noyau `K`.
La sortie contient maintenant 3 canaux.
Le résultat du premier canal est cohérent
avec le résultat du tenseur d'entrée précédent `X`
et le noyau de canal à entrées multiples,
à sortie unique.

```{.python .input}
%%tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ Couche convolutionnelle
:label:`subsec_1x1` 

Au départ, un [**$1 \times 1$ convolution**], i.e., $k_h = k_w = 1$,
does not seem to make much sense.
After all, a convolution correlates adjacent pixels.
A $1 \times 1$ convolution obviously does not.
Nonetheless, they are popular operations that are sometimes included
in the designs of complex deep networks :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`
Let's see in some detail what it actually does.

Because the minimum window is used,
the $1\times 1$ convolution loses the ability
of larger convolutional layers
to recognize patterns consisting of interactions
among adjacent elements in the height and width dimensions.
The only computation of the $1\times 1$ convolution occurs
on the channel dimension.

:numref:`fig_conv_1x1` shows the cross-correlation computation
using the $1\times 1$ convolution kernel
with 3 input channels and 2 output channels.
Note that the inputs and outputs have the same height and width.
Each element in the output is derived
from a linear combination of elements *at the same position*
in the input image.
You could think of the $1\times 1$ convolutional layer
as constituting a fully connected layer applied at every single pixel location
to transform the $c_i$ corresponding input values into $c_o$ output values.
Because this is still a convolutional layer,
the weights are tied across pixel location.
Thus the $1\times 1$ convolutional layer requires $c_o\times c_i$ weights
(plus the bias). Notez également que les couches convolutionnelles sont généralement suivies 
de non-linéarités. Cela permet de s'assurer que les convolutions $1 \times 1$ ne peuvent pas simplement être 
repliées dans d'autres convolutions 

![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Vérifions si cela fonctionne en pratique :
nous implémentons une convolution $1 \times 1$
en utilisant une couche entièrement connectée.
La seule chose que nous devons faire est d'apporter quelques ajustements
à la forme des données avant et après la multiplication de la matrice.

```{.python .input}
%%tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

Lors de l'exécution des convolutions $1\times 1$,
la fonction ci-dessus est équivalente à la fonction de corrélation croisée précédemment implémentée `corr2d_multi_in_out`.
Vérifions cela à l'aide d'un échantillon de données.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
%%tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Discussion

Les canaux nous permettent de combiner le meilleur des deux mondes : les MLP qui permettent des non-linéarités significatives et les convolutions qui permettent une analyse *localisée* des caractéristiques. En particulier, les canaux permettent au CNN de raisonner avec plusieurs caractéristiques, comme les détecteurs de bords et de formes en même temps. Ils offrent également un compromis pratique entre la réduction drastique des paramètres découlant de l'invariance de la traduction et de la localité, et le besoin de modèles expressifs et diversifiés en vision par ordinateur. 

Notez toutefois que cette flexibilité a un prix. Pour une image de taille $(h \times w)$, le coût du calcul d'une convolution $k \times k$ est de $O(h \cdot w \cdot k^2)$. Pour les canaux d'entrée et de sortie $c_i$ et $c_o$ respectivement, ce coût passe à $O(h \cdot w \cdot k^2 \cdot c_i \cdot c_o)$. Pour une image de $256 \times 256$ pixels avec un noyau $5 \times 5$ et des canaux d'entrée et de sortie $128$ respectivement, cela représente plus de 53 milliards d'opérations (nous comptons les multiplications et les additions séparément). Plus tard, nous rencontrerons des stratégies efficaces pour réduire ce coût, par exemple en exigeant que les opérations par canal soient en diagonale de bloc, ce qui conduit à des architectures telles que ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`. 

## Exercices

1. Supposons que nous ayons deux noyaux de convolution de taille $k_1$ et $k_2$, respectivement 
 (sans non-linéarité entre les deux).
   1. Prouvez que le résultat de l'opération peut être exprimé par une seule convolution.
   1. Quelle est la dimensionnalité de la convolution unique équivalente ?
   1. L'inverse est-il vrai, c'est-à-dire qu'il est toujours possible de décomposer une convolution en deux convolutions plus petites ?
1. Supposons une entrée de forme $c_i\times h\times w$ et un noyau de convolution de forme 
$c_o\times c_i\times k_h\times k_w$ , padding de $(p_h, p_w)$, et stride de $(s_h, s_w)$.
 1. Quel est le coût de calcul (multiplications et additions) pour la propagation vers l'avant ?
   1. Quelle est l'empreinte mémoire ?
   1. Quelle est l'empreinte mémoire pour le calcul en arrière ?
   1. Quel est le coût de calcul pour la rétropropagation ?
1. De quel facteur le nombre de calculs augmente-t-il si nous doublons le nombre de canaux d'entrée 
$c_i$ et le nombre de canaux de sortie $c_o$? Que se passe-t-il si on double le padding ?
1. Les variables `Y1` et `Y2` dans le dernier exemple de cette section sont-elles exactement les mêmes ? Pourquoi ?
1. Exprimer les convolutions comme une multiplication matricielle, même lorsque la fenêtre de convolution n'est pas $1 \times 1$? 
1. Votre tâche consiste à mettre en œuvre des convolutions rapides avec un noyau $k \times k$. L'un des algorithmes candidats 
consiste à balayer horizontalement la source, en lisant une bande large $k$ et en calculant la bande de sortie $1$-wide 
une valeur à la fois. L'autre solution consiste à lire une bande large $k + \Delta$ et à calculer une bande de sortie large $\Delta$ 
 . Pourquoi cette dernière solution est-elle préférable ? Y a-t-il une limite à la taille de la bande $\Delta$?
1. Supposons que nous ayons une matrice $c \times c$. 
    1. Combien de fois est-il plus rapide de multiplier avec une matrice diagonale en bloc si la matrice est décomposée en blocs $b$?
   1. Quel est l'inconvénient d'avoir des blocs $b$? Comment pourriez-vous y remédier, du moins en partie ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:
