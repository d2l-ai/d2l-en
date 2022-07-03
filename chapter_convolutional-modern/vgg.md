```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Réseaux utilisant des blocs (VGG)
:label:`sec_vgg` 

 Si AlexNet a offert des preuves empiriques que les CNN profonds
peuvent obtenir de bons résultats, il n'a pas fourni de modèle général
pour guider les chercheurs ultérieurs dans la conception de nouveaux réseaux.
Dans les sections suivantes, nous allons présenter plusieurs concepts heuristiques
couramment utilisés pour concevoir des réseaux profonds.

Les progrès réalisés dans ce domaine reflètent ceux de l'intégration à très grande échelle (VLSI) 
dans la conception des puces
où les ingénieurs sont passés du placement des transistors
aux éléments logiques, puis aux blocs logiques :cite:`Mead.1980` .
De même, la conception d'architectures de réseaux neuronaux
est devenue progressivement plus abstraite,
les chercheurs passant de la réflexion en termes de neurones individuels
à des couches entières,
et maintenant à des blocs, c'est-à-dire des motifs répétitifs de couches. 

L'idée d'utiliser des blocs est apparue pour la première fois sur le site
[Visual Geometry Group](http://www.robots.ox.ac.uk/~)vgg/) (VGG)
de l'université d'Oxford,
dans leur réseau éponyme *VGG* :cite:`Simonyan.Zisserman.2014` .
Il est facile d'implémenter ces structures répétées dans le code
avec n'importe quel cadre moderne d'apprentissage profond en utilisant des boucles et des sous-routines.

## (**VGG Blocks**)
:label:`subsec_vgg-blocks` 

 Le bloc de construction de base des CNN
est une séquence des éléments suivants :
(i) une couche convolutive
avec un remplissage pour maintenir la résolution,
(ii) une non-linéarité telle qu'un ReLU,
(iii) une couche de mise en commun telle que
max-pooling pour réduire la résolution. L'un des problèmes de cette approche, 
, est que la résolution spatiale diminue assez rapidement. En particulier, 
cela impose au réseau une limite stricte de couches convolutionnelles $\log_2 d$ avant que toutes les dimensions 
($d$) ne soient utilisées. Par exemple, dans le cas d'ImageNet, il serait impossible d'avoir 
plus de 8 couches convolutives de cette manière. 

L'idée principale de Simonyan et Zisserman était d'utiliser des convolutions *multiples* entre le sous-échantillonnage de
via le max-pooling sous la forme d'un bloc. Ils se sont principalement intéressés à la question de savoir si les réseaux profonds ou larges 
donnent de meilleurs résultats. Par exemple, l'application successive de deux convolutions $3 \times 3$
 touche les mêmes pixels qu'une seule convolution $5 \times 5$. En même temps, cette dernière utilise environ 
autant de paramètres ($25 \cdot c^2$) que trois convolutions $3 \times 3$ ($3 \cdot 9 \cdot c^2$). 
Dans une analyse assez détaillée, ils ont montré que les réseaux profonds et étroits étaient nettement plus performants que leurs homologues peu profonds. L'apprentissage profond s'est ainsi lancé dans une quête de réseaux toujours plus profonds, avec plus de 100 couches pour les applications typiques.
Alors que
empilage $3 \times 3$ convolutions
a été un standard d'or dans les réseaux profonds ultérieurs,
implémentations de telles opérations
ont également été efficaces sur les GPU :cite:`lavin2016fast` . 



Revenons au VGG : un bloc VGG consiste en une *séquence* de convolutions avec $3\times3$ noyaux avec un padding de 1 
(conservant la hauteur et la largeur) suivi d'une couche de max-pooling $2 \times 2$ avec un stride de 2
(divisant par deux la hauteur et la largeur après chaque bloc).
Dans le code ci-dessous, nous définissons une fonction appelée `vgg_block`
 pour implémenter un bloc VGG.

:begin_tab:`mxnet`
La fonction ci-dessous prend deux arguments,
correspondant au nombre de couches convolutionnelles `num_convs`
 et au nombre de canaux de sortie `num_channels`.
:end_tab:

:begin_tab:`pytorch`
La fonction ci-dessous prend trois arguments correspondant au nombre
de couches convolutives `num_convs`, au nombre de canaux d'entrée `in_channels`
 et au nombre de canaux de sortie `out_channels`.
:end_tab:

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## [**VGG Network**]
:label:`subsec_vgg-network`

Like AlexNet and LeNet, 
the VGG Network can be partitioned into two parts:
the first consisting mostly of convolutional and pooling layers
and the second consisting of fully connected layers that are identical to those in AlexNet. 
The key difference is 
that the convolutional layers are grouped in nonlinear transformations that 
leave the dimensonality unchanged, followed by a resolution-reduction step, as 
depicted in :numref:`fig_vgg`. 

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

The convolutional part of the network connects several VGG blocks from :numref:`fig_vgg` (also defined in the `vgg_block` function)
en succession. Ce regroupement de convolutions est un modèle qui 
est resté pratiquement inchangé au cours de la dernière décennie, bien que le choix spécifique des opérations 
ait subi des modifications considérables. 
La variable `conv_arch` est constituée d'une liste de tuples (un par bloc),
où chacun contient deux valeurs : le nombre de couches convolutionnelles
et le nombre de canaux de sortie,
qui sont précisément les arguments nécessaires pour appeler
la fonction `vgg_block`. En tant que tel, VGG définit une *famille* de réseaux plutôt que de se contenter de 
une manifestation spécifique. Pour construire un réseau spécifique, il suffit d'itérer sur `arch` pour composer les blocs.

```{.python .input}
%%tab all
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            in_channels = 1
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

Le réseau VGG original avait 5 blocs convolutifs,
parmi lesquels les deux premiers ont une couche convolutive chacun
et les trois derniers contiennent deux couches convolutives chacun.
Le premier bloc comporte 64 canaux de sortie
et chaque bloc suivant double le nombre de canaux de sortie,
jusqu'à ce que ce nombre atteigne 512.
Comme ce réseau utilise 8 couches convolutionnelles
et 3 couches entièrement connectées, il est souvent appelé VGG-11.

```{.python .input}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

Comme vous pouvez le voir, nous divisons par deux la hauteur et la largeur à chaque bloc,
pour finalement atteindre une hauteur et une largeur de 7
avant d'aplatir les représentations
pour le traitement par la partie entièrement connectée du réseau.

## Training

[**Comme le VGG-11 est plus lourd en termes de calcul que l'AlexNet
nous construisons un réseau avec un plus petit nombre de canaux.**]
C'est plus que suffisant pour l'entraînement sur Fashion-MNIST.
Le processus de [**formation du modèle**] est similaire à celui d'AlexNet dans :numref:`sec_alexnet` .

```{.python .input}
%%tab mxnet, pytorch
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## Résumé

On pourrait dire que le VGG est le premier réseau de neurones convolutifs véritablement moderne. Bien qu'AlexNet ait introduit de nombreux éléments qui rendent l'apprentissage profond efficace à grande échelle, c'est le VGG qui a sans doute introduit des propriétés clés telles que les blocs de convolutions multiples et une préférence pour les réseaux profonds et étroits. C'est également le premier réseau qui est en fait une famille entière de modèles paramétrés de manière similaire, ce qui permet au praticien de trouver un compromis entre complexité et vitesse. C'est aussi l'endroit où les cadres modernes d'apprentissage profond brillent. Il n'est plus nécessaire de générer des fichiers de configuration XML pour spécifier un réseau, mais plutôt de configurer ces réseaux à l'aide d'un simple code Python. 

Très récemment, ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021` a démontré qu'il est possible d'obtenir des performances compétitives en utilisant une architecture beaucoup moins profonde grâce à un grand nombre de calculs parallèles. Il s'agit là d'un développement passionnant qui, espérons-le, influencera la conception des architectures à l'avenir. Pour le reste du chapitre, cependant, nous suivrons la voie du progrès scientifique au cours de la dernière décennie. 

## Exercices


 1. Comparé à AlexNet, VGG est beaucoup plus lent en termes de calcul, et il a également besoin de plus de mémoire GPU. 
    1. Comparez le nombre de paramètres nécessaires pour AlexNet et VGG.
   1. Comparez le nombre d'opérations en virgule flottante utilisées dans les couches convolutionnelles et dans les couches entièrement connectées. 
    1. Comment pourriez-vous réduire le coût de calcul créé par les couches entièrement connectées ?
1. Lorsque l'on affiche les dimensions associées aux différentes couches du réseau, on ne voit que les informations 
 associées à 8 blocs (plus quelques transformations auxiliaires), alors que le réseau comporte 11 couches. Où sont passées les 3 couches restantes, 
?
1. Le suréchantillonnage de la résolution dans Fashion-MNIST par un facteur de $8 \times 8$, de 28 à 224 dimensions, est très coûteux 
. Essayez plutôt de modifier l'architecture du réseau et la conversion de la résolution, par exemple à 56 ou à 84 dimensions 
 pour son entrée. Pouvez-vous le faire sans réduire la précision du réseau ?
1. Utilisez le tableau 1 du document VGG :cite:`Simonyan.Zisserman.2014` pour construire d'autres modèles courants, 
 tels que le VGG-16 ou le VGG-19.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
