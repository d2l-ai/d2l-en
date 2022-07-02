```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Conception d'architectures de réseaux de convolution
:label:`sec_cnn-design` 

 Les années 2010 ont été marquées par le passage
de l'ingénierie des *caractéristiques* à l'ingénierie des *réseaux*
en vision par ordinateur.
Depuis qu'AlexNet (:numref:`sec_alexnet` )
a battu les modèles de vision par ordinateur conventionnels sur ImageNet,
construire des réseaux très profonds
en empilant les mêmes blocs,
et surtout $3 \times 3$ convolutions,
a été popularisé par les réseaux VGG (:numref:`sec_vgg` ).
Le réseau du réseau (:numref:`sec_nin` )
ajoute des non-linéarités locales via $1 \times 1$ convolutions
et utilise la mise en commun de la moyenne globale
pour agréger les informations
sur tous les sites.
GoogLeNet (:numref:`sec_googlenet` )
est un réseau multi-branches qui
combine les avantages du réseau VGG

 et du réseau dans le réseau,
où son bloc d'initialisation
adopte la stratégie des transformations parallèles concaténées
.
ResNets (:numref:`sec_resnet` )
empile des blocs résiduels,
qui sont des sous-réseaux à deux branches
utilisant le mappage d'identité dans une branche.
Les DenseNets (:numref:`sec_densenet` )
généralisent les architectures résiduelles.
Parmi les autres architectures notables
, citons
MobileNets, qui utilise l'apprentissage en réseau pour obtenir une précision élevée dans
des contextes où les ressources sont limitées :cite:`Howard.Sandler.Chu.ea.2019` ,
les Squeeze-and-Excitation Networks (SENets) qui
permettent un transfert d'informations efficace entre les canaux
:cite:`Hu.Shen.Sun.2018` ,
et EfficientNets :cite:`tan2019efficientnet` 
 qui mettent les réseaux à l'échelle via la recherche d'une architecture neuronale.

Plus précisément, la *recherche d'architecture neuronale* (NAS) :cite:`zoph2016neural,liu2018darts` 
 est le processus d'automatisation des architectures de réseaux neuronaux.
Étant donné un espace de recherche fixe,
NAS utilise une stratégie de recherche
pour sélectionner automatiquement
une architecture dans l'espace de recherche
sur la base de l'estimation des performances retournées.
Le résultat de NAS
est une instance de réseau unique.

Au lieu de se concentrer sur la conception de telles instances individuelles,
une approche alternative
consiste à *concevoir des espaces de conception de réseaux*
qui caractérisent des populations de réseaux :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` .
Cette méthode
combine la force de la conception manuelle et des NAS.
Par le biais de procédures semi-automatiques (comme dans les SNA),
conception d'espaces de conception de réseaux
explore l'aspect structure de la conception de réseaux
à partir de l'espace de conception initial *AnyNet*.
Elle procède ensuite à la découverte de principes de conception (comme dans la conception manuelle)
qui conduisent à des réseaux simples et réguliers : *RegNets*.
Avant de faire la lumière sur ces principes de conception,
commençons par
l'espace de conception initial.

## L'espace de conception AnyNet

L'espace de conception initial est appelé *AnyNet*,
un espace de conception relativement non contraint,
où nous pouvons nous concentrer sur
l'exploration de la structure du réseau
en supposant des blocs standard et fixes tels que ResNeXt (:numref:`subsec_resnext` ).
Plus précisément,
la structure du réseau
comprend
éléments
tels que le nombre de blocs
et le nombre de canaux de sortie
dans chaque étape,
et le nombre de groupes (largeur de groupe) et le ratio de goulots d'étranglement
au sein de
chaque bloc ResNeXt.



![The AnyNet design space. Besides the number of groups and bottleneck ratio within each block, design choices include depth $d_i$ and the number of output channels $w_i$ for any stage $i$.](../img/anynet.svg)
:label:`fig_anynet`

L'espace de conception AnyNet
est illustré dans :numref:`fig_anynet` .
Ce réseau
commence par une *tige*,
suivie d'un *corps* avec $n$ étapes de transformation,
et une *tête* finale.
Plus concrètement,
la tige du réseau
est une convolution $3 \times 3$ avec stride 2
qui divise par deux la hauteur et la largeur d'une image d'entrée.
La tête du réseau
est une mise en commun de la moyenne globale suivie
par une couche entièrement connectée pour prédire
la classe de sortie.
Notez que
la tige et la tête du réseau
sont maintenues fixes et simples,
afin que la conception se concentre sur
le corps du réseau qui est central
aux performances.
Plus précisément,
le corps du réseau
se compose de $n$ étapes de transformation
($n$ est donné),
où l'étape $i$
 se compose de $d_i$ blocs ResNeXt
avec des canaux de sortie $w_i$,
et progressivement
réduit de moitié la hauteur et la largeur via le premier bloc
(en fixant `use_1x1conv=True, strides=2` dans `d2l.ResNeXtBlock` dans :numref:`subsec_resnext` ).
En outre,
désigne
le taux de goulot d'étranglement et
le nombre de groupes (largeur de groupe) 
dans
chaque bloc ResNeXt pour l'étape $i$
 comme $b_i$ et $g_i$, respectivement.
Globalement,
malgré la structure simple du réseau,
en variant $b_i$, $g_i$, $w_i$, et $d_i$
 résulte en
un grand nombre de
réseaux possibles dans l'espace de conception d'AnyNet.


Pour mettre en œuvre AnyNet,
nous définissons d'abord sa tige de réseau.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

Chaque étape est constituée de `depth` blocs ResNeXt,
où `num_channels` spécifie la largeur du bloc.
Notez que le premier bloc divise par deux la hauteur et la largeur des images d'entrée.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

En assemblant la tige, le corps et la tête du réseau,
nous complétons l'implémentation d'AnyNet.

```{.python .input}
%%tab all
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

## Contrainte des espaces de conception avec des distributions d'erreurs inférieures

Pour toute étape $i$ d'AnyNet,
les choix de conception sont 
le ratio de goulot d'étranglement $b_i$ 
 et le nombre de groupes $g_i$
 dans chaque bloc,
la largeur du bloc $w_i$,
et la profondeur $d_i$.
Le processus de conception des espaces de conception du réseau
commence
à partir d'une structure de réseau
relativement sans contrainte, caractérisée
par ($b_i$, $g_i$, $w_i$, $d_i$)
dans l'espace de conception initial d'AnyNet.
Ensuite, ce processus
échantillonne progressivement les modèles
de l'espace de conception d'entrée
pour évaluer la distribution des erreurs :cite:`radosavovic2019network` 
 comme indicateur de qualité
pour produire un espace de conception
plus contraint avec des modèles plus simples qui peuvent avoir
une meilleure qualité. 

Détaillons
cet indicateur de qualité pour les espaces de conception.
Étant donné $n$ modèles échantillonnés à partir d'un certain espace de conception,
la *fonction de distribution empirique des erreurs* $F(e)$
 mesure la fraction de modèles
avec des erreurs $e_i$ inférieures à $e$:

$$F(e) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i < e).$$ 

 
 En partant de l'espace de conception AnyNet initial non contraint ($\text{AnyNetX}_A$ dans :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` ),
partageant le rapport de réseau de bouteilles $b_i = b$ pour toutes les étapes $i$ résulte en un espace de conception plus contraint $\text{AnyNetX}_B$.
L'échantillonnage et l'entraînement des modèles $n=500$ à partir de $\text{AnyNetX}_A$ et $\text{AnyNetX}_B$ chacun,
à gauche de :numref:`fig_regnet-paper-fig5` 
 montre que les deux espaces de conception ont une qualité similaire.
Puisque plus c'est simple, mieux c'est,
nous continuons à chercher à partir de $\text{AnyNetX}_B$
 en partageant en plus le nombre de groupes $g_i = g$.
Cela conduit à un espace de conception encore plus simplifié
$\text{AnyNetX}_C$ avec pratiquement aucun changement
dans les distributions d'erreurs (à droite de :numref:`fig_regnet-paper-fig5` ).

![Comparing error empirical distribution functions of design spaces. The legends show the min error and mean error. Sharing bottleneck ratio (from $\text{AnyNetX}_A$ to  $\text{AnyNetX}_B$) et le partage du nombre de groupes (de $\text{AnyNetX}_B$ à $\text{AnyNetX}_C$) simplifient l'espace de conception sans pratiquement aucun changement dans les distributions d'erreurs (figure tirée de :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` ).](../img/regnet-paper-fig5.png)
:width:`600px` 
 :label:`fig_regnet-paper-fig5` 

 L'étude des bons et des mauvais modèles de $\text{AnyNetX}_C$ suggère qu'il peut être utile d'augmenter la largeur entre les étapes :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` .
De manière empirique, la simplification de
$\text{AnyNetX}_C$ en $\text{AnyNetX}_D$
 avec $w_{i} \leq w_{i+1}$
 améliore la qualité des espaces de conception (à gauche de :numref:`fig_regnet-paper-fig7` ).
De même,
en ajoutant d'autres contraintes de $d_{i} \leq d_{i+1}$
 pour augmenter la profondeur du réseau entre les étapes
donne un $\text{AnyNetX}_E$
 encore meilleur (à droite de :numref:`fig_regnet-paper-fig7` ).

![Comparing error empirical distribution functions of design spaces. The legends show the min error and mean error. Increasing network width across stages (from $\text{AnyNetX}_C$ to  $\text{AnyNetX}_D$) et l'augmentation de la profondeur du réseau entre les étapes (de $\text{AnyNetX}_D$ à $\text{AnyNetX}_E$) simplifie l'espace de conception avec des distributions d'erreurs améliorées (figure tirée de :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` ).](../img/regnet-paper-fig7.png)
:width:`600px` 
 :label:`fig_regnet-paper-fig7` 

 

 ## RegNet

L'espace de conception $\text{AnyNetX}_E$ résultant
se compose de réseaux simples
suivant des principes de conception faciles à interpréter :

* Partager le ratio du réseau de bouteilles $b_i = b$ pour toutes les étapes $i$;
* Partager le nombre de groupes $g_i = g$ pour toutes les étapes $i$;
* Augmenter la largeur du réseau à travers les étapes : $w_{i} \leq w_{i+1}$;
* Augmenter la profondeur du réseau à travers les étapes : $d_{i} \leq d_{i+1}$.

Suivant ces principes de conception, :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` a proposé des contraintes linéaires quantifiées à
$w_i$ et $d_i$ croissantes,
conduisant à
RegNetX utilisant les blocs ResNeXt
et RegNetY qui utilise en plus les opérateurs de SENets :cite:`Hu.Shen.Sun.2018` .
A titre d'exemple,
nous implémentons une variante RegNetX à 32 couches
caractérisée par

* $b_i = 1;$
 * $g_i = 16;$
 * $w_1 = 32, w_2=80;$
 * $d_1 = 4, d_2=6.$

```{.python .input}
%%tab all
class RegNet32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

Nous pouvons constater que chaque étape RegNet réduit progressivement la résolution et augmente les canaux de sortie.

```{.python .input}
%%tab mxnet, pytorch
RegNet32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNet32().layer_summary((1, 96, 96, 1))
```

## Training

L'entraînement du RegNet à 32 couches sur le jeu de données Fashion-MNIST se fait comme précédemment.

```{.python .input}
%%tab mxnet, pytorch
model = RegNet32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNet32(lr=0.01)
    trainer.fit(model, data)
```

## Discussion

Avec des propriétés souhaitables comme la localité et l'invariance de traduction (:numref:`sec_why-conv` )
pour la vision,
CNNs ont été les architectures dominantes dans ce domaine.
Récemment, les transformateurs
(qui seront traités dans :numref:`sec_transformer` ) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` 
 et les MLP :cite:`tolstikhin2021mlp` 
 ont également suscité des recherches au-delà de
les architectures CNN bien établies pour la vision.
Plus précisément,
bien que dépourvus des biais inductifs inhérents aux CNN mentionnés plus haut
,
les transformateurs de vision
ont atteint des performances de pointe
dans la classification d'images à grande échelle au début des années 2020,
montrant que
*l'évolutivité l'emporte sur les biais inductifs*
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` .

En d'autres termes,
il est souvent possible d'entraîner de grands transformateurs
pour surpasser de grands CNN sur de grands ensembles de données.
Cependant,
la complexité quadratique
de l'auto-attention (qui sera traitée dans :numref:`sec_self-attention-and-positional-encoding` )
rend l'architecture des transformateurs
moins adaptée aux images à haute résolution.
Pour résoudre ce problème, les transformateurs de Swin

 introduisent des fenêtres décalées pour
atteindre des performances de pointe
dans une gamme plus large de tâches de vision au-delà de la classification d'images :cite:`liu2021swin` .
Inspiré par
par le comportement supérieur de mise à l'échelle des transformateurs
avec auto-attention multi-têtes (à couvrir dans :numref:`sec_multihead-attention` ),
le processus d'amélioration progressive
d'une architecture ResNet standard
vers la conception d'un transformateur de vision
conduit à une famille de modèles CNN appelés ConvNeXts
qui rivalisent favorablement avec les transformateurs Swin :cite:`liu2022convnet` .
Nous renvoyons les lecteurs intéressés
aux discussions sur la conception des CNN
dans le document ConvNeXt :cite:`liu2022convnet` .



## Exercices

1. Augmentez le nombre d'étages à 4. Pouvez-vous concevoir un RegNet plus profond et plus performant ?
1. Dé-ResNeXt-ifiez les RegNets en remplaçant le bloc ResNeXt par le bloc ResNet. Quelles sont les performances de votre nouveau modèle ?
1. Implémentez plusieurs instances d'une famille "VioNet" en *violant* les principes de conception des RegNet. Quelles sont leurs performances ? Lequel de ($d_i$, $w_i$, $g_i$, $b_i$) est le facteur le plus important ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8738)
:end_tab:
