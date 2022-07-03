```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Réseau en réseau (NiN)
:label:`sec_nin` 

LeNet, AlexNet et VGG partagent tous un modèle de conception commun :
extraire des caractéristiques exploitant la structure *spatiale*
via une séquence de convolutions et de couches de mise en commun
et post-traiter les représentations via des couches entièrement connectées.
Les améliorations apportées à LeNet par AlexNet et VGG résident principalement
dans la manière dont ces réseaux ultérieurs élargissent et approfondissent ces deux modules.

Cette conception pose deux défis majeurs. 
Premièrement, les couches entièrement connectées à l'extrémité
de l'architecture consomment un nombre énorme de paramètres. Par exemple, même un modèle simple
tel que le VGG-11 nécessite une matrice monstrueuse $25088 \times 4096$, occupant près de
400 Mo de RAM. Il s'agit d'un obstacle important à la rapidité des calculs, en particulier sur
les appareils mobiles et embarqués. Deuxièmement, il est également impossible d'ajouter des couches entièrement connectées
plus tôt dans le réseau pour augmenter le degré de non-linéarité : cela détruirait la structure spatiale
et nécessiterait potentiellement encore plus de mémoire.

Les blocs *réseau dans réseau* (*NiN*) de :cite:`Lin.Chen.Yan.2013` offrent une alternative,
capable de résoudre les deux problèmes en une seule stratégie simple.
Ils ont été proposés sur la base d'une intuition très simple : (i) utiliser $1 \times 1$ convolutions pour ajouter
non-linéarités locales à travers les activations de canaux et (ii) utiliser le pooling moyen global pour intégrer
à travers tous les emplacements dans la dernière couche de représentation. Notez que la mise en commun de la moyenne globale ne serait pas
efficace, si ce n'était pour les non-linéarités ajoutées. Voyons cela en détail.


## (**NiN Blocks**)

Rappelons-nous :numref:`subsec_1x1`. Nous y avons vu que les entrées et les sorties des couches convolutionnelles
sont constituées de tenseurs quadridimensionnels dont les axes
correspondent à l'exemple, au canal, à la hauteur et à la largeur.
Rappelons également que les entrées et sorties des couches entièrement connectées
sont généralement des tenseurs bidimensionnels correspondant à l'exemple et à la caractéristique.
L'idée derrière NiN est d'appliquer une couche entièrement connectée
à chaque emplacement de pixel (pour chaque hauteur et largeur).
La convolution $1 \times 1$ qui en résulte peut être considérée comme
une couche entièrement connectée agissant indépendamment sur chaque emplacement de pixel.

:numref:`fig_nin` illustre les principales différences structurelles
entre VGG et NiN, et leurs blocs.
Notez à la fois la différence dans les blocs NiN (la convolution initiale est suivie de $1 \times 1$ convolutions, alors que VGG conserve $3 \times 3$ convolutions) et à la fin où nous n'avons plus besoin d'une couche géante entièrement connectée.

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

## [**Modèle NiN**]

NiN utilise les mêmes tailles de convolution initiale qu'AlexNet (il a été proposé peu après).
Les tailles des noyaux sont respectivement $11\times 11$, $5\times 5$, et $3\times 3$,
et le nombre de canaux de sortie correspond à celui d'AlexNet. Chaque bloc NiN est suivi d'une couche de max-pooling
avec un stride de 2 et une forme de fenêtre de $3\times 3$.

La deuxième différence significative entre NiN et AlexNet et VGG
est que NiN évite complètement les couches entièrement connectées.
Au lieu de cela, NiN utilise un bloc NiN avec un nombre de canaux de sortie égal au nombre de classes d'étiquettes, suivi d'une couche de mise en commun de la moyenne *globale*,
produisant un vecteur de logits.
Cette conception réduit considérablement le nombre de paramètres de modèle requis, mais au prix d'une augmentation potentielle du temps de formation.

```{.python .input}
%%tab all
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

Nous créons un exemple de données pour voir [**la forme de la sortie de chaque bloc**].

```{.python .input}
%%tab mxnet, pytorch
model = NiN()
X = d2l.randn(1, 1, 224, 224)
for layer in model.net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
%%tab tensorflow
model = NiN()
X = d2l.normal((1, 224, 224, 1))
for layer in model.net.layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**Training**]

Comme précédemment, nous utilisons Fashion-MNIST pour entraîner le modèle.
L'entraînement de NiN est similaire à celui d'AlexNet et de VGG.

```{.python .input}
%%tab mxnet, pytorch
model = NiN(lr=0.05)
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
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## Résumé

NiN a beaucoup moins de paramètres qu'AlexNet et VGG. Cela vient du fait qu'il n'a pas besoin de couches entièrement connectées géantes et de moins de convolutions avec des noyaux larges. Au lieu de cela, il utilise des convolutions locales $1 \times 1$ et une mise en commun de la moyenne globale. Ces choix de conception ont influencé de nombreuses conceptions ultérieures de CNN.

## Exercices

1. Pourquoi y a-t-il deux $1\times 1$ couches convolutionnelles par bloc NiN ? Que se passe-t-il si on en ajoute une ? Que se passe-t-il si l'on réduit à une seule ?
1. Que se passe-t-il si vous remplacez le pooling moyen global par une couche entièrement connectée (vitesse, précision, nombre de paramètres) ?
1. Calculez l'utilisation des ressources pour NiN.
   1. Quel est le nombre de paramètres ?
   1. Quelle est la quantité de calcul ?
   1. Quelle est la quantité de mémoire nécessaire pendant l'entraînement ?
   1. Quelle est la quantité de mémoire nécessaire pendant la prédiction ?
1. Quels sont les problèmes éventuels liés à la réduction de la représentation $384 \times 5 \times 5$ en une représentation $10 \times 5 \times 5$ en une seule étape ?
1. Utilisez les décisions de conception structurelle du VGG qui ont conduit au VGG-11, VGG-16 et VGG-19 pour concevoir une famille de réseaux de type NiN.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:
