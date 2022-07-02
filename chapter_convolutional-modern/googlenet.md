```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Multi-Branch Networks  (GoogLeNet)
:label:`sec_googlenet`

In 2014, *GoogLeNet*
won the ImageNet Challenge :cite:`Szegedy.Liu.Jia.ea.2015`, using a structure
that combined the strengths of NiN :cite:`Lin.Chen.Yan.2013`, repeated blocks :cite:`Simonyan.Zisserman.2014`,
and a cocktail of convolution kernels. It is arguably also the first network that exhibits a clear distinction among the stem, body, and head in a CNN. This design pattern has persisted ever since in the design of deep networks: the *stem* is given by the first 2-3 convolutions that operate on the image. They extract low-level features from the underlying images. This is followed by a *body* of convolutional blocks. Finally, the *head* maps the features obtained so far to the required classification, segmentation, detection, or tracking problem at hand.

The key contribution in GoogLeNet was the design of the network body. It solved the problem of selecting
convolution kernels in an ingenious way. While other works tried to identify which convolution, ranging from $1 \times 1$ to $11 \times 11$ would be best, it simply *concatenated* multi-branch convolutions.
In what follows we introduce a slightly simplified version of GoogLeNet. The simplifications are due to the fact that  tricks to stabilize training, in particular intermediate loss functions, are no longer needed due to the availability of improved training algorithms.

## (**Inception Blocks**)

The basic convolutional block in GoogLeNet is called an *Inception block*,
stemming from the meme "we need to go deeper" of the movie *Inception*.

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

As depicted in :numref:`fig_inception`,
the inception block consists of four parallel branches.
The first three branches use convolutional layers
with window sizes of $1\times 1$, $3\times 3$, and $5\times 5$
to extract information from different spatial sizes.
The middle two branches also add a $1\times 1$ convolution of the input
to reduce the number of channels, reducing the model's complexity.
The fourth branch uses a $3\times 3$ max-pooling layer,
followed by a $1\times 1$ convolutional layer
to change the number of channels.
The four branches all use appropriate padding to give the input and output the same height and width.
Finally, the outputs along each branch are concatenated
along the channel dimension and comprise the block's output.
The commonly-tuned hyperparameters of the Inception block
are the number of output channels per layer.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Branch 2
        self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Branch 3
        self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Branch 4
        self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return np.concatenate((b1, b2, b3, b4), axis=1)
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])
```

To gain some intuition for why this network works so well,
consider the combination of the filters.
They explore the image in a variety of filter sizes.
This means that details at different extents
can be recognized efficiently by filters of different sizes.
At the same time, we can allocate different amounts of parameters
for different filters.


## [**GoogLeNet Model**]

As shown in :numref:`fig_inception_full`, GoogLeNet uses a stack of a total of 9 inception blocks, arranged into 3 groups with max-pooling in between,
and global average pooling in its head to generate its estimates.
Max-pooling between inception blocks reduces the dimensionality.
At its stem, the first module is similar to AlexNet and LeNet.

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

We can now implement GoogLeNet piece by piece. Let's begin with the stem.
The first module uses a 64-channel $7\times 7$ convolutional layer.

```{.python .input}
%%tab all
class GoogleNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                              activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

Le deuxième module utilise deux couches convolutives :
d'abord, une couche convolutive de 64 canaux $1\times 1$,
suivie d'une couche convolutive $3\times 3$ qui triple le nombre de canaux. Cela correspond à la deuxième branche du bloc Inception et conclut la conception du corps. À ce stade, nous avons 192 canaux.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b2(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Le troisième module connecte en série deux blocs Inception complets.
Le nombre de canaux de sortie du premier bloc d'Inception est
$64+128+32+32=256$ . Cela revient à 
un rapport du nombre de canaux de sortie
entre les quatre branches de $2:4:1:1$. Pour y parvenir, nous réduisons d'abord les dimensions d'entrée
par $\frac{1}{2}$ et par $\frac{1}{12}$ dans la deuxième et la troisième branche respectivement
pour arriver aux canaux $96 = 192/2$ et $16 = 192/12$ respectivement.

Le nombre de canaux de sortie du deuxième bloc d'Inception
est augmenté à $128+192+96+64=480$, ce qui donne un rapport de $128:192:96:64 = 4:6:3:2$. Comme précédemment,
nous devons réduire le nombre de dimensions intermédiaires dans la deuxième et la troisième branche. Une échelle
de $\frac{1}{2}$ et $\frac{1}{8}$ respectivement suffit, ce qui donne les canaux $128$ et $32$
 respectivement. Ceci est capturé par les arguments des constructeurs de blocs `Inception` suivants.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b3(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                             Inception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Le quatrième module est plus compliqué.
Il connecte cinq blocs d'Inception en série,
et ils ont respectivement les canaux de sortie $192+208+48+64=512$, $160+224+64+64=512$,
$128+256+64+64=512$ , $112+288+64+64=528$,
et $256+320+128+128=832$.
Le nombre de canaux assignés à ces branches est similaire
à celui du troisième module :
la deuxième branche avec la couche convolutive $3\times 3$
 sort le plus grand nombre de canaux,
suivi de la première branche avec seulement la couche convolutive $1\times 1$,
la troisième branche avec la couche convolutive $5\times 5$,
et la quatrième branche avec la couche max-pooling $3\times 3$.
Les deuxième et troisième branches réduiront d'abord
le nombre de canaux en fonction du ratio.
Ces ratios sont légèrement différents dans les différents blocs d'Inception.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b4(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Le cinquième module possède deux blocs d'Inception avec $256+320+128+128=832$
 et $384+384+128+128=1024$ canaux de sortie.
Le nombre de canaux attribués à chaque branche
est le même que celui des troisième et quatrième modules,
mais diffère par des valeurs spécifiques.
Il convient de noter que le cinquième bloc est suivi de la couche de sortie.
Ce bloc utilise la couche de mise en commun de la moyenne globale
pour changer la hauteur et la largeur de chaque canal à 1, tout comme dans NiN.
Enfin, nous transformons la sortie en un tableau bidimensionnel
suivi d'une couche entièrement connectée
dont le nombre de sorties est le nombre de classes d'étiquettes.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b5(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D())
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.Sequential([
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            tf.keras.layers.Dense(num_classes)])
```

Le modèle GoogLeNet est complexe sur le plan informatique. Notez le grand nombre de
hyperparamètres relativement arbitraires en termes de nombre de canaux choisis.
Ce travail a été réalisé avant que les scientifiques ne commencent à utiliser des outils automatiques pour
optimiser la conception des réseaux.

Pour l'instant, la seule modification que nous allons effectuer est de
[**réduire la hauteur et la largeur d'entrée de 224 à 96
pour avoir un temps d'apprentissage raisonnable sur Fashion-MNIST.**]
Cela simplifie le calcul. Examinons les changements de forme de la sortie entre les différents modules (
).

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
model = GoogleNet().layer_summary((1, 96, 96, 1))
```

## [**Training**]

Comme précédemment, nous entraînons notre modèle en utilisant le jeu de données Fashion-MNIST.
 Nous le transformons en une résolution de $96 \times 96$ pixels
 avant d'invoquer la procédure d'entraînement.

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = GoogleNet(lr=0.01)
    trainer.fit(model, data)
```

## Discussion

L'une des principales caractéristiques de GoogLeNet est qu'il est *moins cher* à calculer que ses prédécesseurs
tout en offrant une meilleure précision. Ceci marque le début d'une conception de réseau beaucoup plus délibérée
qui échange le coût d'évaluation d'un réseau avec une réduction des erreurs. Il marque également le début de l'expérimentation au niveau du bloc avec les hyperparamètres de conception du réseau, même si elle était entièrement manuelle à l'époque. Cela est dû en grande partie au fait que les cadres d'apprentissage profond de 2015 manquaient encore d'une grande partie de la flexibilité de conception
que nous considérons maintenant comme acquise. De plus, l'optimisation complète du réseau est coûteuse et, à l'époque, l'entraînement sur ImageNet s'avérait encore
difficile sur le plan informatique.

Au cours des sections suivantes, nous rencontrerons un certain nombre de choix de conception (par exemple, la normalisation des lots, les connexions résiduelles et le regroupement des canaux) qui nous permettent d'améliorer considérablement les réseaux. Pour l'instant, vous pouvez être fier d'avoir implémenté ce qui est sans doute le premier CNN vraiment moderne.

## Exercices

1. GoogLeNet a connu un tel succès qu'il est passé par un certain nombre d'itérations. Il existe plusieurs itérations
 de GoogLeNet qui ont progressivement amélioré la vitesse et la précision. Essayez d'implémenter et d'exécuter certaines d'entre elles.
  Voici quelques-unes d'entre elles :
  1. Ajout d'une couche de normalisation par lot :cite:`Ioffe.Szegedy.2015` , tel que décrit
 plus loin dans :numref:`sec_batch_norm` .
 1. Effectuez des ajustements au bloc Inception (largeur, choix et ordre des convolutions), comme décrit dans
 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` .
  1. Utilisez le lissage des étiquettes pour la régularisation du modèle, comme décrit dans
 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` .
  1. Effectuez d'autres ajustements au bloc Inception en ajoutant une connexion résiduelle
 :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017` , comme décrit ultérieurement dans
 :numref:`sec_resnet` .
1. Quelle est la taille minimale d'une image pour que GoogLeNet fonctionne ?
1. Pouvez-vous concevoir une variante de GoogLeNet qui fonctionne sur la résolution native de Fashion-MNIST, soit $28 \times 28$ pixels ? Comment devriez-vous modifier la tige, le corps et la tête du réseau, le cas échéant ?
1. Comparez la taille des paramètres des modèles AlexNet, VGG, NiN et GoogLeNet. Comment les deux dernières architectures de réseau
 réduisent-elles de manière significative la taille des paramètres du modèle ?
1. Comparez la quantité de calculs nécessaires dans GoogLeNet et AlexNet. Comment cela affecte-t-il la conception d'une puce accélératrice, par exemple, en termes de taille de mémoire, de quantité de calcul et d'avantages des opérations spécialisées ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
