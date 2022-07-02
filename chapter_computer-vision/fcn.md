# Réseaux entièrement convolutifs
:label:`sec_fcn` 

 Comme nous l'avons vu dans :numref:`sec_semantic_segmentation` ,
la segmentation sémantique
classe les images au niveau des pixels.
Un réseau entièrement convolutif (FCN)
utilise un réseau neuronal convolutif pour
transformer les pixels de l'image en classes de pixels :cite:`Long.Shelhamer.Darrell.2015` .
Contrairement aux CNN que nous avons rencontrés précédemment
pour la classification d'images 
ou la détection d'objets,
un réseau entièrement convolutif
transforme 
la hauteur et la largeur des cartes de caractéristiques intermédiaires
pour les ramener à celles de l'image d'entrée :
ceci est réalisé par
la couche convolutive transposée
introduite dans :numref:`sec_transposed_conv` .
Par conséquent,
la sortie de classification
et l'image d'entrée 
ont une correspondance biunivoque 
au niveau du pixel :
la dimension du canal à n'importe quel pixel de sortie 
contient les résultats de classification
pour le pixel d'entrée à la même position spatiale.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## Le modèle

Nous décrivons ici la conception de base du modèle de réseau entièrement convolutif. 
Comme indiqué dans :numref:`fig_fcn` ,
ce modèle utilise d'abord un CNN pour extraire les caractéristiques de l'image,
puis transforme le nombre de canaux en
le nombre de classes
via une couche convolutive $1\times 1$,
et enfin transforme la hauteur et la largeur de
les cartes de caractéristiques
en celles de
de l'image d'entrée via
la convolution transposée introduite dans :numref:`sec_transposed_conv` . 
Par conséquent,
la sortie du modèle a la même hauteur et la même largeur que l'image d'entrée,
où le canal de sortie contient les classes prédites
pour le pixel d'entrée à la même position spatiale.


![Fully convolutional network.](../img/fcn.svg)
:label:`fig_fcn`

Ci-dessous, nous [**utilisons un modèle ResNet-18 pré-entraîné sur le jeu de données ImageNet pour extraire les caractéristiques des images**]
et désignons l'instance du modèle par `pretrained_net`.
Les dernières couches de ce modèle
comprennent une couche de mise en commun de la moyenne globale
et une couche entièrement connectée :
elles ne sont pas nécessaires
dans le réseau entièrement convolutif.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

Ensuite, nous [**créons l'instance du réseau entièrement convolutif `net`**].
Elle copie toutes les couches pré-entraînées dans le ResNet-18
sauf la couche finale de mise en commun de la moyenne globale
et la couche entièrement connectée qui sont les plus proches
de la sortie.

```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

Étant donné une entrée dont la hauteur et la largeur sont respectivement de 320 et 480,
la propagation vers l'avant de `net`
 réduit la hauteur et la largeur de l'entrée à 1/32 de l'original, soit 10 et 15.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

Ensuite, nous [**utilisons une couche convolutive $1\times 1$ pour transformer le nombre de canaux de sortie en nombre de classes (21) du jeu de données Pascal VOC2012.**]
Enfin, nous devons (**augmenter la hauteur et la largeur des cartes de caractéristiques de 32 fois**) pour les ramener à la hauteur et à la largeur de l'image d'entrée. 
Rappelez-vous comment calculer 
la forme de sortie d'une couche convolutive dans :numref:`sec_padding` . 
Depuis $(320-64+16\times2+32)/32=10$ et $(480-64+16\times2+32)/32=15$, nous construisons une couche convolutive transposée avec un pas de $32$, 
en fixant
la hauteur et la largeur du noyau
à $64$, le remplissage à $16$.
En général,
nous pouvons voir que
pour stride $s$,
padding $s/2$ (en supposant que $s/2$ est un entier),
et la hauteur et la largeur du noyau $2s$, 
la convolution transposée augmentera
la hauteur et la largeur de l'entrée par $s$ fois.

```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**Initialisation des couches convolutionnelles transposées**]


 Nous savons déjà que
les couches convolutionnelles transposées peuvent augmenter
la hauteur et la largeur des cartes de caractéristiques
.
En traitement d'images, nous pouvons avoir besoin de mettre à l'échelle
une image, c'est-à-dire de *suréchantillonner*.
*L'interpolation bilinéaire*
est l'une des techniques de suréchantillonnage les plus utilisées.
Elle est également souvent utilisée pour initialiser des couches convolutives transposées.

Pour expliquer l'interpolation bilinéaire,
disons que 
étant donné une image d'entrée
nous voulons 
calculer chaque pixel 
de l'image de sortie suréchantillonnée.
Afin de calculer le pixel de l'image de sortie
à la coordonnée $(x, y)$, 
commence par mapper $(x, y)$ à la coordonnée $(x', y')$ sur l'image d'entrée, par exemple, en fonction du rapport entre la taille de l'entrée et la taille de la sortie. 
Notez que les coordonnées $x'$ et $y'$ sont des nombres réels. 
Ensuite, trouvez les quatre pixels les plus proches de la coordonnée
$(x', y')$ sur l'image d'entrée. 
Enfin, le pixel de l'image de sortie aux coordonnées $(x, y)$ est calculé sur la base de ces quatre pixels les plus proches
sur l'image d'entrée et de leur distance relative par rapport à $(x', y')$. 

Le suréchantillonnage de l'interpolation bilinéaire
peut être mis en œuvre par la couche convolutive transposée 
avec le noyau construit par la fonction `bilinear_kernel` suivante. 
En raison des limitations d'espace, nous ne fournissons que l'implémentation de la fonction `bilinear_kernel` ci-dessous
sans discussions sur la conception de son algorithme.

```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

Faisons [**l'expérience d'un suréchantillonnage de l'interpolation bilinéaire**] 
qui est mis en œuvre par une couche convolutive transposée. 
Nous construisons une couche convolutive transposée qui 
double la taille et le poids,
et initialisons son noyau avec la fonction `bilinear_kernel`.

```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

Lisez l'image `X` et affectez la sortie de suréchantillonnage à `Y`. Afin d'imprimer l'image, nous devons ajuster la position de la dimension du canal.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

Comme nous pouvons le constater, la couche convolutive transposée augmente à la fois la hauteur et la largeur de l'image d'un facteur deux.
À l'exception des différentes échelles en coordonnées,
l'image mise à l'échelle par interpolation bilinéaire et l'image originale imprimée dans :numref:`sec_bbox` se ressemblent.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

Dans un réseau entièrement convolutif, nous [**initialisons la couche convolutive transposée avec un suréchantillonnage par interpolation bilinéaire. Pour la couche convolutive $1\times 1$, nous utilisons l'initialisation de Xavier.**]

```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**Lire le jeu de données**]

Nous lisons
le jeu de données de segmentation sémantique
tel que présenté dans :numref:`sec_semantic_segmentation` . 
La forme de l'image de sortie du recadrage aléatoire est
spécifiée comme $320\times 480$: la hauteur et la largeur sont toutes deux divisibles par $32$.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**Training**]


 Nous pouvons maintenant entraîner notre réseau entièrement convolutif construit
. 
La fonction de perte et le calcul de la précision ici
ne sont pas essentiellement différents de ceux de la classification d'images des chapitres précédents. 
Comme nous utilisons le canal de sortie de la couche convolutive transposée
pour
prédire la classe de chaque pixel,
la dimension du canal est spécifiée dans le calcul de la perte.
En outre, la précision est calculée
sur la base de l'exactitude
de la classe prédite pour tous les pixels.

```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**Prédiction**]


 Lors de la prédiction, nous devons normaliser l'image d'entrée
dans chaque canal et transformer l'image dans le format d'entrée quadridimensionnel requis par le CNN.

```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

Pour [**visualiser la classe prédite**] de chaque pixel, nous faisons correspondre la classe prédite à la couleur de son étiquette dans l'ensemble de données.

```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

Les images de l'ensemble de données de test varient en taille et en forme.
Comme le modèle utilise une couche convolutive transposée avec un stride de 32,
lorsque la hauteur ou la largeur d'une image d'entrée est indivisible par 32,
la hauteur ou la largeur de sortie de la couche convolutive transposée
s'écartera de la forme de l'image d'entrée.
Afin de résoudre ce problème,
nous pouvons recadrer plusieurs zones rectangulaires dont la hauteur et la largeur sont des multiples entiers de 32 dans l'image,
et effectuer la propagation vers l'avant
sur les pixels de ces zones séparément.
Notez que
l'union de ces zones rectangulaires doit couvrir complètement l'image d'entrée.
Lorsqu'un pixel est couvert par plusieurs zones rectangulaires,
la moyenne des sorties de convolution transposées
dans des zones séparées pour ce même pixel
peut être entrée dans
l'opération softmax
pour prédire la classe.


Pour simplifier, nous ne lisons que quelques grandes images de test,
et recadrons une zone $320\times480$ pour la prédiction en partant du coin supérieur gauche d'une image.
Pour ces images de test, nous imprimons les zones rognées
, les résultats de prédiction
,
et la vérité du sol ligne par ligne.

```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Résumé

* Le réseau entièrement convolutif utilise d'abord un CNN pour extraire les caractéristiques de l'image, puis transforme le nombre de canaux en nombre de classes via une couche convolutive $1\times 1$, et enfin transforme la hauteur et la largeur des cartes de caractéristiques en celles de l'image d'entrée via la convolution transposée.
* Dans un réseau entièrement convolutif, nous pouvons utiliser un suréchantillonnage d'interpolation bilinéaire pour initialiser la couche convolutive transposée.


## Exercices

1. Si nous utilisons l'initialisation de Xavier pour la couche convolutionnelle transposée dans l'expérience, comment le résultat change-t-il ?
1. Pouvez-vous améliorer davantage la précision du modèle en réglant les hyperparamètres ?
1. Prédisez les classes de tous les pixels des images de test.
1. L'article original sur les réseaux entièrement convolutifs utilise également les sorties de certaines couches CNN intermédiaires :cite:`Long.Shelhamer.Darrell.2015` . Essayez d'implémenter cette idée.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:
