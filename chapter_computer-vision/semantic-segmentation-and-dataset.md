# Segmentation sémantique et ensemble de données
:label:`sec_semantic_segmentation` 

Lorsque l'on aborde les tâches de détection d'objets
dans :numref:`sec_bbox` --:numref:`sec_rcnn`,
les boîtes de délimitation rectangulaires
sont utilisées pour étiqueter et prédire les objets dans les images.
Cette section aborde le problème de la *segmentation sémantique*,
qui se concentre sur la manière de diviser une image en régions appartenant à différentes classes sémantiques.
Contrairement à la détection d'objets,
la segmentation sémantique
reconnaît et comprend
ce qui se trouve dans les images au niveau du pixel :
son étiquetage et sa prédiction des régions sémantiques sont
au niveau du pixel.
:numref:`fig_segmentation` montre les étiquettes
du chien, du chat et de l'arrière-plan de l'image dans la segmentation sémantique.
Par rapport à la détection d'objets,
les frontières au niveau du pixel étiquetées
dans la segmentation sémantique sont évidemment plus fines.


![Labels of the dog, cat, and background of the image in semantic segmentation.](../img/segmentation.svg)
:label:`fig_segmentation`


## Segmentation d'images et segmentation d'instances

Il existe également deux tâches importantes
dans le domaine de la vision par ordinateur qui sont similaires à la segmentation sémantique,
à savoir la segmentation d'images et la segmentation d'instances.
Nous allons brièvement
les distinguer de la segmentation sémantique comme suit.

* *La segmentation d'image* divise une image en plusieurs régions constitutives. Les méthodes pour ce type de problème utilisent généralement la corrélation entre les pixels de l'image. Elles n'ont pas besoin d'informations sur les étiquettes des pixels de l'image pendant l'apprentissage et ne peuvent pas garantir que les régions segmentées auront la sémantique que nous espérons obtenir pendant la prédiction. En prenant l'image dans :numref:`fig_segmentation` comme entrée, la segmentation d'image peut diviser le chien en deux régions : l'une couvre la bouche et les yeux qui sont principalement noirs, et l'autre couvre le reste du corps qui est principalement jaune.
* *La segmentation d'instance* est également appelée *détection et segmentation simultanées*. Elle étudie comment reconnaître les régions au niveau du pixel de chaque instance d'objet dans une image. Contrairement à la segmentation sémantique, la segmentation d'instance doit distinguer non seulement la sémantique, mais aussi les différentes instances d'objets. Par exemple, s'il y a deux chiens dans l'image, la segmentation d'instance doit distinguer auquel des deux chiens appartient un pixel.



## Le jeu de données de segmentation sémantique Pascal VOC2012

[**L'un des jeux de données de segmentation sémantique les plus importants
est [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**]
Dans ce qui suit,
nous allons examiner ce jeu de données.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

Le fichier tar de l'ensemble de données est d'environ 2 Go.
Le téléchargement du fichier peut donc prendre un certain temps.
L'ensemble de données extrait est situé à l'adresse `../data/VOCdevkit/VOC2012`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

Après avoir entré le chemin `../data/VOCdevkit/VOC2012`,
nous pouvons voir les différents composants de l'ensemble de données.
Le chemin `ImageSets/Segmentation` contient des fichiers texte
qui spécifient les échantillons d'entraînement et de test,
tandis que les chemins `JPEGImages` et `SegmentationClass`
stockent l'image d'entrée et l'étiquette pour chaque exemple, respectivement.
Ici, l'étiquette est également au format image,
avec la même taille
que son image d'entrée étiquetée.
En outre, les pixels
ayant la même couleur dans toute image d'étiquette appartiennent à la même classe sémantique.
Le texte suivant définit la fonction `read_voc_images` pour [**lire toutes les images d'entrée et les étiquettes dans la mémoire**].

```{.python .input}
#@tab mxnet
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

Nous [**dessinons les cinq premières images d'entrée et leurs étiquettes**].
Dans les images d'étiquettes, le blanc et le noir représentent respectivement les bordures et le fond, tandis que les autres couleurs correspondent aux différentes classes.

```{.python .input}
#@tab mxnet
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

Ensuite, nous [**énumérons
les valeurs des couleurs RVB et les noms des classes**]
pour toutes les étiquettes de cet ensemble de données.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

Avec les deux constantes définies ci-dessus,
nous pouvons facilement
[**trouver l'indice de classe pour chaque pixel dans une étiquette**].
Nous définissons la fonction `voc_colormap2label`
pour construire le mappage des valeurs de couleur RVB ci-dessus
aux indices de classe,
et la fonction `voc_label_indices`
pour mapper toute valeur RVB à ses indices de classe dans cet ensemble de données Pascal VOC2012.

```{.python .input}
#@tab mxnet
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**Par exemple**], dans la première image d'exemple,
l'indice de classe pour la partie avant de l'avion est 1,
alors que l'indice de fond est 0.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### Prétraitement des données

Dans les expériences précédentes,
comme dans :numref:`sec_alexnet` --:numref:`sec_googlenet`,
les images sont redimensionnées
pour s'adapter à la forme d'entrée requise par le modèle.
Cependant, dans le cadre de la segmentation sémantique,
cette opération
nécessite de redimensionner les classes de pixels prédites
pour les ramener à la forme originale de l'image d'entrée.
Une telle remise à l'échelle peut être inexacte,
surtout pour les régions segmentées avec des classes différentes. Pour éviter ce problème,
nous recadrons l'image à une forme *fixe* au lieu de la remettre à l'échelle. Plus précisément, [**en utilisant un recadrage aléatoire à partir de l'augmentation de l'image, nous recadrons la même zone de
l'image d'entrée et l'étiquette**].

```{.python .input}
#@tab mxnet
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab mxnet
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**Classe de jeu de données de segmentation sémantique personnalisée**]

Nous définissons une classe de jeu de données de segmentation sémantique personnalisée `VOCSegDataset` en héritant de la classe `Dataset` fournie par les API de haut niveau.
En implémentant la fonction `__getitem__`,
nous pouvons accéder arbitrairement à l'image d'entrée indexée comme `idx` dans le jeu de données et à l'indice de classe de chaque pixel dans cette image.
Étant donné que certaines images de l'ensemble de données
ont une taille plus petite
que la taille de sortie du recadrage aléatoire,
ces exemples sont filtrés
par une fonction personnalisée `filter`.

En outre, nous définissons également la fonction `normalize_image` pour
normaliser les valeurs des trois canaux RVB des images d'entrée.

```{.python .input}
#@tab mxnet
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**Lire l'ensemble de données**]

Nous utilisons la classe personnalisée `VOCSegDatase`t pour
créer des instances de l'ensemble d'entraînement et de l'ensemble de test, respectivement.
Supposons que
nous spécifions que la forme de sortie des images recadrées de manière aléatoire est $320\times 480$.
Nous pouvons voir ci-dessous le nombre d'exemples
qui sont retenus dans l'ensemble d'entraînement et l'ensemble de test.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

En fixant la taille du lot à 64,
nous définissons l'itérateur de données pour l'ensemble d'entraînement.
Imprimons la forme du premier minibatch.
Contrairement à la classification d'images ou à la détection d'objets, les étiquettes sont ici des tenseurs tridimensionnels.

```{.python .input}
#@tab mxnet
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**Putting All Things Together**]

Enfin, nous définissons la fonction suivante `load_data_voc`
pour télécharger et lire le jeu de données de segmentation sémantique Pascal VOC2012.
Elle renvoie des itérateurs de données pour les ensembles de données d'entraînement et de test.

```{.python .input}
#@tab mxnet
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## Résumé

* La segmentation sémantique reconnaît et comprend ce qui se trouve dans une image au niveau du pixel en divisant l'image en régions appartenant à différentes classes sémantiques.
* L'un des plus importants jeux de données de segmentation sémantique est le Pascal VOC2012.
* Dans la segmentation sémantique, puisque l'image d'entrée et l'étiquette correspondent de manière biunivoque au niveau du pixel, l'image d'entrée est recadrée de manière aléatoire à une forme fixe plutôt que d'être redimensionnée.


## Exercices

1. Comment la segmentation sémantique peut-elle être appliquée aux véhicules autonomes et aux diagnostics d'images médicales ? Pouvez-vous imaginer d'autres applications ?
1. Rappelez-vous les descriptions de l'augmentation des données dans :numref:`sec_image_augmentation`. Parmi les méthodes d'augmentation des données utilisées dans la classification d'images, lesquelles ne pourraient pas être appliquées à la segmentation sémantique ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab:
