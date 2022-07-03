# Le jeu de données de détection d'objets
:label:`sec_object-detection-dataset` 

 Il n'existe pas de petit jeu de données tel que MNIST et Fashion-MNIST dans le domaine de la détection d'objets.
Afin de démontrer rapidement les modèles de détection d'objets,
[**nous avons collecté et étiqueté un petit jeu de données**].
Tout d'abord, nous avons pris des photos de bananes gratuites dans notre bureau
et généré
1000 images de bananes avec différentes rotations et tailles.
Nous avons ensuite placé chaque image de banane
à une position aléatoire sur une image de fond.
Au final, nous avons étiqueté des boîtes englobantes pour ces bananes sur les images.


## [**Téléchargement du jeu de données**]

Le jeu de données de détection de bananes avec toutes les images et les fichiers d'étiquettes csv
peut être téléchargé directement sur Internet.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Lire le jeu de données

Nous allons [**lire le jeu de données de détection des bananes**] dans la fonction `read_data_bananas`
 ci-dessous.
Le jeu de données comprend un fichier csv pour les étiquettes de classe d'objet
et les coordonnées de la boîte de délimitation de terrain véridique

 aux coins supérieur gauche et inférieur droit.

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `cible` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `cible` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

En utilisant la fonction `read_data_bananas` pour lire les images et les étiquettes,
la classe `BananasDataset` suivante
nous permettra de [**créer une instance `Dataset` personnalisée**]
pour charger le jeu de données de détection de bananes.

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Enfin, nous définissons
la fonction `load_data_bananas` pour [**renvoyer deux instances
d'itérateur de données pour les ensembles de formation et de test.**]
Pour l'ensemble de données de test,
il n'est pas nécessaire de le lire dans un ordre aléatoire.

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

Lisons [**un minibatch et imprimons les formes de
à la fois des images et des étiquettes**] dans ce minibatch.
La forme du minibatch d'images,
(taille du lot, nombre de canaux, hauteur, largeur),
semble familière :
est la même que dans nos tâches précédentes de classification d'images.
La forme du minilot d'étiquettes est
(taille du lot, $m$, 5),
où $m$ est le plus grand nombre possible de boîtes de délimitation
qu'une image possède dans l'ensemble de données.

Bien que le calcul en minibatchs soit plus efficace,
il faut que tous les exemples d'images
contiennent le même nombre de boîtes englobantes pour former un minibatch par concaténation.
En général,
images peuvent avoir un nombre variable de boîtes englobantes ;
ainsi,
images avec moins de $m$ boîtes englobantes
seront complétées par des boîtes englobantes illégales
jusqu'à ce que $m$ soit atteint.
Ensuite,
l'étiquette de chaque bounding box est représentée par un tableau de longueur 5.
Le premier élément du tableau est la classe de l'objet dans le bounding box,
où -1 indique un bounding box illégal pour le remplissage.
Les quatre autres éléments du tableau sont
les valeurs des coordonnées ($x$, $y$)
du coin supérieur gauche et du coin inférieur droit
de la boîte englobante (la plage est comprise entre 0 et 1).
Pour l'ensemble de données sur les bananes,
comme il n'y a qu'une seule boîte englobante sur chaque image,
nous avons $m=1$.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Démonstration**]

Démontrons dix images avec leurs boîtes englobantes étiquetées de vérité.
Nous pouvons constater que les rotations, les tailles et les positions des bananes varient sur toutes ces images.
Bien entendu, il ne s'agit que d'un simple ensemble de données artificielles.
Dans la pratique, les ensembles de données du monde réel sont généralement beaucoup plus complexes.

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Résumé

* Le jeu de données de détection de bananes que nous avons collecté peut être utilisé pour démontrer les modèles de détection d'objets.
* Le chargement des données pour la détection d'objets est similaire à celui de la classification d'images. Cependant, dans la détection d'objets, les étiquettes contiennent également des informations sur les boîtes de délimitation de la vérité du sol, ce qui manque dans la classification d'images.


## Exercices

1. Présentez d'autres images avec des boîtes de délimitation véridiques au sol dans l'ensemble de données de détection de bananes. Comment diffèrent-elles en ce qui concerne les boîtes englobantes et les objets ?
1. Supposons que nous voulions appliquer l'augmentation des données, comme le recadrage aléatoire, à la détection d'objets. En quoi cela peut-il être différent de ce qui se passe dans la classification d'images ? Indice : que se passe-t-il si une image recadrée ne contient qu'une petite partie d'un objet ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
