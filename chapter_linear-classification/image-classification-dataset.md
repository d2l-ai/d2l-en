```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Le jeu de données de classification d'images
:label:`sec_fashion_mnist` 

 (~)~)Le jeu de données MNIST est l'un des jeux de données les plus utilisés pour la classification d'images, mais il est trop simple comme jeu de données de référence. Nous utiliserons le jeu de données Fashion-MNIST similaire, mais plus complexe ~)~))

L'un des jeux de données les plus utilisés pour la classification d'images est le jeu de données [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998` de chiffres manuscrits. Au moment de sa publication dans les années 1990, il représentait un formidable défi pour la plupart des algorithmes d'apprentissage automatique, puisqu'il comprenait 60 000 images d'une résolution de $28 \times 28$ pixels (plus un ensemble de données de test de 10 000 images). Pour mettre les choses en perspective, à l'époque, une Sun SPARCStation 5 dotée d'une énorme mémoire vive de 64 Mo et d'une vitesse fulgurante de 5 MFLOP était considérée comme un équipement de pointe pour l'apprentissage automatique aux AT&T Bell Laboratories en 1995. L'obtention d'une grande précision dans la reconnaissance des chiffres était un élément clé de l'automatisation du tri des lettres pour l'USPS dans les années 1990. Les réseaux profonds tels que LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`, les machines à vecteurs de support avec invariances :cite:`Scholkopf.Burges.Vapnik.1996`, et les classifieurs à distance tangente :cite:`Simard.LeCun.Denker.ea.1998` ont tous permis d'atteindre des taux d'erreur inférieurs à 1 %. 

Pendant plus d'une décennie, MNIST a servi de *point de référence pour la comparaison des algorithmes d'apprentissage automatique. 
S'il a bien fonctionné en tant que jeu de données de référence,
même les modèles simples selon les normes actuelles atteignent une précision de classification supérieure à 95 %,
ce qui ne permet pas de distinguer les modèles les plus forts des plus faibles. Qui plus est, l'ensemble de données permet d'atteindre des niveaux de précision *très* élevés, ce qui n'est généralement pas le cas dans de nombreux problèmes de classification. Cela a orienté le développement algorithmique vers des familles spécifiques d'algorithmes qui peuvent tirer parti de jeux de données propres, tels que les méthodes d'ensembles actifs et les algorithmes d'ensembles actifs à recherche de limites.
Aujourd'hui, MNIST sert davantage de vérification de l'intégrité que de référence. ImageNet :cite:`Deng.Dong.Socher.ea.2009` pose un défi beaucoup plus pertinent .
Malheureusement, ImageNet est trop grand pour la plupart des exemples et illustrations de ce livre, car l'entraînement prendrait trop de temps pour rendre les exemples interactifs. En remplacement, nous allons concentrer notre discussion dans les sections suivantes sur le jeu de données Fashion-MNIST,
qualitativement similaire mais beaucoup plus petit, :cite:`Xiao.Rasul.Vollgraf.2017`, qui a été publié en 2017. Il est composé d'images de 10 catégories de vêtements à une résolution de $28 \times 28$ pixels.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## Chargement du jeu de données

Comme il s'agit d'un jeu de données si fréquemment utilisé, tous les principaux frameworks en fournissent des versions prétraitées. Nous pouvons [**télécharger et lire le jeu de données Fashion-MNIST en mémoire en utilisant les fonctions intégrées du framework.**]

```{.python .input  n=5}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input  n=6}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input  n=7}
%%tab tensorflow
class FashionMNIST(d2l.DataModule):  #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST se compose d'images de 10 catégories, chacune représentée
par 6 000 images dans l'ensemble de données d'entraînement et par 1 000 dans l'ensemble de données de test.
Un *ensemble de données de test* est utilisé pour évaluer les performances du modèle (il ne doit pas être utilisé pour la formation).
Par conséquent, l'ensemble de entrainement et l'ensemble de test
contiennent respectivement 60 000 et 10 000 images.

```{.python .input  n=8}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input  n=9}
%%tab tensorflow
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

Les images sont en niveaux de gris et mises à l'échelle à $32 \times 32$ pixels en résolution ci-dessus. Ceci est similaire à l'ensemble de données MNIST original qui était composé d'images (binaires) en noir et blanc. Notez cependant que la plupart des données d'image modernes ont 3 canaux (rouge, vert, bleu) et que les images hyperspectrales peuvent avoir plus de 100 canaux (le capteur HyMap a 126 canaux).
Par convention, nous stockons l'image sous la forme d'un tenseur $c \times h \times w$, où $c$ est le nombre de canaux de couleur, $h$ est la hauteur et $w$ est la largeur.

```{.python .input  n=10}
%%tab all
data.train[0][0].shape
```

[~)~)Deux fonctions utilitaires pour visualiser l'ensemble de données~)~)]

Les catégories de Fashion-MNIST ont des noms compréhensibles par l'homme. 
La fonction utilitaire suivante permet de convertir les étiquettes numériques en noms.

```{.python .input  n=11}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Lecture d'un mini-lot

Pour nous faciliter la vie lors de la lecture des ensembles d'entraînement et de test,
nous utilisons l'itérateur de données intégré plutôt que d'en créer un de toutes pièces.
Rappelons qu'à chaque itération, un itérateur de données
[**lit un mini-batch de données de taille `batch_size`.**]
Nous mélangeons également de manière aléatoire les exemples pour l'itérateur de données d'apprentissage.

```{.python .input  n=12}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input  n=13}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input  n=14}
%%tab tensorflow
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
        self.batch_size).map(resize_fn).shuffle(shuffle_buf)
```

Pour voir comment cela fonctionne, chargeons un minibatch d'images en invoquant la nouvelle méthode `train_dataloader`. Il contient 64 images.

```{.python .input  n=15}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Regardons le temps qu'il faut pour lire les images. Bien qu'il s'agisse d'un chargeur intégré, il n'est pas très rapide. Néanmoins, c'est suffisant puisque le traitement des images avec un réseau profond prend un peu plus de temps. C'est donc suffisant pour que l'entrainement d'un réseau ne soit pas une contrainte d'IO.

```{.python .input  n=16}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Visualisation

Nous utiliserons l'ensemble de données Fashion-MNIST assez fréquemment. Une fonction pratique `show_images` peut être utilisée pour visualiser les images et les étiquettes associées. Les détails de son implémentation sont reportés en annexe.

```{.python .input  n=17}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

Faisons-en bon usage. En général, c'est une bonne idée de visualiser et d'inspecter les données sur lesquelles vous vous entraînez. 
Les humains sont très doués pour repérer les aspects inhabituels et, à ce titre, la visualisation sert de garantie supplémentaire contre les erreurs et les fautes dans la conception des expériences. Voici [**les images et leurs étiquettes correspondantes**] (en texte)
pour les premiers exemples de l'ensemble de données d'entraînement.

```{.python .input  n=18}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet') or tab.selected('pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(X, nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

Nous sommes maintenant prêts à travailler avec le jeu de données Fashion-MNIST dans les sections qui suivent.

## Résumé

Nous disposons maintenant d'un jeu de données un peu plus réaliste à utiliser pour la classification. Fashion-MNIST est un jeu de données de classification de vêtements composé d'images représentant 10 catégories. Nous utiliserons cet ensemble de données dans les sections et chapitres suivants pour évaluer différentes conceptions de réseaux, d'un modèle linéaire simple à des réseaux résiduels avancés. Comme nous le faisons habituellement avec les images, nous les lisons comme un tenseur de forme (taille du lot, nombre de canaux, hauteur, largeur). Pour l'instant, nous n'avons qu'un seul canal car les images sont en niveaux de gris (la visualisation ci-dessus utilise une palette de fausses couleurs pour une meilleure visibilité). 

Enfin, les itérateurs de données sont un élément clé pour des performances efficaces. Par exemple, nous pouvons utiliser les GPU pour décompresser efficacement les images, transcoder les vidéos ou effectuer d'autres prétraitements. Dans la mesure du possible, vous devez vous appuyer sur des itérateurs de données bien implémentés qui exploitent le calcul haute performance pour éviter de ralentir votre boucle d'apprentissage.


## Exercices

1. La réduction de `batch_size` (par exemple, à 1) affecte-t-elle les performances de lecture ?
1. Les performances de l'itérateur de données sont importantes. Pensez-vous que l'implémentation actuelle est suffisamment rapide ? Explorez diverses options pour l'améliorer. Utilisez un profileur de système pour déterminer où se trouvent les goulots d'étranglement.
1. Consultez la documentation en ligne de l'API du cadre. Quels autres ensembles de données sont disponibles ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
