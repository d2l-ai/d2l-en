```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Convolutional Neural Networks (LeNet)
:label:`sec_lenet`

Nous avons maintenant tous les ingrédients nécessaires pour assembler
un CNN entièrement fonctionnel.
Dans notre rencontre précédente avec des données d'image, nous avons appliqué
un modèle linéaire avec régression softmax (:numref:`sec_softmax_scratch`)
et un MLP (:numref:`sec_mlp_scratch`)
à des images de vêtements dans l'ensemble de données Fashion-MNIST.
Pour rendre ces données exploitables, nous avons d'abord aplati chaque image à partir d'une matrice de 28\times28$.
en un vecteur de longueur fixe de 784$ dimensions,
et nous les avons ensuite traitées dans des couches entièrement connectées.
Maintenant que nous maîtrisons les couches convolutionnelles,
nous pouvons conserver la structure spatiale de nos images.
Un avantage supplémentaire du remplacement des couches entièrement connectées par des couches convolutives,
nous bénéficierons de modèles plus parcimonieux qui nécessitent beaucoup moins de paramètres.

Dans cette section, nous allons présenter *LeNet*,
parmi les premiers CNNs publiés
publié qui a attiré l'attention pour ses performances dans les tâches de vision par ordinateur.
Le modèle a été introduit par (et nommé pour) Yann LeCun,
alors chercheur chez AT&T Bell Labs,
dans le but de reconnaître des chiffres manuscrits dans des images :cite:`LeCun.Bottou.Bengio.ea.1998`.
Ce travail représentait l'aboutissement
d'une décennie de recherche pour développer cette technologie.
En 1989, l'équipe de LeCun a publié la première étude visant à former avec succès des CNN par rétro-propagation.
former des CNN par rétropropagation :cite:`LeCun.Boser.Denker.ea.1989`.

A l'époque, LeNet a obtenu des résultats exceptionnels
égalant les performances des machines à vecteurs de support,
alors une approche dominante dans l'apprentissage supervisé, atteignant un taux d'erreur de moins de 1% par chiffre.
LeNet a finalement été adapté pour reconnaître les chiffres
pour le traitement des dépôts dans les guichets automatiques.
À ce jour, certains guichets automatiques utilisent encore le code
que Yann LeCun et son collègue Leon Bottou ont écrit dans les années 1990 !


## LeNet

À un niveau élevé, (**LeNet (LeNet-5) se compose de deux parties:
(i) un codeur convolutif composé de deux couches convolutives; et
(ii) un bloc dense composé de trois couches entièrement connectées**);
L'architecture est résumée dans :numref:`img_lenet`.

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

Les unités de base dans chaque bloc convolutif
sont une couche convolutive, une fonction d'activation sigmoïde,
et une opération ultérieure de mise en commun des moyennes.
Notez que bien que les ReLUs et le max-pooling fonctionnent mieux,
ces découvertes n'avaient pas encore été faites à l'époque.
Chaque couche convolutionnelle utilise un noyau de 5\fois 5
et une fonction d'activation sigmoïde.
Ces couches transforment les entrées disposées dans l'espace
à un certain nombre de cartes de caractéristiques bidimensionnelles, augmentant typiquement
en augmentant le nombre de canaux.
La première couche convolutive a 6 canaux de sortie,
tandis que la seconde en a 16.
Chaque opération de mise en commun $2\times2$ (stride 2)
réduit la dimensionnalité par un facteur de 4$ via un sous-échantillonnage spatial.
Le bloc convolutif émet une sortie dont la forme est donnée par
(taille du lot, nombre de canaux, hauteur, largeur).

Afin de passer la sortie du bloc convolutif
au bloc dense,
nous devons aplatir chaque exemple dans le minibatch.
En d'autres termes, nous prenons cette entrée quadridimensionnelle et la transformons
en une entrée bidimensionnelle attendue par les couches entièrement connectées :
pour rappel, la représentation bidimensionnelle que nous désirons utilise la première dimension pour indexer les exemples dans le minibatch
et la seconde pour donner la représentation vectorielle plate de chaque exemple.
Le bloc dense de LeNet a trois couches entièrement connectées,
avec 120, 84, et 10 sorties, respectivement.
Comme nous effectuons toujours une classification,
la couche de sortie à 10 dimensions correspond
au nombre de classes de sortie possibles.

Même si arriver au point où vous comprenez vraiment
ce qui se passe dans LeNet peut avoir demandé un peu de travail,
j'espère que l'extrait de code suivant vous convaincra
que la mise en œuvre de tels modèles avec les cadres d'apprentissage profond modernes
est remarquablement simple.
Il suffit d'instancier un bloc `Sequential`.
et enchaîner les couches appropriées,
en utilisant l'initialisation de Xavier comme
introduite dans :numref:`subsec_xavier`.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab all
class LeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

Nous prenons quelques libertés dans la reproduction de LeNet dans la mesure où nous remplaçons la couche d'activation gaussienne par
une couche softmax. Cela simplifie grandement l'implémentation, notamment en raison du fait que le décodeur gaussien est rarement utilisé de nos jours.
fait que le décodeur gaussien est rarement utilisé de nos jours. Pour le reste, ce réseau correspond à
l'architecture originale de LeNet-5.

Voyons ce qui se passe à l'intérieur du réseau. En passant un
un seul canal (noir et blanc)
28$ \times 28$ image à travers le réseau
et en imprimant la forme de sortie à chaque couche,
on peut [**inspecter le modèle**] pour s'assurer
que ses opérations correspondent à
ce que nous attendons de :numref:`img_lenet_vert`.

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

Notez que la hauteur et la largeur de la représentation
à chaque couche tout au long du bloc convolutif
sont réduites (par rapport à la couche précédente).
La première couche convolutive utilise 2 pixels de remplissage
pour compenser la réduction de la hauteur et de la largeur
qui résulterait autrement de l'utilisation d'un noyau $5 \times 5$.
Pour l'anecdote, la taille de l'image de $28 \times 28$ pixels dans l'ensemble de données OCR MNIST original
est le résultat de *l'élagage* de lignes (et de colonnes) de 2 pixels des scans originaux
qui mesuraient $32 \times 32$ pixels. Cela a été fait principalement pour
économiser de l'espace (une réduction de 30%) à une époque où les mégaoctets étaient importants.

En revanche, la deuxième couche convolutive renonce au remplissage,
et donc la hauteur et la largeur sont toutes deux réduites de 4 pixels.
À mesure que l'on monte dans la pile de couches,
le nombre de canaux augmente couche par couche,
passant de 1 dans l'entrée à 6 après la première couche convolutive
et à 16 après la deuxième couche convolutive.
Cependant, chaque couche de mise en commun réduit de moitié la hauteur et la largeur.
Enfin, chaque couche entièrement connectée réduit la dimensionnalité,
émettant finalement une sortie dont la dimension
correspond au nombre de classes.


## Entraînement

Maintenant que nous avons implémenté le modèle,
faisons [**une expérience pour voir comment le modèle LeNet-5 se comporte sur Fashion-MNIST**].

Bien que les CNN aient moins de paramètres,
ils peuvent néanmoins être plus coûteux à calculer
que les MLP de profondeur similaire
car chaque paramètre participe à beaucoup plus de multiplications
.
Si vous avez accès à un GPU, c'est peut-être le bon moment
de le mettre en action pour accélérer la formation.
Notez que
la classe `d2l.Trainer` s'occupe de tous les détails.
Par défaut, elle initialise les paramètres du modèle sur les
dispositifs disponibles.
Comme pour les MLP, notre fonction de perte est l'entropie croisée,
et nous la minimisons via une descente de gradient stochastique en minibatch.

```{.python .input}
%%tab pytorch, mxnet
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## Résumé

Dans ce chapitre, nous avons fait des progrès significatifs. Nous sommes passés des MLP des années 1980 aux CNN des années 1990 et du début des années 2000. Les architectures proposées, par exemple sous la forme de LeNet-5, restent significatives, même à ce jour. Il est intéressant de comparer les taux d'erreur sur Fashion-MNIST obtenus avec LeNet-5 aux meilleurs taux obtenus avec les MLP (:numref:`sec_mlp_scratch` ) et ceux obtenus avec des architectures beaucoup plus avancées comme ResNet (:numref:`sec_resnet` ). LeNet est beaucoup plus proche de ces dernières que des premières. L'une des principales différences, comme nous le verrons, est que de plus grandes quantités de calcul permettent des architectures beaucoup plus complexes.

Une deuxième différence est la facilité relative avec laquelle nous avons pu implémenter LeNet. Ce qui était auparavant un défi d'ingénierie représentant des mois de code C++ et d'assemblage, d'ingénierie pour améliorer SN, un outil précoce d'apprentissage profond basé sur Lisp :cite:`Bottou.Le-Cun.1988`, et enfin l'expérimentation de modèles, peut maintenant être accompli en quelques minutes. C'est cet incroyable gain de productivité qui a considérablement démocratisé le développement de modèles d'apprentissage profond. Dans le chapitre suivant, nous allons suivre ce terrier de lapin pour voir où il nous mène.

## Exercices

1. Modernisons LeNet. Implémentez et testez les changements suivants :
   1. Remplacer le pooling moyen par le max-pooling.
   1. Remplacer la couche softmax par ReLU.
1. Essayez de modifier la taille du réseau de style LeNet pour améliorer sa précision en plus du max-pooling et de ReLU.
   1. Ajustez la taille de la fenêtre de convolution.
   1. Ajustez le nombre de canaux de sortie.
   1. Ajustez le nombre de couches de convolution.
   1. Ajustez le nombre de couches entièrement connectées.
   1. Ajustez les taux d'apprentissage et d'autres détails de entrainement (par exemple, l'initialisation et le nombre d'époques).
1. Essayez le réseau amélioré sur l'ensemble de données MNIST original.
1. Affichez les activations de la première et de la deuxième couche de LeNet pour différentes entrées (par exemple, des pulls et des manteaux).
1. Qu'arrive-t-il aux activations lorsque vous introduisez des images très différentes dans le réseau (par exemple, des chats, des voitures, ou même du bruit aléatoire) ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
