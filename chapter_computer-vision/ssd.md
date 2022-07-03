# Single Shot Multibox Detection
:label:`sec_ssd` 

Dans :numref:`sec_bbox` --:numref:`sec_object-detection-dataset`,
nous avons présenté les boîtes englobantes, les boîtes d'ancrage,
la détection d'objets multi-échelle, et le jeu de données pour la détection d'objets.
Nous sommes maintenant prêts à utiliser ces connaissances de base
pour concevoir un modèle de détection d'objets :
single shot multibox detection
(SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`.
Ce modèle est simple, rapide et largement utilisé.
Bien qu'il ne s'agisse que d'un des nombreux modèles de détection d'objets,

certains des principes de conception
et des détails de mise en œuvre de cette section
sont également applicables à d'autres modèles.


## Modèle

:numref:`fig_ssd` fournit une vue d'ensemble de
la conception de la détection multiboxes à un seul coup.
Ce modèle se compose principalement de
un réseau de base
suivi de
plusieurs blocs de cartes de caractéristiques multi-échelles.
Le réseau de base
sert à extraire les caractéristiques de l'image d'entrée,
et peut donc utiliser un CNN profond.
Par exemple,
l'article original sur la détection de boîtes multiples à un seul coup
adopte un réseau VGG tronqué avant la couche de classification
 :cite:`Liu.Anguelov.Erhan.ea.2016`,
tandis que ResNet a également été couramment utilisé.
Grâce à notre conception,
 nous pouvons faire en sorte que le réseau de base produise
des cartes de caractéristiques plus grandes
afin de générer plus de boîtes d'ancrage
pour détecter des objets plus petits.
Par la suite,
chaque bloc de cartes de caractéristiques multi-échelles
réduit (par exemple, de moitié)
la hauteur et la largeur des cartes de caractéristiques
du bloc précédent,
et permet à chaque unité
des cartes de caractéristiques
d'augmenter son champ réceptif sur l'image d'entrée.


Rappelons la conception
de la détection d'objets multi-échelle
par le biais de représentations en couches des images par
des réseaux neuronaux profonds
dans :numref:`sec_multiscale-object-detection`.
Étant donné que les cartes de caractéristiques multi-échelles
plus proches du sommet de :numref:`fig_ssd` 
sont plus petites mais ont des champs réceptifs plus grands,
elles conviennent pour détecter
des objets moins nombreux mais plus grands.

En bref,
par l'intermédiaire de son réseau de base et de plusieurs blocs de cartes de caractéristiques multi-échelles,
détection multi-boîtes à un seul coup
génère un nombre variable de boîtes d'ancrage de différentes tailles,
et détecte des objets de taille variable
en prédisant les classes et les décalages
de ces boîtes d'ancrage (donc les boîtes englobantes) ;
il s'agit donc d'un modèle de détection d'objets multi-échelles.


![As a multiscale object detection model, single-shot multibox detection mainly consists of a base network followed by several multiscale feature map blocks.](../img/ssd.svg)
:label:`fig_ssd`


Dans ce qui suit,
nous décrivons les détails de la mise en œuvre
de différents blocs dans :numref:`fig_ssd`. Pour commencer, nous verrons comment implémenter
la prédiction de classe et de boîte englobante.



### [**Couche de prédiction de classe**]

Soit le nombre de classes d'objets $q$.
Les boîtes d'ancrage ont alors $q+1$ classes,
où la classe 0 est le fond.
À une certaine échelle,
supposons que la hauteur et la largeur des cartes de caractéristiques
soient respectivement $h$ et $w$.
Lorsque $a$ boîtes d'ancrage
sont générées avec
chaque position spatiale de ces cartes de caractéristiques comme centre,
un total de $hwa$ boîtes d'ancrage doivent être classées.
Cela rend souvent la classification avec des couches entièrement connectées infaisable en raison des coûts de paramétrage probablement élevés de
.
Rappelez-vous comment nous avons utilisé les canaux des couches convolutionnelles

pour prédire les classes dans :numref:`sec_nin`.
La détection multi-boîtes à un coup utilise la même technique
pour réduire la complexité du modèle.

Plus précisément,
la couche de prédiction des classes utilise une couche convolutive
sans modifier la largeur ou la hauteur des cartes de caractéristiques.
De cette façon,
il peut y avoir une correspondance biunivoque
entre les sorties et les entrées
aux mêmes dimensions spatiales (largeur et hauteur)
des cartes de caractéristiques.
Plus concrètement, les canaux
des cartes de caractéristiques de sortie
à n'importe quelle position spatiale ($x$, $y$)
représentent des prédictions de classe
pour toutes les cases d'ancrage centrées sur
($x$, $y$) des cartes de caractéristiques d'entrée.
Pour produire des prédictions valides,
il doit y avoir des canaux de sortie $a(q+1)$,
où, pour la même position spatiale,
le canal de sortie avec l'indice $i(q+1) + j$
représente la prédiction de
la classe $j$ ($0 \leq j \leq q$)
pour la boîte d'ancrage $i$ ($0 \leq i < a$).

Nous définissons ci-dessous une telle couche de prédiction de classe,
spécifiant $a$ et $q$ via les arguments `num_anchors` et `num_classes`, respectivement.
Cette couche utilise une couche convolutive $3\times3$ avec un padding
de 1.
La largeur et la hauteur de l'entrée et de la sortie de cette couche convolutive
restent inchangées.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**Couche de prédiction de boîte englobante**)

La conception de la couche de prédiction de boîte englobante est similaire à celle de la couche de prédiction de classe.
La seule différence réside dans le nombre de sorties pour chaque boîte d'ancrage :
ici, nous devons prédire quatre décalages plutôt que des classes $q+1$.

```{.python .input}
#@tab mxnet
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**Concaténation des prédictions pour plusieurs échelles**]

Comme nous l'avons mentionné, la détection multi-boîtes à un seul coup
utilise des cartes de caractéristiques multi-échelles pour générer des boîtes d'ancrage et prédire leurs classes et leurs décalages.
À différentes échelles,
les formes des cartes de caractéristiques
ou le nombre de boîtes d'ancrage centrées sur la même unité
peuvent varier.
Par conséquent,
les formes des sorties de prédiction
à différentes échelles peuvent varier.

Dans l'exemple suivant,
nous construisons des cartes de caractéristiques à deux échelles différentes,
`Y1` et `Y2`,
pour le même mini-lot,
où la hauteur et la largeur de `Y2`
sont la moitié de celles de `Y1`.
Prenons la prédiction de classe comme exemple.
Supposons que
5 et 3 boîtes d'ancrage
soient générées pour chaque unité de `Y1` et `Y2`, respectivement.
Supposons en outre que
le nombre de classes d'objets est de 10.
Pour les cartes de caractéristiques `Y1` et `Y2`
le nombre de canaux dans les sorties de prédiction de classe
est respectivement $5\times(10+1)=55$ et $3\times(10+1)=33$,
où la forme de l'une ou l'autre sortie est
(taille du lot, nombre de canaux, hauteur, largeur).

```{.python .input}
#@tab mxnet
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

Comme nous pouvons le constater, à l'exception de la dimension de la taille du lot,
les trois autres dimensions ont toutes des tailles différentes.
Afin de concaténer ces deux sorties de prédiction pour un calcul plus efficace,
nous allons transformer ces tenseurs dans un format plus cohérent.

Notez que
la dimension du canal contient les prédictions pour
boîtes d'ancrage avec le même centre.
Nous déplaçons d'abord cette dimension vers la plus intérieure.
Puisque la taille du lot reste la même pour différentes échelles,
nous pouvons transformer la sortie de prédiction
en un tenseur bidimensionnel
avec la forme (taille du lot, hauteur $\times$ largeur $\times$ nombre de canaux).
Nous pouvons ensuite concaténer
de telles sorties à différentes échelles
selon la dimension 1.

```{.python .input}
#@tab mxnet
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

De cette façon,
même si `Y1` et `Y2` ont des tailles différentes
en termes de canaux, de hauteurs et de largeurs,
nous pouvons toujours concaténer ces deux sorties de prédiction à deux échelles différentes pour le même minibatch.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**Bloc de sous-échantillonnage**]

Afin de détecter des objets à plusieurs échelles,
nous définissons le bloc de sous-échantillonnage suivant `down_sample_blk` qui
divise par deux la hauteur et la largeur des cartes de caractéristiques d'entrée.
En fait,
ce bloc applique la conception des blocs VGG
dans :numref:`subsec_vgg-blocks`.
Plus concrètement,
chaque bloc de sous-échantillonnage se compose de
deux $3\times3$ couches convolutionnelles avec un padding de 1
suivi d'une $2\times2$ couche de max-pooling avec un stride de 2.
Comme nous le savons, $3\times3$ couches convolutionnelles avec un padding de 1 ne modifient pas la forme des cartes de caractéristiques.
Cependant, le max-pooling suivant $2\times2$ réduit de moitié la hauteur et la largeur des cartes de caractéristiques d'entrée.
Pour les cartes de caractéristiques d'entrée et de sortie de ce bloc de sous-échantillonnage,
car $1\times 2+(3-1)+(3-1)=6$,
chaque unité de la sortie
a un champ récepteur $6\times6$ sur l'entrée.
Par conséquent, le bloc de sous-échantillonnage agrandit le champ réceptif de chaque unité dans ses cartes de caractéristiques de sortie.

```{.python .input}
#@tab mxnet
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

Dans l'exemple suivant, le bloc de sous-échantillonnage que nous avons construit modifie le nombre de canaux d'entrée et réduit de moitié la hauteur et la largeur des cartes de caractéristiques d'entrée.

```{.python .input}
#@tab mxnet
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**Bloc réseau de base**]

Le bloc réseau de base est utilisé pour extraire les caractéristiques des images d'entrée.
Pour simplifier,
nous construisons un petit réseau de base
composé de trois blocs de sous-échantillonnage
qui doublent le nombre de canaux à chaque bloc.
Pour une image d'entrée $256\times256$,
ce bloc de réseau de base produit $32 \times 32$ des cartes de caractéristiques ($256/2^3=32$).

```{.python .input}
#@tab mxnet
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### Le modèle complet


 [**Le modèle complet de détection multiboxes à un seul coup

se compose de cinq blocs.**]
Les cartes de caractéristiques produites par chaque bloc
sont utilisées à la fois pour
(i) générer des boîtes d'ancrage
et (ii) prédire les classes et les décalages de ces boîtes d'ancrage.

Parmi ces cinq blocs,
le premier
est le bloc de réseau de base,
les deuxième à quatrième sont
des blocs de sous-échantillonnage,
et le dernier bloc
utilise le max-pooling global
pour réduire la hauteur et la largeur à 1.
Techniquement,
les deuxième à cinquième blocs
sont tous
ces blocs de cartes de caractéristiques multi-échelles
dans :numref:`fig_ssd`.

```{.python .input}
#@tab mxnet
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

Maintenant, nous [**définissons la propagation vers l'avant**]
pour chaque bloc.
Contrairement à
dans les tâches de classification d'images, les résultats de
comprennent
(i) les cartes de caractéristiques CNN `Y`,
(ii) les boîtes d'ancrage générées à l'aide de `Y` à l'échelle actuelle,
et (iii) les classes et les décalages prédits (sur la base de `Y`)
pour ces boîtes d'ancrage.

```{.python .input}
#@tab mxnet
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

Rappelez-vous que
dans :numref:`fig_ssd` 
un bloc de carte de caractéristiques multi-échelle
qui est plus proche du sommet
est destiné à détecter des objets plus grands ;
doit donc générer des boîtes d'ancrage plus grandes.
Dans la propagation vers l'avant ci-dessus,
à chaque bloc de carte de caractéristiques multi-échelle,
 nous transmettons une liste de deux valeurs d'échelle
via l'argument `sizes`
de la fonction `multibox_prior` invoquée (décrite dans :numref:`sec_anchor` ).
Dans ce qui suit,
l'intervalle entre 0,2 et 1,05
est divisé uniformément
en cinq sections pour déterminer les
plus petites valeurs d'échelle aux cinq blocs : 0.2, 0,37, 0,54, 0,71 et 0,88.
Ensuite, leurs valeurs à plus grande échelle
sont données par
$\sqrt{0.2 \times 0.37} = 0.272$ , $\sqrt{0.37 \times 0.54} = 0.447$, et ainsi de suite.

[~)~)Hyperparamètres pour chaque bloc~)~)]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

Nous pouvons maintenant [**définir le modèle complet**] `TinySSD` comme suit.

```{.python .input}
#@tab mxnet
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

Nous [**créons une instance de modèle
et l'utilisons pour effectuer une propagation directe**]
sur un minibatch d'images $256 \times 256$ `X`.

Comme nous l'avons montré précédemment dans cette section,
le premier bloc produit des cartes de caractéristiques $32 \times 32$.
Rappelons que
les deuxième à quatrième blocs de sous-échantillonnage
divisent par deux la hauteur et la largeur
et que le cinquième bloc utilise la mise en commun globale.
Étant donné que 4 boîtes d'ancrage
sont générées pour chaque unité le long des dimensions spatiales
des cartes de caractéristiques,
aux cinq échelles,
un total de $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ boîtes d'ancrage est généré pour chaque image.

```{.python .input}
#@tab mxnet
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

### Entraînement

Nous allons maintenant expliquer
comment entraîner le modèle de détection multiboxes à un seul coup
pour la détection d'objets.


### Lecture du jeu de données et initialisation du modèle

Pour commencer,
[**lisons
le jeu de données de détection de bananes**]
décrit dans :numref:`sec_object-detection-dataset`.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

Il n'y a qu'une seule classe dans le jeu de données de détection de bananes. Après avoir défini le modèle,
nous devons (**initialiser ses paramètres et définir
l'algorithme d'optimisation**).

```{.python .input}
#@tab mxnet
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**Définir les fonctions de perte et d'évaluation**]

La détection d'objets présente deux types de pertes.
La première perte concerne les classes de boîtes d'ancrage :
son calcul
peut simplement réutiliser
la fonction de perte d'entropie croisée
que nous avons utilisée pour la classification des images.
La seconde perte
concerne les décalages des boîtes d'ancrage positives (hors arrière-plan) :
il s'agit d'un problème de régression.
Pour ce problème de régression,
cependant,
ici nous n'utilisons pas la perte au carré
décrite dans :numref:`subsec_normal_distribution_and_squared_loss`.
Au lieu de cela,
nous utilisons la perte normalisée $\ell_1$,
la valeur absolue de la différence entre
la prédiction et la vérité du sol.
La variable de masque `bbox_masks` permet de filtrer les boîtes d'ancrage négatives
et les boîtes d'ancrage illégales (remplies)
dans le calcul de la perte.
Au final, nous additionnons
la perte de classe des boîtes d'ancrage
et la perte de décalage des boîtes d'ancrage
pour obtenir la fonction de perte du modèle.

```{.python .input}
#@tab mxnet
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

Nous pouvons utiliser la précision pour évaluer les résultats de la classification.
En raison de la perte de norme utilisée $\ell_1$ pour les décalages,
nous utilisons l'erreur absolue moyenne * pour évaluer les boîtes englobantes prédites
.
Ces résultats de prédiction sont obtenus
à partir des boîtes d'ancrage générées et des décalages prédits
pour celles-ci.

```{.python .input}
#@tab mxnet
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**Entrainement du modèle**]

Lors de l'entrainement du modèle,
nous devons générer des boîtes d'ancrage multi-échelles (`anchors`)
et prédire leurs classes (`cls_preds`) et leurs décalages (`bbox_preds`) dans la propagation vers l'avant.
Ensuite, nous étiquetons les classes (`cls_labels`) et les décalages (`bbox_labels`) de ces boîtes d'ancrage générées
sur la base des informations de l'étiquette `Y`.
Enfin, nous calculons la fonction de perte
en utilisant les valeurs prédites et étiquetées
des classes et des décalages.
Pour des implémentations concises, l'évaluation de l'ensemble de données de test
est omise ici.

```{.python .input}
#@tab mxnet
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**Prédiction**]

Pendant la prédiction,
le but est de détecter tous les objets d'intérêt
sur l'image.
Ci-dessous
nous lisons et redimensionnons une image test,
en la convertissant en
un tenseur à quatre dimensions qui est
requis par les couches convolutionnelles.

```{.python .input}
#@tab mxnet
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

À l'aide de la fonction `multibox_detection` ci-dessous,
les boîtes limites prédites
sont obtenues
à partir des boîtes d'ancrage et de leurs décalages prédits.
Ensuite, la suppression non maximale est utilisée
pour éliminer les boîtes limites prédites similaires.

```{.python .input}
#@tab mxnet
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

Enfin, nous [**affichons
toutes les boîtes englobantes prédites avec un niveau de confiance de 0,9 ou plus pour
**]
comme résultat.

```{.python .input}
#@tab mxnet
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## Résumé

* Single shot multibox detection est un modèle de détection d'objets multi-échelles. Grâce à son réseau de base et à plusieurs blocs de cartes de caractéristiques multi-échelles, le modèle de détection multi-boîtes à un coup génère un nombre variable de boîtes d'ancrage de différentes tailles et détecte les objets de taille variable en prédisant les classes et les décalages de ces boîtes d'ancrage (donc les boîtes englobantes).
* Lors de l'apprentissage du modèle de détection multiboxes à un coup, la fonction de perte est calculée sur la base des valeurs prédites et étiquetées des classes et des décalages des boîtes d'ancrage.



## Exercices

1. Pouvez-vous améliorer la détection des boîtes multiples à un seul coup en améliorant la fonction de perte ? Par exemple, remplacez la perte normalisée $\ell_1$ par une perte normalisée lisse $\ell_1$ pour les décalages prédits. Cette fonction de perte utilise une fonction carrée autour de zéro pour la régularité, qui est contrôlée par l'hyperparamètre $\sigma$:

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

Lorsque $\sigma$ est très grand, cette perte est similaire à la perte de norme $\ell_1$. Lorsque sa valeur est plus petite, la fonction de perte est plus lisse.

```{.python .input}
#@tab mxnet
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

En outre, dans l'expérience, nous avons utilisé la perte d'entropie croisée pour la prédiction de classe :
désignant par $p_j$ la probabilité prédite pour la classe de vérité du sol $j$, la perte d'entropie croisée est $-\log p_j$. Nous pouvons également utiliser la perte focale
:cite:`Lin.Goyal.Girshick.ea.2017` : étant donné les hyperparamètres $\gamma > 0$
et $\alpha > 0$, cette perte est définie comme suit :

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$ 

Comme nous pouvons le constater, l'augmentation de $\gamma$
peut réduire efficacement la perte relative
pour les exemples bien classés (par exemple, $p_j > 0.5$)
de sorte que la formation
peut se concentrer davantage sur les exemples difficiles qui sont mal classés.

```{.python .input}
#@tab mxnet
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. En raison du manque d'espace, nous avons omis certains détails de mise en œuvre du modèle de détection multiboxes à un seul coup dans cette section. Pouvez-vous encore améliorer le modèle dans les aspects suivants :
   1. Lorsqu'un objet est beaucoup plus petit que l'image, le modèle pourrait redimensionner l'image d'entrée.
   1. Il y a généralement un grand nombre de boîtes d'ancrage négatives. Pour rendre la distribution des classes plus équilibrée, nous pourrions réduire l'échantillonnage des boîtes d'ancrage négatives.
   1. Dans la fonction de perte, attribuez des hyperparamètres de poids différents à la perte de classe et à la perte de décalage.
   1. Utilisez d'autres méthodes pour évaluer le modèle de détection d'objets, telles que celles présentées dans l'article sur la détection multiboxes à un seul coup :cite:`Liu.Anguelov.Erhan.ea.2016`.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:
