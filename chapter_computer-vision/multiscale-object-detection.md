# Détection d'objets multi-échelle
:label:`sec_multiscale-object-detection` 

 
Dans :numref:`sec_anchor`,
nous avons généré plusieurs boîtes d'ancrage centrées sur chaque pixel d'une image d'entrée. 
Essentiellement, ces boîtes d'ancrage 
représentent des échantillons de
différentes régions de l'image.
Cependant, 
nous pouvons nous retrouver avec trop de boîtes d'ancrage pour calculer
si elles sont générées pour *chaque* pixel.
Pensez à une image d'entrée $561 \times 728$.
Si cinq boîtes d'ancrage 
de formes différentes
sont générées pour chaque pixel comme centre,
plus de deux millions de boîtes d'ancrage ($561 \times 728 \times 5$) doivent être étiquetées et prédites sur l'image.

## Boîtes d'ancrage multi-échelles
:label:`subsec_multiscale-anchor-boxes` 

Vous pouvez vous rendre compte que
il n'est pas difficile de réduire les boîtes d'ancrage sur une image.
Par exemple, nous pouvons simplement 
échantillonner uniformément une petite partie des pixels
de l'image d'entrée
pour générer des boîtes d'ancrage centrées sur eux.
En outre, 
à différentes échelles
nous pouvons générer différents nombres de boîtes d'ancrage
de différentes tailles.
Intuitivement,
les objets plus petits sont plus susceptibles
d'apparaître sur une image que les objets plus grands.
Par exemple, les objets
$1 \times 1$ , $1 \times 2$, et $2 \times 2$ 
peuvent apparaître sur une image $2 \times 2$
de 4, 2 et 1 façons possibles, respectivement.
Par conséquent, en utilisant des boîtes d'ancrage plus petites pour détecter des objets plus petits, nous pouvons échantillonner plus de régions,
alors que pour des objets plus grands, nous pouvons échantillonner moins de régions.

Pour démontrer comment générer des boîtes d'ancrage
à plusieurs échelles, lisons une image.
Sa hauteur et sa largeur sont respectivement de 561 et 728 pixels.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

Rappelez-vous que dans :numref:`sec_conv_layer` 
nous appelons un tableau bidimensionnel issu de 
une couche convolutive une carte de caractéristiques.
En définissant la forme de la carte de caractéristiques,
nous pouvons déterminer les centres des boîtes d'ancrage échantillonnées uniformément sur n'importe quelle image.


La fonction `display_anchors` est définie ci-dessous.
[**Nous générons des boîtes d'ancrage (`anchors`) sur la carte des caractéristiques (`fmap`), chaque unité (pixel) étant le centre de la boîte d'ancrage.**]
Les valeurs des coordonnées de l'axe $(x, y)$
dans les boîtes d'ancrage (`anchors`) ayant été divisées par la largeur et la hauteur de la carte des caractéristiques (`fmap`),
ces valeurs sont comprises entre 0 et 1,
ce qui indique les positions relatives des boîtes d'ancrage
dans la carte des caractéristiques.

Puisque les centres des boîtes d'ancrage (`anchors`)
sont répartis sur toutes les unités de la carte des caractéristiques (`fmap`),
ces centres doivent être *uniformément* distribués
sur toute image d'entrée
en termes de positions spatiales relatives.
Plus concrètement,
étant donné la largeur et la hauteur de la carte de caractéristiques `fmap_w` et `fmap_h`, respectivement,
la fonction suivante échantillonnera *uniformément*
pixels dans `fmap_h` lignes et `fmap_w` colonnes
sur toute image d'entrée.
Centrées sur ces pixels échantillonnés uniformément, les boîtes d'ancrage
d'échelle `s` (en supposant que la longueur de la liste `s` est de 1) et de différents rapports d'aspect (`ratios`)
seront générées.

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Tout d'abord, examinons [**
la détection de petits objets**].
Afin de faciliter la distinction lors de l'affichage, les boîtes d'ancrage avec des centres différents ne se chevauchent pas :
l'échelle de la boîte d'ancrage est fixée à 0,15
et la hauteur et la largeur de la carte de caractéristiques sont fixées à 4. Nous pouvons voir
que les centres des boîtes d'ancrage dans 4 lignes et 4 colonnes sur l'image sont uniformément répartis.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Nous passons à [**réduire de moitié la hauteur et la largeur de la carte de caractéristiques et utiliser des boîtes d'ancrage plus grandes pour détecter des objets plus grands**]. Lorsque l'échelle est fixée à 0,4,
certaines boîtes d'ancrage se chevauchent les unes les autres.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Enfin, nous [**réduisons encore de moitié la hauteur et la largeur de la carte des caractéristiques et augmentons l'échelle des boîtes d'ancrage à 0,8**]. Maintenant, le centre de la boîte d'ancrage est le centre de l'image.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Détection multi-échelle


Puisque nous avons généré des boîtes d'ancrage multi-échelle,
nous allons les utiliser pour détecter des objets de différentes tailles
à différentes échelles.
Dans ce qui suit,
 nous présentons une méthode de détection d'objets multi-échelle
basée sur CNN que nous mettrons en œuvre
dans :numref:`sec_ssd`.

À une certaine échelle,
disons que nous avons $c$ des cartes de caractéristiques de forme $h \times w$.
En utilisant la méthode décrite dans :numref:`subsec_multiscale-anchor-boxes`,
nous générons $hw$ ensembles de boîtes d'ancrage,
où chaque ensemble a $a$ boîtes d'ancrage avec le même centre.
Par exemple, 
à la première échelle dans les expériences de :numref:`subsec_multiscale-anchor-boxes`,
étant donné dix (nombre de canaux) $4 \times 4$ cartes de caractéristiques,
nous avons généré 16 ensembles de boîtes d'ancrage,
où chaque ensemble contient 3 boîtes d'ancrage avec le même centre.
Ensuite, chaque boîte d'ancrage est étiquetée avec
la classe et le décalage basé sur les boîtes de délimitation de la réalité du terrain. À l'échelle actuelle, le modèle de détection d'objets doit prédire les classes et les décalages de $hw$ ensembles de boîtes d'ancrage sur l'image d'entrée, où les différents ensembles ont des centres différents.


Supposons que les cartes de caractéristiques $c$ ici
sont les sorties intermédiaires obtenues
par la propagation avant CNN basée sur l'image d'entrée. Puisqu'il existe $hw$ différentes positions spatiales sur chaque carte de caractéristiques,
la même position spatiale peut être 
considérée comme ayant $c$ unités.
Selon la définition de
du champ réceptif dans :numref:`sec_conv_layer`,
ces $c$ unités à la même position spatiale
des cartes de caractéristiques
ont le même champ réceptif sur l'image d'entrée :
elles représentent les informations de l'image d'entrée
dans le même champ réceptif.
Par conséquent, nous pouvons transformer les unités $c$
des cartes de caractéristiques à la même position spatiale
en classes et décalages
des boîtes d'ancrage $a$
générées en utilisant cette position spatiale.
En substance,
nous utilisons les informations de l'image d'entrée dans un certain champ réceptif
pour prédire les classes et les décalages des boîtes d'ancrage
qui sont
proches de ce champ réceptif
sur l'image d'entrée.


Lorsque les cartes de caractéristiques des différentes couches
ont des champs réceptifs de tailles différentes sur l'image d'entrée, elles peuvent être utilisées pour détecter des objets de tailles différentes.
Par exemple, nous pouvons concevoir un réseau neuronal dans lequel les unités
des cartes de caractéristiques les plus proches de la couche de sortie
ont des champs réceptifs plus larges,
de sorte qu'elles peuvent détecter des objets plus grands à partir de l'image d'entrée.

En bref, nous pouvons exploiter
les représentations en couches d'images à plusieurs niveaux
par des réseaux neuronaux profonds
pour la détection d'objets à plusieurs échelles.
Nous allons montrer comment cela fonctionne à travers un exemple concret
dans :numref:`sec_ssd`.




## Résumé

* À plusieurs échelles, nous pouvons générer des boîtes d'ancrage de différentes tailles pour détecter des objets de différentes tailles.
* En définissant la forme des cartes de caractéristiques, nous pouvons déterminer les centres des boîtes d'ancrage uniformément échantillonnées sur n'importe quelle image.
* Nous utilisons les informations de l'image d'entrée dans un certain champ réceptif pour prédire les classes et les décalages des boîtes d'ancrage qui sont proches de ce champ réceptif sur l'image d'entrée.
* Grâce à l'apprentissage profond, nous pouvons exploiter ses représentations en couches des images à plusieurs niveaux pour la détection d'objets multi-échelles.


## Exercices

1. D'après nos discussions sur :numref:`sec_alexnet`, les réseaux neuronaux profonds apprennent des caractéristiques hiérarchiques avec des niveaux d'abstraction croissants pour les images. Dans la détection d'objets multi-échelles, les cartes de caractéristiques à différentes échelles correspondent-elles à différents niveaux d'abstraction ? Pourquoi ou pourquoi pas ?
1. À la première échelle (`fmap_w=4, fmap_h=4`), les expériences présentées dans :numref:`subsec_multiscale-anchor-boxes` génèrent des boîtes d'ancrage uniformément distribuées qui peuvent se chevaucher.
1. Étant donné une variable de carte de caractéristiques de forme $1 \times c \times h \times w$, où $c$, $h$ et $w$ sont le nombre de canaux, la hauteur et la largeur des cartes de caractéristiques, respectivement. Comment pouvez-vous transformer cette variable en classes et décalages de boîtes d'ancrage ? Quelle est la forme de la sortie ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
