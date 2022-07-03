# Détection d'objets et boîtes englobantes
:label:`sec_bbox` 

 
Dans les sections précédentes (par exemple, :numref:`sec_alexnet` --:numref:`sec_googlenet` ),
nous avons présenté divers modèles de classification d'images.
Dans les tâches de classification d'images,
nous supposons qu'il n'y a qu'un *un*
objet principal
dans l'image et nous nous concentrons uniquement sur la manière de 
reconnaître sa catégorie.
Cependant, il y a souvent *plusieurs* objets
dans l'image d'intérêt.
Nous voulons non seulement connaître leurs catégories, mais aussi leurs positions spécifiques dans l'image.
Dans le domaine de la vision par ordinateur, nous appelons ces tâches la *détection d'objets* (ou la *reconnaissance d'objets*).

La détection d'objets a été
largement appliquée dans de nombreux domaines.
Par exemple, la conduite autonome doit planifier 
des itinéraires de déplacement
en détectant les positions
des véhicules, des piétons, des routes et des obstacles dans les images vidéo capturées.
Par ailleurs,
les robots peuvent utiliser cette technique
pour détecter et localiser des objets d'intérêt
tout au long de leur navigation dans un environnement.
De plus, les systèmes de sécurité

peuvent avoir besoin de détecter des objets anormaux, tels que des intrus ou des bombes.

Dans les prochaines sections, nous allons présenter 
plusieurs méthodes d'apprentissage profond pour la détection d'objets.
Nous commencerons par une introduction
aux *positions* (ou *emplacements*) des objets.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Nous allons charger l'image d'exemple qui sera utilisée dans cette section. Nous pouvons voir qu'il y a un chien sur le côté gauche de l'image et un chat sur le côté droit.
Ce sont les deux principaux objets de cette image.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Boîtes englobantes


En détection d'objets,
nous utilisons généralement une *boîte englobante* pour décrire l'emplacement spatial d'un objet.
La boîte englobante est rectangulaire et est déterminée par les coordonnées $x$ et $y$ du coin supérieur gauche du rectangle et par les coordonnées du coin inférieur droit. 
Une autre représentation couramment utilisée de la boîte englobante est constituée des coordonnées $(x, y)$-axe
du centre de la boîte englobante, ainsi que de la largeur et de la hauteur de la boîte.

[**Nous définissons ici des fonctions pour convertir entre**] ces (**deux représentations
**) :
`box_corner_to_center` convertit la représentation à deux coins
en la présentation centre-largeur-hauteur,
et `box_center_to_corner` vice versa.
L'argument d'entrée `boxes` doit être un tenseur bidimensionnel de la forme
($n$, 4), où $n$ est le nombre de boîtes englobantes.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Nous allons [**définir les boîtes englobantes du chien et du chat dans l'image**] en nous basant sur
sur les informations de coordonnées.
L'origine des coordonnées dans l'image
est le coin supérieur gauche de l'image, et à droite et en bas sont les directions positives
des axes $x$ et $y$, respectivement.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Nous pouvons vérifier l'exactitude des deux fonctions de conversion de boîte englobante
en effectuant deux conversions.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Dessinons [**les boîtes englobantes dans l'image**] pour vérifier leur exactitude.
Avant de dessiner, nous allons définir une fonction d'aide `bbox_to_rect`. Elle représente la boîte englobante dans le format de boîte englobante du paquet `matplotlib`.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Après avoir ajouté les boîtes englobantes sur l'image,
nous pouvons voir que le contour principal des deux objets se trouve essentiellement à l'intérieur des deux boîtes.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Résumé

* La détection d'objets ne reconnaît pas seulement tous les objets d'intérêt dans l'image, mais aussi leurs positions. La position est généralement représentée par une boîte de délimitation rectangulaire.
* Nous pouvons convertir entre deux représentations de boîte englobante couramment utilisées.

## Exercices

1. Trouvez une autre image et essayez d'étiqueter une boîte englobante qui contient l'objet. Comparez l'étiquetage des boîtes englobantes et des catégories : lequel prend le plus de temps ?
1. Pourquoi la dimension la plus intérieure de l'argument d'entrée `boxes` de `box_corner_to_center` et `box_center_to_corner` est-elle toujours 4 ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
