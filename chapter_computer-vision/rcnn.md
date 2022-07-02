# Les CNN basés sur les régions (R-CNN)
:label:`sec_rcnn` 

 Outre la détection multi-boîtes à un seul coup
décrite dans :numref:`sec_ssd` ,
les CNN basés sur les régions ou les régions avec des caractéristiques CNN (R-CNN)
font également partie des nombreuses approches pionnières
de 
appliquant
l'apprentissage profond à la détection d'objets
:cite:`Girshick.Donahue.Darrell.ea.2014` .
Dans cette section, nous allons présenter
le R-CNN et sa série d'améliorations : le R-CNN rapide
:cite:`Girshick.2015` , le R-CNN plus rapide :cite:`Ren.He.Girshick.ea.2015` , et le R-CNN de masque
:cite:`He.Gkioxari.Dollar.ea.2017` .
En raison de l'espace limité, nous nous concentrerons uniquement sur la conception de ces modèles
.



## R-CNNs


 Le *R-CNN* extrait d'abord
de nombreuses (par exemple, 2000) *propositions de régions*
de l'image d'entrée
(par exemple, les boîtes d'ancrage peuvent également être considérées
comme des propositions de régions),
en étiquetant leurs classes et leurs boîtes limites (par exemple, les décalages).
:cite:`Girshick.Donahue.Darrell.ea.2014`
Ensuite, un CNN est utilisé pour 
effectuer une propagation vers l'avant sur chaque proposition de région
pour extraire ses caractéristiques.
Ensuite, les caractéristiques de chaque proposition de région
sont utilisées pour
prédire la classe et la boîte englobante
de cette proposition de région.


 ![The R-CNN model.](../img/r-cnn.svg) 
 :label:`fig_r-cnn` 

 :numref:`fig_r-cnn` montre le modèle R-CNN. Plus concrètement, le R-CNN comprend les quatre étapes suivantes :

1. Effectuer une *recherche sélective* pour extraire plusieurs propositions de régions de haute qualité sur l'image d'entrée :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013` . Ces régions proposées sont généralement sélectionnées à plusieurs échelles avec des formes et des tailles différentes. Chaque proposition de région sera étiquetée avec une classe et une boîte de délimitation de vérité.
1. Choisissez un CNN pré-entraîné et tronquez-le avant la couche de sortie. Redimensionnez chaque proposition de région à la taille d'entrée requise par le réseau, et produisez les caractéristiques extraites pour la proposition de région par propagation directe. 
1. Prenez les caractéristiques extraites et la classe étiquetée de chaque proposition de région comme exemple. Entraîner plusieurs machines à vecteurs de support pour classer les objets, où chaque machine à vecteurs de support détermine individuellement si l'exemple contient une classe spécifique.
1. Prenez les caractéristiques extraites et la boîte de délimitation étiquetée de chaque proposition de région comme exemple. Entraînez un modèle de régression linéaire pour prédire le rectangle de délimitation de la vérité du terrain.


Bien que le modèle R-CNN utilise des CNN pré-entraînés pour extraire efficacement les caractéristiques des images, 
est lent.
Imaginez que nous sélectionnions
des milliers de propositions de régions à partir d'une seule image d'entrée :
cela nécessite des milliers de propagations vers l'avant de
CNN pour effectuer la détection d'objets.
Cette charge de calcul massive
rend infaisable l'utilisation à grande échelle des R-CNN dans les applications du monde réel
.

## R-CNN rapide

Le principal goulot d'étranglement des performances de 
un R-CNN réside dans
la propagation avant CNN indépendante
pour chaque proposition de région, 
sans partage des calculs.
Comme ces régions ont généralement des chevauchements
, les extractions de caractéristiques indépendantes
conduisent à
beaucoup de calculs répétés.
L'une des principales améliorations de 
le *R-CNN rapide* par rapport au R-CNN
est que 
la propagation vers l'avant du CNN
est uniquement effectuée sur 
l'image entière :cite:`Girshick.2015` . 

![The fast R-CNN model.](../img/fast-rcnn.svg) 
 :label:`fig_fast_r-cnn` 

 :numref:`fig_fast_r-cnn` décrit le modèle R-CNN rapide. Ses principaux calculs sont les suivants :


 1. Par rapport au R-CNN, dans le R-CNN rapide, l'entrée du CNN pour l'extraction de caractéristiques est l'image entière, plutôt que des propositions de régions individuelles. De plus, ce CNN est entraînable. Étant donné une image d'entrée, la forme de la sortie du CNN est $1 \times c \times h_1  \times w_1$.
1. Supposons que la recherche sélective génère $n$ propositions de régions. Ces propositions de régions (de formes différentes) marquent des régions d'intérêt (de formes différentes) sur la sortie du CNN. Ces régions d'intérêt extraient ensuite des caractéristiques de la même forme (disons que la hauteur $h_2$ et la largeur $w_2$ sont spécifiées) afin d'être facilement concaténées. Pour ce faire, le R-CNN rapide introduit la couche de mise en commun * des régions d'intérêt (RoI) : la sortie du CNN et les propositions de régions sont introduites dans cette couche, qui produit des caractéristiques concaténées de forme $n \times c \times h_2 \times w_2$ qui sont ensuite extraites pour toutes les propositions de régions.
1. En utilisant une couche entièrement connectée, transformez les caractéristiques concaténées en une sortie de forme $n \times d$, où $d$ dépend de la conception du modèle.
1. Prédire la classe et la boîte englobante pour chacune des propositions de régions $n$. Plus concrètement, dans la prédiction de la classe et de la zone limite, transformez la sortie de la couche entièrement connectée en une sortie de forme $n \times q$ ($q$ est le nombre de classes) et une sortie de forme $n \times 4$, respectivement. La prédiction de classe utilise la régression softmax.


La couche de mise en commun de la région d'intérêt proposée dans le R-CNN rapide est différente de la couche de mise en commun introduite dans :numref:`sec_pooling` . 
Dans la couche de mise en commun,
nous contrôlons indirectement la forme de la sortie
en spécifiant les tailles de
la fenêtre de mise en commun, le remplissage et le pas.
En revanche,
nous pouvons spécifier directement la forme de sortie
dans la couche de mise en commun des régions d'intérêt.

Par exemple, spécifions pour
la hauteur et la largeur de sortie 
pour chaque région, respectivement $h_2$ et $w_2$.
Pour toute fenêtre de région d'intérêt
de forme $h \times w$,
cette fenêtre est divisée en une grille $h_2 \times w_2$
 de sous-fenêtres,
où la forme de chaque sous-fenêtre est approximativement 
$(h/h_2) \times (w/w_2)$ .
Dans la pratique,
la hauteur et la largeur de toute sous-fenêtre sont arrondies vers le haut, et l'élément le plus grand est utilisé comme sortie de la sous-fenêtre.
Par conséquent, la couche de mise en commun des régions d'intérêt peut extraire des caractéristiques de la même forme 
même lorsque les régions d'intérêt ont des formes différentes.


A titre d'exemple,
dans :numref:`fig_roi` ,
la région d'intérêt $3\times 3$ supérieure gauche 
est sélectionnée sur une entrée $4 \times 4$.
Pour cette région d'intérêt,
, nous utilisons une couche de regroupement de régions d'intérêt $2\times 2$ pour obtenir
et $2\times 2$ en sortie.
Notez que 
chacune des quatre sous-fenêtres divisées
contient les éléments
0, 1, 4 et 5 (5 est le maximum) ;
2 et 6 (6 est le maximum) ;
8 et 9 (9 est le maximum) ;
et 10.

![A $2\times 2$ region of interest pooling layer.](../img/roi.svg)
:label:`fig_roi`

Nous démontrons ci-dessous le calcul de la couche de mise en commun des régions d'intérêt. Supposons que la hauteur et la largeur des caractéristiques extraites par CNN `X` soient toutes deux de 4, et qu'il n'y ait qu'un seul canal.

```{.python .input}
#@tab mxnet
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

Supposons également que
que la hauteur et la largeur de l'image d'entrée sont toutes deux de 40 pixels et que la recherche sélective génère deux propositions de régions sur cette image.
Chaque proposition de région
est exprimée en cinq éléments :
sa classe d'objet suivie des coordonnées $(x, y)$ de ses coins supérieur gauche et inférieur droit.

```{.python .input}
#@tab mxnet
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

Comme la hauteur et la largeur de `X` sont $1/10$ de la hauteur et de la largeur de l'image d'entrée,
les coordonnées des deux propositions de régions
sont multipliées par 0,1 selon l'argument spécifié `spatial_scale`.
Ensuite, les deux régions d'intérêt sont marquées sur `X` comme `X[:, :, 0:3, 0:3]` et `X[:, :, 1:4, 0:4]`respectivement. 
Enfin, dans le regroupement des régions d'intérêt $2\times 2$,
chaque région d'intérêt est divisée
en une grille de sous-fenêtres pour
extraire davantage de caractéristiques de la même forme $2\times 2$.

```{.python .input}
#@tab mxnet
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

## R-CNN rapide

Pour être plus précis dans la détection d'objets,
le modèle R-CNN rapide
doit généralement générer 
un grand nombre de propositions de régions dans la recherche sélective.
Pour réduire les propositions de régions
sans perte de précision,
le *R-CNN plus rapide*
propose de remplacer la recherche sélective par un *réseau de propositions de régions* :cite:`Ren.He.Girshick.ea.2015` .



 ![The faster R-CNN model.](../img/faster-rcnn.svg) 
 :label:`fig_faster_r-cnn` 

 
 :numref:`fig_faster_r-cnn` montre le modèle R-CNN plus rapide. Par rapport au modèle R-CNN rapide,
le modèle R-CNN rapide ne modifie que
la méthode de proposition de région
de la recherche sélective à un réseau de proposition de région.
Le reste du modèle reste inchangé
.
Le réseau de proposition de région 
fonctionne selon les étapes suivantes :

1. Utilisez une couche convolutive $3\times 3$ avec un padding de 1 pour transformer la sortie du CNN en une nouvelle sortie avec $c$ canaux. De cette façon, chaque unité le long des dimensions spatiales des cartes de caractéristiques extraites par le CNN obtient un nouveau vecteur de caractéristiques de longueur $c$.
1. Centré sur chaque pixel des cartes de caractéristiques, générez plusieurs boîtes d'ancrage de différentes échelles et rapports d'aspect et étiquetez-les.
1. En utilisant le vecteur caractéristique de longueur$c$ au centre de chaque boîte d'ancrage, prédisez la classe binaire (arrière-plan ou objets) et la boîte de délimitation pour cette boîte d'ancrage.
1. Considérez les boîtes de délimitation prédites dont les classes prédites sont des objets. Supprimez les résultats superposés en utilisant la suppression non maximale. Les boîtes englobantes restantes prédites pour les objets constituent les propositions de régions requises par la couche de regroupement des régions d'intérêt.



Il convient de noter que, 
dans le cadre du modèle R-CNN plus rapide,
le réseau de propositions de régions
est entraîné conjointement
avec le reste du modèle. 
En d'autres termes, la fonction objective de 
le R-CNN plus rapide comprend
non seulement la prédiction de la classe et de la boîte de délimitation
dans la détection d'objets,
mais aussi la prédiction binaire de la classe et de la boîte de délimitation
des boîtes d'ancrage dans le réseau de proposition de région.
Grâce à l'apprentissage de bout en bout,
le réseau de propositions de régions apprend
à générer des propositions de régions de haute qualité,
afin de rester précis dans la détection d'objets
avec un nombre réduit de propositions de régions
qui sont apprises à partir de données.




## Masque R-CNN

Dans l'ensemble de données d'apprentissage,
si les positions des objets au niveau des pixels 
sont également étiquetées sur les images, 
le *masque R-CNN* peut efficacement exploiter
ces étiquettes détaillées 
pour améliorer encore la précision de la détection des objets :cite:`He.Gkioxari.Dollar.ea.2017` .


![The mask R-CNN model.](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

Comme le montre :numref:`fig_mask_r-cnn` , 
le masque R-CNN
est modifié sur la base du R-CNN plus rapide. 
Plus précisément,
le masque R-CNN remplace la couche de mise en commun de la région d'intérêt
par la couche d'alignement* de la région d'intérêt (RdI)
*. 
Cette couche d'alignement de la région d'intérêt
utilise l'interpolation bilinéaire
pour préserver les informations spatiales sur les cartes de caractéristiques, ce qui convient mieux à la prédiction au niveau du pixel.
La sortie de cette couche
contient des cartes de caractéristiques de la même forme
pour toutes les régions d'intérêt. 
Ces cartes sont utilisées à l'adresse
pour prédire à l'adresse 
non seulement la classe et la boîte englobante de chaque région d'intérêt,
, mais aussi la position de l'objet au niveau du pixel grâce à un réseau supplémentaire entièrement convolutif.
Plus de détails sur l'utilisation d'un réseau entièrement convolutif pour prédire la sémantique d'une image au niveau du pixel 
seront fournis
dans les sections suivantes de ce chapitre.




## Résumé


 * Le R-CNN extrait de nombreuses propositions de régions de l'image d'entrée, utilise un CNN pour effectuer une propagation vers l'avant sur chaque proposition de région afin d'en extraire les caractéristiques, puis utilise ces caractéristiques pour prédire la classe et la boîte englobante de cette proposition de région.
* L'une des principales améliorations du R-CNN rapide par rapport au R-CNN est que la propagation directe du CNN n'est effectuée que sur l'image entière. Il introduit également la couche de regroupement des régions d'intérêt, de sorte que les caractéristiques de la même forme peuvent être extraites pour les régions d'intérêt qui ont des formes différentes.
* Le R-CNN plus rapide remplace la recherche sélective utilisée dans le R-CNN rapide par un réseau de propositions de régions entraîné conjointement, de sorte que le premier peut rester précis dans la détection d'objets avec un nombre réduit de propositions de régions.
* Basé sur le R-CNN rapide, le R-CNN de masque introduit en plus un réseau entièrement convolutif, afin d'exploiter les étiquettes au niveau des pixels pour améliorer encore la précision de la détection des objets.


## Exercices

1. Peut-on concevoir la détection d'objets comme un problème de régression unique, tel que la prédiction des boîtes englobantes et des probabilités de classe ? Vous pouvez vous référer à la conception du modèle YOLO :cite:`Redmon.Divvala.Girshick.ea.2016` .
1. Comparez la détection multiboxes à un seul coup avec les méthodes présentées dans cette section. Quelles sont leurs principales différences ? Vous pouvez vous référer à la figure 2 de :cite:`Zhao.Zheng.Xu.ea.2019` .

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/374)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1409)
:end_tab:
