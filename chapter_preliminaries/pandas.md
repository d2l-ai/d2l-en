```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Prétraitement des données
:label:`sec_pandas` 

 Jusqu'à présent, nous avons travaillé avec des données synthétiques
qui arrivaient dans des tenseurs prêts à l'emploi.
Cependant, pour appliquer l'apprentissage profond dans la nature
, nous devons extraire des données désordonnées 
stockées dans des formats arbitraires,
et les prétraiter pour répondre à nos besoins.
Heureusement, le programme *pandas* [library](https://pandas.pydata.org/) 
 peut faire une grande partie de ce travail.
Cette section, bien qu'elle ne remplace pas 
une *pandas* [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html),
vous donnera un cours intensif
sur certaines des routines les plus courantes.


## Lire l'ensemble de données

Les fichiers CSV (Comma-separated values) sont omniprésents 
pour stocker des données tabulaires (de type feuille de calcul).
Ici, chaque ligne correspond à un enregistrement
et se compose de plusieurs champs (séparés par des virgules), par exemple,
"Albert Einstein,14 mars 1879,Ulm,École polytechnique fédérale,Réalisations dans le domaine de la physique gravitationnelle".
Pour démontrer comment charger des fichiers CSV avec `pandas`, 
nous (**créons un fichier CSV ci-dessous**) `../data/house_tiny.csv`. 
Ce fichier représente un ensemble de données de maisons,
où chaque ligne correspond à une maison distincte
et les colonnes correspondent au nombre de pièces (`NumRooms`),
au type de toit (`RoofType`), et au prix (`Price`).

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Importons maintenant `pandas` et chargeons le jeu de données avec `read_csv`.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Préparation des données

Dans le cadre de l'apprentissage supervisé, nous formons des modèles
pour prédire une valeur *cible* désignée,
étant donné un ensemble de valeurs *d'entrée*. 
La première étape du traitement de l'ensemble de données
consiste à séparer les colonnes correspondant
aux valeurs d'entrée et aux valeurs cibles. 
Nous pouvons sélectionner les colonnes soit par leur nom, soit par
via une indexation basée sur l'emplacement des entiers (`iloc`).

Vous avez peut-être remarqué que `pandas` a remplacé
toutes les entrées CSV par la valeur `NA`
 par une valeur spéciale `NaN` (*not a number*). 
Cela peut également se produire lorsqu'une entrée est vide,
par exemple, "3,,,,270000".
C'est ce qu'on appelle les *valeurs manquantes* 
et ce sont les "punaises de lit" de la science des données,
une menace persistante à laquelle vous serez confronté
tout au long de votre carrière. 
Selon le contexte, 
les valeurs manquantes peuvent être traitées
soit par *imputation* soit par *suppression*.
L'imputation remplace les valeurs manquantes 
par des estimations de leurs valeurs
tandis que la suppression élimine simplement 
soit les lignes, soit les colonnes
qui contiennent des valeurs manquantes. 

Voici quelques heuristiques d'imputation courantes.
[**Pour les champs de saisie catégoriels, 
nous pouvons traiter `NaN` comme une catégorie.**]
Puisque la colonne `RoofType` prend les valeurs `Slate` et `NaN`,
`pandas` peut convertir cette colonne 
en deux colonnes `RoofType_Slate` et `RoofType_nan`.
Une ligne dont le type d'allée est `Slate` donnera aux valeurs 
de `RoofType_Slate` et `RoofType_nan` la valeur 1 et 0, respectivement.
L'inverse est vrai pour une ligne dont la valeur `RoofType` est manquante.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Pour les valeurs numériques manquantes, 
une heuristique courante consiste à 
[**remplacer les entrées `NaN` par 
la valeur moyenne de la colonne correspondante**].

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Conversion au format tenseur

Maintenant que [**toutes les entrées de `inputs` et `targets` sont numériques,
nous pouvons les charger dans un tenseur**] (rappelons :numref:`sec_ndarray` ).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.values), np.array(targets.values)
X, y
```

```{.python .input}
%%tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(targets.values)
X, y
```

## Discussion

Vous savez maintenant comment partitionner des colonnes de données, 
imputer des variables manquantes, 
et charger `pandas` des données dans des tenseurs. 
Dans :numref:`sec_kaggle_house` , vous allez
acquérir d'autres compétences en matière de traitement des données. 
Bien que ce cours accéléré ait gardé les choses simples, le traitement des données
peut être difficile.
Par exemple, au lieu d'arriver dans un seul fichier CSV,
notre ensemble de données peut être réparti sur plusieurs fichiers
extraits d'une base de données relationnelle.
Par exemple, dans une application de commerce électronique,
les adresses des clients peuvent se trouver dans une table
et les données d'achat dans une autre.
En outre, les praticiens sont confrontés à une myriade de types de données
autres que catégoriques et numériques. 
Parmi les autres types de données figurent les chaînes de texte, les images, les données audio
et les nuages de points. 
Souvent, des outils avancés et des algorithmes efficaces 
sont nécessaires pour éviter que le traitement des données ne devienne
le principal goulot d'étranglement du pipeline d'apprentissage automatique. 
Ces problèmes se poseront lorsque nous aborderons la vision par ordinateur 
et le traitement du langage naturel. 
Enfin, nous devons prêter attention à la qualité des données.
Les ensembles de données du monde réel sont souvent entachés 
de valeurs aberrantes, de mesures erronées provenant de capteurs et d'erreurs d'enregistrement, 
. Ces problèmes doivent être résolus avant 
d'intégrer les données dans un modèle. 
Les outils de visualisation de données tels que [seaborn](https://seaborn.pydata.org/), 
[Bokeh](https://docs.bokeh.org/) , ou [matplotlib](https://matplotlib.org/)
 peuvent vous aider à inspecter manuellement les données 
et à développer des intuitions sur 
les problèmes que vous devrez peut-être résoudre.


## Exercices

1. Essayez de charger des ensembles de données, par exemple, Abalone à partir du site [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) et inspectez leurs propriétés. Quelle fraction d'entre eux a des valeurs manquantes ? Quelle fraction des variables est numérique, catégorique ou textuelle ?
1. Essayez d'indexer et de sélectionner les colonnes de données par leur nom plutôt que par leur numéro. La documentation de pandas sur [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) contient plus de détails sur la façon de procéder.
1. Quelle taille d'ensemble de données pensez-vous pouvoir charger de cette manière ? Quelles pourraient être les limitations ? Conseil : tenez compte du temps de lecture des données, de la représentation, du traitement et de l'empreinte mémoire. Essayez cette méthode sur votre ordinateur portable. Qu'est-ce qui change si vous l'essayez sur un serveur ? 
1. Comment traiteriez-vous des données comportant un très grand nombre de catégories ? Que se passe-t-il si les étiquettes des catégories sont toutes uniques ? Devriez-vous inclure ces derniers ?
1. Quelles alternatives aux pandas pouvez-vous imaginer ? Que pensez-vous de [loading NumPy tensors from a file](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Consultez [Pillow](https://python-pillow.org/), la bibliothèque d'imagerie Python 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:
