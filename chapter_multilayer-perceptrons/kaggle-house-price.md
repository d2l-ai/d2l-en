```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Prédire les prix des maisons sur Kaggle
:label:`sec_kaggle_house` 

Maintenant que nous avons présenté quelques outils de base
pour construire et former des réseaux profonds
et les régulariser avec des techniques telles que
weight decay and dropout,
nous sommes prêts à mettre toutes ces connaissances en pratique
en participant à une compétition Kaggle.
La compétition de prédiction du prix des maisons
est un excellent point de départ.
Les données sont assez génériques et ne présentent pas de structure exotique
qui pourrait nécessiter des modèles spécialisés (comme l'audio ou la vidéo).
Cet ensemble de données, collecté par Bart de Cock en 2011 :cite:`De-Cock.2011`,
couvre les prix des maisons à Ames, IA, sur la période 2006-2010.
Il est considérablement plus important que le célèbre [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) de Harrison et Rubinfeld (1978),
offrant à la fois plus d'exemples et plus de caractéristiques.


Dans cette section, nous vous guiderons dans les détails du prétraitement des données de,
de la conception du modèle et de la sélection des hyperparamètres.
Nous espérons qu'à travers une approche pratique,
vous acquerrez certaines intuitions qui vous guideront
dans votre carrière de scientifique des données.


## Téléchargement de données

Tout au long du livre, nous entraînerons et testerons des modèles
sur divers ensembles de données téléchargés.
Ici, nous (**implémentons deux fonctions utilitaires**)
pour télécharger des fichiers et extraire des fichiers zip ou tar.
Encore une fois, nous reportons leurs implémentations dans :numref:`sec_utils`.

```{.python .input  n=2}
%%tab all

def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""

def extract(filename, folder):
    """Extract a zip/tar file into folder."""
```

## Kaggle

[Kaggle](https://www.kaggle.com) est une plateforme populaire
qui accueille des compétitions d'apprentissage automatique.
Chaque concours est centré sur un ensemble de données et beaucoup
sont sponsorisés par des parties prenantes qui offrent des prix
aux solutions gagnantes.
La plateforme permet aux utilisateurs d'interagir
via des forums et du code partagé,
favorisant à la fois la collaboration et la compétition.
Bien que la chasse au classement soit souvent hors de contrôle,
avec des chercheurs qui se concentrent de manière myope sur les étapes de prétraitement
plutôt que de poser des questions fondamentales,
l'objectivité d'une plateforme
qui facilite les comparaisons quantitatives directes
entre les approches concurrentes ainsi que le partage du code
afin que chacun puisse apprendre ce qui a fonctionné et ce qui n'a pas fonctionné, présente également un intérêt considérable.
Si vous souhaitez participer à un concours Kaggle,
vous devez d'abord créer un compte
(voir :numref:`fig_kaggle` ).

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

Sur la page du concours de prédiction du prix des logements, comme illustré par
dans :numref:`fig_house_pricing`,
vous pouvez trouver l'ensemble de données (sous l'onglet "Données"),
soumettre des prédictions, et voir votre classement,
L'URL est ici :

&gt; https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house-pricing.png) 
:width:`400px` 
:label:`fig_house_pricing` 

## Accéder au jeu de données et le lire

Notez que les données du concours sont séparées
en jeux d'entraînement et de test.
Chaque enregistrement comprend la valeur foncière de la maison
et des attributs tels que le type de rue, l'année de construction,
le type de toit, l'état du sous-sol, etc.
Les caractéristiques sont constituées de divers types de données.
Par exemple, l'année de construction
est représentée par un nombre entier,
le type de toit par des affectations catégorielles discrètes,
et d'autres caractéristiques par des nombres à virgule flottante.
Et c'est là que la réalité complique les choses :
pour certains exemples, certaines données sont carrément manquantes
avec la valeur manquante marquée simplement comme "na".
Le prix de chaque maison est inclus
pour l'ensemble d'entraînement uniquement
(il s'agit d'un concours après tout).
Nous voudrons partitionner l'ensemble d'entraînement
pour créer un ensemble de validation,
mais nous ne pourrons évaluer nos modèles que sur l'ensemble de test officiel
après avoir téléchargé les prédictions sur Kaggle.
L'onglet "Données" de l'onglet de la compétition
dans :numref:`fig_house_pricing` 
a des liens pour télécharger les données.

```{.python .input  n=14}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

Pour commencer, nous allons [**lire et traiter les données en utilisant `pandas`**], que nous avons présenté dans :numref:`sec_pandas`.
Pour plus de commodité, nous pouvons télécharger et mettre en cache
l'ensemble de données sur les logements de Kaggle.
Si un fichier correspondant à cet ensemble de données existe déjà dans le répertoire de cache et que son SHA-1 correspond à `sha1_hash`, notre code utilisera le fichier en cache pour éviter d'encombrer votre Internet avec des téléchargements redondants.

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

Le jeu de données d'entraînement comprend 1460 exemples,
80 caractéristiques et 1 étiquette, tandis que les données de validation
contiennent 1459 exemples et 80 caractéristiques.

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## Prétraitement des données

Regardons [**les quatre premières et les deux dernières caractéristiques ainsi que l'étiquette (SalePrice)**] des quatre premiers exemples.

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

Nous pouvons voir que dans chaque exemple, la première caractéristique est l'ID.
Cela aide le modèle à identifier chaque exemple de formation.
Bien que cette caractéristique soit pratique, elle n'apporte aucune information à
à des fins de prédiction.
Par conséquent, nous la supprimerons de l'ensemble de données
avant d'introduire les données dans le modèle.
En outre, étant donné la grande variété de types de données,
nous devrons prétraiter les données avant de pouvoir commencer la modélisation.


Commençons par les caractéristiques numériques.
Tout d'abord, nous appliquons une heuristique,
[**remplaçant toutes les valeurs manquantes par la moyenne de la caractéristique correspondante.**]
Ensuite, pour placer toutes les caractéristiques sur une échelle commune,
nous (normalisons les données en
remettant à l'échelle les caractéristiques à moyenne nulle et variance unitaire) :

$$x \leftarrow \frac{x - \mu}{\sigma},$$ 

où $\mu$ et $\sigma$ désignent respectivement la moyenne et l'écart type.
Pour vérifier que cela transforme effectivement
notre caractéristique (variable) de sorte qu'elle ait une moyenne nulle et une variance unitaire,
notez que $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$
et que $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Intuitivement, nous normalisons les données
pour deux raisons.
Premièrement, cela s'avère pratique pour l'optimisation.
Deuxièmement, comme nous ne savons pas *a priori*
quelles caractéristiques seront pertinentes,
nous ne voulons pas pénaliser les coefficients
attribués à une caractéristique plus qu'à une autre.

(**Ensuite, nous traitons des valeurs discrètes.**)
Cela inclut des caractéristiques telles que "MSZoning".
( **Nous les remplaçons par un codage à un coup** )
de la même manière que nous avons précédemment transformé
les étiquettes multi-classes en vecteurs (voir :numref:`subsec_classification-problem` ).
Par exemple, "MSZoning" prend les valeurs "RL" et "RM".
En abandonnant la caractéristique "MSZoning",
deux nouvelles caractéristiques indicatrices
"MSZoning_RL" et "MSZoning_RM" sont créées avec des valeurs de 0 ou 1.
Selon le codage à un coup,
si la valeur originale de "MSZoning" est "RL",
alors "MSZoning_RL" est 1 et "MSZoning_RM" est 0.
Le paquet `pandas` fait cela automatiquement pour nous.

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding.
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

Vous pouvez voir que cette conversion fait passer
le nombre de caractéristiques de 79 à 331 (sans compter les colonnes ID et label).

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## Mesure d'erreur

Pour commencer, nous allons entraîner un modèle linéaire avec une perte au carré. Comme on pouvait s'y attendre, notre modèle linéaire ne mènera pas à une soumission gagnante du concours, mais il permet de vérifier si les données contiennent des informations significatives. Si nous ne pouvons pas faire mieux qu'une supposition aléatoire, il y a de fortes chances que nous ayons un bug dans le traitement des données. Et si les choses fonctionnent, le modèle linéaire servira de référence, nous donnant une idée de la proximité du modèle simple par rapport aux meilleurs modèles rapportés, nous donnant une idée du gain que nous devrions attendre de modèles plus sophistiqués.

Avec les prix de l'immobilier, comme avec les prix des actions,
nous nous intéressons aux quantités relatives
plus qu'aux quantités absolues.
Ainsi, [**nous avons tendance à nous soucier davantage de l'erreur relative $\frac{y - \hat{y}}{y}$**]
que de l'erreur absolue $y - \hat{y}$.
Par exemple, si notre prédiction est erronée de 100 000 USD
lors de l'estimation du prix d'une maison dans l'Ohio rural,
où la valeur d'une maison typique est de 125 000 USD,
alors nous faisons probablement un travail horrible.
En revanche, si nous nous trompons de ce montant
à Los Altos Hills, en Californie,
cela pourrait représenter une prédiction étonnamment précise
(là-bas, le prix médian des maisons dépasse 4 millions USD).

(**Une façon de résoudre ce problème est de mesurer l'écart dans le logarithme des estimations de prix.**)
En fait, c'est également la mesure d'erreur officielle
utilisée par le concours pour évaluer la qualité des soumissions.
Après tout, une petite valeur $\delta$ pour $|\log y - \log \hat{y}| \leq \delta$
se traduit par $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
Cela conduit à l'erreur quadratique moyenne suivante entre le logarithme du prix prédit et le logarithme du prix de l'étiquette :

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values, dtype=d2l.float32)
    # Logarithm of prices 
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$-Fold Cross-Validation

Vous vous souvenez peut-être que nous avons introduit [**cross-validation**]
dans :numref:`subsec_generalization-model-selection`, où nous avons discuté de la façon de traiter
avec la sélection de modèles.
Nous allons en faire bon usage pour sélectionner la conception du modèle
et pour ajuster les hyperparamètres.
Nous avons d'abord besoin d'une fonction qui renvoie
le pli $i^\mathrm{th}$ des données
dans une procédure de validation croisée $K$-fold.
Elle procède par découpage du segment $i^\mathrm{th}$
comme données de validation et renvoie le reste comme données de formation.
Notez que ce n'est pas la manière la plus efficace de traiter les données
et nous ferions certainement quelque chose de beaucoup plus intelligent
si notre ensemble de données était considérablement plus grand.
Mais cette complexité supplémentaire pourrait obscurcir inutilement notre code.
Nous pouvons donc l'omettre ici en toute sécurité en raison de la simplicité de notre problème.

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

[**L'erreur moyenne de validation est retournée**]
lorsque nous nous entraînons $K$ fois dans la validation croisée $K$-fold.

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**Sélection du modèle**]

Dans cet exemple, nous choisissons un ensemble non accordé d'hyperparamètres
et laissons au lecteur le soin d'améliorer le modèle.
Trouver un bon choix peut prendre du temps,
en fonction du nombre de variables sur lesquelles on optimise.
Avec un ensemble de données suffisamment grand,
et les types normaux d'hyperparamètres,
$K$ -fold cross-validation tend à être
raisonnablement résistant aux tests multiples.
Cependant, si nous essayons un nombre déraisonnable d'options,
 nous pouvons avoir de la chance et constater que notre performance de validation
n'est plus représentative de l'erreur réelle.

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

Remarquez que parfois le nombre d'erreurs de formation
pour un ensemble d'hyperparamètres peut être très faible,
même si le nombre d'erreurs sur $K$-fold cross-validation
est considérablement plus élevé.
Cela indique que l'ajustement est excessif.
Tout au long de la formation, vous voudrez surveiller ces deux chiffres.
Un surajustement (overfitting) moindre pourrait indiquer que nos données peuvent supporter un modèle plus puissant.
Un surajustement (overfitting) massif pourrait suggérer que nous pouvons gagner
en incorporant des techniques de régularisation.

## [**Soumettre des prédictions sur Kaggle**]

Maintenant que nous savons ce que devrait être un bon choix d'hyperparamètres,
nous pouvons 
calculer les prédictions moyennes 
sur l'ensemble de test
par tous les modèles $K$.
L'enregistrement des prédictions dans un fichier csv
simplifiera le téléchargement des résultats sur Kaggle.
Le code suivant va générer un fichier appelé `submission.csv`.

```{.python .input}
%%tab all
preds = [model(d2l.tensor(data.val.values, dtype=d2l.float32))
         for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

Ensuite, comme démontré dans :numref:`fig_kaggle_submit2`,
nous pouvons soumettre nos prédictions sur Kaggle
et voir comment elles se comparent aux prix réels des maisons (étiquettes)
sur l'ensemble de test.
Les étapes sont assez simples :

* Connectez-vous au site Web de Kaggle et visitez la page du concours de prédiction du prix des maisons.
* Cliquez sur le bouton "Submit Predictions" ou "Late Submission" (au moment de la rédaction de cet article, le bouton est situé à droite).
* Cliquez sur le bouton "Télécharger le fichier de soumission" dans la case en pointillés en bas de la page et sélectionnez le fichier de prédiction que vous souhaitez télécharger.
* Cliquez sur le bouton "Make Submission" en bas de la page pour afficher vos résultats.

![Submitting data to Kaggle](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Résumé

* Les données réelles contiennent souvent un mélange de différents types de données et doivent être prétraitées.
* La remise à l'échelle des données à valeurs réelles à une moyenne nulle et une variance unitaire est un bon défaut. Il en va de même pour le remplacement des valeurs manquantes par leur moyenne.
* Transformer les caractéristiques catégorielles en caractéristiques indicatrices nous permet de les traiter comme des vecteurs à un coup.
* Nous pouvons utiliser $K$-fold cross-validation pour sélectionner le modèle et ajuster les hyperparamètres.
* Les logarithmes sont utiles pour les erreurs relatives.


## Exercices

1. Soumettez vos prédictions pour cette section à Kaggle. Quelle est la qualité de vos prédictions ?
1. Est-ce toujours une bonne idée de remplacer les valeurs manquantes par leur moyenne ? Indice : pouvez-vous construire une situation où les valeurs ne sont pas manquantes au hasard ?
1. Améliorez le score sur Kaggle en ajustant les hyperparamètres par le biais de la validation croisée $K$.
1. Améliorez le score en améliorant le modèle (par exemple, les couches, la décroissance des poids et l'abandon).
1. Que se passe-t-il si nous ne normalisons pas les caractéristiques numériques continues comme nous l'avons fait dans cette section ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
