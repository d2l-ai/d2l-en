# Systèmes de recommandation riches en fonctionnalités

Les données d'interaction constituent l'indication la plus fondamentale des préférences et des intérêts des utilisateurs. Elles jouent un rôle essentiel dans les anciens modèles présentés. Pourtant, les données d'interaction sont généralement très éparses et peuvent parfois être bruitées. Pour résoudre ce problème, nous pouvons intégrer dans le modèle de recommandation des informations secondaires telles que les caractéristiques des articles, les profils des utilisateurs et même le contexte dans lequel l'interaction a eu lieu. L'utilisation de ces caractéristiques est utile pour faire des recommandations, car elles peuvent être un prédicteur efficace des intérêts des utilisateurs, en particulier lorsque les données d'interaction font défaut. En tant que tel, il est essentiel que les modèles de recommandation aient également la capacité de traiter ces caractéristiques et de donner au modèle une certaine conscience du contenu/contexte. Pour démontrer ce type de modèles de recommandation, nous introduisons une autre tâche sur le taux de clics (CTR) pour les recommandations de publicité en ligne :cite:`McMahan.Holt.Sculley.ea.2013` et présentons un ensemble de données de publicité anonyme. Les services de publicité ciblée ont suscité une grande attention et sont souvent considérés comme des moteurs de recommandation. La recommandation de publicités qui correspondent aux goûts et aux intérêts personnels des utilisateurs est importante pour améliorer le taux de clics.


Les spécialistes du marketing numérique utilisent la publicité en ligne pour afficher des annonces aux clients. Le taux de clics est un indicateur qui mesure le nombre de clics que les annonceurs reçoivent sur leurs publicités par rapport au nombre d'impressions et il est exprimé en pourcentage calculé avec la formule : 

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

Le taux de clics est un signal important qui indique l'efficacité des algorithmes de prédiction. La prédiction du taux de clics est une tâche qui consiste à prédire la probabilité qu'un élément d'un site Web soit cliqué. Les modèles de prédiction du taux de clics peuvent être utilisés non seulement dans les systèmes de publicité ciblée, mais aussi dans les systèmes de recommandation d'articles généraux (par exemple, films, actualités, produits), les campagnes de courrier électronique et même les moteurs de recherche. Il est également étroitement lié à la satisfaction de l'utilisateur, au taux de conversion, et peut être utile pour définir les objectifs de la campagne, car il peut aider les annonceurs à fixer des attentes réalistes.

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Un ensemble de données sur la publicité en ligne

Avec les progrès considérables de l'Internet et de la technologie mobile, la publicité en ligne est devenue une importante source de revenus et génère la grande majorité des recettes dans l'industrie de l'Internet. Il est important d'afficher des publicités pertinentes ou des publicités qui suscitent l'intérêt des utilisateurs afin que les visiteurs occasionnels puissent être convertis en clients payants. Le jeu de données que nous avons introduit est un jeu de données sur la publicité en ligne. Il se compose de 34 champs, la première colonne représentant la variable cible qui indique si une publicité a été cliquée (1) ou non (0). Toutes les autres colonnes sont des caractéristiques catégorielles. Les colonnes peuvent représenter l'identifiant de la publicité, l'identifiant du site ou de l'application, l'identifiant du dispositif, l'heure, les profils des utilisateurs, etc. La sémantique réelle des caractéristiques n'est pas divulguée pour des raisons d'anonymat et de confidentialité.

Le code suivant télécharge l'ensemble de données depuis notre serveur et l'enregistre dans le dossier de données local.

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Il y a un ensemble d'entraînement et un ensemble de test, composés de 15000 et 3000 échantillons/lignes, respectivement.

## Enveloppeur de jeu de données

Pour faciliter le chargement des données, nous implémentons un `CTRDataset` qui charge le jeu de données publicitaires à partir du fichier CSV et peut être utilisé par `DataLoader`.

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

L'exemple suivant charge les données d'entraînement et imprime le premier enregistrement.

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Comme on peut le voir, les 34 champs sont des caractéristiques catégorielles. Chaque valeur représente l'indice à un coup de l'entrée correspondante. L'étiquette $0$ signifie qu'elle n'est pas cliquée. Ce `CTRDataset` peut également être utilisé pour charger d'autres jeux de données tels que le défi de la publicité par affichage de Criteo [dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) et la prédiction du taux de clics d'Avazu [dataset](https://www.kaggle.com/c/avazu-ctr-prediction). 

## Résumé 
* Le taux de clics est une mesure importante utilisée pour évaluer l'efficacité des systèmes publicitaires et des systèmes de recommandation.
* La prédiction du taux de clics est généralement convertie en un problème de classification binaire. L'objectif est de prédire si une annonce/un élément sera cliqué ou non en fonction de caractéristiques données.

## Exercices

* Pouvez-vous charger les jeux de données Criteo et Avazu avec le fichier fourni `CTRDataset`. Il est important de noter que le jeu de données Criteo est composé de caractéristiques à valeur réelle, il se peut donc que vous deviez réviser un peu le code.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:
