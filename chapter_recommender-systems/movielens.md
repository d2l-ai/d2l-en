# The MovieLens Dataset

Il existe un certain nombre de jeux de données disponibles pour la recherche sur les recommandations. Parmi eux, le jeu de données [MovieLens](https://movielens.org/) est probablement l'un des plus populaires. MovieLens est un système de recommandation de films non commercial basé sur le Web. Il a été créé en 1997 et est géré par GroupLens, un laboratoire de recherche de l'Université du Minnesota, dans le but de recueillir des données de classement de films à des fins de recherche.  Les données de MovieLens ont été essentielles pour plusieurs études de recherche, notamment sur la recommandation personnalisée et la psychologie sociale.


## Obtenir les données


 Le jeu de données MovieLens est hébergé par le site [GroupLens](https://grouplens.org/datasets/movielens/). Plusieurs versions sont disponibles. Nous utiliserons le jeu de données MovieLens 100K :cite:`Herlocker.Konstan.Borchers.ea.1999` .  Ce jeu de données est composé de $100,000$ évaluations, allant de 1 à 5 étoiles, de 943 utilisateurs sur 1682 films. Il a été nettoyé de manière à ce que chaque utilisateur ait évalué au moins 20 films. Quelques informations démographiques simples telles que l'âge, le sexe, les genres pour les utilisateurs et les articles sont également disponibles.  Nous pouvons télécharger le dossier [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) et extraire le fichier `u.data`, qui contient toutes les évaluations de $100,000$ au format csv. Il y a beaucoup d'autres fichiers dans le dossier, une description détaillée de chaque fichier peut être trouvée dans le fichier [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) de l'ensemble de données.

Pour commencer, nous importons les paquets nécessaires à l'exécution des expériences de cette section.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

Ensuite, nous téléchargeons le jeu de données MovieLens 100k et chargeons les interactions sous `DataFrame`.

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## Statistiques de l'ensemble de données

Chargeons les données et inspectons manuellement les cinq premiers enregistrements. C'est un moyen efficace d'apprendre la structure des données et de vérifier qu'elles ont été chargées correctement.

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

Nous pouvons voir que chaque ligne est constituée de quatre colonnes, dont "user id" 1-943, "item id" 1-1682, "rating" 1-5 et "timestamp". Nous pouvons construire une matrice d'interaction de taille $n \times m$, où $n$ et $m$ sont respectivement le nombre d'utilisateurs et le nombre d'articles. Cet ensemble de données n'enregistre que les évaluations existantes, nous pouvons donc l'appeler matrice d'évaluation et nous utiliserons indifféremment matrice d'interaction et matrice d'évaluation dans le cas où les valeurs de cette matrice représentent des évaluations exactes. La plupart des valeurs de la matrice d'évaluation sont inconnues car les utilisateurs n'ont pas évalué la majorité des films. Nous montrons également la sparsité de cet ensemble de données. La sparsité est définie comme `1 - number of nonzero entries / ( number of users * number of items)`. Il est clair que la matrice d'interaction est extrêmement éparse (sparsité = 93,695 %). Les ensembles de données du monde réel peuvent souffrir d'une plus grande sparsité, ce qui constitue un défi de longue date pour la création de systèmes de recommandation. Une solution viable consiste à utiliser des informations secondaires supplémentaires, telles que les caractéristiques de l'utilisateur et de l'article, pour atténuer la sparsité.

Nous traçons ensuite la distribution du nombre d'évaluations différentes. Comme prévu, il s'agit d'une distribution normale, avec la plupart des évaluations centrées sur 3-4.

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## Division de l'ensemble de données

Nous divisons l'ensemble de données en ensembles de formation et de test. La fonction suivante propose deux modes de fractionnement :`random` et `seq-aware`. Dans le mode `random`, la fonction fractionne les 100 000 interactions de manière aléatoire sans tenir compte de l'horodatage et utilise par défaut 90 % des données comme échantillons d'entraînement et les 10 % restants comme échantillons de test. Dans le mode `seq-aware`, nous laissons de côté l'élément qu'un utilisateur a évalué le plus récemment pour le test, et les interactions historiques des utilisateurs comme ensemble de formation.  Les interactions historiques des utilisateurs sont triées du plus ancien au plus récent en fonction de l'horodatage. Ce mode sera utilisé dans la section sur la recommandation sensible aux séquences.

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

Notez qu'il est bon d'utiliser un ensemble de validation dans la pratique, en plus du seul ensemble de test. Cependant, nous l'omettons pour des raisons de concision. Dans ce cas, notre ensemble de test peut être considéré comme notre ensemble de validation retenu.

## Chargement des données

Après avoir divisé l'ensemble de données, nous allons convertir l'ensemble de formation et l'ensemble de test en listes et dictionnaires/matrices pour des raisons de commodité. La fonction suivante lit le cadre de données ligne par ligne et énumère l'index des utilisateurs/articles à partir de zéro. La fonction renvoie ensuite des listes d'utilisateurs, d'articles, d'évaluations et un dictionnaire/matrice qui enregistre les interactions. Nous pouvons spécifier le type de rétroaction à `explicit` ou `implicit`.

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Ensuite, nous assemblons les étapes ci-dessus et elles seront utilisées dans la section suivante. Les résultats sont enveloppés dans `Dataset` et `DataLoader`. Notez que le `last_batch` de `DataLoader` pour les données de formation est réglé sur le mode `rollover` (Les échantillons restants sont reportés à l'époque suivante.) et les ordres sont mélangés.

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## Résumé

* Les jeux de données MovieLens sont largement utilisés pour la recherche sur les recommandations. Ils sont disponibles publiquement et peuvent être utilisés gratuitement.
* Nous définissons des fonctions pour télécharger et prétraiter le jeu de données MovieLens 100k pour une utilisation ultérieure dans les sections suivantes.


## Exercices

* Quels autres jeux de données de recommandation similaires pouvez-vous trouver ?
* Parcourez le site [https://movielens.org/](https://movielens.org/) pour obtenir plus d'informations sur MovieLens.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/399)
:end_tab:
