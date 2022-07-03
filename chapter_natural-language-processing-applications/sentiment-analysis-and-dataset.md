# L'analyse des sentiments et l'ensemble de données
:label:`sec_sentiment` 

 
 Avec la prolifération des médias sociaux en ligne
et des plateformes d'évaluation,
une pléthore de
données d'opinion
a été enregistrée,
présentant un grand potentiel pour
soutenir les processus de prise de décision.
*L'analyse des sentiments*
étudie les sentiments des gens
dans les textes qu'ils produisent,
tels que les critiques de produits,
les commentaires de blogs,
et
les discussions de forums.
Elle trouve de nombreuses applications
dans des domaines aussi divers que 
la politique (par exemple, l'analyse des sentiments du public à l'égard des politiques),
la finance (par exemple, l'analyse des sentiments du marché),
et 
le marketing (par exemple, la recherche de produits et la gestion de la marque).

Puisque les sentiments
peuvent être classés
comme des polarités ou des échelles discrètes (par exemple, positif et négatif),
nous pouvons considérer 
l'analyse des sentiments 
comme une tâche de classification de texte,
qui transforme une séquence de texte de longueur variable
en une catégorie de texte de longueur fixe.
Dans ce chapitre,
nous utiliserons le logiciel [large movie review dataset](https://ai.stanford.edu/~)amaas/data/sentiment/)
 de Stanford pour l'analyse des sentiments. 
Il se compose d'un ensemble d'entraînement et d'un ensemble de test, 
, contenant chacun 25 000 critiques de films téléchargées depuis IMDb.
 
 Dans les deux ensembles de données, 
, il y a un nombre égal d'étiquettes "positives" et "négatives",
indiquant différentes polarités de sentiment.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Lecture du jeu de données

Tout d'abord, téléchargez et extrayez ce jeu de données de critiques IMDb
dans le chemin `../data/aclImdb`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Ensuite, lisez les jeux de données de entrainement et de test. Chaque exemple est une critique et son étiquette : 1 pour "positif" et 0 pour "négatif".

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## Prétraitement du jeu de données

En traitant chaque mot comme un jeton
et en filtrant les mots qui apparaissent moins de 5 fois,
nous créons un vocabulaire à partir du jeu de données d'entraînement.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Après la tokénisation,
traçons l'histogramme de
la longueur des commentaires en tokens.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Comme nous nous y attendions,
les commentaires ont des longueurs variables.
Pour traiter
un mini lot de ces commentaires à chaque fois,
nous fixons la longueur de chaque commentaire à 500 avec troncature et remplissage,
ce qui est similaire à 
l'étape de prétraitement 
pour le jeu de données de traduction automatique
dans :numref:`sec_machine_translation` .

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Création d'itérateurs de données

Nous pouvons maintenant créer des itérateurs de données.
À chaque itération, un minibatch d'exemples est renvoyé.

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Rassembler le tout

Enfin, nous regroupons les étapes précédentes dans la fonction `load_data_imdb`.
Elle renvoie les itérateurs de données d'entraînement et de test ainsi que le vocabulaire de l'ensemble de données de critiques IMDb.

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Résumé

* L'analyse des sentiments étudie les sentiments des personnes dans leur texte produit, ce qui est considéré comme un problème de classification de texte qui transforme une séquence de texte de longueur variable
en une catégorie de texte de longueur fixe.
* Après le prétraitement, nous pouvons charger le grand ensemble de données de critiques de films de Stanford (IMDb review dataset) dans des itérateurs de données avec un vocabulaire.


## Exercices


 1. Quels hyperparamètres de cette section pouvons-nous modifier pour accélérer l'entrainement des modèles d'analyse de sentiments ?
1. Pouvez-vous implémenter une fonction pour charger le jeu de données de [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html) dans des itérateurs de données et des étiquettes pour l'analyse des sentiments ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:
