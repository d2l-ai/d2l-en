# L'inférence en langage naturel et le jeu de données
:label:`sec_natural-language-inference-and-dataset` 

Dans :numref:`sec_sentiment`, nous avons abordé le problème de l'analyse des sentiments.
Cette tâche vise à classer une séquence de texte unique dans des catégories prédéfinies,
telles qu'un ensemble de polarités de sentiments.
Cependant, lorsqu'il est nécessaire de décider si une phrase peut être déduite d'une autre, 
ou d'éliminer la redondance en identifiant les phrases qui sont sémantiquement équivalentes,
savoir comment classer une séquence de texte est insuffisant.
Au lieu de cela, nous devons être capables de raisonner sur des paires de séquences de texte.


## Inférence en langage naturel

*L'inférence en langage naturel* étudie si une *hypothèse*
peut être déduite d'une *prémisse*, où les deux sont une séquence de texte.
En d'autres termes, l'inférence en langage naturel détermine la relation logique entre une paire de séquences de texte.
Ces relations sont généralement de trois types :

* *Entaillement* : l'hypothèse peut être déduite de la prémisse.
* *Contradiction* : la négation de l'hypothèse peut être déduite de la prémisse.
* *Neutre* : tous les autres cas.

L'inférence en langage naturel est également connue sous le nom de tâche de reconnaissance de l'entaillement textuel.
Par exemple, la paire suivante sera étiquetée comme *entailment* car "montrer de l'affection" dans l'hypothèse peut être déduit de "s'étreindre" dans la prémisse.

&gt; Prémisse : Deux femmes se font des câlins.

&gt; Hypothèse : Deux femmes se montrent affectueuses.

L'exemple suivant est un exemple de *contradiction* car "courir l'exemple de codage" indique "ne pas dormir" plutôt que "dormir".

&gt; Prémisse : un homme exécute l'exemple de codage de Dive into Deep Learning.

&gt; Hypothèse : L'homme dort.

Le troisième exemple montre une relation de *neutralité* car ni "célèbre" ni "pas célèbre" ne peuvent être déduits du fait que "se produisent pour nous". 

&gt; Prémisse : Les musiciens se produisent pour nous.

&gt; Hypothèse : Les musiciens sont célèbres.

L'inférence en langage naturel est un sujet central pour la compréhension du langage naturel.
Ses applications sont nombreuses, allant de la recherche d'informations sur
à la réponse à des questions dans un domaine ouvert.
Pour étudier ce problème, nous commencerons par examiner un ensemble de données de référence populaire sur l'inférence en langage naturel.


## Le Stanford Natural Language Inference (SNLI) Dataset

[**Stanford Natural Language Inference (SNLI) Corpus**] est une collection de plus de 500000 paires de phrases anglaises étiquetées :cite:`Bowman.Angeli.Potts.ea.2015`.
Nous téléchargeons et stockons le jeu de données SNLI extrait dans le chemin `../data/snli_1.0`.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**Lire le jeu de données**]

Le jeu de données SNLI original contient des informations beaucoup plus riches que celles dont nous avons réellement besoin dans nos expériences. Ainsi, nous définissons une fonction `read_snli` pour n'extraire qu'une partie du jeu de données, puis renvoyer des listes de prémisses, d'hypothèses et leurs étiquettes.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Maintenant, [**imprimons les 3 premières paires**] de prémisses et d'hypothèses, ainsi que leurs étiquettes ("0", "1", et "2" correspondent à "implication", "contradiction", et "neutre", respectivement).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

L'ensemble d'apprentissage compte environ 550000 paires,
et l'ensemble de test compte environ 10000 paires.
Le tableau suivant montre que 
les trois [**labels "entailment", "contradiction", et "neutral" sont équilibrés**] dans 
à la fois dans l'ensemble d'entraînement et dans l'ensemble de test.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**Définir une classe pour charger le jeu de données**]

Nous définissons ci-dessous une classe pour charger le jeu de données SNLI en héritant de la classe `Dataset` de Gluon. L'argument `num_steps` dans le constructeur de la classe spécifie la longueur d'une séquence de texte afin que chaque minibloc de séquences ait la même forme. 
En d'autres termes, les tokens
après les premiers `num_steps` dans une séquence plus longue sont coupés, tandis que les tokens spéciaux "&lt;pad&gt;" seront ajoutés aux séquences plus courtes jusqu'à ce que leur longueur devienne `num_steps`.
En implémentant la fonction `__getitem__`, nous pouvons accéder arbitrairement aux prémisses, aux hypothèses et aux étiquettes avec l'index `idx`.

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**Putting All Things Together**]

Nous pouvons maintenant invoquer la fonction `read_snli` et la classe `SNLIDataset` pour télécharger l'ensemble de données SNLI et renvoyer `DataLoader` instances pour les ensembles de entrainement et de test, ainsi que le vocabulaire de l'ensemble de formation.
Il convient de noter que nous devons utiliser le vocabulaire construit à partir de l'ensemble d'apprentissage
comme celui de l'ensemble de test. 
Par conséquent, tout nouveau token provenant de l'ensemble de test sera inconnu du modèle formé sur l'ensemble de formation.

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Ici, nous définissons la taille du lot à 128 et la longueur de la séquence à 50,
et invoquons la fonction `load_data_snli` pour obtenir les itérateurs de données et le vocabulaire.
Ensuite, nous imprimons la taille du vocabulaire.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Maintenant nous imprimons la forme du premier minibatch.
Contrairement à l'analyse des sentiments,
nous avons deux entrées `X[0]` et `X[1]` représentant des paires de prémisses et d'hypothèses.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Résumé

* L'inférence en langage naturel étudie si une hypothèse peut être déduite d'une prémisse, où les deux sont une séquence de texte.
* Dans l'inférence en langage naturel, les relations entre les prémisses et les hypothèses comprennent l'implication, la contradiction et la neutralité.
* Le corpus Stanford Natural Language Inference (SNLI) est un jeu de données de référence populaire pour l'inférence en langage naturel.


## Exercices

1. La traduction automatique a longtemps été évaluée sur la base de la correspondance superficielle $n$-gram entre une traduction de sortie et une traduction de référence. Pouvez-vous concevoir une mesure pour évaluer les résultats de la traduction automatique en utilisant l'inférence en langage naturel ?
1. Comment pouvons-nous modifier les hyperparamètres pour réduire la taille du vocabulaire ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:
