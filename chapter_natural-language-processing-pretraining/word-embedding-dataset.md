# Le jeu de données pour le pré-entraînement des incorporations de mots
:label:`sec_word2vec_data` 

 Maintenant que nous connaissons les détails techniques de 
les modèles word2vec et les méthodes d'entraînement approximatives,
nous allons passer en revue leurs implémentations. 
Plus précisément,
nous prendrons comme exemple le modèle de saut de programme dans :numref:`sec_word2vec` 
 et l'échantillonnage négatif dans :numref:`sec_approx_train` 
 .
Dans cette section,
nous commençons par l'ensemble de données
pour le pré-entraînement du modèle d'intégration des mots :
le format original des données
sera transformé
en minibatchs
qui pourront être itérés pendant l'entraînement.

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## Lecture du jeu de données

Le jeu de données que nous utilisons ici
est [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42). 
Ce corpus est échantillonné
à partir d'articles du Wall Street Journal,
divisé en ensembles de formation, de validation et de test.
Dans le format original,
chaque ligne du fichier texte
représente une phrase de mots séparés par des espaces.
Ici, nous traitons chaque mot comme un jeton.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

Après avoir lu l'ensemble d'entraînement,
nous construisons un vocabulaire pour le corpus,
où tout mot qui apparaît 
moins de 10 fois est remplacé par 
le jeton "&lt;unk&gt;".
Notez que l'ensemble de données original
contient également des jetons "&lt;unk&gt;" qui représentent des mots rares (inconnus).

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Sous-échantillonnage

Les données textuelles
contiennent généralement des mots à haute fréquence
tels que " the ", " a " et " in " :
ils peuvent même apparaître des milliards de fois dans
de très grands corpus.
Cependant,
ces mots cooccurrent souvent
avec de nombreux mots différents dans les fenêtres de contexte
, fournissant ainsi peu de signaux utiles.
Par exemple,
considérons le mot "chip" dans une fenêtre de contexte :
intuitivement,
sa cooccurrence avec un mot à basse fréquence "intel"
est plus utile pour l'apprentissage
que 
la cooccurrence avec un mot à haute fréquence "a".
En outre, l'apprentissage avec de grandes quantités de mots (à haute fréquence)
est lent.
Ainsi, lors de la formation de modèles d'intégration de mots, 
les mots à haute fréquence peuvent être *sous-échantillonnés* :cite:`Mikolov.Sutskever.Chen.ea.2013` .
Plus précisément, 
chaque mot indexé $w_i$ 
 dans l'ensemble de données sera écarté avec une probabilité de


 $$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$ 

 où $f(w_i)$ est le rapport entre 
le nombre de mots $w_i$
 et le nombre total de mots dans l'ensemble de données, 
et la constante $t$ est un hyperparamètre
($10^{-4}$ dans l'expérience). 
Nous pouvons voir que ce n'est que lorsque
la fréquence relative
$f(w_i) > t$ que le mot (à haute fréquence) $w_i$ peut être écarté, 
et que plus la fréquence relative du mot, 
est élevée, plus la probabilité d'être écarté est grande.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `jeton` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

L'extrait de code suivant 
trace l'histogramme de
le nombre de tokens par phrase
avant et après le sous-échantillonnage.
Comme prévu, le sous-échantillonnage 
raccourcit considérablement les phrases
en supprimant les mots à haute fréquence,
ce qui accélère la formation.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Pour les tokens individuels, le taux d'échantillonnage du mot à haute fréquence "the" est inférieur à 1/20.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

En revanche, les mots à basse fréquence "join" de 
sont entièrement conservés.

```{.python .input}
#@tab all
compare_counts('join')
```

Après le sous-échantillonnage, nous faisons correspondre les tokens à leurs indices dans le corpus.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Extraction des mots centraux et des mots de contexte


 La fonction suivante `get_centers_and_contexts`
 extrait tous les 
mots centraux et leurs mots de contexte
de `corpus`.
Elle échantillonne uniformément un entier entre 1 et `max_window_size`
 au hasard comme taille de la fenêtre de contexte.
Pour tout mot central,
les mots 
dont la distance par rapport à lui
ne dépasse pas la taille de la fenêtre contextuelle
échantillonnée
sont ses mots contextuels.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Ensuite, nous créons un ensemble de données artificielles contenant deux phrases de 7 et 3 mots, respectivement. 
La taille maximale de la fenêtre de contexte est de 2 
et nous imprimons tous les mots centraux et leurs mots de contexte.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

Lors de l'entraînement sur le jeu de données PTB,
, nous fixons la taille maximale de la fenêtre de contexte à 5. 
Ce qui suit extrait tous les mots centraux et leurs mots contextuels dans l'ensemble de données.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Echantillonnage négatif

Nous utilisons l'échantillonnage négatif pour l'entraînement approximatif. 
Pour échantillonner les mots de bruit selon 
une distribution prédéfinie,
nous définissons la classe suivante `RandomGenerator`,
où la distribution d'échantillonnage (éventuellement non normalisée) est transmise
via l'argument `sampling_weights`.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Par exemple, 
nous pouvons tirer 10 variables aléatoires $X$
 parmi les indices 1, 2 et 3
avec des probabilités d'échantillonnage $P(X=1)=2/9, P(X=2)=3/9$, et $P(X=3)=4/9$ comme suit.

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Pour une paire de mot central et de mot de contexte, 
nous échantillonnons aléatoirement `K` (5 dans l'expérience) mots de bruit. Conformément aux suggestions du document word2vec,
la probabilité d'échantillonnage $P(w)$ de 
un mot de bruit $w$
 est 
fixé à sa fréquence relative 
dans le dictionnaire
élevé à 
la puissance de 0,75 :cite:`Mikolov.Sutskever.Chen.ea.2013` .

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Chargement des exemples d'entraînement en minibatchs
:label:`subsec_word2vec-minibatch-loading` 

 Après que
tous les mots centraux
ainsi que leurs
mots de contexte et mots de bruit échantillonnés aient été extraits,
ils seront transformés en 
minibatchs d'exemples
qui peuvent être chargés itérativement
pendant l'entraînement.



Dans un minibatch,
l'exemple $i^\mathrm{th}$ comprend un mot central
et ses mots de contexte $n_i$ et mots de bruit $m_i$. 
En raison des différentes tailles de fenêtre de contexte,
$n_i+m_i$ varie pour différents $i$.
Ainsi,
pour chaque exemple
, nous concaténons ses mots de contexte et ses mots de bruit dans 
la variable `contexts_negatives`,
et remplissons de zéros jusqu'à ce que la longueur de concaténation
atteigne $\max_i n_i+m_i$ (`max_len`).
Pour exclure les paddings
dans le calcul de la perte,
nous définissons une variable masque `masks`.
Il existe une correspondance biunivoque
entre les éléments de `masks` et les éléments de `contexts_negatives`,
, où les zéros (sinon les uns) de `masks` correspondent aux remplissages de `contexts_negatives`.


Pour distinguer les exemples positifs des exemples négatifs,
nous séparons les mots de contexte des mots de bruit dans `contexts_negatives` via une variable `labels`. 
Comme pour `masks`,
il existe également une correspondance biunivoque
entre les éléments de `labels` et les éléments de `contexts_negatives`,
où les uns (sinon les zéros) de `labels` correspondent aux mots de contexte (exemples positifs) de `contexts_negatives`.


L'idée ci-dessus est mise en œuvre dans la fonction `batchify` suivante.
Son entrée `data` est une liste dont la longueur
est égale à la taille du lot,
où chaque élément est un exemple
composé de
le mot central `center`, ses mots de contexte `context`, et ses mots de bruit `negative`.
Cette fonction renvoie 
un mini-batch qui peut être chargé pour des calculs 
pendant la formation,
comme l'inclusion de la variable de masque.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Testons cette fonction en utilisant un minibatch de deux exemples.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Putting All Things Together

Enfin, nous définissons la fonction `load_data_ptb` qui lit l'ensemble de données PTB et renvoie l'itérateur de données et le vocabulaire.

```{.python .input}
#@tab mxnet
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

Imprimons le premier minibatch de l'itérateur de données.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Résumé

* Les mots à haute fréquence peuvent ne pas être très utiles pour la formation. Nous pouvons les sous-échantillonner pour accélérer la formation.
* Pour l'efficacité du calcul, nous chargeons les exemples en minibatchs. Nous pouvons définir d'autres variables pour distinguer les paddings des non-paddings, et les exemples positifs des négatifs.



## Exercices

1. Comment le temps d'exécution du code de cette section change-t-il si on n'utilise pas le sous-échantillonnage ?
1. La classe `RandomGenerator` met en cache les résultats de l'échantillonnage aléatoire de `k`. Donnez d'autres valeurs à `k` et voyez comment cela affecte la vitesse de chargement des données.
1. Quels autres hyperparamètres dans le code de cette section peuvent affecter la vitesse de chargement des données ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:
