# Séquences textuelles
:label:`sec_text-sequence` 

Nous avons examiné et évalué
les outils statistiques
et les défis de la prédiction
pour les données de séquence.
Ces données peuvent prendre de nombreuses formes.
Plus précisément,
car nous nous concentrerons sur
dans de nombreux chapitres du livre,
le texte est l'un des exemples les plus populaires de données de séquence.
Par exemple,
un article peut être simplement vu comme une séquence de mots, ou même une séquence de caractères.
Pour faciliter nos futures expériences
avec des données de séquence,
nous consacrerons cette section
à expliquer les étapes courantes de prétraitement du texte.
Habituellement, ces étapes sont les suivantes :

1. Charger le texte en tant que chaînes de caractères en mémoire.
1. Diviser les chaînes de caractères en jetons (par exemple, mots et caractères).
1. Construire une table de vocabulaire pour faire correspondre les tokens fractionnés à des indices numériques.
1. Convertir le texte en séquences d'indices numériques afin qu'il puisse être manipulé facilement par des modèles.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Lecture de l'ensemble de données

Pour commencer, nous chargeons le texte 
de l'ouvrage de H. G. Wells [*The Time Machine*](http://www.gutenberg.org/ebooks/35).
Il s'agit d'un corpus assez petit d'un peu plus de 30 000 mots, mais pour ce que nous voulons illustrer, c'est très bien. Des collections de documents plus réalistes contiennent plusieurs milliards de mots.
La fonction suivante 
(**lit le texte brut en une chaîne de caractères**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root, 
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

## Prétraitement

Pour simplifier, nous ignorons la ponctuation et les majuscules lors du prétraitement du texte brut.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenization


*Les tokens* sont les unités atomiques (indivisibles) du texte
et ce qui constitue un token 
(par exemple, des caractères ou des mots)
est un choix de conception.
Bien qu'il soit issu du traitement du langage naturel,
le concept de token devient également populaire dans le domaine de la vision par ordinateur,
 
par exemple pour faire référence à des patchs d'images :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Ci-dessous, nous convertissons notre texte prétraité en caractères.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Vocabulaire

Alors que ces tokens sont toujours des chaînes de caractères,
nos modèles nécessitent des entrées numériques.
[**Ainsi, nous aurons besoin d'une classe
pour construire un *vocabulaire*
qui attribue un index unique 
à chaque token distinct.**]
À cette fin,
nous comptons d'abord les tokens uniques dans tous les documents de l'ensemble d'entraînement, à savoir un *corpus*,
et attribuons ensuite un index numérique à chaque token unique.
Les tokens qui apparaissent rarement sont souvent supprimés pour réduire la complexité. Tout token qui n'existe pas dans le corpus ou qui a été supprimé est mappé dans un token inconnu spécial "&lt;unk&gt;". 
Dans le futur,
nous pourrons compléter le vocabulaire
avec une liste de tokens réservés.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)        
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]        
    
    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

Nous pouvons maintenant [**construire le vocabulaire**] pour notre ensemble de données, 
en l'utilisant pour convertir une séquence de texte
en une liste d'indices numériques.
Notez que nous n'avons perdu aucune information
et que nous pouvons facilement reconvertir notre ensemble de données 
en sa représentation originale (chaîne de caractères).

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Assemblage de tout

En utilisant les classes et les méthodes ci-dessus, nous [**empaquetons tout dans la méthode suivante `build` de la classe `TimeMachine` **], qui renvoie `corpus`, une liste d'indices de tokens, et `vocab`, le vocabulaire du corpus *The Time Machine*.
Les modifications que nous avons apportées ici sont les suivantes :
(i) nous tokenisons le texte en caractères, et non en mots, pour simplifier l'entraînement dans les sections suivantes ;
(ii) `corpus` est une liste unique, et non une liste de listes de tokens, puisque chaque ligne de texte dans le jeu de données *The Time Machine* n'est pas nécessairement une phrase ou un paragraphe.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):    
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Statistiques du langage naturel
:label:`subsec_natural-lang-stat` 

 
En utilisant le corpus réel et la classe `Vocab`
définie ci-dessus,
étudions également les statistiques des tokens de mots
.
Nous construisons un vocabulaire basé sur le corpus *The Time Machine* et imprimons les 10 mots les plus fréquents.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Comme nous pouvons le constater, (**les mots les plus populaires sont**) en fait assez ennuyeux à regarder.
Ils sont souvent appelés (***mots d'arrêt***) et sont donc filtrés.
Néanmoins, ils sont toujours porteurs de sens et nous les utiliserons toujours.
En outre, il est évident que la fréquence des mots décroît assez rapidement. Le mot $10^{\mathrm{th}}$ le plus fréquent est moins de $1/5$ aussi fréquent que le mot le plus populaire. Pour avoir une meilleure idée, nous [**traçons la figure de la fréquence des mots**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Nous sommes sur quelque chose de tout à fait fondamental ici : la fréquence des mots décroît rapidement d'une manière bien définie.
Après avoir considéré les premiers mots comme des exceptions, tous les autres mots suivent approximativement une ligne droite sur un graphique log-log. Cela signifie que les mots satisfont à la *loi de Zipf*,
qui stipule que la fréquence $n_i$ du mot $i^\mathrm{th}$ le plus fréquent
est :

$$n_i \propto \frac{1}{i^\alpha},$$ 
:eqlabel:`eq_zipf_law` 

ce qui est équivalent à

$$\log n_i = -\alpha \log i + c,$$ 

où $\alpha$ est l'exposant qui caractérise la distribution et $c$ est une constante.
Cela devrait déjà nous faire réfléchir si nous voulons modéliser les mots par des statistiques de comptage.
Après tout, nous allons considérablement surestimer la fréquence de la queue, également connue sous le nom de mots peu fréquents. Mais [**qu'en est-il des autres combinaisons de mots, comme deux mots consécutifs (bigrammes), trois mots consécutifs (trigrammes)**], et au-delà ?
Voyons si la fréquence des bigrammes se comporte de la même manière que la fréquence des mots simples (unigrammes).

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Une chose est notable ici. Sur les dix paires de mots les plus fréquentes, neuf sont composées des deux mots d'arrêt et une seule est pertinente pour le livre en question : "l'heure". En outre, voyons si la fréquence des trigrammes se comporte de la même manière.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Enfin, visualisons [**la fréquence des jetons**] parmi ces trois modèles : unigrammes, bigrammes et trigrammes.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Cette figure est très intéressante.
Premièrement, au-delà des mots unigrammes, les séquences de mots semblent également suivre la loi de Zipf, bien qu'avec un exposant plus petit $\alpha$ in :eqref:`eq_zipf_law`, selon la longueur de la séquence.
Deuxièmement, le nombre de programmes distincts $n$ n'est pas très élevé. Cela nous permet d'espérer qu'il y a beaucoup de structure dans le langage.
Troisièmement, de nombreux $n$-grammes apparaissent très rarement.
Cela rend certaines méthodes inadaptées à la modélisation du langage et motive l'utilisation de modèles d'apprentissage profond.
Nous en discuterons dans la section suivante.


## Résumé

* Le texte est une forme importante de données de séquence.
* Pour prétraiter le texte, on le divise généralement en jetons, on construit un vocabulaire pour convertir les chaînes de jetons en indices numériques et on convertit les données textuelles en indices de jetons pour que les modèles puissent les manipuler.
* La loi de Zipf régit la distribution des mots non seulement pour les unigrammes mais aussi pour les autres $n$-grams.


## Exercices

1. La tokenisation est une étape clé du prétraitement. Elle varie selon les langues. Essayez de trouver trois autres méthodes couramment utilisées pour tokeniser un texte.
1. Dans l'expérience de cette section, tokenisez le texte en mots et faites varier la valeur de l'argument `min_freq` de l'instance `Vocab`. Comment cela affecte-t-il la taille du vocabulaire ?
1. Estimez l'exposant de la loi de Zipf pour les unigrammes, les bigrammes et les trigrammes.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
