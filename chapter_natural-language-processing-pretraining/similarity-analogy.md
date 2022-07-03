# Similitude et analogie des mots
:label:`sec_synonyms` 

Dans :numref:`sec_word2vec_pretraining`, 
nous avons entraîné un modèle word2vec sur un petit ensemble de données, 
et l'avons appliqué
pour trouver des mots sémantiquement similaires 
pour un mot d'entrée.
Dans la pratique, les vecteurs de mots
qui sont pré-entraînés
sur de grands corpus peuvent être
appliqués à des tâches de traitement du langage naturel
en aval,
qui seront traitées ultérieurement
dans :numref:`chap_nlp_app`.
Pour démontrer de manière directe la sémantique de 
des vecteurs de mots pré-entraînés
à partir de grands corpus,
appliquons-les
dans les tâches de similarité et d'analogie de mots.

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

## Chargement des vecteurs de mots prétraités

Vous trouverez ci-dessous une liste d'embeddings GloVe prétraités de dimension 50, 100 et 300,
qui peuvent être téléchargés à partir du site [GloVe website](https://nlp.stanford.edu/projects/glove/).
Les embeddings fastText prétraités sont disponibles en plusieurs langues.
Nous considérons ici une version anglaise (300 dimensions "wiki.en") qui peut être téléchargée sur le site
[fastText website](https://fasttext.cc/) .

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Pour charger ces embeddings GloVe et fastText prétraités, nous définissons la classe suivante `TokenEmbedding`.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Ci-dessous, nous chargeons les 50 encastrements GloVe dimensionnels

 (pré-entraînés sur un sous-ensemble de Wikipédia).
Lors de la création de l'instance `TokenEmbedding`,
le fichier d'intégration spécifié doit être téléchargé si
ne l'a pas encore été.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Sortie de la taille du vocabulaire. Le vocabulaire contient 400000 mots (tokens) et un token inconnu spécial.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Nous pouvons obtenir l'index d'un mot dans le vocabulaire, et vice versa.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Application des vecteurs de mots pré-entraînés

En utilisant les vecteurs GloVe chargés,
nous allons démontrer leur sémantique
en les appliquant
dans les tâches suivantes de similarité et d'analogie de mots.


### Similarité des mots

Comme pour :numref:`subsec_apply-word-embed`,
afin de trouver des mots sémantiquement similaires
pour un mot d'entrée
en se basant sur les similarités en cosinus entre
vecteurs de mots,
nous implémentons la fonction suivante `knn`
 ($k$-voisins les plus proches).

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Ensuite, nous recherchons 
des mots similaires
en utilisant les vecteurs de mots préformés 
de l'instance `TokenEmbedding` `embed`.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

Le vocabulaire des vecteurs de mots pré-entraînés
dans `glove_6b50d` contient 400 000 mots et un jeton inconnu spécial. 
En excluant le mot d'entrée et le jeton inconnu,
parmi ce vocabulaire,
trouvons 
les trois mots les plus sémantiquement similaires
au mot "chip".

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Les sorties ci-dessous contiennent des mots similaires
à "bébé" et "belle".

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Analogie de mots

Outre la recherche de mots similaires,
nous pouvons également appliquer les vecteurs de mots
aux tâches d'analogie de mots.
Par exemple,
"homme":“woman”::“son”: "fille"
est la forme d'une analogie de mots :
"homme" est à "femme" comme "fils" est à "fille".
Plus précisément,
la tâche d'achèvement de l'analogie de mots
peut être définie comme suit :
pour une analogie de mots 
$a : b :: c : d$étant donné les trois premiers mots $a$, $b$ et $c$, trouvez $d$. 
On désigne le vecteur du mot $w$ par $\text{vec}(w)$. 
Pour compléter l'analogie,
nous allons trouver le mot 
dont le vecteur est le plus similaire
au résultat de $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Vérifions l'analogie "homme-femme" en utilisant les vecteurs des mots chargés.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Nous complétons ci-dessous une analogie "capitale-pays"
: 
"beijing":“china”::“tokyo”: "japan".
Ceci démontre la sémantique de 
dans les vecteurs de mots pré-entraînés.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Pour l'analogie
"adjectif-superlatif"
tel que 
"mauvais":“worst”::“big”: "plus grand",
nous pouvons voir que les vecteurs de mots prétraités
peuvent capturer l'information syntaxique.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Pour montrer la notion capturée
de passé dans les vecteurs de mots prétraités,
nous pouvons tester la syntaxe en utilisant l'analogie "présent-passé"
: "faire":“did”::“go”: "aller".

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Résumé

* En pratique, les vecteurs de mots pré-entraînés sur de grands corpus peuvent être appliqués à des tâches de traitement du langage naturel en aval.
* Les vecteurs de mots pré-entraînés peuvent être appliqués aux tâches de similarité et d'analogie de mots.


## Exercices

1. Testez les résultats de fastText en utilisant `TokenEmbedding('wiki.en')`.
1. Lorsque le vocabulaire est extrêmement vaste, comment pouvons-nous trouver des mots similaires ou réaliser une analogie de mots plus rapidement ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
