# Subword Embedding
:label:`sec_fasttext` 

 En anglais,
des mots tels que
"helps", "helped", et "helping" sont 
des formes infléchies du même mot "help".
La relation 
entre "dog" et "dogs"
est la même que 
celle entre "cat" et "cats",
et 
la relation 
entre "boy" et "boyfriend"
est la même que 
celle entre "girl" et "girlfriend".
Dans d'autres langues
comme le français et l'espagnol,
de nombreux verbes ont plus de 40 formes infléchies,
tandis qu'en finnois,
un nom peut avoir jusqu'à 15 cas.
En linguistique,
la morphologie étudie la formation des mots et leurs relations.
Cependant,
la structure interne des mots
n'a été explorée ni dans word2vec
ni dans GloVe.

## Le modèle fastText

Rappelons comment les mots sont représentés dans word2vec.
Dans le modèle de saut de programme
et le modèle de sac de mots continu,
différentes formes infléchies du même mot
sont directement représentées par différents vecteurs
sans paramètres partagés.
Pour utiliser les informations morphologiques,
le modèle *fastText*
a proposé une approche d'intégration de *sous-mots*,
où un sous-mot est un caractère $n$-gramme :cite:`Bojanowski.Grave.Joulin.ea.2017` .
Au lieu d'apprendre des représentations vectorielles au niveau des mots,
fastText peut être considéré comme
le saut-gramme au niveau des sous-mots,
où chaque *mot central* est représenté par la somme de 
ses vecteurs de sous-mots.

Illustrons comment obtenir 
des sous-mots pour chaque mot central dans fastText
en utilisant le mot "où".
Tout d'abord, ajoutez les caractères spéciaux "&lt;" et "&gt;" 
au début et à la fin du mot pour distinguer les préfixes et les suffixes des autres sous-mots. 
Ensuite, nous extrayons les -grammes de caractères $n$ du mot.
Par exemple, lorsque $n=3$,
nous obtenons tous les sous-mots de longueur 3 : "&lt;wh", "whe", "her", "ere", "re&gt;", et le sous-mot spécial "&lt;where&gt;".


Dans fastText, pour tout mot $w$,
désigne par $\mathcal{G}_w$
 l'union de tous ses sous-mots de longueur comprise entre 3 et 6
et son sous-mot spécial.
Le vocabulaire 
est l'union des sous-mots de tous les mots.
En laissant $\mathbf{z}_g$
 être le vecteur du sous-mot $g$ dans le dictionnaire,
le vecteur $\mathbf{v}_w$ pour le mot 
 $w$ comme mot central
dans le modèle skip-gram
est la somme de ses vecteurs de sous-mots :

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$ 

 Le reste de fastText est le même que le modèle skip-gram. Par rapport au modèle de saut de programme, 
le vocabulaire dans fastText est plus large,
ce qui entraîne un plus grand nombre de paramètres de modèle. 
En outre, 
pour calculer la représentation d'un mot,
tous ses vecteurs de sous-mots
doivent être additionnés,
ce qui entraîne une plus grande complexité de calcul.
Cependant,
grâce aux paramètres partagés des sous-mots parmi les mots ayant des structures similaires,
les mots rares et même les mots hors vocabulaire
peuvent obtenir de meilleures représentations vectorielles dans fastText.



## Encodage des paires d'octets
:label:`subsec_Byte_Pair_Encoding` 

 Dans fastText, tous les sous-mots extraits doivent avoir la longueur spécifiée, par exemple $3$ à $6$, et la taille du vocabulaire ne peut donc pas être prédéfinie.
Pour permettre des sous-mots de longueur variable dans un vocabulaire de taille fixe,
nous pouvons appliquer un algorithme de compression
appelé *byte pair encoding* (BPE) pour extraire les sous-mots :cite:`Sennrich.Haddow.Birch.2015` .

Le codage par paires d'octets effectue une analyse statistique de l'ensemble de données d'apprentissage pour découvrir les symboles communs dans un mot,
tels que des caractères consécutifs de longueur arbitraire.
À partir de symboles de longueur 1,
le codage par paires d'octets fusionne de manière itérative la paire la plus fréquente de symboles consécutifs pour produire de nouveaux symboles plus longs.
Notez que pour des raisons d'efficacité, les paires traversant les limites des mots ne sont pas prises en compte.
Au final, nous pouvons utiliser ces symboles comme des sous-mots pour segmenter les mots.
L'encodage de paires d'octets et ses variantes ont été utilisés pour les représentations d'entrée dans des modèles de pré-entraînement de traitement du langage naturel populaires tels que GPT-2 :cite:`Radford.Wu.Child.ea.2019` et RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` .
Dans ce qui suit, nous allons illustrer le fonctionnement de l'encodage de paires d'octets.

Tout d'abord, nous initialisons le vocabulaire de symboles comme étant tous les caractères minuscules anglais, un symbole spécial de fin de mot `'_'`, et un symbole spécial inconnu `'[UNK]'`.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Puisque nous ne considérons pas les paires de symboles qui traversent les limites des mots,
nous avons seulement besoin d'un dictionnaire `raw_token_freqs` qui associe les mots à leurs fréquences (nombre d'occurrences)
dans un ensemble de données.
Notez que le symbole spécial `'_'` est ajouté à chaque mot de sorte que
nous pouvons facilement récupérer une séquence de mots (par exemple, "un homme plus grand")
à partir d'une séquence de symboles de sortie ( par exemple, "un_homme_grand").
Puisque nous commençons le processus de fusion à partir d'un vocabulaire composé uniquement de caractères uniques et de symboles spéciaux, un espace est inséré entre chaque paire de caractères consécutifs dans chaque mot (clés du dictionnaire `token_freqs`).
En d'autres termes, l'espace est le délimiteur entre les symboles dans un mot.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Nous définissons la fonction suivante `get_max_freq_pair` qui
renvoie la paire la plus fréquente de symboles consécutifs dans un mot,
où les mots proviennent des clés du dictionnaire d'entrée `token_freqs`.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `paires` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `paires` with the max value
```

En tant qu'approche avide basée sur la fréquence des symboles consécutifs, le codage par paires d'octets
utilisera la fonction suivante `merge_symbols` pour fusionner la paire la plus fréquente de symboles consécutifs afin de produire de nouveaux symboles.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Nous exécutons maintenant de manière itérative l'algorithme de codage par paires d'octets sur les clés du dictionnaire `token_freqs`. Dans la première itération, la paire la plus fréquente de symboles consécutifs est `'t'` et `'a'`, et l'encodage par paires d'octets les fusionne pour produire un nouveau symbole `'ta'`. Dans la deuxième itération, le codage par paires d'octets continue à fusionner `'ta'` et `'l'` pour produire un autre nouveau symbole `'tal'`.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Après 10 itérations de codage par paires d'octets, nous pouvons constater que la liste `symbols` contient maintenant 10 symboles supplémentaires qui sont fusionnés par itération à partir d'autres symboles.

```{.python .input}
#@tab all
print(symbols)
```

Pour le même ensemble de données spécifié dans les clés du dictionnaire `raw_token_freqs`,
chaque mot de l'ensemble de données est maintenant segmenté en sous-mots "fast_", "fast", "er_", "tall_" et "tall"
à la suite de l'algorithme de codage par paires d'octets.
Par exemple, les mots "faster_" et "taller_" sont segmentés en "fast er_" et "tall er_", respectivement.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Notez que le résultat de l'encodage des paires d'octets dépend de l'ensemble de données utilisé.
Nous pouvons également utiliser les sous-mots appris dans un ensemble de données
pour segmenter les mots d'un autre ensemble de données.
La fonction suivante `segment_BPE` tente de décomposer les mots en sous-mots les plus longs possibles à partir de l'argument d'entrée `symbols`.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

Dans ce qui suit, nous utilisons les sous-mots de la liste `symbols`, qui est apprise à partir de l'ensemble de données susmentionné,
pour segmenter `tokens` qui représente un autre ensemble de données.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Résumé

* Le modèle fastText propose une approche d'intégration de sous-mots. Basé sur le modèle skip-gram de word2vec, il représente un mot central comme la somme des vecteurs de ses sous-mots.
* L'encodage par paires d'octets effectue une analyse statistique de l'ensemble des données d'entraînement pour découvrir les symboles communs dans un mot. En tant qu'approche gourmande, l'encodage de paires d'octets fusionne itérativement la paire la plus fréquente de symboles consécutifs.
* L'incorporation de sous-mots peut améliorer la qualité des représentations des mots rares et des mots hors dictionnaire.

## Exercices

1. A titre d'exemple, il existe environ $3\times 10^8$ -grammes possibles $6$ en anglais. Quel est le problème lorsqu'il y a trop de sous-mots ? Comment résoudre ce problème ? Indice : reportez-vous à la fin de la section 3.2 du document fastText :cite:`Bojanowski.Grave.Joulin.ea.2017` .
1. Comment concevoir un modèle d'intégration de sous-mots basé sur le modèle de sac de mots continu ?
1. Pour obtenir un vocabulaire de taille $m$, combien d'opérations de fusion sont nécessaires lorsque la taille initiale du vocabulaire de symboles est $n$?
1. Comment étendre l'idée de l'encodage de paires d'octets pour extraire des phrases ?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4587)
:end_tab:
