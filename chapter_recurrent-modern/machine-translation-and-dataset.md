```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# La traduction automatique et le jeu de données
:label:`sec_machine_translation` 

 Nous avons utilisé les RNN pour concevoir des modèles de langage,
qui sont essentiels au traitement du langage naturel.
Un autre repère phare est la *traduction automatique*,
un domaine problématique central pour les modèles de *transduction de séquences*
qui transforment les séquences d'entrée en séquences de sortie.
Jouant un rôle crucial dans diverses applications modernes de l'IA, les modèles de transduction de séquence
feront l'objet du reste de ce chapitre
et :numref:`chap_attention` .
À cette fin,
cette section présente le problème de la traduction automatique
et son ensemble de données qui sera utilisé par la suite.


*La traduction automatique* désigne la traduction automatique
d'une séquence
d'une langue à une autre.
En fait, ce domaine
peut remonter aux années 1940
peu après l'invention des ordinateurs numériques,
surtout si l'on considère l'utilisation des ordinateurs
pour le craquage des codes linguistiques pendant la Seconde Guerre mondiale.
Pendant des décennies, les approches statistiques

 ont été dominantes dans ce domaine :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990` 
 avant l'essor
de l'apprentissage de bout en bout
utilisant
les réseaux neuronaux.
Cette dernière
est souvent appelée
*traduction automatique neuronale*
pour se distinguer de
*traduction automatique statistique*
qui implique une analyse statistique
dans des composants tels que
le modèle de traduction et le modèle de langue.


Mettant l'accent sur l'apprentissage de bout en bout,
ce livre se concentrera sur les méthodes de traduction automatique neuronale.
Contrairement à notre problème de modèle de langue
dans :numref:`sec_language-model` 
 dont le corpus est dans une seule langue, les ensembles de données de traduction automatique

 sont composés de paires de séquences de texte
qui sont dans
la langue source et la langue cible, respectivement.
Ainsi,
au lieu de réutiliser la routine de prétraitement
pour la modélisation du langage,
nous avons besoin d'une manière différente de prétraiter
les ensembles de données de traduction automatique.
Dans ce qui suit,
nous montrons comment
charger les données prétraitées
dans des minibatchs pour la formation.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## [**Téléchargement et prétraitement du jeu de données**]

Pour commencer,
nous téléchargeons un jeu de données anglais-français
qui se compose de [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/).
Chaque ligne du jeu de données
est une paire délimitée par des tabulations
d'une séquence de texte anglais
et de la séquence de texte français traduite.
Notez que chaque séquence de texte
peut être une seule phrase ou un paragraphe de plusieurs phrases.
Dans ce problème de traduction automatique
où l'anglais est traduit en français,
l'anglais est la *langue source*
et le français est la *langue cible*.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
            
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Après avoir téléchargé le jeu de données,
nous [**procédons à plusieurs étapes de prétraitement**]
pour les données textuelles brutes.
Par exemple,
nous remplaçons les espaces insécables par des espaces,
nous convertissons les majuscules en minuscules,
et nous insérons des espaces entre les mots et les signes de ponctuation.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)

text = data._preprocess(raw_text)
print(text[:80])
```

## [**Tokenization**]

Différent de la tokenisation au niveau des caractères
dans :numref:`sec_language-model` ,
pour la traduction automatique
nous préférons ici la tokenisation au niveau des mots
(les modèles de pointe peuvent utiliser des techniques de tokenisation plus avancées).
La méthode suivante `_tokenize`
 tokenise les premières paires de séquences de texte `max_examples`,
où
chaque token est soit un mot, soit un signe de ponctuation.
Nous ajoutons le jeton spécial "&lt;eos&gt;"
à la fin de chaque séquence pour indiquer la fin de la séquence
.
Lorsqu'un modèle prédit
en
générant une séquence jeton après jeton,
la génération
du jeton "&lt;eos&gt;"
peut suggérer que
la séquence de sortie est complète.
Au final,
la méthode ci-dessous renvoie
deux listes de tokens :`src` et `tgt`.
Plus précisément,
`src[i]` est une liste de tokens de la séquence de texte
$i^\mathrm{th}$ dans la langue source (ici l'anglais) et `tgt[i]` est celle dans la langue cible (ici le français).

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt

src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

Traçons [**l'histogramme du nombre de tokens par séquence de texte.**]
Dans ce jeu de données simple anglais-français,
la plupart des séquences de texte ont moins de 20 tokens.

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Chargement de séquences de longueur fixe
:label:`subsec_loading-seq-fixed-len` 

 Rappelons que dans la modélisation du langage
[**chaque exemple de séquence**],
soit un segment d'une phrase
ou une étendue sur plusieurs phrases,
(**a une longueur fixe.**))
Cela a été spécifié par l'argument `num_steps`
 (nombre de pas de temps ou de tokens) dans :numref:`sec_language-model` .
Dans la traduction automatique, chaque exemple est
une paire de séquences de texte source et cible,
où chaque séquence de texte peut avoir des longueurs différentes.

Pour des raisons d'efficacité informatique,
nous pouvons toujours traiter un mini lot de séquences de texte
en une seule fois par *troncation* et *remplissage*.
Supposons que chaque séquence du même minilot
ait la même longueur `num_steps`.
Si une séquence de texte comporte moins de `num_steps` tokens,
nous continuerons à ajouter le jeton spécial "&lt;pad&gt;"
à sa fin jusqu'à ce que sa longueur atteigne `num_steps`.
Sinon,
nous tronquerons la séquence de texte
en ne prenant que ses premiers `num_steps` tokens
et en rejetant les autres.
De cette façon,
chaque séquence de texte
aura la même longueur
pour être chargée dans des minibatchs de la même forme.
De plus, nous enregistrons également la longueur de la séquence source en excluant les tokens de remplissage.
Cette information sera nécessaire pour certains modèles que nous aborderons plus tard.


Puisque l'ensemble de données de traduction automatique
est constitué de paires de langues,
nous pouvons construire deux vocabulaires pour
la langue source et
la langue cible séparément.
Avec la tokénisation au niveau du mot,
la taille du vocabulaire sera significativement plus grande
que celle utilisant la tokénisation au niveau du caractère.
Pour pallier ce problème,
nous traitons ici les tokens peu fréquents
qui apparaissent moins de 2 fois
comme le même token inconnu ("&lt;unk&gt;").
Comme nous l'expliquerons plus tard sur 
(:numref:`fig_seq2seq` ),
lors de l'apprentissage avec des séquences cibles,
la sortie du décodeur (jetons d'étiquettes)
peut être la même entrée du décodeur (jetons cibles),
décalée d'un jeton ;
et
le token spécial de début de séquence
"&lt;bos&gt;"
sera utilisé comme premier token d'entrée
pour prédire la séquence cible (:numref:`fig_seq2seq_predict` ).

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())


@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**Lire le jeu de données**]

Enfin, nous définissons la méthode `get_dataloader`
 pour renvoyer l'itérateur de données.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

Lisons [**le premier minibatch de l'ensemble de données anglais-français.**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Nous montrons ci-dessous une paire de séquences source et cible
qui sont traitées par la méthode `_build_arrays` ci-dessus
(dans le format chaîne de caractères).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Résumé

* La traduction automatique fait référence à la traduction automatique d'une séquence d'une langue à une autre.
* En utilisant la tokenisation au niveau des mots, la taille du vocabulaire sera significativement plus grande que celle utilisant la tokenisation au niveau des caractères. Pour pallier ce problème, nous pouvons traiter les tokens peu fréquents comme le même token inconnu.
* Nous pouvons tronquer et capitonner les séquences de texte afin qu'elles aient toutes la même longueur pour être chargées en minibatchs.


## Exercices

1. Essayez différentes valeurs de l'argument `max_examples` dans la méthode `_tokenize`. Comment cela affecte-t-il la taille du vocabulaire de la langue source et de la langue cible ?
1. Dans certaines langues comme le chinois et le japonais, le texte ne comporte pas d'indicateurs de limite de mot (par exemple, un espace). La tokénisation au niveau du mot est-elle toujours une bonne idée dans de tels cas ? Pourquoi ou pourquoi pas ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3863)
:end_tab:
