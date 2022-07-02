# Le jeu de données pour le pré-entraînement de BERT
:label:`sec_bert-dataset` 

 Pour pré-entraîner le modèle BERT tel qu'il est implémenté dans :numref:`sec_bert` ,
nous devons générer le jeu de données dans le format idéal pour faciliter
les deux tâches de pré-entraînement :
modélisation du langage masqué et prédiction de la phrase suivante.
D'une part,
le modèle BERT original est pré-entraîné sur la concaténation de
deux énormes corpus BookCorpus et Wikipedia anglais (voir :numref:`subsec_bert_pretraining_tasks` ),
ce qui le rend difficile à utiliser pour la plupart des lecteurs de ce livre.
D'autre part,
le modèle BERT pré-entraîné prêt à l'emploi
peut ne pas convenir à des applications dans des domaines spécifiques comme la médecine.
C'est pourquoi il est de plus en plus courant de prétraîner BERT sur un jeu de données personnalisé.
Pour faciliter la démonstration du pré-entraînement de BERT,
nous utilisons un corpus plus petit WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016` .

En comparaison avec le jeu de données PTB utilisé pour le pré-entraînement de word2vec dans :numref:`sec_word2vec_data` ,
WikiText-2 (i) conserve la ponctuation originale, ce qui le rend approprié pour la prédiction de la phrase suivante ; (ii) conserve la casse et les nombres originaux ; (iii) est deux fois plus grand.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

Dans [**le jeu de données WikiText-2**],
chaque ligne représente un paragraphe où
un espace est inséré entre toute ponctuation et le jeton qui la précède.
Les paragraphes comportant au moins deux phrases sont conservés.
Pour séparer les phrases, nous utilisons uniquement le point comme délimiteur pour des raisons de simplicité.
Nous laissons les discussions sur les techniques plus complexes de division de phrases dans les exercices
à la fin de cette section.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Définition des fonctions d'aide pour les tâches de pré-entraînement

Dans ce qui suit,
nous commençons par implémenter des fonctions d'aide pour les deux tâches de pré-entraînement de BERT :
la prédiction de la phrase suivante et la modélisation du langage masqué.
Ces fonctions d'aide seront invoquées ultérieurement
lors de la transformation du corpus de texte brut
en un ensemble de données au format idéal pour le pré-entraînement de BERT.

### [**Générer la tâche de prédiction de la prochaine phrase**]

Selon les descriptions de :numref:`subsec_nsp` ,
la fonction `_get_next_sentence` génère un exemple d'entraînement
pour la tâche de classification binaire.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphes` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

La fonction suivante génère des exemples d'entraînement pour la prédiction de la phrase suivante
à partir de l'entrée `paragraph` en invoquant la fonction `_get_next_sentence`.
Ici, `paragraph` est une liste de phrases, où chaque phrase est une liste de tokens.
L'argument `max_len` spécifie la longueur maximale d'une séquence d'entrée BERT pendant le pré-entraînement.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### [**Génération de la tâche de modélisation du langage masqué**]
:label:`subsec_prepare_mlm_data` 

 Afin de générer des exemples d'entraînement
pour la tâche de modélisation du langage masqué
à partir d'une séquence d'entrée BERT,
nous définissons la fonction `_replace_mlm_tokens` suivante.
Dans ses entrées, `tokens` est une liste de tokens représentant une séquence d'entrée BERT,
`candidate_pred_positions` est une liste d'indices de tokens de la séquence d'entrée BERT
excluant ceux des tokens spéciaux (les tokens spéciaux ne sont pas prédits dans la tâche de modélisation du langage masqué),
et `num_mlm_preds` indique le nombre de prédictions (rappel de 15% de tokens aléatoires à prédire).
Suivant la définition de la tâche de modélisation du langage masqué dans :numref:`subsec_mlm` ,
à chaque position de prédiction, l'entrée peut être remplacée par
un token spécial "&lt;mask&gt;" ou un token aléatoire, ou rester inchangée.
Au final, la fonction renvoie les tokens d'entrée après un éventuel remplacement,
les indices des tokens où les prédictions ont lieu et les étiquettes de ces prédictions.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

En invoquant la fonction `_replace_mlm_tokens` mentionnée ci-dessus,
la fonction suivante prend une séquence d'entrée BERT (`tokens`)
comme entrée et renvoie les indices des jetons d'entrée
(après un éventuel remplacement des jetons comme décrit dans :numref:`subsec_mlm` ),
les indices des jetons où les prédictions ont lieu,
et les indices des étiquettes pour ces prédictions.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Transformation du texte en jeu de données de pré-entraînement

Nous sommes maintenant presque prêts à personnaliser une classe `Dataset` pour le pré-entraînement de BERT.
Avant cela, 
nous devons encore définir une fonction d'aide `_pad_bert_inputs`
 pour [**ajouter les tokens spéciaux "&lt;pad&gt;" aux entrées.**]
Son argument `examples` contient les sorties des fonctions d'aide `_get_nsp_data_from_paragraph` et `_get_mlm_data_from_tokens` pour les deux tâches de pré-entraînement.

```{.python .input}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `lentilles_valides` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `lentilles_valides` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

En combinant les fonctions d'aide pour
générant des exemples d'entraînement pour les deux tâches de pré-entraînement,
et la fonction d'aide pour le remplissage des entrées,
, nous personnalisons la classe `_WikiTextDataset` suivante comme [**le jeu de données WikiText-2 pour le pré-entraînement de BERT**].
En implémentant la fonction `__getitem__ `,
nous pouvons accéder arbitrairement aux exemples de pré-entraînement (modélisation du langage masqué et prédiction de la phrase suivante) 
générés à partir d'une paire de phrases du corpus WikiText-2.

Le modèle original de BERT utilise des encastrements WordPiece dont la taille du vocabulaire est de 30000 :cite:`Wu.Schuster.Chen.ea.2016` .
La méthode de tokenisation de WordPiece est une légère modification de
l'algorithme original de codage de paires d'octets dans :numref:`subsec_Byte_Pair_Encoding` .
Par souci de simplicité, nous utilisons la fonction `d2l.tokenize` pour la tokénisation.
Les tokens peu fréquents qui apparaissent moins de cinq fois sont filtrés.

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphes[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphes[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphes[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphes[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

En utilisant la fonction `_read_wiki` et la classe `_WikiTextDataset`,
nous définissons le `load_data_wiki` suivant pour [**télécharger et le jeu de données WikiText-2
et générer des exemples de pré-entraînement**] à partir de celui-ci.

```{.python .input}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

En fixant la taille du lot à 512 et la longueur maximale d'une séquence d'entrée BERT à 64,
nous [**imprimons les formes d'un minilot d'exemples de pré-entraînement BERT**].
Notez que dans chaque séquence d'entrée BERT,
$10$ ($64 \times 0.15$) les positions sont prédites pour la tâche de modélisation du langage masqué.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

Enfin, jetons un coup d'œil à la taille du vocabulaire.
Même après avoir filtré les tokens peu fréquents,
est toujours deux fois plus grand que celui de la base de données PTB.

```{.python .input}
#@tab all
len(vocab)
```

## Résumé

* Comparé au jeu de données de la PTB, le jeu de dates de WikiText-2 conserve la ponctuation, la casse et les chiffres originaux, et est deux fois plus grand.
* Nous pouvons accéder arbitrairement aux exemples de pré-entraînement (modélisation du langage masqué et prédiction de la phrase suivante) générés à partir d'une paire de phrases du corpus WikiText-2.



## Exercices

1. Pour des raisons de simplicité, le point est utilisé comme seul délimiteur pour séparer les phrases. Essayez d'autres techniques de découpage de phrases, comme le spaCy et NLTK. Prenez NLTK comme exemple. Vous devez d'abord installer NLTK :`pip install nltk`. Dans le code, commencez par `import nltk`. Ensuite, téléchargez le tokenizer de phrases Punkt :`nltk.download('punkt')`. Pour diviser des phrases telles que `sentences = 'This is great ! Why not ?'`, l'invocation de `nltk.tokenize.sent_tokenize(sentences)` renverra une liste de deux chaînes de phrases :`['This is great !', 'Why not ?']`.
1. Quelle est la taille du vocabulaire si nous ne filtrons pas les tokens peu fréquents ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:
