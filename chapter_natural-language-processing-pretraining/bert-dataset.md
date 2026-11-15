# Duomenų rinkinys BERT išankstiniam apmokymui
:label:`sec_bert-dataset`

Kad iš anksto apmokytume BERT modelį, įgyvendintą :numref:`sec_bert`,
turime sugeneruoti duomenų rinkinį idealiu formatu, kuris palengvintų
dvi išankstinio apmokymo užduotis:
užmaskuotą kalbos modeliavimą ir kito sakinio prognozavimą.
Viena vertus,
originalus BERT modelis iš anksto apmokytas dviejų milžiniškų tekstynų BookCorpus ir anglų kalbos Vikipedijos
konkatenacijoje (žr. :numref:`subsec_bert_pretraining_tasks`),
todėl daugumai šios knygos skaitytojų jį sunku paleisti.
Kita vertus,
paruoštas iš anksto apmokytas BERT modelis
gali netikti taikymams konkrečiose srityse, pavyzdžiui, medicinoje.
Todėl vis populiaresnis tampa BERT išankstinis apmokymas pritaikytame duomenų rinkinyje.
Kad palengvintume BERT išankstinio apmokymo demonstravimą,
naudojame mažesnį tekstyną WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`.

Palyginti su PTB duomenų rinkiniu, naudotu word2vec išankstiniam apmokymui :numref:`sec_word2vec_data`,
WikiText-2 (i) išsaugo originalią skyrybą, todėl tinka kito sakinio prognozavimui; (ii) išsaugo originalius didžiuosius ir mažuosius raidžių atvejus bei skaičius; (iii) yra daugiau nei dvigubai didesnis.

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

[**WikiText-2 duomenų rinkinyje**]
kiekviena eilutė vaizduoja pastraipą, kurioje
tarpas įterpiamas tarp bet kokio skyrybos ženklo ir jį einančio teksto vieneto.
Paliekamos pastraipos, turinčios bent du sakinius.
Kad suskaidytume sakinius, paprastumo dėlei kaip skirtuką naudojame tik tašką.
Sudėtingesnių sakinių skaidymo technikų aptarimą paliekame pratimams
šio skyrelio pabaigoje.

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

## Pagalbinių funkcijų išankstinio apmokymo užduotims apibrėžimas

Toliau
pradedame nuo pagalbinių funkcijų įgyvendinimo dviem BERT išankstinio apmokymo užduotims:
kito sakinio prognozavimui ir užmaskuotam kalbos modeliavimui.
Šios pagalbinės funkcijos bus iškviečiamos vėliau,
paverčiant neapdorotą teksto tekstyną
į duomenų rinkinį idealiu formatu BERT iš anksto apmokyti.

### [**Kito sakinio prognozavimo užduoties generavimas**]

Pagal :numref:`subsec_nsp` aprašymus,
funkcija `_get_next_sentence` generuoja apmokymo pavyzdį
dvejetainio klasifikavimo užduočiai.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

Toliau pateikta funkcija generuoja apmokymo pavyzdžius kito sakinio prognozavimui
iš įvesties `paragraph` iškviesdama funkciją `_get_next_sentence`.
Čia `paragraph` yra sakinių sąrašas, kur kiekvienas sakinys yra teksto vienetų sąrašas.
Argumentas `max_len` nurodo didžiausią BERT įvesties sekos ilgį išankstinio apmokymo metu.

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

### [**Užmaskuoto kalbos modeliavimo užduoties generavimas**]
:label:`subsec_prepare_mlm_data`

Kad sugeneruotume apmokymo pavyzdžius
užmaskuoto kalbos modeliavimo užduočiai
iš BERT įvesties sekos,
apibrėžiame šią funkciją `_replace_mlm_tokens`.
Jos įvestyse `tokens` yra teksto vienetų sąrašas, vaizduojantis BERT įvesties seką,
`candidate_pred_positions` yra BERT įvesties sekos teksto vienetų indeksų sąrašas,
neįtraukiant specialių teksto vienetų (specialūs teksto vienetai neprognozuojami užmaskuoto kalbos modeliavimo užduotyje),
o `num_mlm_preds` nurodo prognozių skaičių (prisiminkite 15 % atsitiktinių teksto vienetų prognozei).
Laikantis užmaskuoto kalbos modeliavimo užduoties apibrėžimo :numref:`subsec_mlm`,
kiekvienoje prognozavimo pozicijoje įvestis gali būti pakeista
specialiu „&lt;mask&gt;“ teksto vienetu arba atsitiktiniu teksto vienetu, arba likti nepakeista.
Galiausiai funkcija grąžina įvesties teksto vienetus po galimo pakeitimo,
teksto vienetų indeksus, kuriuose vyksta prognozės, ir šių prognozių žymenis.

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

Iškviesdama aukščiau minėtą funkciją `_replace_mlm_tokens`,
toliau pateikta funkcija kaip įvestį priima BERT įvesties seką (`tokens`)
ir grąžina įvesties teksto vienetų indeksus
(po galimo teksto vienetų pakeitimo, kaip aprašyta :numref:`subsec_mlm`),
teksto vienetų indeksus, kuriuose vyksta prognozės,
ir šių prognozių žymenų indeksus.

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

## Teksto transformavimas į išankstinio apmokymo duomenų rinkinį

Dabar esame beveik pasirengę pritaikyti klasę `Dataset` BERT išankstiniam apmokymui.
Prieš tai
dar turime apibrėžti pagalbinę funkciją `_pad_bert_inputs`,
kad [**prie įvesčių pridėtume specialius „&lt;pad&gt;“ teksto vienetus.**]
Jos argumente `examples` yra išvestys iš pagalbinių funkcijų `_get_nsp_data_from_paragraph` ir `_get_mlm_data_from_tokens` dviem išankstinio apmokymo užduotims.

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
        # `valid_lens` excludes count of '<pad>' tokens
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
        # `valid_lens` excludes count of '<pad>' tokens
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

Sudėję kartu pagalbines funkcijas
dviejų išankstinio apmokymo užduočių apmokymo pavyzdžiams generuoti
ir pagalbinę funkciją įvestims papildyti,
pritaikome šią klasę `_WikiTextDataset` kaip [**WikiText-2 duomenų rinkinį BERT išankstiniam apmokymui**].
Įgyvendinę funkciją `__getitem__`,
galime savavališkai pasiekti išankstinio apmokymo (užmaskuoto kalbos modeliavimo ir kito sakinio prognozavimo) pavyzdžius,
sugeneruotus iš WikiText-2 tekstyno sakinių poros.

Originalus BERT modelis naudoja WordPiece įterpinius, kurių žodyno dydis yra 30000 :cite:`Wu.Schuster.Chen.ea.2016`.
WordPiece teksto skaidymo į teksto vienetus metodas yra nedidelė
pradinio baitų porų kodavimo algoritmo iš :numref:`subsec_Byte_Pair_Encoding` modifikacija.
Paprastumo dėlei teksto skaidymui naudojame funkciją `d2l.tokenize`.
Reti teksto vienetai, pasirodantys rečiau nei penkis kartus, atfiltruojami.

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
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
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
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

Naudodami funkciją `_read_wiki` ir klasę `_WikiTextDataset`,
apibrėžiame šią funkciją `load_data_wiki`, kad [**atsisiųstume WikiText-2 duomenų rinkinį
ir iš jo sugeneruotume išankstinio apmokymo pavyzdžius**].

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

Nustatę rinkinio dydį 512, o didžiausią BERT įvesties sekos ilgį 64,
[**atspausdiname BERT išankstinio apmokymo pavyzdžių minirinkinio formas**].
Atkreipkite dėmesį, kad kiekvienoje BERT įvesties sekoje
$10$ ($64 \times 0{,}15$) pozicijų prognozuojama užmaskuoto kalbos modeliavimo užduočiai.

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

Galiausiai pažvelkime į žodyno dydį.
Net atfiltravus retus teksto vienetus,
jis vis dar yra daugiau nei dvigubai didesnis nei PTB duomenų rinkinio.

```{.python .input}
#@tab all
len(vocab)
```

## Santrauka

* Palyginti su PTB duomenų rinkiniu, WikiText-2 duomenų rinkinys išsaugo originalią skyrybą, didžiųjų ir mažųjų raidžių atvejus bei skaičius ir yra daugiau nei dvigubai didesnis.
* Galime savavališkai pasiekti išankstinio apmokymo (užmaskuoto kalbos modeliavimo ir kito sakinio prognozavimo) pavyzdžius, sugeneruotus iš WikiText-2 tekstyno sakinių poros.


## Pratimai

1. Paprastumo dėlei taškas naudojamas kaip vienintelis sakinių skaidymo skirtukas. Išbandykite kitas sakinių skaidymo technikas, pavyzdžiui, spaCy ir NLTK. Paimkime NLTK kaip pavyzdį. Pirmiausia reikia įdiegti NLTK: `pip install nltk`. Kode pirmiausia `import nltk`. Tada atsisiųskite Punkt sakinių skaidymo į teksto vienetus modelį: `nltk.download('punkt')`. Kad suskaidytumėte sakinius, pavyzdžiui, `sentences = 'This is great ! Why not ?'`, iškvietus `nltk.tokenize.sent_tokenize(sentences)` bus grąžintas dviejų sakinių eilučių sąrašas: `['This is great !', 'Why not ?']`.
1. Koks būtų žodyno dydis, jei neatfiltruotume jokio reto teksto vieneto?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1496)
:end_tab:
