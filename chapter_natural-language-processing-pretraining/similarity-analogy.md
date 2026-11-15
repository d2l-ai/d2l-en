# Žodžių panašumas ir analogija
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining`
apmokėme word2vec modelį mažame duomenų rinkinyje
ir pritaikėme jį
semantiškai panašiems žodžiams
duotam žodžiui rasti.
Praktikoje
žodžių vektorius, iš anksto apmokytus
dideliuose tekstynuose, galima
taikyti tolesnėms
natūraliosios kalbos apdorojimo užduotims,
kurios bus aptartos vėliau
:numref:`chap_nlp_app`.
Kad paprastai parodytume
iš anksto apmokytų žodžių vektorių semantiką
iš didelių tekstynų,
pritaikykime juos
žodžių panašumo ir analogijos užduotims.

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

## Iš anksto apmokytų žodžių vektorių įkėlimas

Toliau išvardyti iš anksto apmokyti GloVe įterpiniai, kurių dimensija 50, 100 ir 300,
kuriuos galima atsisiųsti iš [GloVe svetainės](https://nlp.stanford.edu/projects/glove/).
Iš anksto apmokyti fastText įterpiniai prieinami keliomis kalbomis.
Čia nagrinėjame vieną angliškąją versiją (300 dimensijų „wiki.en“), kurią galima atsisiųsti iš
[fastText svetainės](https://fasttext.cc/).

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

Kad įkeltume šiuos iš anksto apmokytus GloVe ir fastText įterpinius, apibrėžiame šią klasę `TokenEmbedding`.

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

Toliau įkeliame
50 dimensijų GloVe įterpinius
(iš anksto apmokytus Wikipedia poaibyje).
Kuriant `TokenEmbedding` egzempliorių,
nurodytą įterpinių failą reikia atsisiųsti, jei jis
dar nebuvo atsisiųstas.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Išveskite žodyno dydį. Žodyne yra 400000 žodžių (teksto vienetų) ir specialus nežinomo teksto vieneto simbolis.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Galime gauti žodžio indeksą žodyne ir atvirkščiai.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Iš anksto apmokytų žodžių vektorių taikymas

Naudodami įkeltus GloVe vektorius,
parodysime jų semantiką
pritaikydami juos
šioms žodžių panašumo ir analogijos užduotims.


### Žodžių panašumas

Panašiai kaip :numref:`subsec_apply-word-embed`,
kad rastume semantiškai panašius žodžius
duotam žodžiui
remiantis žodžių vektorių
kosinuso panašumais,
įgyvendiname šią funkciją `knn`
($k$ artimiausių kaimynų).

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

Tada
ieškome panašių žodžių
naudodami iš anksto apmokytus žodžių vektorius
iš `TokenEmbedding` egzemplioriaus `embed`.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

Iš anksto apmokytų žodžių vektorių žodyne
`glove_6b50d` yra 400000 žodžių ir specialus nežinomo teksto vieneto simbolis.
Neįtraukdami įvesties žodžio ir nežinomo teksto vieneto,
šiame žodyne
raskime
tris semantiškai panašiausius žodžius
žodžiui „chip“.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Toliau išvedami panašūs žodžiai
žodžiams „baby“ ir „beautiful“.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Žodžių analogija

Be panašių žodžių paieškos,
žodžių vektorius taip pat galime taikyti
žodžių analogijos užduotims.
Pavyzdžiui,
„man“ : „woman“ :: „son“ : „daughter“
yra žodžių analogijos forma:
„man“ yra „woman“, kaip „son“ yra „daughter“.
Konkrečiau,
žodžių analogijos užbaigimo užduotį
galima apibrėžti taip:
žodžių analogijai
$a : b :: c : d$, turint pirmuosius tris žodžius $a$, $b$ ir $c$, raskite $d$.
Žodžio $w$ vektorių pažymėkime $\textrm{vec}(w)$.
Kad užbaigtume analogiją,
rasime žodį,
kurio vektorius yra panašiausias
į $\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$ rezultatą.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Patikrinkime „vyriškas–moteriškas“ analogiją naudodami įkeltus žodžių vektorius.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Toliau užbaigiama
„sostinė–šalis“ analogija:
„beijing“ : „china“ :: „tokyo“ : „japan“.
Tai parodo
iš anksto apmokytų žodžių vektorių semantiką.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

„Būdvardis–aukščiausiojo laipsnio būdvardis“ analogijai,
pavyzdžiui,
„bad“ : „worst“ :: „big“ : „biggest“,
matome, kad iš anksto apmokyti žodžių vektoriai
gali užfiksuoti sintaksinę informaciją.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Kad parodytume iš anksto apmokytuose žodžių vektoriuose užfiksuotą
būtojo laiko sąvoką,
sintaksę galime patikrinti naudodami
„esamasis laikas–būtasis laikas“ analogiją: „do“ : „did“ :: „go“ : „went“.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Santrauka

* Praktikoje žodžių vektorius, iš anksto apmokytus dideliuose tekstynuose, galima taikyti tolesnėms natūraliosios kalbos apdorojimo užduotims.
* Iš anksto apmokytus žodžių vektorius galima taikyti žodžių panašumo ir analogijos užduotims.


## Pratimai

1. Išbandykite fastText rezultatus naudodami `TokenEmbedding('wiki.en')`.
1. Kai žodynas yra labai didelis, kaip galime greičiau rasti panašius žodžius arba užbaigti žodžių analogiją?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1336)
:end_tab:
