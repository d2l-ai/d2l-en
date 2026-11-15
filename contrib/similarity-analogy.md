# Sinonimų ir analogijų paieška
:label:`sec_synonyms`

:numref:`sec_word2vec_gluon` skyriuje apmokėme word2vec žodžių įterpimo modelį
mažos apimties duomenų rinkinyje ir ieškojome sinonimų naudodami žodžių vektorių
kosinusinį panašumą. Praktikoje žodžių vektorius, iš anksto apmokytus didelės apimties
tekstynu, dažnai galima taikyti tolesnėms natūraliosios kalbos apdorojimo užduotims. Šiame
skyriuje parodysime, kaip naudoti šiuos iš anksto apmokytus žodžių vektorius sinonimams
ir analogijoms rasti. Iš anksto apmokytus žodžių vektorius toliau taikysime
vėlesniuose skyriuose.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
import matplotlib.pyplot as plt
from mxnet import np, npx
import numpy
import os
from sklearn.decomposition import PCA

npx.set_np()
```

## Iš anksto apmokytų žodžių vektorių naudojimas

Žemiau išvardyti iš anksto apmokyti 50, 100 ir 300 dimensijų GloVe įterpiniai,
kuriuos galima atsisiųsti iš [GloVe svetainės](https://nlp.stanford.edu/projects/glove/).
Iš anksto apmokyti fastText įterpiniai prieinami keliomis kalbomis.
Čia nagrinėjame vieną anglišką versiją (300 dimensijų „wiki.en“), kurią galima atsisiųsti iš
[fastText svetainės](https://fasttext.cc/).

```{.python .input  n=2}
#@tab mxnet
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

Apibrėžiame šią `TokenEmbedding` klasę, kad įkeltume aukščiau minėtus iš anksto apmokytus GloVe ir fastText įterpinius.

```{.python .input  n=3}
#@tab mxnet
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
        return idx_to_token, np.array(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[np.array(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Toliau naudojame 50 dimensijų GloVe įterpinius, iš anksto apmokytus Vikipedijos poaibyje. Atitinkamas žodžių įterpinys automatiškai atsisiunčiamas pirmą kartą sukūrus iš anksto apmokyto žodžių įterpinio objektą.

```{.python .input  n=4}
#@tab mxnet
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Išveskime žodyno dydį. Žodyne yra $400\,000$ žodžių ir specialus nežinomos leksemos simbolis.

```{.python .input  n=5}
#@tab mxnet
len(glove_6b50d)
```

Pagal žodį galime gauti jo indeksą žodyne arba pagal indeksą gauti žodį.

```{.python .input  n=6}
#@tab mxnet
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Iš anksto apmokytų žodžių vektorių taikymas

Toliau, kaip pavyzdį naudodami GloVe, parodome iš anksto apmokytų žodžių vektorių taikymą.

### Sinonimų paieška

Čia iš naujo realizuojame algoritmą, naudojamą sinonimams ieškoti pagal kosinusinį
panašumą, pristatytą :numref:`sec_word2vec`

Kad analogijų paieškoje galėtume pakartotinai naudoti $k$ artimiausių kaimynų paieškos
logiką, šią loginę dalį atskirai apgaubiame funkcija `knn`
($k$ artimiausių kaimynų).

```{.python .input  n=7}
#@tab mxnet
def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

Tada sinonimų ieškome naudodami iš anksto apmokytą žodžių vektorių objektą `embed`.

```{.python .input  n=8}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

Jau sukurto iš anksto apmokytų žodžių vektorių objekto `glove_6b50d` žodyne yra 400 000 žodžių ir specialus nežinomos leksemos simbolis. Neįskaitydami įvesties žodžių ir nežinomų žodžių, ieškome trijų reikšme panašiausių į „chip“ žodžių.

```{.python .input  n=9}
#@tab mxnet
get_similar_tokens('chip', 3, glove_6b50d)
```

Toliau ieškome „baby“ ir „beautiful“ sinonimų.

```{.python .input  n=10}
#@tab mxnet
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input  n=11}
#@tab mxnet
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Analogijų paieška

Be sinonimų paieškos, iš anksto apmokytus žodžių vektorius galime naudoti ir analogijoms tarp žodžių ieškoti. Pavyzdžiui, „man“:„woman“::„son“:„daughter“ yra analogijos pavyzdys: „man“ yra „woman“, kaip „son“ yra „daughter“. Analogijų paieškos uždavinį galima apibrėžti taip: keturiems žodžiams, esantiems analoginiame ryšyje $a : b :: c : d$, žinodami pirmuosius tris žodžius $a$, $b$ ir $c$, norime rasti $d$. Tarkime, žodžio $w$ žodžio vektorius yra $\text{vec}(w)$. Kad išspręstume analogijos uždavinį, turime rasti žodžio vektorių, kuris yra panašiausias į rezultato vektorių $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input  n=12}
#@tab mxnet
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Patikrinkime „vyras–moteris“ analogiją.

```{.python .input  n=13}
#@tab mxnet
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Analogija „sostinė–šalis“: „beijing“ yra „china“, kaip „tokyo“ yra kas? Atsakymas turėtų būti „japan“.

```{.python .input  n=14}
#@tab mxnet
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Analogija „būdvardis–aukščiausiasis laipsnis“: „bad“ yra „worst“, kaip „big“ yra kas? Atsakymas turėtų būti „biggest“.

```{.python .input  n=15}
#@tab mxnet
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Analogija „esamojo laiko veiksmažodis–būtojo laiko veiksmažodis“: „do“ yra „did“, kaip „go“ yra kas? Atsakymas turėtų būti „went“.

```{.python .input  n=16}
#@tab mxnet
get_analogy('do', 'did', 'go', glove_6b50d)
```

```{.python .input  n=51}
#@tab mxnet
def visualization(token_pairs, embed):
    plt.figure(figsize=(7, 5))
    vecs = np.concatenate([embed[pair] for pair in token_pairs])
    vecs_pca = PCA(n_components=2).fit_transform(numpy.array(vecs))
    for i, pair in enumerate(token_pairs):
        x1, y1 = vecs_pca[2 * i]
        x2, y2 = vecs_pca[2 * i + 1]
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        plt.annotate(pair[0], xy=(x1, y1))
        plt.annotate(pair[1], xy=(x2, y2))
        plt.plot([x1, x2], [y1, y2])
    plt.show()
```

```{.python .input  n=57}
#@tab mxnet
token_pairs = [['man', 'woman'], ['son', 'daughter'], ['king', 'queen'],
              ['uncle', 'aunt'], ['sir', 'madam'], ['sister', 'brother']]
visualization(token_pairs, glove_6b50d)
```

## Santrauka

* Žodžių vektorius, iš anksto apmokytus didelės apimties tekstynu, dažnai galima taikyti tolesnėms natūraliosios kalbos apdorojimo užduotims.
* Iš anksto apmokytus žodžių vektorius galime naudoti sinonimams ir analogijoms ieškoti.


## Pratimai

1. Išbandykite fastText rezultatus naudodami `TokenEmbedding('wiki.en')`.
1. Jei žodynas itin didelis, kaip galime pagreitinti sinonimų ir analogijų paiešką?


## [Diskusijos](https://discuss.mxnet.io/t/2390)

![](../img/qr_similarity-analogy.svg)
