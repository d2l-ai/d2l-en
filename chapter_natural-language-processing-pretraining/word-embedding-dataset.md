# Duomenų rinkinys žodžių įterpiniams iš anksto apmokyti
:label:`sec_word2vec_data`

Dabar, kai žinome
word2vec modelių technines detales
ir apytikrius apmokymo metodus,
pereikime prie jų įgyvendinimo.
Konkrečiai,
kaip pavyzdį imsime skip-gram modelį iš :numref:`sec_word2vec`
ir neigiamų pavyzdžių atranką iš :numref:`sec_approx_train`.
Šiame skyrelyje
pradedame nuo duomenų rinkinio
žodžių įterpinių modeliui iš anksto apmokyti:
pradinis duomenų formatas
bus paverstas
minirinkiniais,
kuriuos galima iteruoti apmokymo metu.

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

## Duomenų rinkinio skaitymas

Čia naudojamas duomenų rinkinys yra
[Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42).
Šis tekstynas atrinktas
iš Wall Street Journal straipsnių
ir padalytas į apmokymo, patikros ir testavimo aibes.
Pradiniame formate
kiekviena teksto failo eilutė
yra žodžių sakinys, kurio žodžiai atskirti tarpais.
Čia kiekvieną žodį laikome teksto vienetu.

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

Nuskaitę apmokymo aibę,
sudarome tekstyno žodyną,
kuriame bet kuris žodis, pasirodantis
rečiau nei 10 kartų, pakeičiamas
teksto vienetu „&lt;unk&gt;“.
Atkreipkite dėmesį, kad pradiniame duomenų rinkinyje
taip pat yra „&lt;unk&gt;“ teksto vienetų, vaizduojančių retus (nežinomus) žodžius.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Imties retinimas

Teksto duomenyse
paprastai yra dažnai pasitaikančių žodžių,
tokių kaip „the“, „a“ ir „in“:
jie gali pasirodyti net milijardus kartų
labai dideliuose tekstynuose.
Tačiau
šie žodžiai dažnai pasirodo kartu
su daugeliu skirtingų žodžių
konteksto languose ir suteikia mažai naudingų signalų.
Pavyzdžiui,
panagrinėkime žodį „chip“ konteksto lange:
intuityviai
jo bendras pasirodymas su retu žodžiu „intel“
yra naudingesnis apmokymui nei
bendras pasirodymas su dažnu žodžiu „a“.
Be to, apmokymas su milžiniškais (dažnų) žodžių kiekiais
yra lėtas.
Todėl apmokant žodžių įterpinių modelius
dažnus žodžius galima *išretinti* (angl. *subsample*) :cite:`Mikolov.Sutskever.Chen.ea.2013`.
Konkrečiai,
kiekvienas duomenų rinkinio indeksuotas žodis $w_i$
bus atmetamas su tikimybe


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

kur $f(w_i)$ yra
žodžių $w_i$ skaičiaus
ir bendro duomenų rinkinio žodžių skaičiaus santykis,
o konstanta $t$ yra hiperparametras
(eksperimente $10^{-4}$).
Matome, kad tik tada, kai
santykinis dažnis
$f(w_i) > t$, (dažnas) žodis $w_i$ gali būti atmestas,
ir kuo didesnis santykinis žodžio dažnis,
tuo didesnė tikimybė, kad jis bus atmestas.

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

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Šis kodo fragmentas
nubraižo teksto vienetų skaičiaus viename sakinyje
histogramą
prieš imties retinimą ir po jo.
Kaip ir tikėtasi,
imties retinimas gerokai sutrumpina sakinius
atmesdamas dažnus žodžius,
o tai paspartins apmokymą.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Atskiriems teksto vienetams dažno žodžio „the“ atrankos dažnis yra mažesnis nei 1/20.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Priešingai,
reti žodžiai, pavyzdžiui, „join“, išsaugomi visiškai.

```{.python .input}
#@tab all
compare_counts('join')
```

Po imties retinimo tekstyno teksto vienetus atvaizduojame į jų indeksus.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Centrinių ir konteksto žodžių išgavimas


Toliau pateikta funkcija `get_centers_and_contexts`
iš `corpus` išgauna visus
centrinius žodžius ir jų konteksto žodžius.
Ji atsitiktinai ir tolygiai atrenka sveikąjį skaičių nuo 1 iki `max_window_size`
kaip konteksto lango dydį.
Bet kuriam centriniam žodžiui
tie žodžiai,
kurių atstumas nuo jo
neviršija atrinkto
konteksto lango dydžio,
yra jo konteksto žodžiai.

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

Toliau sukuriame dirbtinį duomenų rinkinį, sudarytą iš dviejų atitinkamai 7 ir 3 žodžių sakinių.
Tegul didžiausias konteksto lango dydis yra 2,
ir atspausdinkime visus centrinius žodžius bei jų konteksto žodžius.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

Apmokant PTB duomenų rinkinyje,
didžiausią konteksto lango dydį nustatome 5.
Toliau išgaunami visi duomenų rinkinio centriniai žodžiai ir jų konteksto žodžiai.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Neigiamų pavyzdžių atranka

Apytikriam apmokymui naudojame neigiamų pavyzdžių atranką.
Kad atrinktume triukšmo žodžius pagal
iš anksto apibrėžtą pasiskirstymą,
apibrėžiame šią klasę `RandomGenerator`,
kurioje (galbūt nenormuotas) atrankos pasiskirstymas perduodamas
per argumentą `sampling_weights`.

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

Pavyzdžiui,
galime išgauti 10 atsitiktinių dydžių $X$
tarp indeksų 1, 2 ir 3
su atrankos tikimybėmis $P(X=1)=2/9, P(X=2)=3/9$ ir $P(X=3)=4/9$ taip.

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Centrinio žodžio ir konteksto žodžio porai
atsitiktinai atrenkame `K` (eksperimente 5) triukšmo žodžių. Pagal word2vec straipsnio pasiūlymus,
triukšmo žodžio $w$ atrankos tikimybė $P(w)$
nustatoma lygi jo santykiniam dažniui
žodyne,
pakeltam
0,75 laipsniu :cite:`Mikolov.Sutskever.Chen.ea.2013`.

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

## Apmokymo pavyzdžių įkėlimas minirinkiniais
:label:`subsec_word2vec-minibatch-loading`

Išgavus
visus centrinius žodžius
kartu su jų
konteksto žodžiais ir atrinktais triukšmo žodžiais,
jie bus paversti
pavyzdžių minirinkiniais,
kuriuos galima iteratyviai įkelti
apmokymo metu.



Minirinkinyje
$i$-asis pavyzdys apima centrinį žodį
ir jo $n_i$ konteksto žodžių bei $m_i$ triukšmo žodžių.
Dėl skirtingų konteksto langų dydžių
$n_i+m_i$ skiriasi skirtingiems $i$.
Todėl
kiekvienam pavyzdžiui
jo konteksto žodžius ir triukšmo žodžius sukonkatenuojame kintamajame
`contexts_negatives`
ir papildome nuliais, kol konkatenacijos ilgis
pasiekia $\max_i n_i+m_i$ (`max_len`).
Kad skaičiuojant nuostolį
papildymai būtų neįtraukiami,
apibrėžiame kaukės kintamąjį `masks`.
Tarp `masks` elementų ir `contexts_negatives` elementų
yra abipus vienareikšmė atitiktis,
kur `masks` nuliai (kitaip – vienetai) atitinka `contexts_negatives` papildymus.


Kad atskirtume teigiamus ir neigiamus pavyzdžius,
`contexts_negatives` konteksto žodžius nuo triukšmo žodžių atskiriame per kintamąjį `labels`.
Panašiai kaip `masks`,
taip pat yra abipus vienareikšmė atitiktis
tarp `labels` elementų ir `contexts_negatives` elementų,
kur `labels` vienetai (kitaip – nuliai) atitinka `contexts_negatives` konteksto žodžius (teigiamus pavyzdžius).


Aukščiau pateikta idėja įgyvendinta šioje funkcijoje `batchify`.
Jos įvestis `data` yra sąrašas, kurio ilgis
lygus rinkinio dydžiui,
kur kiekvienas elementas yra pavyzdys,
sudarytas iš
centrinio žodžio `center`, jo konteksto žodžių `context` ir jo triukšmo žodžių `negative`.
Ši funkcija grąžina
minirinkinį, kurį galima įkelti skaičiavimams
apmokymo metu,
pavyzdžiui, įskaitant kaukės kintamąjį.

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

Išbandykime šią funkciją naudodami dviejų pavyzdžių minirinkinį.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Viskas kartu

Galiausiai apibrėžiame funkciją `load_data_ptb`, kuri nuskaito PTB duomenų rinkinį ir grąžina duomenų iteratorių bei žodyną.

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

Atspausdinkime pirmąjį duomenų iteratoriaus minirinkinį.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Santrauka

* Dažni žodžiai apmokymui gali būti ne tokie naudingi. Galime juos išretinti, kad apmokymas būtų greitesnis.
* Skaičiavimo efektyvumui pasiekti pavyzdžius įkeliame minirinkiniais. Galime apibrėžti kitus kintamuosius, kad atskirtume papildymus nuo ne papildymų ir teigiamus pavyzdžius nuo neigiamų.



## Pratimai

1. Kaip pasikeistų šiame skyrelyje pateikto kodo veikimo laikas, jei nenaudotume imties retinimo?
1. Klasė `RandomGenerator` kaupia `k` atsitiktinės atrankos rezultatų. Nustatykite `k` į kitas reikšmes ir pažiūrėkite, kaip tai veikia duomenų įkėlimo greitį.
1. Kokie kiti šio skyrelio kodo hiperparametrai gali turėti įtakos duomenų įkėlimo greičiui?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1330)
:end_tab:
