# Nuotaikų analizė ir duomenų rinkinys
:label:`sec_sentiment`


Plintant internetinei socialinei žiniasklaidai
ir apžvalgų platformoms,
buvo užfiksuota gausybė
nuomonių duomenų,
turinčių didelį potencialą
palaikyti sprendimų priėmimo procesus.
*Nuotaikų analizė*
tiria žmonių nuotaikas
jų sukurtame tekste,
pavyzdžiui, produktų atsiliepimuose,
tinklaraščių komentaruose
ir
forumų diskusijose.
Ji plačiai taikoma
tokiose skirtingose srityse kaip
politika (pvz., visuomenės nuotaikų politikos atžvilgiu analizė),
finansai (pvz., rinkos nuotaikų analizė)
ir
rinkodara (pvz., produktų tyrimai ir prekių ženklų valdymas).

Kadangi nuotaikas
galima suskirstyti
į diskrečius poliškumus arba skales (pvz., teigiama ir neigiama),
nuotaikų analizę galime laikyti
teksto klasifikavimo užduotimi,
kuri kintamo ilgio teksto seką paverčia
fiksuoto ilgio teksto kategorija.
Šiame skyriuje
nuotaikų analizei naudosime Stanfordo [didelį filmų apžvalgų duomenų rinkinį](https://ai.stanford.edu/%7Eamaas/data/sentiment/).
Jį sudaro apmokymo ir testavimo aibės,
kurių kiekvienoje yra 25000 filmų apžvalgų, atsisiųstų iš IMDb.
Abiejuose duomenų rinkiniuose
yra vienodas skaičius
„teigiamų“ ir „neigiamų“ žymenų,
nurodančių skirtingus nuotaikų poliškumus.

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

## Duomenų rinkinio skaitymas

Pirmiausia atsisiųskite ir išpakuokite šį IMDb apžvalgų duomenų rinkinį
kelyje `../data/aclImdb`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Toliau nuskaitykite apmokymo ir testavimo duomenų rinkinius. Kiekvienas pavyzdys yra apžvalga ir jos žymuo: 1 „teigiamam“ ir 0 „neigiamam“.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## Duomenų rinkinio išankstinis apdorojimas

Kiekvieną žodį laikydami teksto vienetu
ir atfiltruodami žodžius, pasirodančius rečiau nei 5 kartus,
sukuriame žodyną iš apmokymo duomenų rinkinio.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Po teksto skaidymo į teksto vienetus
nubraižykime
apžvalgų ilgių teksto vienetais histogramą.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Kaip ir tikėjomės,
apžvalgos yra skirtingo ilgio.
Kad vienu metu apdorotume
tokių apžvalgų minirinkinį,
kiekvienos apžvalgos ilgį nustatome 500 su nukirtimu ir užpildymu,
panašiai kaip
mašininio vertimo duomenų rinkinio
išankstinio apdorojimo žingsnyje
:numref:`sec_machine_translation`.

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Duomenų iteratorių kūrimas

Dabar galime sukurti duomenų iteratorius.
Kiekvienoje iteracijoje grąžinamas pavyzdžių minirinkinys.

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Viskas kartu

Galiausiai aukščiau pateiktus žingsnius apgaubiame funkcija `load_data_imdb`.
Ji grąžina IMDb apžvalgų duomenų rinkinio apmokymo ir testavimo duomenų iteratorius bei žodyną.

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Santrauka

* Nuotaikų analizė tiria žmonių nuotaikas jų sukurtame tekste, kuri laikoma teksto klasifikavimo problema, paverčiančia kintamo ilgio teksto seką
fiksuoto ilgio teksto kategorija.
* Po išankstinio apdorojimo galime įkelti Stanfordo didelį filmų apžvalgų duomenų rinkinį (IMDb apžvalgų duomenų rinkinį) į duomenų iteratorius su žodynu.


## Pratimai


1. Kokius šio skyrelio hiperparametrus galime keisti, kad paspartintume nuotaikų analizės modelių apmokymą?
1. Ar galite įgyvendinti funkciją, kuri įkeltų [Amazon apžvalgų](https://snap.stanford.edu/data/web-Amazon.html) duomenų rinkinį į duomenų iteratorius ir žymenis nuotaikų analizei?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1387)
:end_tab:
