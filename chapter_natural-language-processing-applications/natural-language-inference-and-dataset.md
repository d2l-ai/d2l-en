# Natūraliosios kalbos išvadų darymas ir duomenų rinkinys
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment` aptarėme nuotaikų analizės problemą.
Ši užduotis siekia suklasifikuoti vieną teksto seką į iš anksto apibrėžtas kategorijas,
pavyzdžiui, nuotaikų poliškumų aibę.
Tačiau kai reikia nuspręsti, ar vieną sakinį galima išvesti iš kito,
arba pašalinti pertekliškumą nustatant semantiškai ekvivalentiškus sakinius,
nepakanka žinoti, kaip suklasifikuoti vieną teksto seką.
Vietoj to turime gebėti samprotauti apie teksto sekų poras.


## Natūraliosios kalbos išvadų darymas

*Natūraliosios kalbos išvadų darymas* tiria, ar *hipotezę*
galima išvesti iš *prielaidos*, kur abi yra teksto sekos.
Kitaip tariant, natūraliosios kalbos išvadų darymas nustato loginį sąryšį tarp teksto sekų poros.
Tokie sąryšiai paprastai skirstomi į tris tipus:

* *Išplaukimas* (angl. *entailment*): hipotezę galima išvesti iš prielaidos.
* *Prieštara* (angl. *contradiction*): hipotezės neigimą galima išvesti iš prielaidos.
* *Neutralumas* (angl. *neutral*): visi kiti atvejai.

Natūraliosios kalbos išvadų darymas taip pat žinomas kaip teksto išplaukimo atpažinimo užduotis.
Pavyzdžiui, ši pora bus pažymėta kaip *išplaukimas*, nes „rodo prieraišumą“ hipotezėje galima išvesti iš „apsikabina“ prielaidoje.

> Prielaida: Dvi moterys apsikabina.

> Hipotezė: Dvi moterys rodo prieraišumą.

Toliau pateiktas *prieštaros* pavyzdys, nes „vykdo kodo pavyzdį“ rodo „nemiega“, o ne „miega“.

> Prielaida: Vyras vykdo kodo pavyzdį iš „Dive into Deep Learning“.

> Hipotezė: Vyras miega.

Trečiasis pavyzdys rodo *neutralumo* sąryšį, nes iš fakto, kad „atlieka mums“, negalima išvesti nei „yra žinomi“, nei „nėra žinomi“.

> Prielaida: Muzikantai mums groja.

> Hipotezė: Muzikantai yra žinomi.

Natūraliosios kalbos išvadų darymas buvo svarbi natūraliosios kalbos supratimo tema.
Jis plačiai taikomas – nuo
informacijos paieškos iki atsakymo į atvirų sričių klausimus.
Kad ištirtume šią problemą, pradėsime nuo populiaraus natūraliosios kalbos išvadų darymo lyginamojo duomenų rinkinio nagrinėjimo.


## Stanfordo natūraliosios kalbos išvadų darymo (SNLI) duomenų rinkinys

[**Stanfordo natūraliosios kalbos išvadų darymo (SNLI) tekstynas**] yra daugiau nei 500000 pažymėtų anglų kalbos sakinių porų rinkinys :cite:`Bowman.Angeli.Potts.ea.2015`.
Atsisiunčiame ir išsaugome išpakuotą SNLI duomenų rinkinį kelyje `../data/snli_1.0`.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**Duomenų rinkinio skaitymas**]

Pradinis SNLI duomenų rinkinys turi daug daugiau informacijos, nei mums iš tikrųjų reikia eksperimentuose. Todėl apibrėžiame funkciją `read_snli`, kad išgautume tik dalį duomenų rinkinio, o tada grąžintume prielaidų, hipotezių ir jų žymenų sąrašus.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Dabar [**atspausdinkime pirmąsias 3 prielaidos ir hipotezės poras**], taip pat jų žymenis („0“, „1“ ir „2“ atitinka atitinkamai „išplaukimas“, „prieštara“ ir „neutralumas“).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

Apmokymo aibėje yra apie 550000 porų,
o testavimo aibėje – apie 10000 porų.
Toliau parodoma, kad
trys [**žymenys „išplaukimas“, „prieštara“ ir „neutralumas“ yra subalansuoti**]
tiek apmokymo, tiek testavimo aibėje.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**Klasės duomenų rinkiniui įkelti apibrėžimas**]

Toliau apibrėžiame klasę SNLI duomenų rinkiniui įkelti paveldėdami iš `Dataset` klasės Gluon. Klasės konstruktoriaus argumentas `num_steps` nurodo teksto sekos ilgį, kad kiekvienas sekų minirinkinys turėtų tą pačią formą.
Kitaip tariant,
teksto vienetai po pirmųjų `num_steps` ilgesnėje sekoje nukertami, o specialūs teksto vienetai „&lt;pad&gt;“ pridedami prie trumpesnių sekų, kol jų ilgis tampa `num_steps`.
Įgyvendinę funkciją `__getitem__`, galime savavališkai pasiekti prielaidą, hipotezę ir žymenį pagal indeksą `idx`.

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**Viskas kartu**]

Dabar galime iškviesti funkciją `read_snli` ir klasę `SNLIDataset`, kad atsisiųstume SNLI duomenų rinkinį ir grąžintume `DataLoader` egzempliorius tiek apmokymo, tiek testavimo aibėms kartu su apmokymo aibės žodynu.
Pažymėtina, kad testavimo aibei turime naudoti žodyną, sukonstruotą iš apmokymo aibės.
Todėl bet kuris naujas teksto vienetas iš testavimo aibės bus nežinomas modeliui, apmokytam apmokymo aibėje.

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Čia rinkinio dydį nustatome 128, o sekos ilgį 50,
ir iškviečiame funkciją `load_data_snli`, kad gautume duomenų iteratorius ir žodyną.
Tada atspausdiname žodyno dydį.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Dabar atspausdiname pirmojo minirinkinio formą.
Priešingai nei nuotaikų analizėje,
turime dvi įvestis `X[0]` ir `X[1]`, vaizduojančias prielaidų ir hipotezių poras.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Santrauka

* Natūraliosios kalbos išvadų darymas tiria, ar hipotezę galima išvesti iš prielaidos, kur abi yra teksto sekos.
* Natūraliosios kalbos išvadų darymo srityje sąryšiai tarp prielaidų ir hipotezių apima išplaukimą, prieštarą ir neutralumą.
* Stanfordo natūraliosios kalbos išvadų darymo (SNLI) tekstynas yra populiarus natūraliosios kalbos išvadų darymo lyginamasis duomenų rinkinys.


## Pratimai

1. Mašininis vertimas ilgą laiką buvo vertinamas remiantis paviršutinišku $n$-gramų sutapimu tarp išvesties vertimo ir tikrojo vertimo. Ar galite suprojektuoti mašininio vertimo rezultatų vertinimo matą naudodami natūraliosios kalbos išvadų darymą?
1. Kaip galime pakeisti hiperparametrus, kad sumažintume žodyno dydį?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1388)
:end_tab:
