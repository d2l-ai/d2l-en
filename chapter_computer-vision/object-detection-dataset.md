# Objektų aptikimo duomenų rinkinys
:label:`sec_object-detection-dataset`

Objektų aptikimo srityje nėra mažo duomenų rinkinio, kaip MNIST ar Fashion-MNIST.
Kad greitai parodytume objektų aptikimo modelius,
[**surinkome ir pažymėjome nedidelį duomenų rinkinį**].
Pirmiausia nufotografavome nemokamus bananus savo biure
ir sugeneravome
1000 bananų vaizdų su skirtingais pasukimais ir dydžiais.
Tada kiekvieną bananų vaizdą patalpinome
atsitiktinėje kai kurio fono vaizdo vietoje.
Galiausiai pažymėjome tų bananų apibrėžiamuosius stačiakampius vaizduose.


## [**Duomenų rinkinio atsisiuntimas**]

Bananų aptikimo duomenų rinkinį su visais vaizdų ir
csv žymenų failais galima atsisiųsti tiesiogiai iš interneto.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Duomenų rinkinio skaitymas

Toliau pateiktoje funkcijoje `read_data_bananas`
[**nuskaitysime bananų aptikimo duomenų rinkinį**].
Duomenų rinkinyje yra csv failas
objektų klasių žymenims ir
tikrųjų apibrėžiamųjų stačiakampių koordinatėms
viršutiniame kairiajame ir apatiniame dešiniajame kampuose.

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

Naudodami funkciją `read_data_bananas` vaizdams ir žymenims skaityti,
toliau pateikta klasė `BananasDataset`
leis mums [**sukurti pritaikytą `Dataset` egzempliorių**]
bananų aptikimo duomenų rinkiniui įkelti.

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Galiausiai apibrėžiame
funkciją `load_data_bananas`, kad [**grąžintume du
duomenų iteratoriaus egzempliorius apmokymo ir testavimo aibėms.**]
Testavimo duomenų rinkiniui
nereikia skaityti atsitiktine tvarka.

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

[**Nuskaitykime minirinkinį ir atspausdinkime
vaizdų ir žymenų formas**] šiame minirinkinyje.
Vaizdų minirinkinio forma
(rinkinio dydis, kanalų skaičius, aukštis, plotis)
atrodo pažįstama:
ji tokia pati kaip ankstesnėse vaizdų klasifikavimo užduotyse.
Žymenų minirinkinio forma yra
(rinkinio dydis, $m$, 5),
kur $m$ yra didžiausias galimas apibrėžiamųjų stačiakampių skaičius,
kurį turi bet kuris duomenų rinkinio vaizdas.

Nors skaičiavimai minirinkiniais yra efektyvesni,
jiems reikia, kad visi vaizdų pavyzdžiai
turėtų tiek pat apibrėžiamųjų stačiakampių, kad konkatenacija būtų galima sudaryti minirinkinį.
Apskritai
vaizdai gali turėti skirtingą apibrėžiamųjų stačiakampių skaičių;
todėl
vaizdai, turintys mažiau nei $m$ apibrėžiamųjų stačiakampių,
bus papildyti neleistinais apibrėžiamaisiais stačiakampiais,
kol bus pasiektas $m$.
Tada
kiekvieno apibrėžiamojo stačiakampio žymuo vaizduojamas 5 ilgio masyvu.
Pirmasis masyvo elementas yra apibrėžiamajame stačiakampyje esančio objekto klasė,
kur -1 rodo neleistiną apibrėžiamąjį stačiakampį papildymui.
Likę keturi masyvo elementai yra
apibrėžiamojo stačiakampio viršutinio kairiojo ir apatinio dešiniojo kampų
($x$, $y$) koordinačių reikšmės
(intervalas tarp 0 ir 1).
Bananų duomenų rinkiniui,
kadangi kiekviename vaizde yra tik vienas apibrėžiamasis stačiakampis,
turime $m=1$.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Demonstracija**]

Parodykime dešimt vaizdų su jų pažymėtais tikraisiais apibrėžiamaisiais stačiakampiais.
Matome, kad bananų pasukimai, dydžiai ir padėtys visuose šiuose vaizduose skiriasi.
Žinoma, tai tik paprastas dirbtinis duomenų rinkinys.
Praktikoje realaus pasaulio duomenų rinkiniai paprastai yra daug sudėtingesni.

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Santrauka

* Mūsų surinktas bananų aptikimo duomenų rinkinys gali būti naudojamas objektų aptikimo modeliams demonstruoti.
* Objektų aptikimo duomenų įkėlimas panašus į vaizdų klasifikavimo. Tačiau objektų aptikime žymenyse taip pat yra informacijos apie tikruosius apibrėžiamuosius stačiakampius, kurios vaizdų klasifikavime nėra.


## Pratimai

1. Parodykite kitus bananų aptikimo duomenų rinkinio vaizdus su tikraisiais apibrėžiamaisiais stačiakampiais. Kuo jie skiriasi apibrėžiamųjų stačiakampių ir objektų atžvilgiu?
1. Tarkime, kad norime objektų aptikimui taikyti duomenų augmentaciją, pavyzdžiui, atsitiktinį apkarpymą. Kuo ji gali skirtis nuo vaizdų klasifikavimo? Užuomina: kas, jei apkarpytame vaizde yra tik nedidelė objekto dalis?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1608)
:end_tab:
