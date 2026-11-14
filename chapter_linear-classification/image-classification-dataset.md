```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Vaizdų klasifikavimo duomenų rinkinys
:label:`sec_fashion_mnist`

(~~MNIST duomenų rinkinys yra vienas plačiai naudojamų vaizdų klasifikavimo duomenų rinkinių, tačiau kaip lyginamasis duomenų rinkinys jis yra per paprastas. Naudosime panašų, bet sudėtingesnį Fashion-MNIST duomenų rinkinį~~)

Vienas plačiai naudojamas vaizdų klasifikavimo duomenų rinkinys yra ranka rašytų skaitmenų [MNIST duomenų rinkinys](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`. Išleistas XX a. dešimtajame dešimtmetyje, jis buvo didelis iššūkis daugumai mašininio mokymosi algoritmų: jį sudaro 60 000 $28 \times 28$ pikselių skyros vaizdų (plius 10 000 vaizdų testavimo duomenų rinkinys). Kad suvoktume mastelį, 1995 m. „Sun SPARCStation 5" su didžiuliais 64 MB RAM ir žaibišku 5 MFLOPs buvo laikoma moderniausia mašininio mokymosi įranga AT&T Bell Laboratories. Aukštas skaitmenų atpažinimo tikslumas buvo svarbi JAV pašto tarnybos (USPS) laiškų rūšiavimo automatizavimo dalis dešimtajame dešimtmetyje. Giliuosius tinklus, tokius kaip LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`, atraminių vektorių metodus su invariantiškumais :cite:`Scholkopf.Burges.Vapnik.1996` ir liestinės atstumo klasifikatorius :cite:`Simard.LeCun.Denker.ea.1998` visus galėjo pasiekti žemesnę nei 1 % klaidų dalį.

Daugiau nei dešimtmetį MNIST buvo *pagrindinis* atskaitos taškas lyginant mašininio mokymosi algoritmus.
Nors kaip lyginamasis duomenų rinkinys jis tarnavo gerai,
net paprasti šiandienos standartais modeliai pasiekia daugiau nei 95 % klasifikavimo tikslumą,
todėl jis netinka stipriems modeliams atskirti nuo silpnesnių. Be to, šis duomenų rinkinys leidžia pasiekti *labai* aukštą tikslumą, kokio paprastai nematome daugelyje klasifikavimo uždavinių. Tai iškreipė algoritmų plėtrą link tam tikrų algoritmų šeimų, galinčių pasinaudoti švariais duomenų rinkiniais, tokių kaip aktyviosios aibės metodai ir ribą ieškantys aktyviosios aibės algoritmai.
Šiandien MNIST tarnauja veikiau kaip sveiko proto patikrinimas, o ne kaip lyginamasis rinkinys. ImageNet :cite:`Deng.Dong.Socher.ea.2009` kelia gerokai
aktualesnį iššūkį. Deja, ImageNet yra per didelis daugeliui šios knygos pavyzdžių ir iliustracijų, nes jo apmokymas truktų per ilgai, kad pavyzdžiai būtų interaktyvūs. Kaip pakaitalą, tolesniuose skyreliuose sutelksime dėmesį į kokybiškai panašų, bet gerokai mažesnį Fashion-MNIST
duomenų rinkinį :cite:`Xiao.Rasul.Vollgraf.2017`, išleistą 2017 m. Jame yra 10 drabužių kategorijų vaizdų $28 \times 28$ pikselių skyros.

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## Duomenų rinkinio įkėlimas

Kadangi Fashion-MNIST duomenų rinkinys toks naudingas, visi pagrindiniai karkasai pateikia jo iš anksto apdorotas versijas. Galime [**atsisiųsti ir nuskaityti jį į atmintį naudodami įtaisytąsias karkaso priemones.**]

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST sudaro 10 kategorijų vaizdai, kurių kiekviena atvaizduojama
6000 vaizdų apmokymo duomenų rinkinyje ir 1000 – testavimo duomenų rinkinyje.
*Testavimo duomenų rinkinys* naudojamas modelio veikimui įvertinti (jo negalima naudoti apmokymui).
Todėl apmokymo ir testavimo aibėse
yra atitinkamai 60 000 ir 10 000 vaizdų.

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

Vaizdai yra pilkumo atspalvių ir aukščiau padidinti iki $32 \times 32$ pikselių skyros. Tai panašu į pradinį MNIST duomenų rinkinį, kurį sudarė (dvejetainiai) juodai balti vaizdai. Tačiau atkreipkite dėmesį, kad dauguma šiuolaikinių vaizdų duomenų turi tris kanalus (raudoną, žalią, mėlyną), o hiperspektriniai vaizdai gali turėti daugiau nei 100 kanalų (HyMap jutiklis turi 126 kanalus).
Pagal susitarimą vaizdą saugome kaip $c \times h \times w$ tenzorių, kur $c$ yra spalvų kanalų skaičius, $h$ – aukštis, o $w$ – plotis.

```{.python .input}
%%tab all
data.train[0][0].shape
```

[~~Dvi pagalbinės funkcijos duomenų rinkiniui vizualizuoti~~]

Fashion-MNIST kategorijos turi žmonėms suprantamus pavadinimus.
Šis patogumo metodas konvertuoja skaitines žymenų reikšmes į jų pavadinimus ir atvirkščiai.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Minirinkinio skaitymas

Kad palengvintume savo gyvenimą skaitydami iš apmokymo ir testavimo aibių,
naudojame įtaisytąjį duomenų iteratorių, užuot kūrę jį nuo nulio.
Prisiminkite, kad kiekvienoje iteracijoje duomenų iteratorius
[**nuskaito `batch_size` dydžio duomenų minirinkinį.**]
Taip pat atsitiktinai permaišome pavyzdžius apmokymo duomenų iteratoriui.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

Kad pamatytume, kaip tai veikia, įkelkime vaizdų minirinkinį iškviesdami metodą `train_dataloader`. Jame yra 64 vaizdai.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Pažiūrėkime, kiek laiko užtrunka vaizdų skaitymas. Nors tai įtaisytasis krautuvas, jis nėra žaibiškai greitas. Vis dėlto to pakanka, nes vaizdų apdorojimas giliuoju tinklu trunka kur kas ilgiau. Todėl pakankamai gerai, kad tinklo apmokymą neribotų įvesties ir išvesties spartos.

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Vizualizavimas

Fashion-MNIST duomenų rinkinį naudosime dažnai. Patogumo funkcija `show_images` gali būti naudojama vaizdams ir susijusiems žymenims vizualizuoti.
Praleisdami įgyvendinimo detales, žemiau tiesiog parodome sąsają: mums reikia žinoti tik kaip iškviesti `d2l.show_images`, o ne kaip ji veikia,
kalbant apie tokias pagalbines funkcijas.

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

Pasinaudokime ja. Apskritai gera mintis vizualizuoti ir apžiūrėti duomenis, su kuriais apmokote.
Žmonės labai gerai pastebi keistenybes, todėl vizualizavimas tarnauja kaip papildoma apsauga nuo klaidų ir trūkumų planuojant eksperimentus. Čia pateikiami [**vaizdai ir jų atitinkami žymenys**] (tekstu)
pirmiesiems keliems apmokymo duomenų rinkinio pavyzdžiams.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

Dabar esame pasirengę dirbti su Fashion-MNIST duomenų rinkiniu tolesniuose skyreliuose.

## Santrauka

Dabar turime šiek tiek realistiškesnį duomenų rinkinį klasifikavimui. Fashion-MNIST yra aprangos klasifikavimo duomenų rinkinys, sudarytas iš vaizdų, atvaizduojančių 10 kategorijų. Šį duomenų rinkinį naudosime tolesniuose skyreliuose ir skyriuose įvairiems tinklų dizainams įvertinti – nuo paprasto tiesinio modelio iki pažangių liekamųjų tinklų. Kaip įprasta dirbant su vaizdais, juos skaitome kaip tenzorių formos (minirinkinio dydis, kanalų skaičius, aukštis, plotis). Kol kas turime tik vieną kanalą, nes vaizdai yra pilkumo atspalvių (aukščiau pateikta vizualizacija naudoja netikrųjų spalvų paletę geresniam matomumui).

Galiausiai duomenų iteratoriai yra pagrindinis efektyvaus veikimo komponentas. Pavyzdžiui, galime naudoti GPU efektyviam vaizdų dekompresavimui, vaizdo įrašų perkodavimui ar kitam išankstiniam apdorojimui. Kai tik įmanoma, turėtumėte remtis gerai įgyvendintais duomenų iteratoriais, išnaudojančiais didelio našumo skaičiavimus, kad nesulėtintumėte apmokymo ciklo.


## Pratimai

1. Ar `batch_size` mažinimas (pavyzdžiui, iki 1) veikia skaitymo našumą?
1. Duomenų iteratoriaus našumas yra svarbus. Ar manote, kad dabartinis įgyvendinimas pakankamai greitas? Išnagrinėkite įvairias galimybes jam pagerinti. Naudokite sistemos profiliuotoją, kad išsiaiškintumėte, kur yra kliūtys.
1. Peržiūrėkite karkaso internetinę API dokumentaciją. Kokie dar duomenų rinkiniai prieinami?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17980)
:end_tab:
