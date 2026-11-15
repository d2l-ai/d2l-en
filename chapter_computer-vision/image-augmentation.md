# Vaizdų augmentacija
:label:`sec_image_augmentation`

:numref:`sec_alexnet`
minėjome, kad dideli duomenų rinkiniai
yra būtina sąlyga
giliųjų neuroninių tinklų sėkmei
įvairiuose taikymuose.
*Vaizdų augmentacija*
sukuria panašius, bet skirtingus apmokymo pavyzdžius
po eilės atsitiktinių apmokymo vaizdų pakeitimų, taip padidindama apmokymo aibės dydį.
Alternatyviai,
vaizdų augmentaciją galima motyvuoti tuo,
kad
atsitiktiniai apmokymo pavyzdžių pakeitimai
leidžia modeliams mažiau pasikliauti
tam tikrais požymiais, taip pagerindami jų apibendrinimo gebą.
Pavyzdžiui, galime apkarpyti vaizdą skirtingais būdais, kad dominantis objektas atsirastų skirtingose vietose, taip sumažindami modelio priklausomybę nuo objekto padėties.
Taip pat galime reguliuoti tokius veiksnius kaip ryškumas ir spalva, kad sumažintume modelio jautrumą spalvai.
Tikriausiai tiesa,
kad vaizdų augmentacija buvo nepakeičiama
AlexNet sėkmei tuo metu.
Šiame skyrelyje aptarsime šią plačiai kompiuterinėje regoje naudojamą techniką.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## Įprasti vaizdų augmentacijos metodai

Nagrinėdami įprastus vaizdų augmentacijos metodus, kaip pavyzdį naudosime šį $400\times 500$ dydžio vaizdą.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

Dauguma vaizdų augmentacijos metodų turi tam tikrą atsitiktinumo laipsnį. Kad būtų lengviau stebėti vaizdų augmentacijos poveikį, toliau apibrėžiame pagalbinę funkciją `apply`. Ši funkcija kelis kartus paleidžia vaizdų augmentacijos metodą `aug` įvesties vaizdui `img` ir parodo visus rezultatus.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Apvertimas ir apkarpymas

:begin_tab:`mxnet`
[**Vaizdo apvertimas kairėn ir dešinėn**] paprastai nekeičia objekto kategorijos.
Tai vienas ankstyviausių ir plačiausiai naudojamų vaizdų augmentacijos metodų.
Toliau naudojame modulį `transforms`, kad sukurtume `RandomFlipLeftRight` egzempliorių, kuris
apverčia vaizdą kairėn ir dešinėn su 50 % tikimybe.
:end_tab:

:begin_tab:`pytorch`
[**Vaizdo apvertimas kairėn ir dešinėn**] paprastai nekeičia objekto kategorijos.
Tai vienas ankstyviausių ir plačiausiai naudojamų vaizdų augmentacijos metodų.
Toliau naudojame modulį `transforms`, kad sukurtume `RandomHorizontalFlip` egzempliorių, kuris
apverčia vaizdą kairėn ir dešinėn su 50 % tikimybe.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**Apvertimas aukštyn ir žemyn**] nėra toks įprastas kaip apvertimas kairėn ir dešinėn. Tačiau bent jau šiam pavyzdiniam vaizdui apvertimas aukštyn ir žemyn netrukdo atpažinimui.
Toliau sukuriame `RandomFlipTopBottom` egzempliorių, kuris apverčia
vaizdą aukštyn ir žemyn su 50 % tikimybe.
:end_tab:

:begin_tab:`pytorch`
[**Apvertimas aukštyn ir žemyn**] nėra toks įprastas kaip apvertimas kairėn ir dešinėn. Tačiau bent jau šiam pavyzdiniam vaizdui apvertimas aukštyn ir žemyn netrukdo atpažinimui.
Toliau sukuriame `RandomVerticalFlip` egzempliorių, kuris apverčia
vaizdą aukštyn ir žemyn su 50 % tikimybe.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

Mūsų naudotame pavyzdiniame vaizde katė yra vaizdo viduryje, tačiau apskritai taip gali nebūti.
:numref:`sec_pooling` paaiškinome, kad sutelkimo sluoksnis gali sumažinti konvoliucinio sluoksnio jautrumą tikslo padėčiai.
Be to, galime atsitiktinai apkarpyti vaizdą, kad objektai skirtingose vaizdo vietose atsirastų skirtingais masteliais, o tai taip pat gali sumažinti modelio jautrumą tikslo padėčiai.

Toliau pateiktame kode [**atsitiktinai apkerpame**] kiekvieną kartą $10\% \sim 100\%$ originalaus ploto sritį, o šios srities pločio ir aukščio santykis atsitiktinai parenkamas iš $0{,}5 \sim 2$. Tada srities plotis ir aukštis abu pakeičiami iki 200 pikselių.
Jei nenurodyta kitaip, atsitiktinis skaičius tarp $a$ ir $b$ šiame skyrelyje reiškia tolydžią reikšmę, gautą atsitiktinai ir tolygiai imant imtį iš intervalo $[a, b]$.

```{.python .input}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Spalvų keitimas

Dar vienas augmentacijos metodas yra spalvų keitimas. Galime keisti keturis vaizdo spalvos aspektus: ryškumą, kontrastą, sodrumą ir atspalvį. Toliau pateiktame pavyzdyje [**atsitiktinai keičiame vaizdo ryškumą**] iki reikšmės tarp 50 % ($1-0{,}5$) ir 150 % ($1+0{,}5$) originalaus vaizdo.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Panašiai galime [**atsitiktinai keisti vaizdo atspalvį**].

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Taip pat galime sukurti `RandomColorJitter` egzempliorių ir nurodyti, kaip [**vienu metu atsitiktinai keisti vaizdo `brightness`, `contrast`, `saturation` ir `hue`**].

```{.python .input}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Kelių vaizdų augmentacijos metodų derinimas

Praktikoje [**derinsime kelis vaizdų augmentacijos metodus**].
Pavyzdžiui,
galime derinti aukščiau apibrėžtus skirtingus vaizdų augmentacijos metodus ir taikyti juos kiekvienam vaizdui per `Compose` egzempliorių.

```{.python .input}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**Mokymas su vaizdų augmentacija**]

Apmokykime modelį su vaizdų augmentacija.
Čia naudojame CIFAR-10 duomenų rinkinį vietoj anksčiau naudoto Fashion-MNIST duomenų rinkinio.
Taip yra todėl, kad objektų padėtis ir dydis Fashion-MNIST duomenų rinkinyje yra normalizuoti, o objektų spalva ir dydis CIFAR-10 duomenų rinkinyje skiriasi žymiai labiau.
Toliau parodomi pirmieji 32 CIFAR-10 duomenų rinkinio apmokymo vaizdai.

```{.python .input}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Kad prognozavimo metu gautume apibrėžtus rezultatus, vaizdų augmentaciją paprastai taikome tik apmokymo pavyzdžiams, o prognozavimo metu nenaudojame vaizdų augmentacijos su atsitiktinėmis operacijomis.
[**Čia naudojame tik paprasčiausią atsitiktinio apvertimo kairėn ir dešinėn metodą**]. Be to, naudojame `ToTensor` egzempliorių, kad vaizdų minirinkinį paverstume formatu, kurio reikalauja giliojo mokymosi karkasas, t. y.
32 bitų slankiojo kablelio skaičiais tarp 0 ir 1 su forma (rinkinio dydis, kanalų skaičius, aukštis, plotis).

```{.python .input}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
Toliau apibrėžiame pagalbinę funkciją, kad palengvintume vaizdo skaitymą ir
vaizdų augmentacijos taikymą.
Gluon duomenų rinkinių pateikiama funkcija `transform_first` taiko vaizdų augmentaciją pirmajam kiekvieno apmokymo
pavyzdžio elementui (vaizdui ir žymeniui), t. y. vaizdui.
Išsamų `DataLoader` pristatymą žr. :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`pytorch`
Toliau [**apibrėžiame pagalbinę funkciją, kad palengvintume vaizdo skaitymą ir
vaizdų augmentacijos taikymą**].
PyTorch duomenų rinkinio pateikiamas argumentas `transform` taiko augmentaciją vaizdams transformuoti.
Išsamų `DataLoader` pristatymą žr. :numref:`sec_fashion_mnist`.
:end_tab:

```{.python .input}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Mokymas keliais GPU

Apmokome ResNet-18 modelį iš
:numref:`sec_resnet`
CIFAR-10 duomenų rinkinyje.
Prisiminkite kelių GPU apmokymo pristatymą
:numref:`sec_multi_gpu_concise`.
Toliau
[**apibrėžiame funkciją modeliui apmokyti ir įvertinti naudojant kelis GPU**].

```{.python .input}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Dabar galime [**apibrėžti funkciją `train_with_data_aug` modeliui apmokyti su vaizdų augmentacija**].
Ši funkcija gauna visus prieinamus GPU,
naudoja Adam kaip optimizavimo algoritmą,
taiko vaizdų augmentaciją apmokymo duomenų rinkiniui
ir galiausiai iškviečia ką tik apibrėžtą funkciją `train_ch13` modeliui apmokyti ir įvertinti.

```{.python .input}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

[**Apmokykime modelį**] naudodami vaizdų augmentaciją, pagrįstą atsitiktiniu apvertimu kairėn ir dešinėn.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Santrauka

* Vaizdų augmentacija sukuria atsitiktinius vaizdus remdamasi esamais apmokymo duomenimis, kad pagerintų modelių apibendrinimo gebą.
* Kad prognozavimo metu gautume apibrėžtus rezultatus, vaizdų augmentaciją paprastai taikome tik apmokymo pavyzdžiams, o prognozavimo metu nenaudojame vaizdų augmentacijos su atsitiktinėmis operacijomis.
* Giliojo mokymosi karkasai pateikia daug skirtingų vaizdų augmentacijos metodų, kuriuos galima taikyti vienu metu.


## Pratimai

1. Apmokykite modelį nenaudodami vaizdų augmentacijos: `train_with_data_aug(test_augs, test_augs)`. Palyginkite apmokymo ir testavimo tikslumą naudojant vaizdų augmentaciją ir jos nenaudojant. Ar šis lyginamasis eksperimentas gali pagrįsti teiginį, kad vaizdų augmentacija gali sušvelninti persimokymą? Kodėl?
1. Derinkite kelis skirtingus vaizdų augmentacijos metodus apmokydami modelį CIFAR-10 duomenų rinkinyje. Ar tai pagerina testavimo tikslumą?
1. Peržiūrėkite giliojo mokymosi karkaso internetinę dokumentaciją. Kokių dar vaizdų augmentacijos metodų jis pateikia?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1404)
:end_tab:
