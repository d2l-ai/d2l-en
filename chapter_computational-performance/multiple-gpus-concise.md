# Glaustas įgyvendinimas keliems GPU
:label:`sec_multi_gpu_concise`

Įgyvendinti lygiagretinimą nuo nulio kiekvienam naujam modeliui nėra malonu. Be to, yra didelė nauda optimizuojant sinchronizavimo įrankius dideliam našumui. Toliau parodysime, kaip tai padaryti naudojant aukšto lygio giliojo mokymosi karkasų API.
Matematika ir algoritmai yra tokie patys kaip :numref:`sec_multi_gpu`.
Visiškai nenuostabu, kad šio skyrelio kodui vykdyti jums reikės bent dviejų GPU.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Žaislinis tinklas**]

Naudokime šiek tiek prasmingesnį tinklą nei LeNet iš :numref:`sec_multi_gpu`, kuris vis dar pakankamai lengvas ir greitai apmokomas.
Pasirenkame ResNet-18 variantą :cite:`He.Zhang.Ren.ea.2016`. Kadangi įvesties vaizdai yra maži, jį šiek tiek modifikuojame. Visų pirma, skirtumas nuo :numref:`sec_resnet` yra tas, kad pradžioje naudojame mažesnį konvoliucijos branduolį, žingsnį ir papildymą.
Be to, pašaliname maksimalaus sutraukimo sluoksnį.

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Tinklo inicializavimas

:begin_tab:`mxnet`
Funkcija `initialize` leidžia mums inicializuoti parametrus pasirinktame įrenginyje.
Inicializavimo metodų priminimui žr. :numref:`sec_numerical_stability`. Ypač patogu tai, kad ji taip pat leidžia inicializuoti tinklą *keliuose* įrenginiuose vienu metu. Pabandykime, kaip tai veikia praktiškai.
:end_tab:

:begin_tab:`pytorch`
Tinklą inicializuosime apmokymo ciklo viduje.
Inicializavimo metodų priminimui žr. :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
Naudodami :numref:`sec_multi_gpu` pristatytą funkciją `split_and_load`, galime padalyti duomenų minirinkinį ir nukopijuoti dalis į įrenginių sąrašą, pateiktą kintamuoju `devices`. Tinklo egzempliorius *automatiškai* naudoja tinkamą GPU tiesioginio sklidimo reikšmei apskaičiuoti. Čia sugeneruojame 4 stebinius ir padalijame juos tarp GPU.
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Kai duomenys praeina per tinklą, atitinkami parametrai inicializuojami *įrenginyje, per kurį praėjo duomenys*.
Tai reiškia, kad inicializavimas vyksta kiekvienam įrenginiui atskirai. Kadangi inicializavimui pasirinkome GPU 0 ir GPU 1, tinklas inicializuojamas tik ten, o ne CPU. Iš tikrųjų parametrų CPU net nėra. Tai galime patikrinti atspausdinę parametrus ir stebėdami galimas klaidas.
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Toliau pakeiskime kodą, skirtą [**tikslumui įvertinti**], tokiu, kuris veikia (**lygiagrečiai keliuose įrenginiuose**). Tai pakeičia :numref:`sec_lenet` funkciją `evaluate_accuracy_gpu`. Pagrindinis skirtumas yra tas, kad prieš iškviesdami tinklą padalijame minirinkinį. Visa kita iš esmės identiška.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Apmokymas**]

Kaip ir anksčiau, apmokymo kodas turi atlikti kelias pagrindines funkcijas efektyviam lygiagretinimui:

* Tinklo parametrai turi būti inicializuoti visuose įrenginiuose.
* Iteruojant per duomenų rinkinio minirinkinius, jie turi būti padalijami tarp visų įrenginių.
* Nuostolį ir jo gradientą apskaičiuojame lygiagrečiai per įrenginius.
* Gradientai agreguojami ir parametrai atitinkamai atnaujinami.

Pabaigoje apskaičiuojame tikslumą (vėlgi lygiagrečiai), kad pateiktume galutinį tinklo našumą. Apmokymo procedūra yra gana panaši į ankstesnių skyrių įgyvendinimus, išskyrus tai, kad turime padalyti ir agreguoti duomenis.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Pažiūrėkime, kaip tai veikia praktiškai. Apšilimui [**apmokome tinklą viename GPU.**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Toliau [**apmokymui naudojame 2 GPU**]. Palyginti su LeNet,
įvertintu :numref:`sec_multi_gpu`,
ResNet-18 modelis yra gerokai sudėtingesnis. Čia lygiagretinimas parodo savo pranašumą. Skaičiavimo laikas yra prasmingai didesnis nei parametrų sinchronizavimo laikas. Tai pagerina mastelio didėjimą, nes lygiagretinimo pridėtinės sąnaudos yra mažiau svarbios.

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Santrauka

:begin_tab:`mxnet`
* Gluon suteikia primityvus modelio inicializavimui keliuose įrenginiuose pateikiant konteksto sąrašą.
:end_tab:
* Duomenys automatiškai įvertinami įrenginiuose, kuriuose jie randami.
* Pasirūpinkite, kad tinklai būtų inicializuoti kiekviename įrenginyje prieš bandant pasiekti to įrenginio parametrus. Kitaip susidursite su klaida.
* Optimizavimo algoritmai automatiškai agreguoja per kelis GPU.



## Pratimai

:begin_tab:`mxnet`
1. Šiame skyrelyje naudojamas ResNet-18. Išbandykite skirtingas epochas, rinkinio dydžius ir mokymosi greičius. Skaičiavimams naudokite daugiau GPU. Kas nutinka, jei tai išbandysite su 16 GPU (pvz., AWS p2.16xlarge egzemplioriuje)?
1. Kartais skirtingi įrenginiai suteikia skirtingą skaičiavimo galią. Galėtume naudoti GPU ir CPU vienu metu. Kaip turėtume padalyti darbą? Ar tai vertas pastangų? Kodėl? Kodėl ne?
1. Kas nutinka, jei atsisakome `npx.waitall()`? Kaip modifikuotumėte apmokymą, kad lygiagretinimui būtų iki dviejų žingsnių persidengimas?
:end_tab:

:begin_tab:`pytorch`
1. Šiame skyrelyje naudojamas ResNet-18. Išbandykite skirtingas epochas, rinkinio dydžius ir mokymosi greičius. Skaičiavimams naudokite daugiau GPU. Kas nutinka, jei tai išbandysite su 16 GPU (pvz., AWS p2.16xlarge egzemplioriuje)?
1. Kartais skirtingi įrenginiai suteikia skirtingą skaičiavimo galią. Galėtume naudoti GPU ir CPU vienu metu. Kaip turėtume padalyti darbą? Ar tai vertas pastangų? Kodėl? Kodėl ne?
:end_tab:



:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1403)
:end_tab:
