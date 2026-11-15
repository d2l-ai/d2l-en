# Mokymosi greičio planavimas
:label:`sec_scheduler`

Iki šiol daugiausia dėmesio skyrėme optimizavimo *algoritmams*, t. y. kaip atnaujinti svorių vektorius, o ne *spartai*, kuria jie atnaujinami. Vis dėlto mokymosi greičio koregavimas dažnai yra lygiai toks pat svarbus kaip pats algoritmas. Reikia atsižvelgti į kelis aspektus:

* Akivaizdžiausia, kad mokymosi greičio *dydis* yra svarbus. Jei jis per didelis, optimizavimas diverguoja, jei per mažas, apmokymas trunka per ilgai arba gauname neoptimalų rezultatą. Anksčiau matėme, kad uždavinio sąlygotumo skaičius yra svarbus (išsamiau žr., pvz., :numref:`sec_momentum`). Intuityviai tai yra pokyčio mažiausiai jautria kryptimi ir jautriausia kryptimi santykis.
* Antra, mažėjimo sparta yra lygiai tokia pat svarbi. Jei mokymosi greitis lieka didelis, galime tiesiog šokinėti aplink minimumą ir taip nepasiekti optimalumo. :numref:`sec_minibatch_sgd` tai aptarta gana išsamiai, o :numref:`sec_sgd` analizavome veikimo garantijas. Trumpai tariant, norime, kad sparta mažėtų, bet tikriausiai lėčiau nei $\mathcal{O}(t^{-\frac{1}{2}})$, kuri būtų geras pasirinkimas iškiliesiems uždaviniams.
* Dar vienas aspektas, kuris yra lygiai taip pat svarbus, yra *inicializacija*. Tai susiję ir su tuo, kaip parametrai nustatomi iš pradžių (išsamiau žr. :numref:`sec_numerical_stability`), ir su tuo, kaip jie iš pradžių kinta. Tai vadinama *apšilimu* (angl. *warmup*), t. y. kaip greitai iš pradžių pradedame judėti sprendinio link. Dideli žingsniai pradžioje gali būti nenaudingi, ypač kadangi pradinė parametrų aibė yra atsitiktinė. Pradinės atnaujinimo kryptys taip pat gali būti gana beprasmės.
* Galiausiai yra nemažai optimizavimo variantų, atliekančių ciklišką mokymosi greičio koregavimą. Tai nepatenka į dabartinio skyriaus apimtį. Skaitytojams rekomenduojame susipažinti su detalėmis :citet:`Izmailov.Podoprikhin.Garipov.ea.2018`, pavyzdžiui, kaip gauti geresnius sprendinius vidurkinant per visą parametrų *trajektoriją*.

Turint omenyje, kad mokymosi greičiams valdyti reikia daug detalių, dauguma giliojo mokymosi karkasų turi įrankius tam atlikti automatiškai. Dabartiniame skyriuje apžvelgsime, kokį poveikį tikslumui turi skirtingi grafikai, ir taip pat parodysime, kaip tai galima efektyviai valdyti naudojant *mokymosi greičio planuoklį*.

## Žaislinis uždavinys

Pradedame nuo žaislinio uždavinio, kurį pakankamai pigu skaičiuoti, bet kuris pakankamai nesudėtingas, kad iliustruotų kai kuriuos pagrindinius aspektus. Tam pasirenkame šiek tiek modernizuotą LeNet versiją (`relu` vietoj `sigmoid` aktyvacijos, MaxPooling vietoj AveragePooling), taikomą Fashion-MNIST. Be to, tinklą hibridizuojame našumui padidinti. Kadangi didžioji dalis kodo yra standartinė, tiesiog pristatome pagrindus be tolesnio išsamaus aptarimo. Prireikus atnaujinti žinias žr. :numref:`chap_cnn`.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Pažiūrėkime, kas nutinka, jei šį algoritmą iškviečiame su numatytaisiais nustatymais, pavyzdžiui, $0.3$ mokymosi greičiu, ir apmokome $30$ iteracijų. Atkreipkite dėmesį, kaip apmokymo tikslumas toliau auga, o testavimo tikslumo pažanga po tam tikro taško sustoja. Skirtumas tarp abiejų kreivių rodo persimokymą.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Planuokliai

Vienas mokymosi greičio koregavimo būdas yra jį aiškiai nustatyti kiekviename žingsnyje. Tai patogiai pasiekiama metodu `set_learning_rate`. Galėtume jį mažinti po kiekvienos epochos (ar net po kiekvieno minirinkinio), pavyzdžiui, dinamiškai reaguodami į tai, kaip vyksta optimizavimas.

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

Bendriau, norime apibrėžti planuoklį. Iškviestas su atnaujinimų skaičiumi, jis grąžina atitinkamą mokymosi greičio reikšmę. Apibrėžkime paprastą, kuris nustato mokymosi greitį $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Nubraižykime jo elgesį reikšmių intervale.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Dabar pažiūrėkime, kaip tai pasireiškia apmokant Fashion-MNIST. Planuoklį tiesiog pateikiame kaip papildomą apmokymo algoritmo argumentą.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Tai veikė gerokai geriau nei anksčiau. Išsiskiria du dalykai: kreivė buvo kur kas glotnesnė nei anksčiau. Antra, persimokymo buvo mažiau. Deja, *teoriškai* nėra gerai išspręstas klausimas, kodėl tam tikros strategijos lemia mažesnį persimokymą. Yra tam tikras argumentas, kad mažesnis žingsnio dydis lems parametrus, esančius arčiau nulio, taigi paprastesnius. Tačiau tai visiškai nepaaiškina reiškinio, nes mes iš tikrųjų nestabdome anksti, o tiesiog švelniai mažiname mokymosi greitį.

## Strategijos

Nors negalime aprėpti visos mokymosi greičio planuoklių įvairovės, toliau bandome pateikti trumpą populiarių strategijų apžvalgą. Įprasti pasirinkimai yra polinominis mažėjimas ir dalimis pastovūs grafikai. Be to, kosinuso mokymosi greičio grafikai, kaip nustatyta, empiriškai gerai veikia kai kuriuose uždaviniuose. Galiausiai kai kuriuose uždaviniuose naudinga optimizatorių apšildyti prieš naudojant didelius mokymosi greičius.

### Daugiklio planuoklis

Viena alternatyva polinominiam mažėjimui būtų daugybinis, t. y. $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$, kai $\alpha \in (0, 1)$. Kad mokymosi greitis nesumažėtų žemiau pagrįstos apatinės ribos, atnaujinimo lygtis dažnai modifikuojama į $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Tai taip pat galima pasiekti MXNet įtaisytu planuokliu per objektą `lr_scheduler.FactorScheduler`. Jis priima dar keletą parametrų, pavyzdžiui, apšilimo laikotarpį, apšilimo režimą (tiesinį ar pastovų), didžiausią pageidaujamą atnaujinimų skaičių ir kt.; toliau pagal poreikį naudosime įtaisytus planuoklius ir čia tik paaiškinsime jų funkcionalumą. Kaip iliustruota, prireikus sukurti savo planuoklį yra gana paprasta.

### Daugelio daugiklių planuoklis

Įprasta giliųjų tinklų apmokymo strategija yra išlaikyti mokymosi greitį dalimis pastovų ir kas kiek laiko sumažinti jį tam tikru dydžiu. Tai yra, turint momentų, kada sumažinti greitį, aibę, pavyzdžiui, $s = \{5, 10, 20\}$, mažiname $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ kaskart, kai $t \in s$. Darant prielaidą, kad reikšmės kiekviename žingsnyje perpus sumažinamos, tai galime įgyvendinti taip.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Intuicija už šio dalimis pastovaus mokymosi greičio grafiko yra ta, kad leidžiame optimizavimui vykti, kol pasiekiamas stacionarus taškas svorių vektorių skirstinio atžvilgiu. Tada (ir tik tada) mažiname greitį, kad gautume aukštesnės kokybės gero lokaliojo minimumo pakaitalą. Toliau pateiktas pavyzdys rodo, kaip tai gali duoti vis šiek tiek geresnius sprendinius.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Kosinuso planuoklis

Gana glumią euristiką pasiūlė :citet:`Loshchilov.Hutter.2016`. Ji remiasi pastebėjimu, kad galbūt nenorime pradžioje per smarkiai mažinti mokymosi greičio ir, be to, kad galbūt norime pabaigoje „patobulinti“ sprendinį labai mažu mokymosi greičiu. Taip gaunamas kosinuso pavidalo grafikas su tokia mokymosi greičių funkcine forma, kai $t \in [0, T]$.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


Čia $\eta_0$ yra pradinis mokymosi greitis, o $\eta_T$ – tikslinis greitis laiko momentu $T$. Be to, kai $t > T$, reikšmę tiesiog užfiksuojame ties $\eta_T$ ir vėl nedidiname. Šiame pavyzdyje didžiausią atnaujinimo žingsnį nustatome $T = 20$.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Kompiuterinės regos kontekste šis grafikas *gali* pagerinti rezultatus. Tačiau atkreipkite dėmesį, kad tokie patobulinimai negarantuojami (kaip matyti toliau).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Apšilimas

Kai kuriais atvejais parametrų inicializavimo nepakanka geram sprendiniui garantuoti. Tai ypač problema kai kuriems pažangiems tinklų dizainams, kurie gali lemti nestabilius optimizavimo uždavinius. Tai galėtume spręsti pasirinkdami pakankamai mažą mokymosi greitį, kad išvengtume divergavimo pradžioje. Deja, tai reiškia lėtą pažangą. Ir atvirkščiai, didelis mokymosi greitis iš pradžių lemia divergavimą.

Gana paprastas šios dilemos sprendimas yra naudoti apšilimo laikotarpį, per kurį mokymosi greitis *didėja* iki savo pradinio maksimumo, ir atvėsinti greitį iki optimizavimo proceso pabaigos. Paprastumo dėlei šiam tikslui paprastai naudojamas tiesinis didėjimas. Taip gaunamas toliau nurodytos formos grafikas.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Atkreipkite dėmesį, kad tinklas pradžioje geriau konverguoja (ypač stebėkite veikimą per pirmąsias 5 epochas).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Apšilimas gali būti taikomas bet kuriam planuokliui (ne tik kosinuso). Išsamesnį mokymosi greičio grafikų aptarimą ir daug daugiau eksperimentų žr. :cite:`Gotmare.Keskar.Xiong.ea.2018`. Visų pirma jie nustatė, kad apšilimo fazė riboja parametrų divergavimo apimtį labai giliuose tinkluose. Tai intuityviai prasminga, nes tikėtume didelio divergavimo dėl atsitiktinės inicializacijos tose tinklo dalyse, kurioms pradžioje daugiausia laiko reikia pažangai padaryti.

## Santrauka

* Mokymosi greičio mažinimas apmokymo metu gali pagerinti tikslumą ir (kas glumiausia) sumažinti modelio persimokymą.
* Dalimis mažinti mokymosi greitį, kai tik pažanga pasiekia plynaukštę, praktikoje yra veiksminga. Iš esmės tai užtikrina, kad efektyviai konverguotume į tinkamą sprendinį ir tik tada sumažintume būdingą parametrų dispersiją mažindami mokymosi greitį.
* Kosinuso planuokliai populiarūs kai kuriems kompiuterinės regos uždaviniams. Išsamiau apie tokį planuoklį žr., pvz., [GluonCV](http://gluon-cv.mxnet.io).
* Apšilimo laikotarpis prieš optimizavimą gali apsaugoti nuo divergavimo.
* Optimizavimas giliajame mokymesi atlieka kelis tikslus. Be apmokymo tikslo minimizavimo, skirtingi optimizavimo algoritmų ir mokymosi greičio planavimo pasirinkimai gali lemti gana skirtingą apibendrinimo ir persimokymo kiekį testavimo aibėje (esant tokiam pačiam apmokymo paklaidos dydžiui).

## Pratimai

1. Eksperimentuokite su optimizavimo elgesiu duotam fiksuotam mokymosi greičiui. Koks yra geriausias modelis, kurį taip galite gauti?
1. Kaip keičiasi konvergavimas, jei keičiate mokymosi greičio mažėjimo eksponentę? Eksperimentuose patogumui naudokite `PolyScheduler`.
1. Pritaikykite kosinuso planuoklį dideliems kompiuterinės regos uždaviniams, pvz., apmokant ImageNet. Kaip jis veikia našumą, palyginti su kitais planuokliais?
1. Kiek ilgai turėtų trukti apšilimas?
1. Ar galite susieti optimizavimą ir atranką? Pradėkite naudodami :citet:`Welling.Teh.2011` rezultatus apie stochastinę Langevino dinamiką su gradientu.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1081)
:end_tab:
