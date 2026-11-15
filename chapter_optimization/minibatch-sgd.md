# Minirinkinių stochastinis gradientinis nusileidimas
:label:`sec_minibatch_sgd`

Iki šiol susidūrėme su dviem kraštutinumais gradientiniu požiūriu pagrįstame mokyme: :numref:`sec_gd` gradientams apskaičiuoti ir parametrams atnaujinti naudoja visą duomenų rinkinį, vieną perėjimą vienu metu. Ir atvirkščiai, :numref:`sec_sgd` pažangai daryti apdoroja po vieną apmokymo pavyzdį.
Kiekvienas iš jų turi savų trūkumų.
Gradientinis nusileidimas nėra ypač *efektyvus duomenų požiūriu*, kai duomenys labai panašūs.
Stochastinis gradientinis nusileidimas nėra ypač *efektyvus skaičiavimo požiūriu*, nes CPU ir GPU negali išnaudoti visos vektorizavimo galios.
Tai leidžia manyti, kad gali egzistuoti kažkas tarp jų,
ir iš tiesų būtent tai iki šiol naudojome aptartuose pavyzdžiuose.

## Vektorizavimas ir spartinančioji atmintinė

Sprendimo naudoti minirinkinius esmė yra skaičiavimo efektyvumas. Tai lengviausia suprasti nagrinėjant lygiagretinimą keliems GPU ir keliems serveriams. Tokiu atveju kiekvienam GPU turime išsiųsti bent po vieną vaizdą. Turint 8 GPU serveryje ir 16 serverių, jau gauname ne mažesnį kaip 128 minirinkinio dydį.

Reikalai kiek subtilesni, kai kalbama apie pavienius GPU ar net CPU. Šie įrenginiai turi kelių tipų atmintį, dažnai kelių tipų skaičiavimo vienetus ir skirtingus pralaidumo apribojimus tarp jų.
Pavyzdžiui, CPU turi nedidelį registrų skaičių, o tada L1, L2 ir kai kuriais atvejais net L3 spartinančiąją atmintinę (kuria dalijasi skirtingi procesoriaus branduoliai).
Šios spartinančiosios atmintinės yra didėjančio dydžio ir delsos (kartu mažėjančio pralaidumo).
Pakanka pasakyti, kad procesorius gali atlikti daug daugiau operacijų, nei gali suteikti pagrindinės atminties sąsaja.

Pirma, 2 GHz CPU su 16 branduolių ir AVX-512 vektorizavimu gali apdoroti iki $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ baitų per sekundę. GPU galimybės šį skaičių lengvai viršija 100 kartų. Kita vertus, vidutinės klasės serverio procesorius gali neturėti daug daugiau nei 100 GB/s pralaidumo, t. y. mažiau nei dešimtadalį to, ko reikėtų procesoriui aprūpinti. Kad reikalai būtų dar blogesni, ne visi atminties prieigos veiksmai yra vienodi: atminties sąsajos paprastai yra 64 bitų pločio ar platesnės (pvz., GPU iki 384 bitų), todėl vieno baito nuskaitymas patiria daug platesnės prieigos kainą.

Antra, pirmajai prieigai būdingos didelės pridėtinės sąnaudos, o nuosekli prieiga yra palyginti pigi (tai dažnai vadinama srautiniu skaitymu). Yra daug daugiau dalykų, kuriuos reikia turėti omenyje, pavyzdžiui, spartinimas turint kelis lizdus, lustelius ir kitas struktūras.
Išsamesnį aptarimą žr. šiame [Vikipedijos straipsnyje](https://en.wikipedia.org/wiki/Cache_hierarchy).

Šiems apribojimams sušvelninti naudojama CPU spartinančiųjų atmintinių hierarchija, kuri iš tikrųjų yra pakankamai greita aprūpinti procesorių duomenimis. Tai *pagrindinė* varomoji jėga, slypinti už rinkinių naudojimo giliajame mokymesi. Kad reikalai būtų paprasti, panagrinėkime matricų dauginimą, tarkime, $\mathbf{A} = \mathbf{B}\mathbf{C}$. Turime kelias $\mathbf{A}$ skaičiavimo galimybes. Pavyzdžiui, galėtume išbandyti šiuos:

1. Galėtume apskaičiuoti $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$, t. y. galėtume skaičiuoti po elementą skaliarinėmis sandaugomis.
1. Galėtume apskaičiuoti $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$, t. y. skaičiuoti po vieną stulpelį. Panašiai galėtume skaičiuoti $\mathbf{A}$ po vieną eilutę $\mathbf{A}_{i,:}$.
1. Galėtume tiesiog apskaičiuoti $\mathbf{A} = \mathbf{B} \mathbf{C}$.
1. Galėtume suskaidyti $\mathbf{B}$ ir $\mathbf{C}$ į mažesnes blokines matricas ir skaičiuoti $\mathbf{A}$ po vieną bloką.

Jei vadovausimės pirmąja galimybe, kiekvieną kartą, kai norėsime apskaičiuoti elementą $\mathbf{A}_{ij}$, turėsime nukopijuoti į CPU vieną eilutės ir vieną stulpelio vektorių. Dar blogiau, kadangi matricos elementai išdėstyti nuosekliai, vienam iš dviejų vektorių skaitydami iš atminties turėsime pasiekti daug nesusijusių vietų. Antroji galimybė daug palankesnė. Taikant ją, galime išlaikyti stulpelio vektorių $\mathbf{C}_{:,j}$ CPU spartinančiojoje atmintinėje, kol toliau pereiname per $\mathbf{B}$. Tai perpus sumažina atminties pralaidumo reikalavimą su atitinkamai greitesne prieiga. Žinoma, trečioji galimybė yra pageidautiniausia. Deja, dauguma matricų gali netilpti spartinančiojoje atmintinėje visiškai (juk būtent tai ir aptariame). Tačiau ketvirtoji galimybė siūlo praktiškai naudingą alternatyvą: galime perkelti matricos blokus į spartinančiąją atmintinę ir sudauginti juos lokaliai. Optimizuotos bibliotekos tuo pasirūpina už mus. Pažiūrėkime, kiek efektyvios yra šios operacijos praktikoje.

Be skaičiavimo efektyvumo, Python ir paties giliojo mokymosi karkaso įnešamos pridėtinės sąnaudos yra didelės. Prisiminkite, kad kiekvieną kartą vykdydami komandą, Python interpretatorius išsiunčia komandą MXNet varikliui, kuris turi ją įterpti į skaičiavimo grafą ir tvarkyti planavimo metu. Tokios pridėtinės sąnaudos gali būti gana žalingos. Trumpai tariant, labai patartina kiek įmanoma naudoti vektorizavimą (ir matricas).

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Kadangi likusioje knygos dalyje dažnai matuosime vykdymo laiką, apibrėžkime laikmatį.

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

Priskyrimas po elementą tiesiog iteruoja per visas atitinkamai $\mathbf{B}$ ir $\mathbf{C}$ eilutes ir stulpelius, kad priskirtų reikšmę $\mathbf{A}$.

```{.python .input}
#@tab mxnet
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Greitesnė strategija yra priskyrimas po stulpelį.

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

Galiausiai veiksmingiausias būdas yra atlikti visą operaciją viename bloke.
Atkreipkite dėmesį, kad sudauginant bet kurias dvi matricas $\mathbf{B} \in \mathbb{R}^{m \times n}$ ir $\mathbf{C} \in \mathbb{R}^{n \times p}$, reikia maždaug $2mnp$ slankiojo kablelio operacijų,
kai skaliarų daugyba ir sudėtis skaičiuojamos kaip atskiros operacijos (praktikoje sujungtos).
Taigi dviejų $256 \times 256$ matricų dauginimas
užima $0.03$ milijardo slankiojo kablelio operacijų.
Pažiūrėkime, kokia yra atitinkama operacijų sparta.

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minirinkiniai

:label:`sec_minibatches`

Anksčiau laikėme savaime suprantamu dalyku, kad parametrams atnaujinti skaitysime duomenų *minirinkinius*, o ne pavienius stebėjimus. Dabar pateikiame trumpą jo pagrindimą. Apdorojant pavienius stebėjimus, tenka atlikti daug pavienių matricos ir vektoriaus (ar net vektoriaus ir vektoriaus) daugybų, o tai yra gana brangu ir sukelia didelių pridėtinių sąnaudų, kurias patiria pagrindinis giliojo mokymosi karkasas. Tai taikoma ir tinklo vertinimui, kai jis taikomas duomenims (dažnai vadinamam inferencija), ir gradientų skaičiavimui parametrams atnaujinti. Tai yra, tai taikoma kaskart, kai atliekame $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$, kur

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Šios operacijos *skaičiavimo* efektyvumą galime padidinti taikydami ją vienu metu stebėjimų minirinkiniui. Tai yra, gradientą $\mathbf{g}_t$ vienam stebėjimui pakeičiame gradientu mažam rinkiniui

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Pažiūrėkime, ką tai daro statistinėms $\mathbf{g}_t$ savybėms: kadangi ir $\mathbf{x}_t$, ir visi minirinkinio $\mathcal{B}_t$ elementai atrenkami tolygiai atsitiktinai iš apmokymo aibės, gradiento vidurkis nesikeičia. Tačiau dispersija sumažėja gerokai. Kadangi minirinkinio gradientas sudarytas iš $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ nepriklausomų gradientų, kurie vidurkinami, jo standartinis nuokrypis sumažėja $b^{-\frac{1}{2}}$ kartų. Tai pats savaime yra gerai, nes tai reiškia, kad atnaujinimai patikimiau sulygiuoti su pilnu gradientu.

Naiviai tai rodytų, kad pasirinkti didelį minirinkinį $\mathcal{B}_t$ būtų visuotinai pageidautina. Deja, po tam tikro taško papildomas standartinio nuokrypio sumažėjimas yra minimalus, palyginti su tiesiniu skaičiavimo kainos augimu. Praktikoje pasirenkame minirinkinį, kuris yra pakankamai didelis, kad užtikrintų gerą skaičiavimo efektyvumą, bet vis tiek tilptų į GPU atmintį. Kad iliustruotume sutaupymą, pažvelkime į kodą. Jame atliekame tą pačią matricų daugybą, bet šįkart suskaidytą į „minirinkinius“ po 64 stulpelius vienu metu.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

Kaip matome, skaičiavimas minirinkinyje yra iš esmės toks pat efektyvus kaip su visa matrica. Būtina įspėti. :numref:`sec_batch_norm` naudojome reguliarizavimo tipą, kuris stipriai priklausė nuo dispersijos kiekio minirinkinyje. Didinant pastarąjį, dispersija mažėja, o kartu ir triukšmo įnešimo nauda dėl rinkinio normalizacijos. Išsamiau, kaip keisti mastelį ir apskaičiuoti atitinkamus narius, žr., pvz., :citet:`Ioffe.2017`.

## Duomenų rinkinio skaitymas

Pažiūrėkime, kaip iš duomenų efektyviai generuojami minirinkiniai. Toliau šiems optimizavimo algoritmams palyginti naudojame NASA sukurtą duomenų rinkinį, skirtą [skirtingų orlaivių sparno triukšmui](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) tikrinti. Patogumui naudojame tik pirmuosius $1,500$ pavyzdžių. Duomenys išankstiniam apdorojimui išbalinami, t. y. pašaliname vidurkį ir kiekvienai koordinatei dispersiją pakeičiame į $1$.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Įgyvendinimas nuo nulio

Prisiminkite minirinkinių stochastinio gradientinio nusileidimo įgyvendinimą iš :numref:`sec_linear_scratch`. Toliau pateikiame kiek bendresnį įgyvendinimą. Patogumui jis turi tą patį iškvietimo parašą kaip kiti vėliau šiame skyriuje pristatomi optimizavimo algoritmai. Konkrečiai, pridedame būsenos
įvestį `states` ir hiperparametrą dedame į žodyną `hyperparams`. Be to,
apmokymo funkcijoje vidurkinsime kiekvieno minirinkinio pavyzdžio nuostolį,
todėl optimizavimo algoritme gradiento nereikia
dalyti iš rinkinio dydžio.

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Toliau įgyvendiname bendrąją apmokymo funkciją, kad būtų lengviau naudoti kitus vėliau šiame skyriuje pristatomus optimizavimo algoritmus. Ji inicializuoja tiesinės regresijos modelį ir gali būti naudojama modeliui apmokyti minirinkinių stochastiniu gradientiniu nusileidimu ir kitais vėliau pristatytais algoritmais.

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Pažiūrėkime, kaip optimizavimas vyksta rinkiniu pagrįstam gradientiniam nusileidimui. Tai galima pasiekti nustačius minirinkinio dydį 1500 (t. y. visų pavyzdžių skaičių). Dėl to modelio parametrai atnaujinami tik vieną kartą per epochą. Pažangos mažai. Iš tikrųjų po 6 žingsnių pažanga sustoja.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Kai rinkinio dydis lygus 1, optimizavimui naudojame stochastinį gradientinį nusileidimą. Paprastumo dėlei įgyvendinant pasirinkome pastovų (nors ir mažą) mokymosi greitį. Stochastiniame gradientiniame nusileidime modelio parametrai atnaujinami kaskart apdorojus pavyzdį. Mūsų atveju tai sudaro 1500 atnaujinimų per epochą. Kaip matome, tikslo funkcijos reikšmės mažėjimas sulėtėja po vienos epochos. Nors abi procedūros per vieną epochą apdorojo 1500 pavyzdžių, mūsų eksperimente stochastinis gradientinis nusileidimas sunaudoja daugiau laiko nei gradientinis nusileidimas. Taip yra todėl, kad stochastinis gradientinis nusileidimas atnaujino parametrus dažniau ir kad mažiau efektyvu apdoroti pavienius stebėjimus po vieną.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Galiausiai, kai rinkinio dydis lygus 100, optimizavimui naudojame minirinkinių stochastinį gradientinį nusileidimą. Laikas, reikalingas per epochą, yra trumpesnis nei laikas, reikalingas stochastiniam gradientiniam nusileidimui, ir laikas rinkiniu pagrįstam gradientiniam nusileidimui.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Sumažinus rinkinio dydį iki 10, kiekvienos epochos laikas padidėja, nes kiekvieno rinkinio darbo krūvį vykdyti yra mažiau efektyvu.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Dabar galime palyginti laiką ir nuostolį ankstesniuose keturiuose eksperimentuose. Kaip matyti, nors stochastinis gradientinis nusileidimas konverguoja greičiau nei GD pagal apdorotų pavyzdžių skaičių, tam pačiam nuostoliui pasiekti jis sunaudoja daugiau laiko nei GD, nes gradiento skaičiavimas pavyzdys po pavyzdžio yra mažiau efektyvus. Minirinkinių stochastinis gradientinis nusileidimas gali suderinti konvergavimo greitį ir skaičiavimo efektyvumą. 10 dydžio minirinkinis yra efektyvesnis už stochastinį gradientinį nusileidimą; 100 dydžio minirinkinis pagal vykdymo laiką net lenkia GD.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Glaustas įgyvendinimas

Gluon galime naudoti klasę `Trainer` optimizavimo algoritmams iškviesti. Tai naudojama bendrajai apmokymo funkcijai įgyvendinti. Tai naudosime visame dabartiniame skyriuje.

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Pakartojus paskutinį eksperimentą naudojant Gluon, pastebimas identiškas elgesys.

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Santrauka

* Vektorizavimas daro kodą efektyvesnį dėl sumažėjusių pridėtinių sąnaudų, kylančių iš giliojo mokymosi karkaso, ir dėl geresnės atminties lokalumo bei spartinimo CPU ir GPU.
* Yra kompromisas tarp statistinio efektyvumo, kylančio iš stochastinio gradientinio nusileidimo, ir skaičiavimo efektyvumo, kylančio apdorojant didelius duomenų rinkinius vienu metu.
* Minirinkinių stochastinis gradientinis nusileidimas siūlo geriausia iš abiejų pasaulių: skaičiavimo ir statistinį efektyvumą.
* Minirinkinių stochastiniame gradientiniame nusileidime apdorojame duomenų rinkinius, gautus atsitiktine apmokymo duomenų permutacija (t. y. kiekvienas stebėjimas apdorojamas tik vieną kartą per epochą, nors ir atsitiktine tvarka).
* Apmokymo metu patartina mažinti mokymosi greičius.
* Apskritai minirinkinių stochastinis gradientinis nusileidimas yra greitesnis nei stochastinis gradientinis nusileidimas ir gradientinis nusileidimas konverguojant į mažesnę riziką, matuojant sienos laikrodžio laiku.

## Pratimai

1. Keiskite rinkinio dydį ir mokymosi greitį ir stebėkite tikslo funkcijos reikšmės mažėjimo spartą bei kiekvienai epochai sunaudotą laiką.
1. Perskaitykite MXNet dokumentaciją ir naudokite klasės `Trainer` funkciją `set_learning_rate`, kad po kiekvienos epochos minirinkinių stochastinio gradientinio nusileidimo mokymosi greitį sumažintumėte iki 1/10 ankstesnės reikšmės.
1. Palyginkite minirinkinių stochastinį gradientinį nusileidimą su variantu, kuris iš tikrųjų *atrenka su grąžinimu* iš apmokymo aibės. Kas nutinka?
1. Piktoji džinė jums nesakydama replikuoja jūsų duomenų rinkinį (t. y. kiekvienas stebėjimas pasitaiko du kartus, o jūsų duomenų rinkinys padidėja iki dvigubo pradinio dydžio, bet niekas jums to nepasakė). Kaip pasikeičia stochastinio gradientinio nusileidimo, minirinkinių stochastinio gradientinio nusileidimo ir gradientinio nusileidimo elgesys?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1069)
:end_tab:
