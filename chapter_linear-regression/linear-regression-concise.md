```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Glaustas tiesinės regresijos įgyvendinimas
:label:`sec_linear_concise`

Pastarąjį dešimtmetį giliojo mokymosi srityje įvyko tarsi
Kambro sprogimas.
Metodų, taikymų ir algoritmų gausa gerokai pranoksta
ankstesnių dešimtmečių pažangą.
Tai lėmė laimingas kelių veiksnių derinys,
vienas iš kurių yra galingi nemokami įrankiai,
kuriuos siūlo keli atvirojo kodo giliojo mokymosi karkasai.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`
ir Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
galima laikyti
pirmąja tokių karkasų karta,
sulaukusia plataus pritaikymo.
Skirtingai nuo ankstesnių (pamatinių) darbų, tokių kaip
SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`,
kuris teikė į Lisp panašią programavimo patirtį,
šiuolaikiniai karkasai siūlo automatinį diferencijavimą
ir Python patogumą.
Šie karkasai leidžia mums automatizuoti ir moduliarizuoti
pasikartojantį gradientiniu mokymusi pagrįstų algoritmų įgyvendinimo darbą.

:numref:`sec_linear_scratch` rėmėmės tik
(i) tenzoriais duomenims saugoti ir tiesinei algebrai;
ir (ii) automatiniu diferencijavimu gradientams skaičiuoti.
Praktikoje, kadangi duomenų iteratoriai, nuostolių funkcijos, optimizatoriai
ir neuroninių tinklų sluoksniai
yra tokie dažni, šiuolaikinės bibliotekos įgyvendina ir šiuos komponentus už mus.
Šiame skyrelyje (**parodysime, kaip įgyvendinti
tiesinės regresijos modelį**) iš :numref:`sec_linear_scratch`
(**glaustai naudojant aukšto lygio giliojo mokymosi karkasų API**).

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Modelio apibrėžimas

Kai :numref:`sec_linear_scratch` įgyvendinome tiesinę regresiją
nuo nulio,
aiškiai apibrėžėme savo modelio parametrus
ir užkodavome skaičiavimus išvesčiai gauti
naudodami pagrindines tiesinės algebros operacijas.
Jūs *turėtumėte* mokėti tai daryti.
Bet kai jūsų modeliai taps sudėtingesni
ir kai tai reikės daryti beveik kasdien,
džiaugsitės pagalba.
Situacija panaši į savo tinklaraščio kūrimą nuo nulio.
Padaryti tai vieną ar du kartus yra naudinga ir pamokoma,
bet būtumėte prastas interneto kūrėjas,
jei mėnesį praleistumėte išradinėdami dviratį.

Standartinėms operacijoms
galime [**naudoti karkaso iš anksto apibrėžtus sluoksnius,**]
kurie leidžia mums sutelkti dėmesį
į modeliui sudaryti naudojamus sluoksnius,
o ne rūpintis jų įgyvendinimu.
Prisiminkite vieno sluoksnio tinklo architektūrą,
aprašytą :numref:`fig_single_neuron`.
Sluoksnis vadinamas *pilnai sujungtu*,
nes kiekviena jo įvestis sujungta
su kiekviena jo išvestimi
per matricos ir vektoriaus daugybą.

:begin_tab:`mxnet`
Gluon pilnai sujungtas sluoksnis apibrėžiamas klasėje `Dense`.
Kadangi norime sugeneruoti tik vieną skaliarinę išvestį,
tą skaičių nustatome lygų 1.
Verta pažymėti, kad dėl patogumo
Gluon nereikalauja, kad nurodytume
kiekvieno sluoksnio įvesties formą.
Todėl mums nereikia pasakyti Gluon,
kiek įvesčių patenka į šį tiesinį sluoksnį.
Kai pirmą kartą perduodame duomenis per savo modelį,
pvz., kai vėliau vykdome `net(X)`,
Gluon automatiškai nustatys kiekvieno sluoksnio įvesčių skaičių ir
taip sukurs teisingą modelį.
Kaip tai veikia, aprašysime išsamiau vėliau.
:end_tab:

:begin_tab:`pytorch`
PyTorch pilnai sujungtas sluoksnis apibrėžiamas klasėse `Linear` ir `LazyLinear` (prieinamose nuo versijos 1.8.0).
Pastaroji
leidžia vartotojams nurodyti *tik*
išvesties dimensiją,
o pirmoji
papildomai klausia,
kiek įvesčių patenka į šį sluoksnį.
Įvesties formų nurodymas yra nepatogus ir gali reikalauti netrivialių skaičiavimų
(pvz., konvoliuciniuose sluoksniuose).
Todėl dėl paprastumo, kai tik galėsime, naudosime tokius „tingius" sluoksnius.
:end_tab:

:begin_tab:`tensorflow`
Keras pilnai sujungtas sluoksnis apibrėžiamas klasėje `Dense`.
Kadangi norime sugeneruoti tik vieną skaliarinę išvestį,
tą skaičių nustatome lygų 1.
Verta pažymėti, kad dėl patogumo
Keras nereikalauja, kad nurodytume
kiekvieno sluoksnio įvesties formą.
Mums nereikia pasakyti Keras,
kiek įvesčių patenka į šį tiesinį sluoksnį.
Kai pirmą kartą bandome perduoti duomenis per savo modelį,
pvz., kai vėliau vykdome `net(X)`,
Keras automatiškai nustatys
kiekvieno sluoksnio įvesčių skaičių.
Kaip tai veikia, aprašysime išsamiau vėliau.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

Metode `forward` tiesiog iškviečiame iš anksto apibrėžtų sluoksnių įtaisytąjį metodą `__call__`, kad apskaičiuotume išvestis.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## Nuostolių funkcijos apibrėžimas

:begin_tab:`mxnet`
Modulis `loss` apibrėžia daug naudingų nuostolių funkcijų.
Dėl spartos ir patogumo atsisakome įgyvendinti savo
ir vietoj to pasirenkame įtaisytąją `loss.L2Loss`.
Kadangi jos grąžinami `loss` yra
kiekvieno pavyzdžio kvadratinė paklaida,
naudojame `mean`, kad suvidurkintume nuostolius per minirinkinį.
:end_tab:

:begin_tab:`pytorch`
[**Klasė `MSELoss` apskaičiuoja vidutinę kvadratinę paklaidą (be $1/2$ daugiklio :eqref:`eq_mse`).**]
Pagal numatytuosius nustatymus `MSELoss` grąžina vidutinius nuostolius per pavyzdžius.
Tai greičiau (ir lengviau naudoti) nei įgyvendinti savo.
:end_tab:

:begin_tab:`tensorflow`
Klasė `MeanSquaredError` apskaičiuoja vidutinę kvadratinę paklaidą (be $1/2$ daugiklio :eqref:`eq_mse`).
Pagal numatytuosius nustatymus ji grąžina vidutinius nuostolius per pavyzdžius.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## Optimizavimo algoritmo apibrėžimas

:begin_tab:`mxnet`
Minirinkinio SGD yra standartinė priemonė
neuroniniams tinklams optimizuoti,
todėl Gluon jį palaiko kartu su keliomis
šio algoritmo variacijomis per savo klasę `Trainer`.
Atkreipkite dėmesį, kad Gluon klasė `Trainer` reiškia
optimizavimo algoritmą,
o :numref:`sec_oo-design` sukurta klasė `Trainer`
talpina apmokymo metodą,
t. y. pakartotinai kviečia optimizatorių
modelio parametrams atnaujinti.
Kai kuriame `Trainer` egzempliorių,
nurodome parametrus, kuriuos optimizuosime,
gaunamus iš mūsų modelio `net` per `net.collect_params()`,
optimizavimo algoritmą, kurį norime naudoti (`sgd`),
ir hiperparametrų žodyną,
reikalingą mūsų optimizavimo algoritmui.
:end_tab:

:begin_tab:`pytorch`
Minirinkinio SGD yra standartinė priemonė
neuroniniams tinklams optimizuoti,
todėl PyTorch jį palaiko kartu su keliomis
šio algoritmo variacijomis modulyje `optim`.
Kai (**kuriame `SGD` egzempliorių,**)
nurodome parametrus, kuriuos optimizuosime,
gaunamus iš mūsų modelio per `self.parameters()`,
ir mokymosi spartą (`self.lr`),
reikalingą mūsų optimizavimo algoritmui.
:end_tab:

:begin_tab:`tensorflow`
Minirinkinio SGD yra standartinė priemonė
neuroniniams tinklams optimizuoti,
todėl Keras jį palaiko kartu su keliomis
šio algoritmo variacijomis modulyje `optimizers`.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## Apmokymas

Galbūt pastebėjote, kad modelio išreiškimas per
giliojo mokymosi karkaso aukšto lygio API
reikalauja mažiau kodo eilučių.
Mums nereikėjo atskirai skirstyti parametrų,
apibrėžti nuostolių funkcijos ar įgyvendinti minirinkinio SGD.
Kai pradėsime dirbti su gerokai sudėtingesniais modeliais,
aukšto lygio API pranašumai gerokai išaugs.

Dabar, kai turime visas pagrindines dalis,
[**pats apmokymo ciklas yra toks pat,
kaip ir įgyvendintasis nuo nulio.**]
Todėl tiesiog iškviečiame metodą `fit` (pristatytą :numref:`oo-design-training`),
kuris remiasi metodo `fit_epoch` įgyvendinimu
:numref:`sec_linear_scratch`,
kad apmokytume savo modelį.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Žemiau
[**palyginame modelio parametrus, išmoktus
apmokant baigtiniuose duomenyse,
su tikraisiais parametrais**],
sugeneravusiais mūsų duomenų rinkinį.
Parametrams pasiekti
kreipiamės į reikiamo sluoksnio
svorius ir poslinkį.
Kaip ir įgyvendinime nuo nulio,
atkreipkite dėmesį, kad mūsų įvertinti parametrai
artimi tikriesiems.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Santrauka

Šiame skyrelyje yra pirmasis (šioje knygoje)
gilaus tinklo įgyvendinimas, pasinaudojantis patogumais,
kuriuos suteikia šiuolaikiniai giliojo mokymosi karkasai,
tokie kaip MXNet :cite:`Chen.Li.Li.ea.2015`,
JAX :cite:`Frostig.Johnson.Leary.2018`,
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`
ir Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`.
Duomenims įkelti, sluoksniui, nuostolių funkcijai, optimizatoriui ir apmokymo ciklui apibrėžti
naudojome karkaso numatytuosius nustatymus.
Kai tik karkasas pateikia visas reikalingas funkcijas,
paprastai gera mintis jas naudoti,
nes bibliotekų šių komponentų įgyvendinimai
paprastai būna labai optimizuoti spartai
ir tinkamai išbandyti patikimumui.
Kartu stenkitės nepamiršti,
kad šiuos modulius *galima* įgyvendinti tiesiogiai.
Tai ypač svarbu siekiantiems tapti tyrėjais,
norintiems būti modelių kūrimo priešakyje,
kur išrasite naujų komponentų,
kurių negali būti jokioje dabartinėje bibliotekoje.

:begin_tab:`mxnet`
Gluon modulis `data` teikia duomenų apdorojimo įrankius,
modulis `nn` apibrėžia daugybę neuroninių tinklų sluoksnių,
o modulis `loss` apibrėžia daug įprastų nuostolių funkcijų.
Be to, `initializer` suteikia prieigą
prie daugelio parametrų inicializavimo pasirinkimų.
Vartotojui patogiai
dimensija ir saugojimas nustatomi automatiškai.
Šios tingios inicializacijos pasekmė yra ta,
kad negalima bandyti pasiekti parametrų,
kol jie nesukurti (ir neinicializuoti).
:end_tab:

:begin_tab:`pytorch`
PyTorch modulis `data` teikia duomenų apdorojimo įrankius,
modulis `nn` apibrėžia daugybę neuroninių tinklų sluoksnių ir įprastų nuostolių funkcijų.
Parametrus galime inicializuoti pakeisdami jų reikšmes
metodais, kurių pavadinimas baigiasi `_`.
Atkreipkite dėmesį, kad turime nurodyti tinklo įvesties dimensijas.
Nors kol kas tai nesudėtinga, tai gali turėti reikšmingų grandininių pasekmių,
kai norime suprojektuoti sudėtingus tinklus su daug sluoksnių.
Norint užtikrinti perkeliamumą, reikia kruopščiai apsvarstyti,
kaip parametrizuoti šiuos tinklus.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow modulis `data` teikia duomenų apdorojimo įrankius,
modulis `keras` apibrėžia daugybę neuroninių tinklų sluoksnių ir įprastų nuostolių funkcijų.
Be to, modulis `initializers` teikia įvairius modelio parametrų inicializavimo metodus.
Tinklų dimensija ir saugojimas nustatomi automatiškai
(bet būkite atsargūs ir nebandykite pasiekti parametrų, kol jie neinicializuoti).
:end_tab:

## Pratimai

1. Kaip reikėtų pakeisti mokymosi spartą, jei suminius nuostolius per minirinkinį
   pakeistumėte nuostolių vidurkiu per minirinkinį?
1. Peržiūrėkite karkaso dokumentaciją ir pažiūrėkite, kokios nuostolių funkcijos pateikiamos. Visų pirma
   pakeiskite kvadratinius nuostolius Huberio robastine nuostolių funkcija. Tai yra, naudokite nuostolių funkciją
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
1. Kaip pasiekti modelio svorių gradientą?
1. Kokį poveikį sprendiniui turi mokymosi spartos ir epochų skaičiaus keitimas? Ar jis ir toliau gerėja?
1. Kaip keičiasi sprendinys, keičiant sugeneruotų duomenų kiekį?
    1. Nubraižykite įvertinimo paklaidą $\hat{\mathbf{w}} - \mathbf{w}$ ir $\hat{b} - b$ kaip duomenų kiekio funkciją. Užuomina: didinkite duomenų kiekį logaritmiškai, o ne tiesiškai, t. y. 5, 10, 20, 50, ..., 10 000, o ne 1000, 2000, ..., 10 000.
    2. Kodėl užuominoje pateiktas patarimas yra tinkamas?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17977)
:end_tab:
