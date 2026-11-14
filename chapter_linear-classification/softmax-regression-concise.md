```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Glaustas softmax regresijos įgyvendinimas
:label:`sec_softmax_concise`



Kaip aukšto lygio giliojo mokymosi karkasai
palengvino tiesinės regresijos įgyvendinimą
(žr. :numref:`sec_linear_concise`),
taip pat patogūs jie ir čia.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Modelio apibrėžimas

Kaip ir :numref:`sec_linear_concise`,
savo pilnai sujungtą sluoksnį sukonstruojame
naudodami įtaisytąjį sluoksnį.
Tada įtaisytasis metodas `__call__` iškviečia `forward`,
kai tik reikia taikyti tinklą kokiai nors įvesčiai.

:begin_tab:`mxnet`
Nors įvestis `X` yra ketvirtos eilės tenzorius,
įtaisytasis sluoksnis `Dense`
automatiškai konvertuos `X` į antros eilės tenzorių,
nekeisdamas dimensijos pagal pirmąją ašį.
:end_tab:

:begin_tab:`pytorch`
Naudojame sluoksnį `Flatten`, kad ketvirtos eilės tenzorių `X` konvertuotume į antros eilės,
nekeisdami dimensijos pagal pirmąją ašį.

:end_tab:

:begin_tab:`tensorflow`
Naudojame sluoksnį `Flatten`, kad ketvirtos eilės tenzorių `X` konvertuotume,
nekeisdami dimensijos pagal pirmąją ašį.
:end_tab:

:begin_tab:`jax`
Flax leidžia naudotojams tinklo klasę rašyti glaustesniu būdu,
naudojant dekoratorių `@nn.compact`. Su `@nn.compact`
galima tiesiog užrašyti visą tinklo logiką viename „tiesioginio sklidimo"
metode, nereikia apibrėžti standartinio metodo `setup`
duomenų klasėje.
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## Dar kartą apie softmax
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch` apskaičiavome savo modelio išvestį
ir pritaikėme kryžminės entropijos nuostolius. Nors tai visiškai
pagrįsta matematiškai, skaičiavimo požiūriu tai rizikinga dėl
skaitinio nepakankamumo ir perpildymo kėlimo eksponente metu.

Prisiminkite, kad softmax funkcija apskaičiuoja tikimybes pagal
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$.
Jei kai kurie $o_k$ yra labai dideli, t. y. labai teigiami,
tada $\exp(o_k)$ gali būti didesnis už didžiausią skaičių,
kurį galima turėti tam tikriems duomenų tipams. Tai vadinama *perpildymu*. Panašiai,
jei kiekvienas argumentas yra labai didelis neigiamas skaičius, gausime *nepakankamumą*.
Pavyzdžiui, vienguba tikslumo slankiojo kablelio skaičiai maždaug
apima diapazoną nuo $10^{-38}$ iki $10^{38}$. Todėl jei didžiausias $\mathbf{o}$ narys
yra už intervalo $[-90, 90]$ ribų, rezultatas nebus stabilus.
Būdas apeiti šią problemą yra atimti $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ iš
visų įrašų:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

Pagal konstrukciją žinome, kad $o_j - \bar{o} \leq 0$ visiems $j$. Todėl $q$ klasių
klasifikavimo uždaviniui vardiklis yra intervale $[1, q]$. Be to,
skaitiklis niekada neviršija $1$, taip išvengiant skaitinio perpildymo. Skaitinis nepakankamumas
atsiranda tik tada, kai $\exp(o_j - \bar{o})$ skaitiškai įvertinamas kaip $0$. Vis dėlto po kelių žingsnių
galime patekti į bėdą, kai norėsime apskaičiuoti $\log \hat{y}_j$ kaip $\log 0$.
Visų pirma atgaliniame sklidime
galime susidurti su ekranu
baisių `NaN` (Not a Number, ne skaičius) rezultatų.

Laimei, mus išgelbsti tai, kad
nors skaičiuojame eksponentines funkcijas,
galiausiai ketiname imti jų logaritmą
(skaičiuodami kryžminės entropijos nuostolius).
Sujungę softmax ir kryžminę entropiją,
galime visiškai išvengti skaitinio stabilumo problemų. Turime:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

Taip išvengiama ir perpildymo, ir nepakankamumo.
Norėsime po ranka turėti įprastą softmax funkciją,
jei kada nors norėtume įvertinti mūsų modelio išvesties tikimybes.
Bet užuot perdavę softmax tikimybes į naują nuostolių funkciją,
tiesiog
[**perduodame logitus ir apskaičiuojame softmax bei jo logaritmą
visa tai iš karto kryžminės entropijos nuostolių funkcijos viduje,**]
kuri daro protingus dalykus, pavyzdžiui, [„LogSumExp triuką"](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## Apmokymas

Toliau apmokome savo modelį. Naudojame Fashion-MNIST vaizdus, išlygintus į 784 dimensijų požymių vektorius.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Kaip ir anksčiau, šis algoritmas konverguoja į sprendinį,
kuris yra pakankamai tikslus,
nors šį kartą naudojant mažiau kodo eilučių nei anksčiau.


## Santrauka

Aukšto lygio API labai patogiai slepia nuo naudotojo potencialiai pavojingus aspektus, tokius kaip skaitinis stabilumas. Be to, jos leidžia naudotojams glaustai projektuoti modelius naudojant labai mažai kodo eilučių. Tai ir palaima, ir prakeiksmas. Akivaizdi nauda yra ta, kad viskas tampa labai prieinama, net inžinieriams, kurie gyvenime nėra lankę nė vienos statistikos paskaitos (iš tiesų jie yra dalis šios knygos tikslinės auditorijos). Tačiau aštrių kampų slėpimas turi ir kainą: nenorą patiems pridėti naujų ir skirtingų komponentų, nes mažai turima įpročio tai daryti. Be to, tai apsunkina dalykų *taisymą*, kai karkaso apsauginis sluoksnis nepadengia visų kraštutinių atvejų. Vėlgi, tai kyla dėl nepakankamo susipažinimo.

Todėl primygtinai raginame peržiūrėti *abi* – ir paprastąją, ir elegantiškąją – daugelio tolesnių įgyvendinimų versijas. Nors pabrėžiame supratimo paprastumą, įgyvendinimai vis dėlto paprastai yra gana našūs (konvoliucijos čia yra didelė išimtis). Mūsų ketinimas – leisti jums remtis jais, kai išrasite ką nors naujo, ko joks karkasas negali suteikti.


## Pratimai

1. Giliajame mokyme naudojama daug skirtingų skaičių formatų, įskaitant FP64 dvigubą tikslumą (naudojamas itin retai),
FP32 vienguba tikslumu, BFLOAT16 (tinka suspaustiems atvaizdams), FP16 (labai nestabilus), TF32 (naujas NVIDIA formatas) ir INT8. Apskaičiuokite mažiausią ir didžiausią eksponentinės funkcijos argumentą, kuriam rezultatas nesukelia skaitinio nepakankamumo ar perpildymo.
1. INT8 yra labai ribotas formatas, sudarytas iš nenulinių skaičių nuo $1$ iki $255$. Kaip galėtumėte išplėsti jo dinaminį diapazoną nenaudodami daugiau bitų? Ar standartinė daugyba ir sudėtis vis dar veikia?
1. Padidinkite apmokymo epochų skaičių. Kodėl validavimo tikslumas po kurio laiko gali sumažėti? Kaip tai galėtume ištaisyti?
1. Kas nutinka didinant mokymosi spartą? Palyginkite nuostolių kreives esant kelioms mokymosi spartoms. Kuri veikia geriau? Kada?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17983)
:end_tab:
