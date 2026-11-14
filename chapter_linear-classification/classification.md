```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Bazinis klasifikavimo modelis
:label:`sec_classification`

Galbūt pastebėjote, kad regresijos atveju įgyvendinimai nuo nulio ir glaustas įgyvendinimas naudojant karkaso funkcionalumą buvo gana panašūs. Tas pats pasakytina apie klasifikavimą. Kadangi daugelis šios knygos modelių nagrinėja klasifikavimą, verta pridėti funkcionalumo, specialiai palaikančio šią aplinką. Šiame skyrelyje pateikiama bazinė klasifikavimo modelių klasė, supaprastinsianti būsimą kodą.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Klasė `Classifier`

:begin_tab:`pytorch, mxnet, tensorflow`
Žemiau apibrėžiame klasę `Classifier`. Metode `validation_step` pateikiame ir nuostolių reikšmę, ir klasifikavimo tikslumą validavimo minirinkinyje. Atnaujinimą braižome kas `num_val_batches` minirinkinių. Tai turi privalumą: gaunami suvidurkinti nuostoliai ir tikslumas visuose validavimo duomenyse. Šie vidurkiai nėra visiškai tikslūs, jei paskutiniame minirinkinyje yra mažiau pavyzdžių, bet šio nedidelio skirtumo nepaisome, kad kodas būtų paprastas.
:end_tab:


:begin_tab:`jax`
Žemiau apibrėžiame klasę `Classifier`. Metode `validation_step` pateikiame ir nuostolių reikšmę, ir klasifikavimo tikslumą validavimo minirinkinyje. Atnaujinimą braižome kas `num_val_batches` minirinkinių. Tai turi privalumą: gaunami suvidurkinti nuostoliai ir tikslumas visuose validavimo duomenyse. Šie vidurkiai nėra visiškai tikslūs, jei paskutiniame minirinkinyje yra mažiau pavyzdžių, bet šio nedidelio skirtumo nepaisome, kad kodas būtų paprastas.

JAX taip pat iš naujo apibrėžiame metodą `training_step`, nes visi modeliai, kurie
vėliau paveldės iš `Classifier`, turės nuostolius, grąžinančius pagalbinius duomenis.
Šie pagalbiniai duomenys gali būti naudojami modeliams su paketine normalizacija
(bus paaiškinta :numref:`sec_batch_norm`), o visais kitais atvejais
nuostoliai taip pat grąžins vietos rezervavimo ženklą (tuščią žodyną),
kad atvaizduotų pagalbinius duomenis.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

Pagal numatytuosius nustatymus naudojame stochastinio gradientinio nusileidimo optimizatorių, veikiantį minirinkiniuose, kaip ir tiesinės regresijos atveju.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Tikslumas

Turint prognozuotą tikimybių skirstinį `y_hat`,
paprastai pasirenkame klasę su didžiausia prognozuota tikimybe,
kai tik turime pateikti griežtą prognozę.
Iš tiesų daugeliui taikymų reikia, kad atliktume pasirinkimą.
Pavyzdžiui, „Gmail" turi suskirstyti el. laišką į „Pagrindinis", „Socialiniai", „Naujienos", „Forumai" arba „Šlamštas".
Jis gali viduje vertinti tikimybes,
bet galiausiai turi pasirinkti vieną iš klasių.

Kai prognozės atitinka žymens klasę `y`, jos yra teisingos.
Klasifikavimo tikslumas yra teisingų prognozių dalis tarp visų prognozių.
Nors tikslumą tiesiogiai optimizuoti gali būti sunku (jis nėra diferencijuojamas),
dažnai tai yra veikimo matas, kuris mums rūpi labiausiai. Jis dažnai yra *pats*
svarbiausias dydis lyginamuosiuose rinkiniuose. Todėl jį beveik visada pateiksime apmokydami klasifikatorius.

Tikslumas apskaičiuojamas taip.
Pirmiausia, jei `y_hat` yra matrica,
darome prielaidą, kad antrojoje dimensijoje saugomi kiekvienos klasės prognozės balai.
Naudojame `argmax`, kad gautume prognozuotą klasę pagal didžiausio įrašo kiekvienoje eilutėje indeksą.
Tada [**palyginame prognozuotą klasę su tikrąja `y` elementas po elemento.**]
Kadangi lygybės operatorius `==` jautrus duomenų tipams,
`y_hat` duomenų tipą konvertuojame, kad atitiktų `y` tipą.
Rezultatas yra tenzorius, kurio įrašai yra 0 (netiesa) ir 1 (tiesa).
Susumavus gaunamas teisingų prognozių skaičius.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Santrauka

Klasifikavimas yra pakankamai dažnas uždavinys, kad nusipelno savo paties patogumo funkcijų. Klasifikavime pagrindinę reikšmę turi klasifikatoriaus *tikslumas*. Atkreipkite dėmesį, kad nors mums dažnai pirmiausia rūpi tikslumas, klasifikatorius apmokome optimizuoti įvairius kitus tikslus dėl statistinių ir skaičiavimo priežasčių. Tačiau nepriklausomai nuo to, kuri nuostolių funkcija buvo minimizuota apmokymo metu, naudinga turėti patogumo metodą mūsų klasifikatoriaus tikslumui empiriškai įvertinti.


## Pratimai

1. Pažymėkime $L_\textrm{v}$ validavimo nuostolius, o $L_\textrm{v}^\textrm{q}$ – jų greitą ir nešvarų įvertį, apskaičiuotą šiame skyrelyje nuostolių vidurkinimu. Galiausiai pažymėkime $l_\textrm{v}^\textrm{b}$ nuostolius paskutiniame minirinkinyje. Išreikškite $L_\textrm{v}$ per $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$ ir imties bei minirinkinio dydžius.
1. Parodykite, kad greitas ir nešvarus įvertis $L_\textrm{v}^\textrm{q}$ yra nepaslinktas. T. y. parodykite, kad $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Kodėl vis tiek norėtumėte naudoti $L_\textrm{v}$?
1. Turint daugiaklasės klasifikacijos nuostolius, kur $l(y,y')$ žymi baudą už įvertį $y'$, kai matome $y$, ir turint tikimybę $p(y \mid x)$, suformuluokite optimalaus $y'$ pasirinkimo taisyklę. Užuomina: išreikškite tikėtinus nuostolius naudodami $l$ ir $p(y \mid x)$.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17981)
:end_tab:
