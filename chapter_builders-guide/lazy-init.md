```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Atidėtasis inicializavimas
:label:`sec_lazy_init`

Iki šiol gali atrodyti, kad išsisukome
nerūpestingai kurdami savo tinklus.
Konkrečiai, darėme šiuos neintuityvius dalykus,
kurie gali neatrodyti tokie, kad turėtų veikti:

* Apibrėžėme tinklų architektūras
  nenurodydami įvesties dimensijos.
* Pridėjome sluoksnius nenurodydami
  ankstesnio sluoksnio išvesties dimensijos.
* Net „inicializavome“ šiuos parametrus
  prieš pateikdami pakankamai informacijos, kad būtų galima nustatyti,
  kiek parametrų turėtų turėti mūsų modeliai.

Galbūt nustebsite, kad mūsų kodas apskritai veikia.
Juk giliojo mokymosi karkasas jokiu būdu negalėtų
žinoti, kokia bus tinklo įvesties dimensija.
Gudrybė čia ta, kad karkasas *atideda inicializavimą*,
laukdamas, kol pirmą kartą perduosime duomenis per modelį,
kad lėkdamas išvestų kiekvieno sluoksnio dydžius.


Vėliau, dirbant su konvoliuciniais neuroniniais tinklais,
ši technika taps dar patogesnė,
nes įvesties dimensija
(pvz., vaizdo skyra)
darys įtaką kiekvieno paskesnio sluoksnio
dimensijai.
Todėl galimybė nustatyti parametrus
nereikalaujant žinoti,
rašant kodą,
dimensijos reikšmės,
gali labai supaprastinti mūsų modelių nurodymo
ir vėlesnio keitimo užduotį.
Toliau gilinamės į inicializavimo mechaniką.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

Pradžioje sukurkime MLP egzempliorių.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

Šiuo metu tinklas jokiu būdu negali žinoti
įvesties sluoksnio svorių dimensijų,
nes įvesties dimensija lieka nežinoma.

:begin_tab:`mxnet, pytorch, tensorflow`
Todėl karkasas dar neinicializavo jokių parametrų.
Tai patvirtiname žemiau bandydami pasiekti parametrus.
:end_tab:

:begin_tab:`jax`
Kaip minėta :numref:`subsec_param-access`, parametrai ir tinklo apibrėžimas yra atskirti
Jax ir Flax, ir naudotojas tvarko abu rankiniu būdu. Flax modeliai yra be būsenos,
todėl atributo `parameters` nėra.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Atkreipkite dėmesį, kad nors parametrų objektai egzistuoja,
kiekvieno sluoksnio įvesties dimensija nurodyta kaip -1.
MXNet naudoja ypatingą reikšmę -1 žymėti,
kad parametro dimensija lieka nežinoma.
Šiuo metu bandymai pasiekti `net[0].weight.data()`
sukeltų vykdymo klaidą, nurodančią, kad tinklas
turi būti inicializuotas prieš pasiekiant parametrus.
Dabar pažiūrėkime, kas nutinka, kai bandome inicializuoti
parametrus metodu `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Atkreipkite dėmesį, kad kiekvieno sluoksnio objektai egzistuoja, bet svoriai yra tušti.
Naudojant `net.get_weights()` būtų išmesta klaida, nes svoriai
dar neinicializuoti.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Kaip matome, nieko nepasikeitė.
Kai įvesties dimensijos nežinomos,
inicializavimo kvietimai iš tikrųjų neinicializuoja parametrų.
Vietoj to šis kvietimas MXNet užregistruoja, kad norime
(ir pasirinktinai pagal kokį skirstinį)
inicializuoti parametrus.
:end_tab:

Toliau perduokime duomenis per tinklą,
kad karkasas pagaliau inicializuotų parametrus.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

Vos tik sužinome įvesties dimensiją,
20,
karkasas gali nustatyti pirmojo sluoksnio svorių matricos formą, įstatydamas reikšmę 20.
Atpažinęs pirmojo sluoksnio formą, karkasas pereina
prie antrojo sluoksnio
ir taip toliau per skaičiavimo grafą,
kol bus žinomos visos formos.
Atkreipkite dėmesį, kad šiuo atveju
atidėtojo inicializavimo reikia tik pirmajam sluoksniui,
bet karkasas inicializuoja nuosekliai.
Kai tik žinomos visų parametrų formos,
karkasas pagaliau gali inicializuoti parametrus.

:begin_tab:`pytorch`
Šis metodas
perduoda fiktyvias įvestis
per tinklą
bandomajam paleidimui,
kad išvestų visų parametrų formas
ir vėliau inicializuotų parametrus.
Jis bus naudojamas vėliau, kai numatytieji atsitiktiniai inicializavimai nepageidaujami.
:end_tab:

:begin_tab:`jax`
Parametrų inicializavimas Flax visada atliekamas rankiniu būdu ir jį tvarko
naudotojas. Šis metodas kaip argumentus priima fiktyvią įvestį ir raktų žodyną.
Šiame raktų žodyne yra ATS (atsitiktinių skaičių generatoriai) modelio parametrams inicializuoti
ir dropout ATS dropout kaukei generuoti modeliuose
su dropout sluoksniais. Daugiau apie dropout bus aptarta vėliau :numref:`sec_dropout`.
Galiausiai metodas inicializuoja modelį, grąžindamas parametrus.
Jį naudojome „po gaubtu“ ir ankstesniuose skyreliuose.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## Santrauka

Atidėtasis inicializavimas gali būti patogus, leisdamas karkasui automatiškai išvesti parametrų formas, todėl lengva keisti architektūras ir pašalinamas vienas dažnas klaidų šaltinis.
Galime perduoti duomenis per modelį, kad karkasas pagaliau inicializuotų parametrus.


## Pratimai

1. Kas nutinka, jei nurodote įvesties dimensijas pirmajam sluoksniui, bet ne paskesniems? Ar inicializavimas įvyksta iš karto?
1. Kas nutinka, jei nurodote nesutampančias dimensijas?
1. Ką turėtumėte daryti, jei turite kintamos dimensijos įvestį? Užuomina: pažiūrėkite į parametrų susiejimą.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17992)
:end_tab:
