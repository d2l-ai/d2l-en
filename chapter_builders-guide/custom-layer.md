```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Pasirinktiniai sluoksniai

Vienas iš giliojo mokymosi sėkmės veiksnių
yra platus sluoksnių, kuriuos galima kūrybiškai derinti,
spektras, leidžiantis projektuoti architektūras,
tinkamas labai įvairioms užduotims.
Pavyzdžiui, tyrėjai išrado sluoksnius,
skirtus būtent vaizdams, tekstui,
ciklams per sekų duomenis
ir
dinaminiam programavimui atlikti.
Anksčiau ar vėliau jums prireiks
sluoksnio, kurio dar nėra giliojo mokymosi karkase.
Tokiais atvejais turite sukurti pasirinktinį sluoksnį.
Šiame skyrelyje parodome, kaip tai padaryti.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

## (**Sluoksniai be parametrų**)

Pradžioje sukonstruojame pasirinktinį sluoksnį,
kuris neturi savų parametrų.
Tai turėtų atrodyti pažįstama, jei prisimenate mūsų
modulių pristatymą :numref:`sec_model_construction`.
Ši klasė `CenteredLayer` tiesiog
atima vidurkį iš savo įvesties.
Kad ją sukurtume, tereikia paveldėti
iš bazinės sluoksnio klasės ir įgyvendinti tiesioginio sklidimo funkciją.

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

Patikrinkime, ar mūsų sluoksnis veikia, kaip numatyta, paduodami jam šiek tiek duomenų.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Dabar galime [**įtraukti savo sluoksnį kaip komponentą
konstruodami sudėtingesnius modelius.**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

Kaip papildomą sveiko proto patikrą, galime paduoti atsitiktinius duomenis
per tinklą ir patikrinti, ar vidurkis iš tiesų yra 0.
Kadangi dirbame su slankiojo kablelio skaičiais,
vis tiek galime pamatyti labai mažą nenulinį skaičių
dėl kvantavimo.

:begin_tab:`jax`
Čia naudojame metodą `init_with_output`, kuris grąžina ir tinklo išvestį,
ir parametrus. Šiuo atveju daugiausia dėmesio skiriame
išvesčiai.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**Sluoksniai su parametrais**]

Dabar, kai žinome, kaip apibrėžti paprastus sluoksnius,
pereikime prie sluoksnių su parametrais,
kuriuos galima koreguoti apmokymo metu, apibrėžimo.
Parametrams kurti galime naudoti įtaisytąsias funkcijas, kurios
suteikia šiek tiek pagrindinių buities darbų funkcionalumo.
Visų pirma jos tvarko modelio parametrų pasiekimą, inicializavimą,
bendrinimą, išsaugojimą ir įkėlimą.
Tokiu būdu, be kitų privalumų, mums nereikės rašyti
pasirinktinių serializavimo rutinų kiekvienam pasirinktiniam sluoksniui.

Dabar įgyvendinkime savo pilnai sujungto sluoksnio versiją.
Prisiminkite, kad šiam sluoksniui reikia dviejų parametrų:
vieno svoriui atvaizduoti, kito – poslinkiui.
Šiame įgyvendinime ReLU aktyvaciją įdedame kaip numatytąją.
Šiam sluoksniui reikia dviejų įvesties argumentų: `in_units` ir `units`, kurie
žymi atitinkamai įvesčių ir išvesčių skaičių.

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
Toliau sukuriame klasės `MyDense` egzempliorių
ir pasiekiame jo modelio parametrus.
:end_tab:

:begin_tab:`pytorch`
Toliau sukuriame klasės `MyLinear` egzempliorių
ir pasiekiame jo modelio parametrus.
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

Galime [**tiesiogiai atlikti tiesioginio sklidimo skaičiavimus naudodami pasirinktinius sluoksnius.**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

Taip pat galime (**konstruoti modelius naudodami pasirinktinius sluoksnius.**)
Kai tai turime, galime jį naudoti lygiai taip pat kaip įtaisytąjį pilnai sujungtą sluoksnį.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## Santrauka

Pasirinktinius sluoksnius galime projektuoti per bazinę sluoksnio klasę. Tai leidžia apibrėžti lanksčius naujus sluoksnius, kurie elgiasi kitaip nei bet kurie bibliotekoje esantys sluoksniai.
Apibrėžti pasirinktiniai sluoksniai gali būti kviečiami savavališkuose kontekstuose ir architektūrose.
Sluoksniai gali turėti vietinių parametrų, kuriuos galima sukurti įtaisytosiomis funkcijomis.


## Pratimai

1. Suprojektuokite sluoksnį, kuris priima įvestį ir apskaičiuoja tenzoriaus redukciją,
   t. y. grąžina $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Suprojektuokite sluoksnį, kuris grąžina pirmąją duomenų Furjė koeficientų pusę.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17993)
:end_tab:
