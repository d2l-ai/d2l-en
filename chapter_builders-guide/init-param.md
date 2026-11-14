```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Parametrų inicializavimas

Dabar, kai žinome, kaip pasiekti parametrus,
pažiūrėkime, kaip juos tinkamai inicializuoti.
Tinkamo inicializavimo būtinybę aptarėme :numref:`sec_numerical_stability`.
Giliojo mokymosi karkasas savo sluoksniams numato atsitiktinius inicializavimus.
Tačiau dažnai norime inicializuoti savo svorius
pagal įvairius kitus protokolus. Karkasas suteikia dažniausiai
naudojamus protokolus ir taip pat leidžia sukurti pasirinktinį inicializatorių.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
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

:begin_tab:`mxnet`
Pagal numatytuosius nustatymus MXNet inicializuoja svorių parametrus atsitiktinai semdamas iš tolygiojo skirstinio $U(-0.07, 0.07)$,
o poslinkio parametrus nustato lygius nuliui.
MXNet modulis `init` suteikia įvairių
iš anksto nustatytų inicializavimo metodų.
:end_tab:

:begin_tab:`pytorch`
Pagal numatytuosius nustatymus PyTorch inicializuoja svorių ir poslinkių matricas
tolygiai semdamas iš intervalo, kuris apskaičiuojamas pagal įvesties ir išvesties dimensiją.
PyTorch modulis `nn.init` suteikia įvairių
iš anksto nustatytų inicializavimo metodų.
:end_tab:

:begin_tab:`tensorflow`
Pagal numatytuosius nustatymus Keras inicializuoja svorių matricas tolygiai semdamas iš intervalo, kuris apskaičiuojamas pagal įvesties ir išvesties dimensiją, o poslinkio parametrai visi nustatomi lygūs nuliui.
TensorFlow suteikia įvairių inicializavimo metodų ir šakniniame modulyje, ir modulyje `keras.initializers`.
:end_tab:

:begin_tab:`jax`
Pagal numatytuosius nustatymus Flax inicializuoja svorius naudodamas `jax.nn.initializers.lecun_normal`,
t. y. semdamas imtis iš sutrumpinto normaliojo skirstinio, centruoto ties 0, kurio
standartinis nuokrypis nustatytas kaip kvadratinė šaknis iš $1 / \textrm{fan}_{\textrm{in}}$,
kur `fan_in` yra svorių tenzoriaus įvesties vienetų skaičius. Poslinkio
parametrai visi nustatomi lygūs nuliui.
Jax modulis `nn.initializers` suteikia įvairių
iš anksto nustatytų inicializavimo metodų.
:end_tab:

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**Įtaisytasis inicializavimas**]

Pradėkime kviesdami įtaisytuosius inicializatorius.
Žemiau pateiktas kodas inicializuoja visus svorių parametrus
kaip Gauso atsitiktinius dydžius
su standartiniu nuokrypiu 0,01, o poslinkio parametrus nustato lygius nuliui.

```{.python .input}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

Taip pat galime inicializuoti visus parametrus
tam tikra pastovia reikšme (tarkime, 1).

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**Taip pat galime taikyti skirtingus inicializatorius tam tikriems blokams.**]
Pavyzdžiui, žemiau inicializuojame pirmąjį sluoksnį
Xavier inicializatoriumi,
o antrąjį sluoksnį –
pastovia reikšme 42.

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**Pasirinktinis inicializavimas**]

Kartais mums reikalingų inicializavimo metodų
giliojo mokymosi karkasas nesuteikia.
Žemiau pateiktame pavyzdyje apibrėžiame inicializatorių
bet kuriam svorio parametrui $w$, naudodami šį keistą skirstinį:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Čia apibrėžiame klasės `Initializer` poklasę.
Paprastai mums tereikia įgyvendinti funkciją `_init_weight`,
kuri priima tenzoriaus argumentą (`data`)
ir priskiria jam norimas inicializuotas reikšmes.
:end_tab:

:begin_tab:`pytorch`
Vėlgi, įgyvendiname funkciją `my_init`, kad pritaikytume ją `net`.
:end_tab:

:begin_tab:`tensorflow`
Čia apibrėžiame `Initializer` poklasę ir įgyvendiname funkciją `__call__`,
kuri grąžina norimą tenzorių, turint formą ir duomenų tipą.
:end_tab:

:begin_tab:`jax`
Jax inicializavimo funkcijos kaip argumentus priima `PRNGKey`, `shape` ir
`dtype`. Čia įgyvendiname funkciją `my_init`, kuri grąžina norimą
tenzorių, turint formą ir duomenų tipą.
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
Atkreipkite dėmesį, kad visada turime galimybę
parametrus nustatyti tiesiogiai.
:end_tab:

:begin_tab:`jax`
Inicializuojant parametrus JAX ir Flax, grąžinamas parametrų žodynas
yra tipo `flax.core.frozen_dict.FrozenDict`. JAX ekosistemoje nerekomenduojama
tiesiogiai keisti masyvo reikšmių, todėl duomenų tipai
paprastai yra nekeičiami. Pakeitimams atlikti galima naudoti `params.unfreeze()`.
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Santrauka

Parametrus galime inicializuoti naudodami įtaisytuosius ir pasirinktinius inicializatorius.

## Pratimai

Susiraskite internetinę dokumentaciją, kad rastumėte daugiau įtaisytųjų inicializatorių.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17991)
:end_tab:
