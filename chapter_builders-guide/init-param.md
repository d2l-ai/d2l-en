# Initialisation des paramètres

Maintenant que nous savons comment accéder aux paramètres,
voyons comment les initialiser correctement.
Nous avons discuté de la nécessité d'une initialisation correcte dans :numref:`sec_numerical_stability` .
Le cadre d'apprentissage profond fournit des initialisations aléatoires par défaut à ses couches.
Cependant, nous souhaitons souvent initialiser nos poids
en fonction de divers autres protocoles. Le cadre fournit les protocoles les plus couramment utilisés
et permet également de créer un initialisateur personnalisé.

:begin_tab:`mxnet`
Par défaut, MXNet initialise les paramètres de poids en tirant au hasard à partir d'une distribution uniforme $U(-0.07, 0.07)$,
en remettant les paramètres de biais à zéro.
Le module `init` de MXNet fournit une variété
de méthodes d'initialisation prédéfinies.
:end_tab:

:begin_tab:`pytorch`
Par défaut, PyTorch initialise les matrices de poids et de biais
uniformément en tirant d'une plage calculée en fonction de la dimension d'entrée et de sortie.
Le module `nn.init` de PyTorch fournit une variété
de méthodes d'initialisation prédéfinies.
:end_tab:

:begin_tab:`tensorflow`
Par défaut, Keras initialise les matrices de poids de manière uniforme en les tirant d'une plage calculée en fonction de la dimension d'entrée et de sortie, et les paramètres de biais sont tous définis à zéro.
TensorFlow fournit une variété de méthodes d'initialisation à la fois dans le module racine et dans le module `keras.initializers`.
:end_tab:

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input  n=3}
%%tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input  n=4}
%%tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

## [**Initialisation intégrée**]

Commençons par faire appel aux initialisateurs intégrés.
Le code ci-dessous initialise tous les paramètres de poids
en tant que variables aléatoires gaussiennes
avec un écart-type de 0,01, tandis que les paramètres de biais sont remis à zéro.

```{.python .input  n=5}
%%tab mxnet
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input  n=6}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.LazyLinear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input  n=7}
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

Nous pouvons également initialiser tous les paramètres
à une valeur constante donnée (disons, 1).

```{.python .input  n=8}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input  n=9}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.LazyLinear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input  n=10}
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

[**Nous pouvons également appliquer des initialisateurs différents pour certains blocs.**]
Par exemple, ci-dessous, nous initialisons la première couche
avec l'initialisateur Xavier
et nous initialisons la deuxième couche
à une valeur constante de 42.

```{.python .input  n=11}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input  n=12}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.LazyLinear:
        nn.init.xavier_uniform_(module.weight)
def init_42(module):
    if type(module) == nn.LazyLinear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input  n=13}
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

### [**Initialisation personnalisée**]

Parfois, les méthodes d'initialisation dont nous avons besoin
ne sont pas fournies par le cadre d'apprentissage profond.
Dans l'exemple ci-dessous, nous définissons un initialisateur
pour tout paramètre de poids $w$ en utilisant la distribution étrange suivante :

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Nous définissons ici une sous-classe de la classe `Initializer`.
Habituellement, il suffit d'implémenter la fonction `_init_weight`
 qui prend un argument tensoriel (`data`)
et lui attribue les valeurs initialisées souhaitées.
:end_tab:

:begin_tab:`pytorch`
Encore une fois, nous implémentons une fonction `my_init` à appliquer à `net`.
:end_tab: 

 :begin_tab:`tensorflow` 
 Ici, nous définissons une sous-classe de `Initializer` et implémentons la fonction `__call__`
 qui renvoie un tenseur souhaité en fonction de la forme et du type de données.
:end_tab:

```{.python .input  n=14}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input  n=15}
%%tab pytorch
def my_init(module):
    if type(module) == nn.LazyLinear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input  n=16}
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

Notez que nous avons toujours la possibilité
de définir directement les paramètres.

```{.python .input  n=17}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input  n=18}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input  n=19}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Résumé

Nous pouvons initialiser des paramètres à l'aide d'initialisateurs intégrés et personnalisés.

## Exercices

Recherchez dans la documentation en ligne d'autres initialisateurs intégrés.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8091)
:end_tab:
