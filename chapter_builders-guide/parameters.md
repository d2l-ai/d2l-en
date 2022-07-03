# Gestion des paramètres

Après avoir choisi une architecture
et défini nos hyperparamètres,
nous passons à la boucle d'apprentissage,
où notre objectif est de trouver les valeurs des paramètres
qui minimisent notre fonction de perte.
Après l'apprentissage, nous aurons besoin de ces paramètres
afin de faire des prédictions futures.

En outre, nous souhaiterons parfois
extraire les paramètres
soit pour les réutiliser dans un autre contexte,
pour sauvegarder notre modèle sur le disque afin qu'il puisse être exécuté dans un autre logiciel,
ou pour l'examiner dans l'espoir de
gagner en compréhension scientifique.

La plupart du temps, nous pourrons
ignorer les détails minutieux
de la manière dont les paramètres sont déclarés
et manipulés, en nous appuyant sur les cadres d'apprentissage profond
pour faire le gros du travail.
Cependant, lorsque nous nous éloignons des architectures empilées
avec des couches standard,
nous devrons parfois entrer dans les détails
de la déclaration et de la manipulation des paramètres.
Dans cette section, nous abordons les points suivants :

* Accès aux paramètres pour le débogage, les diagnostics et les visualisations.
* Partage des paramètres entre différents composants du modèle.

(**Nous commençons par nous concentrer sur un MLP avec une couche cachée.**)

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
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

```{.python .input}
%%tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
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

## [**Accès aux paramètres**]

Commençons par voir comment accéder aux paramètres
des modèles que vous connaissez déjà.
Lorsqu'un modèle est défini via la classe `Sequential`,
nous pouvons d'abord accéder à n'importe quelle couche en indexant
dans le modèle comme s'il s'agissait d'une liste.
Les paramètres de chaque couche sont commodément situés
dans son attribut.
Nous pouvons inspecter les paramètres de la deuxième couche entièrement connectée comme suit.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

Nous pouvons voir que cette couche entièrement connectée
contient deux paramètres,
correspondant aux poids et aux biais de cette couche,
respectivement.


### [**Paramètres ciblés**]

Notez que chaque paramètre est représenté
comme une instance de la classe de paramètres.
Pour faire quoi que ce soit d'utile avec les paramètres,
nous devons d'abord accéder aux valeurs numériques sous-jacentes.
Il existe plusieurs façons de le faire.
Certaines sont plus simples, d'autres plus générales.
Le code suivant extrait le biais
de la deuxième couche du réseau neuronal, qui renvoie une instance de classe de paramètre, et
accède ensuite à la valeur de ce paramètre.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

:begin_tab:`mxnet,pytorch`
Les paramètres sont des objets complexes,
contenant des valeurs, des gradients,
et des informations supplémentaires.
C'est pourquoi nous devons demander la valeur de manière explicite.

En plus de la valeur, chaque paramètre nous permet également d'accéder au gradient. Comme nous n'avons pas encore invoqué la rétropropagation pour ce réseau, celui-ci est dans son état initial.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**Tous les paramètres à la fois**]

Lorsque nous devons effectuer des opérations sur tous les paramètres,
y accéder un par un peut devenir fastidieux.
La situation peut devenir particulièrement difficile
lorsque nous travaillons avec des modules plus complexes (par exemple, des modules imbriqués),
puisque nous devrions récurer
à travers l'arbre entier pour extraire
les paramètres de chaque sous-module. Nous démontrons ci-dessous l'accès aux paramètres de toutes les couches.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

## [**Paramètres liés**]

Souvent, nous voulons partager des paramètres entre plusieurs couches.
Voyons comment le faire de manière élégante.
Dans ce qui suit, nous attribuons une couche entièrement connectée
et utilisons ensuite ses paramètres spécifiquement
pour définir ceux d'une autre couche.
Ici, nous devons exécuter la propagation vers l'avant
`net(X)` avant d'accéder aux paramètres.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])
net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

Cet exemple montre que les paramètres
de la deuxième et de la troisième couche sont égaux.
Ils ne sont pas seulement égaux, ils sont
représentés par le même tenseur exact.
Ainsi, si nous modifions l'un des paramètres,
l'autre change également.
Vous vous demandez peut-être :
lorsque les paramètres sont liés
qu'advient-il des gradients ?
Puisque les paramètres du modèle contiennent des gradients,
les gradients de la deuxième couche cachée
et de la troisième couche cachée sont ajoutés ensemble
pendant la rétropropagation.

## Résumé

Nous avons plusieurs façons d'accéder aux paramètres du modèle et de les lier.


## Exercices

1. Utilisez le modèle `NestMLP` défini dans :numref:`sec_model_construction` et accédez aux paramètres des différentes couches.
1. Construisez un MLP contenant une couche à paramètres partagés et entraînez-le. Pendant le processus d'entraînement, observez les paramètres du modèle et les gradients de chaque couche.
1. Pourquoi le partage des paramètres est-il une bonne idée ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
