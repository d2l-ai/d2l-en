# Couches personnalisées

L'un des facteurs du succès de l'apprentissage profond
est la disponibilité d'un large éventail de couches
qui peuvent être composées de manière créative
pour concevoir des architectures adaptées
à une grande variété de tâches.
Par exemple, les chercheurs ont inventé des couches
spécialement conçues pour traiter des images, du texte,
boucler des données séquentielles,
et
effectuer de la programmation dynamique.
Tôt ou tard, vous rencontrerez ou inventerez
une couche qui n'existe pas encore dans le cadre de l'apprentissage profond.
Dans ce cas, vous devrez construire une couche personnalisée.
Dans cette section, nous vous montrons comment.

## (**Couches sans paramètres**)

Pour commencer, nous construisons une couche personnalisée
qui n'a pas de paramètres propres.
Cela devrait vous sembler familier si vous vous souvenez de notre
introduction au module dans :numref:`sec_model_construction` .
La classe `CenteredLayer` suivante
soustrait simplement la moyenne de son entrée.
Pour la construire, nous devons simplement hériter de
de la classe de couche de base et implémenter la fonction de propagation vers l'avant.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

Vérifions que notre couche fonctionne comme prévu en lui fournissant des données.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Nous pouvons maintenant [**incorporer notre couche comme composant
dans la construction de modèles plus complexes.**]

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

À titre de contrôle supplémentaire, nous pouvons envoyer des données aléatoires
à travers le réseau et vérifier que la moyenne est bien égale à 0.
Comme nous traitons des nombres à virgule flottante,
nous pouvons toujours voir un très petit nombre non nul
en raison de la quantification.

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

## [**Couches avec paramètres**]

Maintenant que nous savons comment définir des couches simples,
passons à la définition de couches avec des paramètres
qui peuvent être ajustés par l'apprentissage.
Nous pouvons utiliser les fonctions intégrées pour créer des paramètres, qui
fournissent certaines fonctionnalités de base.
En particulier, elles régissent l'accès, l'initialisation, le partage
, la sauvegarde et le chargement des paramètres du modèle.
De cette façon, entre autres avantages, nous n'aurons pas besoin d'écrire
des routines de sérialisation personnalisées pour chaque couche personnalisée.

Implémentons maintenant notre propre version de la couche entièrement connectée.
Rappelons que cette couche nécessite deux paramètres,
l'un pour représenter le poids et l'autre pour le biais.
Dans cette implémentation, nous intégrons l'activation ReLU par défaut.
Cette couche requiert deux arguments d'entrée :`in_units` et `units`, dont
représente le nombre d'entrées et de sorties, respectivement.

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

:begin_tab:`mxnet, tensorflow`
Ensuite, nous instançons la classe `MyDense`
 et accédons aux paramètres de son modèle.
:end_tab:

:begin_tab:`pytorch`
Ensuite, nous instançons la classe `MyLinear`
 et accédons à ses paramètres de modèle.
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

Nous pouvons [**effectuer directement des calculs de propagation vers l'avant en utilisant des couches personnalisées.**]

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

Nous pouvons également (**construire des modèles en utilisant des couches personnalisées.**)
Une fois que nous l'avons, nous pouvons l'utiliser comme la couche entièrement connectée intégrée.

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

## Résumé

* Nous pouvons concevoir des couches personnalisées via la classe de couche de base. Cela nous permet de définir de nouvelles couches flexibles qui se comportent différemment de toutes les couches existantes dans la bibliothèque.
* Une fois définies, les couches personnalisées peuvent être invoquées dans des contextes et des architectures arbitraires.
* Les couches peuvent avoir des paramètres locaux, qui peuvent être créés par des fonctions intégrées.


## Exercices

1. Concevez une couche qui prend une entrée et calcule une réduction tensorielle,
 c'est-à-dire qu'elle renvoie $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Concevez une couche qui renvoie la moitié supérieure des coefficients de Fourier des données.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
