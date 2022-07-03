# File I/O

Jusqu'à présent, nous avons vu comment traiter les données et comment
construire, former et tester des modèles d'apprentissage profond.
Cependant, à un moment donné, nous espérons être suffisamment satisfaits
des modèles appris pour vouloir
enregistrer les résultats pour une utilisation ultérieure dans divers contextes
(peut-être même pour faire des prédictions dans le déploiement).
En outre, lors de l'exécution d'un long processus de formation,
la meilleure pratique consiste à sauvegarder périodiquement les résultats intermédiaires (checkpointing)
pour s'assurer que nous ne perdons pas plusieurs jours de calcul
si nous trébuchons sur le cordon d'alimentation de notre serveur.
Il est donc temps d'apprendre à charger et à stocker
à la fois des vecteurs de poids individuels et des modèles entiers.
Cette section aborde les deux questions.

## (**Chargement et sauvegarde de tenseurs**)

Pour les tenseurs individuels, nous pouvons directement
invoquer les fonctions `load` et `save`
pour les lire et les écrire respectivement.
Les deux fonctions exigent que nous fournissions un nom,
et `save` exige comme entrée la variable à sauvegarder.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save('x-file.npy', x)
```

Nous pouvons maintenant relire en mémoire les données du fichier enregistré.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

Nous pouvons [**stocker une liste de tenseurs et les relire en mémoire.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Nous pouvons même [**écrire et lire un dictionnaire qui fait correspondre
des chaînes de caractères aux tenseurs.**]
Ceci est pratique lorsque nous voulons
lire ou écrire tous les poids d'un modèle.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Chargement et sauvegarde des paramètres du modèle**]

Sauvegarder des vecteurs de poids individuels (ou d'autres tenseurs) est utile,
mais cela devient très fastidieux si nous voulons sauvegarder
(et charger plus tard) un modèle entier.
Après tout, il se peut que nous ayons des centaines de groupes de paramètres
disséminés un peu partout.
C'est pourquoi le cadre d'apprentissage profond fournit des fonctionnalités intégrées
pour charger et sauvegarder des réseaux entiers.
Un détail important à noter est que cette
sauvegarde les *paramètres* du modèle et non le modèle entier.
Par exemple, si nous avons un MLP à 3 couches,
nous devons spécifier l'architecture séparément.
La raison en est que les modèles eux-mêmes peuvent contenir du code arbitraire,
; ils ne peuvent donc pas être sérialisés de manière naturelle.
Ainsi, afin de réintégrer un modèle, nous devons
générer l'architecture en code
et ensuite charger les paramètres depuis le disque.
(**Commençons par notre MLP familier.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

Ensuite, nous [**stockons les paramètres du modèle dans un fichier**] portant le nom "mlp.params".

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

Pour récupérer le modèle, nous instancions un clone
du modèle MLP original.
Au lieu d'initialiser aléatoirement les paramètres du modèle,
nous [**lisons les paramètres stockés dans le fichier directement**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

Puisque les deux instances ont les mêmes paramètres de modèle,
le résultat du calcul de la même entrée `X` devrait être le même.
Vérifions cela.

```{.python .input}
%%tab all
Y_clone = clone(X)
Y_clone == Y
```

## Résumé

* Les fonctions `save` et `load` peuvent être utilisées pour effectuer des E/S de fichiers pour les objets tenseurs.
* Nous pouvons sauvegarder et charger les ensembles complets de paramètres pour un réseau via un dictionnaire de paramètres.
* La sauvegarde de l'architecture doit se faire en code plutôt qu'en paramètres.

## Exercices

1. Même s'il n'est pas nécessaire de déployer les modèles formés sur un autre appareil, quels sont les avantages pratiques du stockage des paramètres du modèle ?
1. Supposons que nous voulions réutiliser uniquement des parties d'un réseau pour les incorporer dans un réseau d'une architecture différente. Comment procéderiez-vous pour utiliser, par exemple, les deux premières couches d'un réseau précédent dans un nouveau réseau ?
1. Comment procéder pour sauvegarder l'architecture et les paramètres du réseau ? Quelles restrictions imposeriez-vous à l'architecture ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
