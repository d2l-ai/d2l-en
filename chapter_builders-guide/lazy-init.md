# Lazy Initialization
:label:`sec_lazy_init` 

 Jusqu'à présent, il pourrait sembler que nous nous en sommes sortis
en étant négligents dans la configuration de nos réseaux.
Plus précisément, nous avons fait les choses peu intuitives suivantes,
qui peuvent sembler ne pas devoir fonctionner :

* Nous avons défini les architectures de réseau
 sans spécifier la dimensionnalité d'entrée.
* Nous avons ajouté des couches sans spécifier
 la dimension de sortie de la couche précédente.
* Nous avons même "initialisé" ces paramètres
 avant de fournir suffisamment d'informations pour déterminer
 le nombre de paramètres que nos modèles devraient contenir.

Vous pourriez être surpris que notre code fonctionne.
Après tout, il n'y a aucun moyen pour le cadre d'apprentissage profond
de savoir quelle serait la dimensionnalité d'entrée d'un réseau.
L'astuce ici est que le cadre *diffère l'initialisation*,
attendant jusqu'à la première fois que nous passons des données dans le modèle,
pour déduire les tailles de chaque couche à la volée.


Plus tard, lorsque nous travaillerons avec des réseaux neuronaux convolutifs,
cette technique deviendra encore plus pratique
puisque la dimensionnalité d'entrée
(c'est-à-dire la résolution d'une image)
affectera la dimensionnalité
de chaque couche suivante.
Ainsi, la possibilité de définir des paramètres
sans avoir besoin de savoir,
au moment de l'écriture du code,
quelle est la dimensionnalité
peut grandement simplifier la tâche de spécification
et de modification ultérieure de nos modèles.
Ensuite, nous allons approfondir les mécanismes d'initialisation.


Pour commencer, instancions un MLP.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

À ce stade, le réseau ne peut pas connaître
les dimensions des poids de la couche d'entrée
car la dimension de l'entrée reste inconnue.
Par conséquent, le cadre n'a pas encore initialisé de paramètres.
Nous confirmons en essayant d'accéder aux paramètres ci-dessous.

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
Notez que bien que les objets paramètres existent,
la dimension d'entrée de chaque couche est listée comme -1.
MXNet utilise la valeur spéciale -1 pour indiquer
que la dimension du paramètre reste inconnue.
À ce stade, toute tentative d'accès à `net[0].weight.data()`
 déclencherait une erreur d'exécution indiquant que le réseau
doit être initialisé avant de pouvoir accéder aux paramètres.
Voyons maintenant ce qui se passe lorsque nous tentons d'initialiser les paramètres
via la méthode `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Notez que les objets de chaque couche existent mais que les poids sont vides.
L'utilisation de `net.get_weights()` entraînerait une erreur puisque les poids
n'ont pas encore été initialisés.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Comme nous pouvons le constater, rien n'a changé.
Lorsque les dimensions d'entrée sont inconnues, les appels à l'initialisation de
n'initialisent pas vraiment les paramètres.
Au lieu de cela, cet appel enregistre auprès de MXNet que nous souhaitons que
(et éventuellement, selon quelle distribution)
initialise les paramètres.
:end_tab:

Passons ensuite les données à travers le réseau
pour que le framework initialise finalement les paramètres.

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

Dès que nous connaissons la dimensionnalité d'entrée,
20,
le cadre peut identifier la forme de la matrice de poids de la première couche en introduisant la valeur de 20.
Ayant reconnu la forme de la première couche, le cadre passe à
à la deuxième couche,
et ainsi de suite à travers le graphe de calcul
jusqu'à ce que toutes les formes soient connues.
Notez que dans ce cas,
seule la première couche nécessite une initialisation paresseuse,
mais le framework initialise séquentiellement.
Une fois que toutes les formes de paramètres sont connues,
le cadre peut enfin initialiser les paramètres.

:begin_tab:`pytorch`
La méthode suivante
fait passer des entrées fictives
par le réseau
pour un essai
afin de déduire toutes les formes de paramètres
et d'initialiser ensuite les paramètres.
Elle sera utilisée plus tard lorsque les initialisations aléatoires par défaut ne sont pas souhaitées.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

## Résumé

* L'initialisation paresseuse peut être pratique, permettant au cadre de déduire automatiquement les formes des paramètres, facilitant la modification des architectures et éliminant une source commune d'erreurs.
* Nous pouvons passer des données à travers le modèle pour que le framework initialise finalement les paramètres.


## Exercices

1. Que se passe-t-il si vous spécifiez les dimensions d'entrée à la première couche mais pas aux couches suivantes ? Obtenez-vous une initialisation immédiate ?
1. Que se passe-t-il si vous spécifiez des dimensions non concordantes ?
1. Que devez-vous faire si vous avez des entrées de dimensions variables ? Indice : regardez la liaison des paramètres.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
