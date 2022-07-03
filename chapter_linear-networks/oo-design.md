```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Conception orientée objet pour l'implémentation
:label:`sec_oo-design` 

Dans notre introduction à la régression linéaire,
nous avons parcouru différents composants
dont
les données, le modèle, la fonction de perte,
et l'algorithme d'optimisation.
En effet,
la régression linéaire est
l'un des modèles d'apprentissage automatique les plus simples.
Sa formation,
toutefois, fait appel à un grand nombre de composants identiques à ceux des autres modèles présentés dans cet ouvrage.
Par conséquent, 
avant de plonger dans les détails de la mise en œuvre,
il est utile 
de concevoir certaines des API
utilisées tout au long de cet ouvrage. 
En traitant les composants de l'apprentissage profond
comme des objets,
nous pouvons commencer par
définir des classes pour ces objets
et leurs interactions.
Cette conception orientée objet
pour la mise en œuvre
simplifiera grandement
la présentation et vous aurez peut-être même envie de l'utiliser dans vos projets.


Inspiré par des bibliothèques open-source telles que [PyTorch Lightning](https://www.pytorchlightning.ai/),
à un haut niveau
nous souhaitons avoir trois classes : 
(i) `Module` contient les modèles, les pertes et les méthodes d'optimisation ; 
(ii) `DataModule` fournit des chargeurs de données pour l'entrainement et la validation ; 
(iii) les deux classes sont combinées à l'aide de la classe `Trainer`, qui nous permet de
former des modèles sur une variété de plateformes matérielles. 
La plupart des codes de ce livre adaptent `Module` et `DataModule`. Nous n'aborderons la classe `Trainer` que lorsque nous parlerons des GPU, des CPU, de l'entrainement parallèle et des algorithmes d'optimisation.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import torch as d2l
import tensorflow as tf
```

## Utilitaires
:label:`oo-design-utilities` 

Nous avons besoin de quelques utilitaires pour simplifier la programmation orientée objet dans les carnets Jupyter. L'un des défis est que les définitions de classe ont tendance à être des blocs de code assez longs. La lisibilité des notebooks exige des fragments de code courts, entrecoupés d'explications, une exigence incompatible avec le style de programmation commun aux bibliothèques Python. La première fonction utilitaire
nous permet d'enregistrer des fonctions en tant que méthodes dans une classe *après* la création de la classe. En fait, nous pouvons le faire *même après* avoir créé des instances de la classe ! Cela nous permet de diviser l'implémentation d'une classe en plusieurs blocs de code.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Voyons rapidement comment l'utiliser. Nous prévoyons d'implémenter une classe `A` avec une méthode `do`. Au lieu d'avoir du code pour `A` et `do` dans le même bloc de code, nous pouvons d'abord déclarer la classe `A` et créer une instance `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Ensuite, nous définissons la méthode `do` comme nous le ferions normalement, mais pas dans la portée de la classe `A`. Au lieu de cela, nous décorons cette méthode par `add_to_class` avec la classe `A` comme argument. Ce faisant, la méthode est capable d'accéder aux variables membres de `A` comme nous l'aurions souhaité si elle avait été définie dans le cadre de la définition de `A`. Voyons ce qui se passe lorsque nous l'invoquons pour l'instance `a`.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

La seconde est une classe utilitaire qui enregistre tous les arguments de la méthode `__init__` d'une classe en tant qu'attributs de classe. Cela nous permet d'étendre les signatures d'appel des constructeurs de manière implicite sans code supplémentaire.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Nous reportons son implémentation dans :numref:`sec_utils`. Pour l'utiliser, nous définissons notre classe qui hérite de `HyperParameters` et appelle `save_hyperparameters` dans la méthode `__init__`.

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Le dernier utilitaire nous permet de tracer la progression de l'expérience de manière interactive pendant qu'elle se déroule. En référence à [TensorBoard](https://www.tensorflow.org/tensorboard), beaucoup plus puissant (et complexe), nous le nommons `ProgressBoard`. L'implémentation est reportée à :numref:`sec_utils`. Pour l'instant, nous allons simplement la voir en action.

La fonction `draw` trace un point `(x, y)` dans la figure, avec `label` spécifié dans la légende. L'option `every_n` lisse la ligne en ne montrant que les points $1/n$ dans la figure. Leurs valeurs sont calculées à partir de la moyenne des points voisins $n$ dans la figure originale.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """Plot data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

Dans l'exemple suivant, nous dessinons `sin` et `cos` avec un lissage différent. Si vous exécutez ce bloc de code, vous verrez les lignes grandir en animation.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Modèles
:label:`oo-design-models` 

La classe `Module` est la classe de base de tous les modèles que nous allons implémenter. Au minimum, nous devons définir trois méthodes. La méthode `__init__` stocke les paramètres apprenables, la méthode `training_step` accepte un lot de données pour renvoyer la valeur de perte, la méthode `configure_optimizers` renvoie la méthode d'optimisation, ou une liste de celles-ci, qui est utilisée pour mettre à jour les paramètres apprenables. En option, nous pouvons définir `validation_step` pour rapporter les mesures d'évaluation.
Parfois, nous plaçons le code de calcul de la sortie dans une méthode distincte `forward` pour le rendre plus réutilisable.

```{.python .input}
%%tab all
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('pytorch'):
            self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
Vous pouvez remarquer que `Module` est une sous-classe de `nn.Block`, la classe de base des réseaux neuronaux dans Gluon.
Elle fournit des fonctionnalités pratiques pour la gestion des réseaux neuronaux. Par exemple, si nous définissons une méthode `forward`, comme `forward(self, X)`, pour une instance `a`, nous pouvons invoquer cette fonction par `a(X)`. Cela fonctionne puisqu'il appelle la méthode `forward` dans la méthode intégrée `__call__`. Vous trouverez plus de détails et d'exemples sur `nn.Block` dans :numref:`sec_model_construction`.
:end_tab: 

 :begin_tab:`pytorch` 
Vous avez peut-être remarqué que `Module` est une sous-classe de `nn.Module`, la classe de base des réseaux neuronaux dans PyTorch.
Elle fournit des fonctionnalités pratiques pour manipuler les réseaux neuronaux. Par exemple, si nous définissons une méthode `forward`, comme `forward(self, X)`, pour une instance `a`, nous pouvons invoquer cette fonction par `a(X)`. Cela fonctionne puisqu'il appelle la méthode `forward` dans la méthode intégrée `__call__`. Vous trouverez plus de détails et d'exemples sur `nn.Module` dans :numref:`sec_model_construction`.
:end_tab: 

 :begin_tab:`tensorflow` 
Vous remarquerez peut-être que `Module` est une sous-classe de `tf.keras.Model`, la classe de base des réseaux neuronaux dans TensorFlow.
Elle fournit des fonctionnalités pratiques pour manipuler les réseaux neuronaux. Par exemple, elle invoque la méthode `call` dans la méthode intégrée `__call__`. Ici, nous redirigeons `call` vers la fonction `forward`, en enregistrant ses arguments comme un attribut de classe. Nous faisons cela pour rendre notre code plus similaire aux autres implémentations du framework.
:end_tab:

## Données
:label:`oo-design-data` 

La classe `DataModule` est la classe de base pour les données. Très souvent, la méthode `__init__` est utilisée pour préparer les données. Cela inclut le téléchargement et le prétraitement si nécessaire. La méthode `train_dataloader` renvoie le chargeur de données pour l'ensemble de données d'apprentissage. Un chargeur de données est un générateur (Python) qui produit un lot de données à chaque fois qu'il est utilisé. Ce lot est ensuite introduit dans la méthode `training_step` de `Module` pour calculer la perte. Il existe une option `val_dataloader` pour renvoyer le chargeur de données de validation. Il se comporte de la même manière, sauf qu'il produit des lots de données pour la méthode `validation_step` dans `Module`.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Entrainement
:label:`oo-design-training` 

La classe `Trainer` forme les paramètres apprenables de la classe `Module` avec les données spécifiées dans `DataModule`. La méthode clé est `fit`, qui accepte deux arguments :`model` une instance de `Module`, et `data`, une instance de `DataModule`. Elle itère ensuite sur l'ensemble complet de données `max_epochs` fois pour entraîner le modèle. Comme précédemment, nous reporterons l'implémentation de cette fonction à des chapitres ultérieurs.

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Résumé

Pour mettre en évidence la conception orientée objet
de notre future implémentation d'apprentissage profond,
les classes ci-dessus montrent simplement comment leurs objets 
stockent les données et interagissent entre eux.
Nous continuerons à enrichir les implémentations de ces classes,
comme via `@add_to_class`,
dans le reste du livre.
De plus,
ces classes entièrement implémentées
sont sauvegardées dans le fichier [d2l library](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
une *boîte à outils légère* qui facilite la modélisation structurée pour l'apprentissage profond. 
En particulier, elle facilite la réutilisation de nombreux composants entre les projets sans changer grand-chose. Par exemple, nous pouvons remplacer uniquement l'optimiseur, uniquement le modèle, uniquement le jeu de données, etc. ;
ce degré de modularité porte ses fruits tout au long du livre en termes de concision et de simplicité (c'est pourquoi nous l'avons ajouté) et il peut en faire de même pour vos propres projets. 


## Exercices

1. Localisez les implémentations complètes des classes ci-dessus qui sont sauvegardées dans le répertoire [d2l library](https://github.com/d2l-ai/d2l-en/tree/master/d2l). Nous vous recommandons fortement d'examiner l'implémentation en détail une fois que vous aurez acquis une certaine familiarité avec la modélisation de l'apprentissage profond.
1. Supprimez l'instruction `save_hyperparameters` dans la classe `B`. Pouvez-vous toujours imprimer `self.a` et `self.b`? Facultatif : si vous vous êtes plongé dans l'implémentation complète de la classe `HyperParameters`, pouvez-vous expliquer pourquoi ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6647)
:end_tab:
