```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Mise en œuvre 
:label:`sec_mlp_scratch` 

 Les perceptrons multicouches (MLP) ne sont pas beaucoup plus complexes à mettre en œuvre que les modèles linéaires simples. La principale différence conceptuelle 
est que nous concaténons maintenant plusieurs couches.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

### Implémentation à partir de zéro

Commençons par implémenter un tel réseau à partir de zéro.

### Initialisation des paramètres du modèle

Rappelons que Fashion-MNIST contient 10 classes,
et que chaque image consiste en une grille $28 \times 28 = 784$
 de valeurs de pixels en niveaux de gris.
Comme précédemment, nous ne tiendrons pas compte de la structure spatiale
entre les pixels pour le moment, 
. Nous pouvons donc considérer ceci comme un ensemble de données de classification
avec 784 caractéristiques d'entrée et 10 classes.
Pour commencer, nous allons [**implémenter un MLP
avec une couche cachée et 256 unités cachées.**]
Le nombre de couches et leur largeur sont tous deux réglables 
(ils sont considérés comme des hyperparamètres). 
En règle générale, nous choisissons des largeurs de couche divisibles par des puissances de 2 plus importantes. 
C'est un calcul efficace en raison de la manière dont la mémoire 
est allouée et adressée dans le matériel.

Une fois encore, nous représenterons nos paramètres par plusieurs tenseurs.
Notez que *pour chaque couche*, nous devons garder la trace de
une matrice de poids et un vecteur de biais.
Comme toujours, nous allouons la mémoire
pour les gradients de la perte par rapport à ces paramètres.

```{.python .input  n=5}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input  n=6}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input  n=7}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

### Modèle

Pour nous assurer que nous savons comment tout fonctionne,
nous allons [**implémenter l'activation ReLU**] nous-mêmes
plutôt que d'invoquer directement la fonction intégrée `relu`.

```{.python .input  n=8}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input  n=9}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input  n=10}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

Puisque nous ne tenons pas compte de la structure spatiale,
nous `reshape` chaque image bidimensionnelle en
un vecteur plat de longueur `num_inputs`.
Enfin, nous (**mettons en œuvre notre modèle**)
avec seulement quelques lignes de code. Puisque nous utilisons le cadre intégré autograd, c'est tout ce qu'il faut.

```{.python .input  n=11}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Formation

Heureusement, [**la boucle de entrainement pour les MLP
est exactement la même que pour la régression softmax.**] Nous définissons le modèle, les données, le formateur et enfin nous invoquons la fonction `fit` sur le modèle et les données.

```{.python .input  n=12}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

### Implémentation concise

Comme on peut s'y attendre, en s'appuyant sur les API de haut niveau, nous pouvons implémenter les MLP de manière encore plus concise.

### Modèle

Par rapport à notre implémentation concise
de l'implémentation de la régression softmax
(:numref:`sec_softmax_concise` ),
la seule différence est que nous ajoutons
*deux* couches entièrement connectées là où nous n'en avions ajouté qu'une*une* auparavant.
La première est [**la couche cachée**],
la seconde est la couche de sortie.

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])  
```

### Training

[**La boucle d'entraînement**] est exactement la même
que lorsque nous avons implémenté la régression softmax.
Cette modularité nous permet de séparer
les questions concernant l'architecture du modèle
des considérations orthogonales.

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Résumé

Maintenant que nous avons plus de pratique dans la conception de réseaux profonds, le passage d'une couche unique à plusieurs couches de réseaux profonds ne pose plus un défi aussi important. En particulier, nous pouvons réutiliser l'algorithme de entrainement et le chargeur de données. Notez cependant que l'implémentation de MLP à partir de zéro est néanmoins désordonnée : nommer et garder la trace des paramètres du modèle rend difficile l'extension des modèles. Par exemple, imaginons que l'on veuille insérer une autre couche entre les couches 42 et 43. Il pourrait s'agir maintenant de la couche 42b, à moins que nous ne soyons prêts à effectuer un renommage séquentiel. De plus, si nous implémentons le réseau à partir de zéro, il est beaucoup plus difficile pour le framework d'effectuer des optimisations de performance significatives. 

Néanmoins, vous avez maintenant atteint l'état de l'art de la fin des années 1980, lorsque les réseaux profonds entièrement connectés étaient la méthode de choix pour la modélisation des réseaux neuronaux. Notre prochaine étape conceptuelle consistera à considérer les images. Avant de le faire, nous devons revoir un certain nombre de bases statistiques et de détails sur la façon de calculer des modèles de manière efficace. 


## Exercices

1. Modifiez le nombre d'unités cachées `num_hiddens` et indiquez comment ce nombre affecte la précision du modèle. Quelle est la meilleure valeur de cet hyperparamètre ?
1. Essayez d'ajouter une couche cachée pour voir comment cela affecte les résultats.
1. Pourquoi est-ce une mauvaise idée d'insérer une couche cachée avec un seul neurone ? Qu'est-ce qui pourrait mal tourner ?
1. Comment la modification du taux d'apprentissage modifie-t-elle vos résultats ? Tous les autres paramètres étant fixes, quel taux d'apprentissage vous donne les meilleurs résultats ? Quel est le lien avec le nombre d'époques ?
1. Optimisons tous les hyperparamètres conjointement, c'est-à-dire le taux d'apprentissage, le nombre d'époques, le nombre de couches cachées et le nombre d'unités cachées par couche. 
    1. Quel est le meilleur résultat que vous pouvez obtenir en optimisant sur tous ces paramètres ?
   1. Pourquoi est-il beaucoup plus difficile de traiter plusieurs hyperparamètres ?
   1. Décrivez une stratégie efficace d'optimisation conjointe sur plusieurs paramètres. 
1. Comparer la rapidité du cadre et de l'implémentation " from scratch " pour un problème difficile. Comment cela change-t-il avec la complexité du réseau ?
1. Mesurez la vitesse des multiplications de matrices tensives pour des matrices bien alignées et désalignées. Par exemple, testez les matrices de dimension 1024, 1025, 1026, 1028 et 1032.
   1. Comment cela change-t-il entre les GPU et les CPU ?
   1. Déterminez la largeur du bus mémoire de votre CPU et de votre GPU. 
1. Essayez différentes fonctions d'activation. Laquelle fonctionne le mieux ?
1. Y a-t-il une différence entre les initialisations de poids du réseau ? Cela a-t-il de l'importance ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
