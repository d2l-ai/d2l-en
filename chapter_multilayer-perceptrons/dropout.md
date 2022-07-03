```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Dropout
:label:`sec_dropout` 

 
Réfléchissons brièvement à ce que nous
attendons d'un bon modèle prédictif.
Nous voulons qu'il soit performant sur des données non vues.
La théorie classique de la généralisation
suggère que pour combler l'écart entre
les performances de l'entrainement et du test,
nous devons viser un modèle simple.
La simplicité peut prendre la forme
d'un petit nombre de dimensions.
Nous avons exploré ce point lors de l'examen des fonctions de base monomiales
des modèles linéaires
dans :numref:`sec_model_selection`.
En outre, comme nous l'avons vu lors de l'examen de la décroissance du poids
($\ell_2$ régularisation) dans :numref:`sec_weight_decay`,
la norme (inverse) des paramètres
représente également une mesure utile de la simplicité.
Une autre notion utile de la simplicité est la régularité,
c'est-à-dire que la fonction ne doit pas être sensible
à de petites modifications de ses entrées.
Par exemple, lorsque nous classons des images,
nous nous attendons à ce que l'ajout d'un bruit aléatoire
aux pixels soit pratiquement inoffensif.

En 1995, Christopher Bishop a formalisé
cette idée en prouvant que l'entraînement avec du bruit en entrée
est équivalent à une régularisation de Tikhonov :cite:`Bishop.1995`.
Ce travail a établi un lien mathématique clair
entre l'exigence qu'une fonction soit lisse (et donc simple),
et l'exigence qu'elle soit résiliente
aux perturbations de l'entrée.

Puis, en 2014, Srivastava et al. :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` 
 ont développé une idée astucieuse pour appliquer l'idée de Bishop
également aux couches internes d'un réseau.
Leur idée, appelée *dropout*, consiste à
injecter du bruit tout en calculant
chaque couche interne pendant la propagation vers l'avant,
et elle est devenue une technique standard
pour l'entrainement des réseaux neuronaux.
La méthode est appelée *dropout* parce que nous
*dropout* littéralement certains neurones pendant la formation.
Tout au long de la formation, à chaque itération,
le dropout standard consiste à remettre à zéro
une certaine fraction des nœuds de chaque couche
avant de calculer la couche suivante.

Pour être clair, nous imposons
notre propre récit avec le lien à Bishop.
L'article original sur le dropout
offre une intuition grâce à une analogie surprenante
avec la reproduction sexuelle.
Les auteurs affirment que la suradaptation des réseaux neuronaux
est caractérisée par un état dans lequel
chaque couche repose sur un modèle spécifique
d'activations dans la couche précédente,
appelant cette condition *co-adaptation*.
dropout, affirment-ils, brise la co-adaptation
tout comme la reproduction sexuelle est censée
briser les gènes co-adaptés.
Si l'explication de cette théorie est certainement sujette à débat,
la technique du dropout elle-même s'est avérée durable,
et diverses formes de dropout sont implémentées
dans la plupart des bibliothèques d'apprentissage profond. 


Le principal défi est de savoir comment injecter ce bruit.
Une idée consiste à injecter le bruit de manière *non biaisée*
de sorte que la valeur attendue de chaque couche - tout en fixant
les autres - soit égale à la valeur qu'elle aurait prise en l'absence de bruit.
Dans les travaux de Bishop, il a ajouté un bruit gaussien
aux entrées d'un modèle linéaire.
À chaque itération d'apprentissage, il ajoute le bruit
échantillonné à partir d'une distribution de moyenne zéro
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ à l'entrée \mathbf{x},
produisant un point perturbé \mathbf{x} $\mathbf{x}' = \mathbf{x} + \epsilon$.
En espérance, $E[\mathbf{x}'] = \mathbf{x}$.

Dans la régularisation par abandon standard,
on élimine à zéro une certaine fraction des nœuds dans chaque couche
, puis on *débiaise* chaque couche en normalisant
par la fraction de nœuds qui ont été conservés (non éliminés).
En d'autres termes,
avec une *probabilité d'abandon* $p$,
chaque activation intermédiaire $h$ est remplacée par
une variable aléatoire $h'$ comme suit :

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

 Par construction, l'espérance reste inchangée, c'est-à-dire que  $E[h'] = h$.

## Dropout en pratique

Rappelez-vous le MLP avec une couche cachée et 5 unités cachées
dans :numref:`fig_mlp`.
Lorsque nous appliquons le dropout à une couche cachée,
en mettant à zéro chaque unité cachée avec une probabilité $p$,
le résultat peut être considéré comme un réseau
contenant uniquement un sous-ensemble des neurones d'origine.
Dans :numref:`fig_dropout2` , $h_2$ et $h_5$ sont supprimés.
Par conséquent, le calcul des sorties
ne dépend plus de $h_2$ ou $h_5$
et leur gradient respectif disparaît également
lors de la rétropropagation.

De cette façon, le calcul de la couche de sortie
ne peut pas être trop dépendant d'un élément quelconque de $h_1, \ldots, h_5$.

![MLP avant et après dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

En général, nous désactivons le dropout au moment du test.
Étant donné un modèle entraîné et un nouvel exemple,
nous ne supprimons aucun nœud
et n'avons donc pas besoin de normaliser.
Il existe toutefois quelques exceptions :
certains chercheurs utilisent le dropout au moment du test comme une heuristique
pour estimer l'*incertitude* des prédictions du réseau neuronal :
si les prédictions concordent avec de nombreux masques de dropout différents,
nous pouvons alors dire que le réseau est plus confiant.

## Implémentation from Scratch

Pour implémenter la fonction de dropout pour une seule couche,
nous devons tirer autant d'échantillons
d'un Bernoulli (binaire) variable aléatoire
comme notre couche a des dimensions,
où la variable aléatoire prend la valeur $1$ (conserver)
avec la probabilité $1-p$ et $0$ (abandonner) avec la probabilité $p$.
Une façon simple de mettre cela en œuvre est de tirer d'abord des échantillons
de la distribution uniforme $U[0, 1]$.
Ensuite, nous pouvons conserver les nœuds pour lesquels l'échantillon correspondant à
est supérieur à $p$, en abandonnant les autres.

Dans le code suivant, nous (**implémentons une fonction `dropout_layer`
 qui élimine les éléments de l'entrée du tenseur `X`
 avec la probabilité `dropout`**),
en remettant à l'échelle le reste comme décrit ci-dessus :
en divisant les survivants par `1.0-dropout`.

```{.python .input  n=5}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

Nous pouvons [**tester la fonction `dropout_layer` sur quelques exemples**].
Dans les lignes de code suivantes,
nous faisons passer notre entrée `X` par l'opération d'abandon,
avec des probabilités de 0, 0,5 et 1, respectivement.

```{.python .input  n=6}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### Définition du modèle

Le modèle ci-dessous applique le dropout à la sortie
de chaque couche cachée (suivant la fonction d'activation).
Nous pouvons définir les probabilités d'abandon pour chaque couche séparément.
Une tendance commune consiste à définir
une probabilité d'exclusion plus faible à proximité de la couche d'entrée.
Nous veillons à ce que le décrochage ne soit actif que pendant la formation.

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)
        
    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**Training**]

Ce qui suit est similaire à l'entraînement des MLP décrit précédemment.

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256, 
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

### [**Mise en œuvre concise**]

Avec des API de haut niveau, tout ce que nous avons à faire est d'ajouter une couche `Dropout`
 après chaque couche entièrement connectée,
en passant la probabilité de décrochage
comme seul argument à son constructeur.

Pendant l'apprentissage, la couche `Dropout` abandonnera de manière aléatoire les sorties de la couche précédente
(ou, de manière équivalente, les entrées de la couche suivante)
en fonction de la probabilité d'abandon spécifiée.
Lorsqu'elle n'est pas en mode formation,
la couche `Dropout` transmet simplement les données pendant les tests.

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

Ensuite, nous [**formons le modèle**].

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## Résumé

* Outre le contrôle du nombre de dimensions et de la taille du vecteur de poids, le dropout est un outil supplémentaire pour éviter l'overfitting. Ils sont souvent utilisés conjointement.
* Le dropout remplace une activation $h$ par une variable aléatoire avec une valeur attendue $h$.
* Le dropout n'est utilisé que pendant la formation.


## Exercices

1. Que se passe-t-il si vous changez les probabilités d'abandon pour la première et la deuxième couche ? En particulier, que se passe-t-il si vous changez celles des deux couches ? Concevez une expérience pour répondre à ces questions, décrivez vos résultats quantitatifs et résumez les conclusions qualitatives.
1. Augmentez le nombre d'époques et comparez les résultats obtenus en utilisant le dropout avec ceux obtenus en ne l'utilisant pas.
1. Quelle est la variance des activations dans chaque couche cachée lorsque le dropout est appliqué et lorsqu'il ne l'est pas ? Tracez un graphique pour montrer comment cette quantité évolue dans le temps pour les deux modèles.
1. Pourquoi le dropout n'est-il généralement pas utilisé au moment du test ?
1. En utilisant le modèle de cette section comme exemple, comparez les effets de l'utilisation du dropout et de la décroissance du poids. Que se passe-t-il lorsque l'abandon et la décroissance du poids sont utilisés en même temps ? Les résultats s'additionnent-ils ? Y a-t-il une diminution des rendements (ou pire) ? S'annulent-ils l'un l'autre ?
1. Que se passe-t-il si nous appliquons le dropout aux poids individuels de la matrice de poids plutôt qu'aux activations ?
1. Inventez une autre technique d'injection de bruit aléatoire à chaque couche, différente de la technique d'exclusion standard. Pouvez-vous développer une méthode qui surpasse le dropout sur l'ensemble de données Fashion-MNIST (pour une architecture fixe) ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab: 

 :begin_tab:`pytorch` 
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab: 

 :begin_tab:`tensorflow` 
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:
