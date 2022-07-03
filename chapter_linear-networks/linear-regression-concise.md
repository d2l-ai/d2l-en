```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Concise Implementation of Linear Regression
:label:`sec_linear_concise` 

L'apprentissage profond a connu une sorte d'explosion cambrienne
au cours de la dernière décennie.
Le nombre de techniques, d'applications et d'algorithmes dépasse de loin les progrès accomplis au cours des décennies précédentes .

Cela est dû à une combinaison fortuite de multiples facteurs,
dont l'un est les puissants outils gratuits
offerts par un certain nombre de cadres d'apprentissage profond open source.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
et Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014` 
représentent sans doute la
première génération de ces modèles 
qui a été largement adoptée.
Contrairement aux travaux antérieurs (séminaux) comme
SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`,
qui offraient une expérience de programmation de type Lisp,
les cadres modernes offrent une différenciation automatique
et la commodité de Python.
Ces cadres nous permettent d'automatiser et de modulariser
le travail répétitif de mise en œuvre des algorithmes d'apprentissage basés sur le gradient.

Dans :numref:`sec_linear_scratch`, nous nous sommes uniquement appuyés sur
(i) les tenseurs pour le stockage des données et l'algèbre linéaire ;
et (ii) la différenciation automatique pour le calcul des gradients.
En pratique, comme les itérateurs de données, les fonctions de perte, les optimiseurs,
et les couches de réseaux neuronaux
sont si courants, les bibliothèques modernes implémentent également ces composants pour nous.
Dans cette section, (**nous allons vous montrer comment implémenter
le modèle de régression linéaire**) de :numref:`sec_linear_scratch` 
 (**de manière concise en utilisant les API de haut niveau**) des cadres d'apprentissage profond.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=1}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input  n=1}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

### Définition du modèle

Lorsque nous avons implémenté la régression linéaire à partir de zéro
dans :numref:`sec_linear_scratch`,
nous avons défini les paramètres de notre modèle de manière explicite
et codé les calculs pour produire la sortie
en utilisant des opérations d'algèbre linéaire de base.
Vous *devriez* savoir comment faire cela.
Mais lorsque vos modèles deviendront plus complexes,
et lorsque vous devrez effectuer ces calculs presque tous les jours,
vous serez heureux de pouvoir bénéficier de cette assistance.
La situation est similaire à celle qui consiste à coder son propre blog à partir de zéro.
Le faire une ou deux fois est gratifiant et instructif,
mais vous seriez un piètre développeur web
si vous passiez un mois à réinventer la roue.

Pour les opérations standard,
nous pouvons [**utiliser les couches prédéfinies d'un framework,**]
ce qui nous permet de nous concentrer
sur les couches utilisées pour construire le modèle
plutôt que de nous soucier de leur mise en œuvre.
Rappelons l'architecture d'un réseau à couche unique
telle que décrite dans :numref:`fig_single_neuron`.
La couche est dite *entièrement connectée*,
puisque chacune de ses entrées est connectée
à chacune de ses sorties
au moyen d'une multiplication matrice-vecteur.

:begin_tab:`mxnet`
Dans Gluon, la couche entièrement connectée est définie dans la classe `Dense`.
Puisque nous ne voulons générer qu'une seule sortie scalaire,
nous fixons ce nombre à 1.
Il est intéressant de noter que, pour des raisons de commodité,
Gluon ne nous demande pas de spécifier
la forme des entrées pour chaque couche.
Nous n'avons donc pas besoin d'indiquer à Gluon
combien d'entrées entrent dans cette couche linéaire.
Lorsque nous faisons passer des données par notre modèle pour la première fois,
par exemple, lorsque nous exécutons `net(X)` plus tard,
Gluon déduira automatiquement le nombre d'entrées pour chaque couche et
instanciera ainsi le modèle correct.
Nous décrirons plus en détail comment cela fonctionne plus tard.
:end_tab:

:begin_tab:`pytorch`
Dans PyTorch, la couche entièrement connectée est définie dans les classes `Linear` et `LazyLinear` (disponibles depuis la version 1.8.0). 
La dernière classe
permet aux utilisateurs de *seulement* spécifier
la dimension de sortie,
tandis que la première classe
demande en plus
combien d'entrées vont dans cette couche.
Il est peu pratique de spécifier les formes d'entrée,
qui peuvent nécessiter des calculs non triviaux
(comme dans les couches convolutionnelles).
Ainsi, pour des raisons de simplicité, nous utiliserons ces couches "paresseuses"
chaque fois que nous le pourrons. 
:end_tab:

:begin_tab:`tensorflow`
Dans Keras, la couche entièrement connectée est définie dans la classe `Dense`.
Puisque nous ne voulons générer qu'une seule sortie scalaire,
nous fixons ce nombre à 1.
Il convient de noter que, pour des raisons de commodité,
Keras ne nous demande pas de spécifier
la forme de l'entrée pour chaque couche.
Nous n'avons pas besoin d'indiquer à Keras
combien d'entrées entrent dans cette couche linéaire.
Lorsque nous essayons pour la première fois de faire passer des données par notre modèle,
par exemple, lorsque nous exécutons `net(X)` plus tard,
Keras déduira automatiquement
le nombre d'entrées de chaque couche.
Nous décrirons plus en détail comment cela fonctionne plus tard.
:end_tab:

```{.python .input}
%%tab all
class LinearRegression(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

Dans la méthode `forward`, nous invoquons simplement la fonction intégrée `__call__` des couches prédéfinies pour calculer les sorties.

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    """The linear regression model."""
    return self.net(X)
```

### Définition de la fonction de perte

:begin_tab:`mxnet` 
Le module `loss` définit de nombreuses fonctions de perte utiles.
Pour des raisons de rapidité et de commodité, nous renonçons à implémenter notre propre
et choisissons plutôt la fonction intégrée `loss.L2Loss`.
Étant donné que l'adresse `loss` qu'il renvoie est
l'erreur quadratique pour chaque exemple,
nous utilisons `mean`pour faire la moyenne de la perte sur le minilot.
:end_tab:

:begin_tab:`pytorch`
[**La classe `MSELoss` calcule l'erreur quadratique moyenne (sans le facteur $1/2$ dans :eqref:`eq_mse` ).**]
Par défaut, `MSELoss` renvoie la perte moyenne sur les exemples.
C'est plus rapide (et plus facile à utiliser) que d'implémenter notre propre méthode.
:end_tab:

:begin_tab:`tensorflow`
La classe `MeanSquaredError` calcule l'erreur quadratique moyenne (sans le facteur $1/2$ dans :eqref:`eq_mse` ).
Par défaut, elle renvoie la perte moyenne sur les exemples.
:end_tab:

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

## Définition de l'algorithme d'optimisation

:begin_tab:`mxnet` 
Minibatch SGD est un outil standard
pour l'optimisation des réseaux neuronaux
et donc Gluon le supporte ainsi qu'un certain nombre de
variations de cet algorithme par le biais de sa classe `Trainer`.
Notez que la classe `Trainer` de Gluon représente
l'algorithme d'optimisation,
alors que la classe `Trainer` que nous avons créée dans :numref:`sec_oo-design` 
contient la fonction d'entraînement,
c'est-à-dire l'appel répété de l'optimiseur
pour mettre à jour les paramètres du modèle.
Lorsque nous instancions `Trainer`,
nous spécifions les paramètres à optimiser,
que nous pouvons obtenir de notre modèle `net` via `net.collect_params()`,
l'algorithme d'optimisation que nous souhaitons utiliser (`sgd`),
et un dictionnaire d'hyperparamètres
requis par notre algorithme d'optimisation.
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD est un outil standard
pour l'optimisation des réseaux neuronaux
et PyTorch le prend donc en charge ainsi qu'un certain nombre de variations
de cet algorithme dans le module `optim`.
Lorsque nous (**instantions une instance `SGD`,**)
nous spécifions les paramètres à optimiser,
que nous pouvons obtenir de notre modèle via `self.parameters()`,
et le taux d'apprentissage (`self.lr`)
requis par notre algorithme d'optimisation.
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGD est un outil standard
pour l'optimisation des réseaux de neurones
et Keras le prend donc en charge ainsi qu'un certain nombre de
variations de cet algorithme dans le module `optimizers`.
:end_tab:

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
```

## Training

Vous avez peut-être remarqué que l'expression de notre modèle via
les API de haut niveau d'un cadre d'apprentissage profond
nécessite moins de lignes de code.
Nous n'avons pas eu à allouer les paramètres individuellement,
à définir notre fonction de perte, ou à implémenter le SGD en minibatch.
Lorsque nous commencerons à travailler avec des modèles beaucoup plus complexes,
les avantages de l'API de haut niveau augmenteront considérablement.
Maintenant que tous les éléments de base sont en place,
[**la boucle d'entraînement elle-même est la même
que celle que nous avons implémentée à partir de zéro.**]
Il nous suffit donc d'appeler la méthode `fit` (introduite dans :numref:`oo-design-training` ),
qui s'appuie sur l'implémentation de la méthode `fit_epoch`
dans :numref:`sec_linear_scratch`,
pour entraîner notre modèle.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Ci-dessous, nous
[**comparons les paramètres du modèle appris
par entraînement sur des données finies
et les paramètres réels**]
qui ont généré notre jeu de données.
Pour accéder aux paramètres,
nous accédons aux poids et au biais
de la couche dont nous avons besoin.
Comme dans notre implémentation à partir de zéro,
notez que nos paramètres estimés
sont proches de leurs vrais homologues.

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Résumé

Cette section contient la première implémentation
d'un réseau profond (dans ce livre)
pour exploiter les commodités offertes
par les frameworks modernes d'apprentissage profond,
tels que Gluon `Chen.Li.Li.ea.2015`, 
JAX :cite:`Frostig.Johnson.Leary.2018`, 
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, 
et Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`.
Nous avons utilisé les valeurs par défaut du framework pour charger les données, définir une couche,
une fonction de perte, un optimiseur et une boucle d'apprentissage.
Lorsque le framework fournit toutes les fonctionnalités nécessaires,
c'est généralement une bonne idée de les utiliser,
puisque les implémentations de la bibliothèque de ces composants
ont tendance à être fortement optimisées pour les performances
et correctement testées pour la fiabilité.
Dans le même temps, essayez de ne pas oublier
que ces modules *peuvent* être implémentés directement.
Ceci est particulièrement important pour les chercheurs en herbe
qui souhaitent vivre à la pointe du développement de modèles,
où vous devrez inventer de nouveaux composants
qui ne peuvent exister dans aucune bibliothèque actuelle.

:begin_tab:`mxnet`
Dans Gluon, le module `data` fournit des outils pour le traitement des données,
le module `nn` définit un grand nombre de couches de réseaux neuronaux,
et le module `loss` définit de nombreuses fonctions de perte courantes.
De plus, le module `initializer` donne accès à
à de nombreux choix pour l'initialisation des paramètres.
De manière pratique pour l'utilisateur, la dimensionnalité et le stockage de
sont automatiquement déduits.
Une conséquence de cette initialisation paresseuse est que
vous ne devez pas essayer d'accéder aux paramètres
avant qu'ils aient été instanciés (et initialisés).
:end_tab:

:begin_tab:`pytorch`
Dans PyTorch, le module `data` fournit des outils pour le traitement des données,
le module `nn` définit un grand nombre de couches de réseaux neuronaux et de fonctions de perte communes.
Nous pouvons initialiser les paramètres en remplaçant leurs valeurs
par des méthodes se terminant par `_`.
Notez que nous devons spécifier les dimensions d'entrée du réseau.
Bien que cela soit trivial pour l'instant, cela peut avoir des répercussions importantes
lorsque nous voulons concevoir des réseaux complexes comportant de nombreuses couches.
Une réflexion approfondie sur la manière de paramétrer ces réseaux
est nécessaire pour permettre la portabilité.
:end_tab:

:begin_tab:`tensorflow`
Dans TensorFlow, le module `data` fournit des outils pour le traitement des données,
; le module `keras` définit un grand nombre de couches de réseaux neuronaux et de fonctions de perte communes.
En outre, le module `initializers` fournit diverses méthodes d'initialisation des paramètres du modèle.
La dimensionnalité et le stockage des réseaux sont automatiquement déduits
(mais veillez à ne pas tenter d'accéder aux paramètres avant qu'ils n'aient été initialisés).
:end_tab:

## Exercices

1. Comment devrez-vous modifier le taux d'apprentissage si vous remplacez la perte globale sur le minibatch
par une moyenne sur la perte sur le minibatch ?
1. Consultez la documentation du framework pour voir quelles fonctions de perte sont fournies. En particulier,
remplacez la perte au carré par la fonction de perte robuste de Huber. Autrement dit, utilisez la fonction de perte
$$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \text{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \text{ otherwise}\end{cases}$$ 
 1. Comment accéder au gradient des poids du modèle ?
1. Comment la solution change-t-elle si vous modifiez le taux d'apprentissage et le nombre d'époques ? Continue-t-elle à s'améliorer ?
1. Comment la solution change-t-elle si vous modifiez la quantité de données générées ?
   1. Tracez l'erreur d'estimation pour $\hat{\mathbf{w}} - \mathbf{w}$ et $\hat{b} - b$ en fonction de la quantité de données. Conseil : augmentez la quantité de données de façon logarithmique plutôt que linéaire, c'est-à-dire 5, 10, 20, 50, ..., 10 000 plutôt que 1 000, 2 000, ..., 10 000.
   2. Pourquoi la suggestion de l'astuce est-elle appropriée ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
