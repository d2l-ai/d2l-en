```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Weight Decay
:label:`sec_weight_decay` 

Maintenant que nous avons caractérisé le problème de l'overfitting,
nous pouvons introduire notre première technique de *régularisation*.
Rappelons que nous pouvons toujours atténuer l'overfitting
en collectant davantage de données d'entraînement.
Toutefois, cela peut s'avérer coûteux, long,
ou totalement hors de notre contrôle,
ce qui rend la chose impossible à court terme.
Pour l'instant, nous pouvons supposer que nous disposons déjà de
autant de données de haute qualité que nos ressources le permettent
et que nous nous concentrons sur les outils à notre disposition
même lorsque l'ensemble de données est considéré comme acquis.

Rappelez-vous que dans notre exemple de régression polynomiale
(:numref:`subsec_polynomial-curve-fitting` )
nous pouvions limiter la capacité de notre modèle
en modifiant le degré
du polynôme ajusté.
En effet, la limitation du nombre de caractéristiques
est une technique populaire pour atténuer l'overfitting.
Cependant, le simple fait d'écarter les caractéristiques
peut s'avérer un instrument trop brutal.
Si l'on s'en tient à l'exemple de la régression polynomiale,
examinons ce qui pourrait se passer
avec des données d'entrée à haute dimension.
Les extensions naturelles des polynômes
aux données multivariées sont appelées *monomiales*,
qui sont simplement des produits de puissances de variables.
Le degré d'un monôme est la somme des puissances.
Par exemple, $x_1^2 x_2$, et $x_3 x_5^2$
sont tous deux des monômes de degré 3.

Notez que le nombre de termes de degré $d$
augmente rapidement lorsque $d$ devient plus grand.
Étant donné $k$ variables, le nombre de monômes
de degré $d$ (c'est-à-dire $k$ multichoose $d$) est ${k - 1 + d} \choose {k - 1}$.
Même de petits changements de degré, par exemple de $2$ à $3$,
augmentent considérablement la complexité de notre modèle.
Nous avons donc souvent besoin d'un outil plus fin
pour ajuster la complexité de la fonction.

## Norms and Weight Decay

(**Plutôt que de manipuler directement le nombre de paramètres,
*weight decay*, opère en restreignant les valeurs 
que les paramètres peuvent prendre.**)
Plus communément appelée $\ell_2$ régularisation
en dehors des cercles d'apprentissage profond
lorsqu'elle est optimisée par la descente de gradient stochastique en minibatch,
la décroissance de poids pourrait être la technique la plus largement utilisée
pour régulariser les modèles paramétriques d'apprentissage automatique.
Cette technique est motivée par l'intuition de base
que parmi toutes les fonctions $f$,
la fonction $f = 0$
 (attribuant la valeur $0$ à toutes les entrées)
est en quelque sorte la *plus simple*,
et que nous pouvons mesurer la complexité
d'une fonction par la distance de ses paramètres par rapport à zéro.
Mais comment mesurer précisément
la distance entre une fonction et zéro ?
Il n'y a pas de bonne réponse unique.
En fait, des branches entières des mathématiques,
y compris des parties de l'analyse fonctionnelle
et la théorie des espaces de Banach,
sont consacrées à l'étude de ces questions.

Une interprétation simple pourrait être
de mesurer la complexité d'une fonction linéaire
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ 
par une certaine norme de son vecteur de poids, par exemple, $\| \mathbf{w} \|^2$.
Rappelons que nous avons introduit la norme $\ell_2$ et la norme $\ell_1$,
qui sont des cas particuliers de la norme plus générale $\ell_p$
dans :numref:`subsec_lin-algebra-norms`.
La méthode la plus courante pour garantir un petit vecteur de poids
consiste à ajouter sa norme comme terme de pénalité
au problème de minimisation de la perte.
Ainsi, nous remplaçons notre objectif initial,
*minimiser la perte de prédiction sur les étiquettes d'apprentissage*,
par un nouvel objectif,
*minimiser la somme de la perte de prédiction et du terme de pénalité*.
Maintenant, si notre vecteur de poids devient trop grand,
notre algorithme d'apprentissage pourrait se concentrer
sur la minimisation de la norme de poids $\| \mathbf{w} \|^2$
plutôt que sur la minimisation de l'erreur de formation.
C'est exactement ce que nous voulons.
Pour illustrer les choses en code,
nous reprenons notre exemple précédent
de :numref:`sec_linear_regression` pour la régression linéaire.
Dans cet exemple, notre perte était donnée par

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$ 

Rappelez-vous que $\mathbf{x}^{(i)}$ sont les caractéristiques,
$y^{(i)}$ est l'étiquette pour tout exemple de données $i$, et $(\mathbf{w}, b)$
sont les paramètres de poids et de biais, respectivement.
Pour pénaliser la taille du vecteur de poids,
nous devons d'une manière ou d'une autre ajouter $\| \mathbf{w} \|^2$ à la fonction de perte,
mais comment le modèle doit-il échanger la perte standard
contre cette nouvelle pénalité additive ?
En pratique, nous caractérisons ce compromis
via la *constante de régularisation* $\lambda$,
un hyperparamètre non négatif
que nous ajustons à l'aide de données de validation :

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$ 

 
Pour $\lambda = 0$, nous récupérons notre fonction de perte originale.
Pour $\lambda > 0$, nous limitons la taille de $\| \mathbf{w} \|$.
Nous divisons par $2$ par convention :
lorsque nous prenons la dérivée d'une fonction quadratique,
les $2$ et $1/2$ s'annulent, ce qui garantit que l'expression
pour la mise à jour est belle et simple.
Le lecteur avisé pourrait se demander pourquoi nous travaillons avec la norme au carré
et non avec la norme standard (c'est-à-dire la distance euclidienne).
Nous le faisons pour des raisons de commodité informatique.
En élevant au carré la norme $\ell_2$, nous supprimons la racine carrée,
laissant la somme des carrés de
chaque composante du vecteur de poids.
Cela permet de calculer facilement la dérivée de la pénalité : 
la somme des dérivées est égale à la dérivée de la somme.


De plus, vous pourriez vous demander pourquoi nous travaillons avec la norme $\ell_2$
en premier lieu et pas, disons, avec la norme $\ell_1$.
En fait, d'autres choix sont valables et
populaires dans les statistiques.
Alors que les modèles linéaires régularisés $\ell_2$ constituent
l'algorithme classique de la *régression ridge*,
$\ell_1$ la régression linéaire régularisée
est une méthode tout aussi fondamentale en statistique, 
communément appelée *régression lasso*.
L'une des raisons de travailler avec la norme $\ell_2$
est qu'elle impose une pénalité trop importante
sur les grandes composantes du vecteur de poids.
Cela oriente notre algorithme d'apprentissage
vers des modèles qui distribuent le poids de manière égale
sur un plus grand nombre de caractéristiques.
En pratique, cela peut les rendre plus robustes
aux erreurs de mesure d'une seule variable.
En revanche, les pénalités $\ell_1$ conduisent à des modèles
qui concentrent les poids sur un petit ensemble de caractéristiques
en ramenant les autres poids à zéro.
Cela nous donne une méthode efficace de *sélection des caractéristiques*,
qui peut être souhaitable pour d'autres raisons.
Par exemple, si notre modèle ne repose que sur quelques caractéristiques,
nous n'aurons peut-être pas besoin de collecter, stocker ou transmettre des données
pour les autres caractéristiques (abandonnées). 

En utilisant la même notation que dans :eqref:`eq_linreg_batch_update`,
les mises à jour de la descente de gradient stochastique en minibatch
pour $\ell_2$-suivre la régression régularisée :

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Comme précédemment, nous mettons à jour $\mathbf{w}$ sur la base de la quantité
par laquelle notre estimation diffère de l'observation.
Cependant, nous réduisons également la taille de $\mathbf{w}$ vers zéro.
C'est pourquoi la méthode est parfois appelée "décroissance du poids" :
étant donné le terme de pénalité seul,
notre algorithme d'optimisation *décroît*
le poids à chaque étape de l'apprentissage.
Contrairement à la sélection des caractéristiques,
la décroissance du poids nous offre un mécanisme continu
pour ajuster la complexité d'une fonction.
Les plus petites valeurs de $\lambda$ correspondent à
à des $\mathbf{w}$,
moins contraints, tandis que les plus grandes valeurs de $\lambda$
contraignent $\mathbf{w}$ plus considérablement.
L'inclusion ou non d'une pénalité de biais correspondante $b^2$ 
peut varier selon les implémentations, 
et peut varier selon les couches d'un réseau neuronal.
Souvent, nous ne régularisons pas le terme de biais.
Par ailleurs,
bien que $\ell_2$ la régularisation ne soit pas équivalente à la décroissance des poids pour d'autres algorithmes d'optimisation,
l'idée de régularisation par
réduction de la taille des poids
reste vraie.



## Régression linéaire à haute dimension

Nous pouvons illustrer les avantages de la décroissance de poids 
par un exemple synthétique simple.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Tout d'abord, nous [**générons des données comme précédemment**] :

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Dans cet ensemble de données synthétiques, notre étiquette est donnée 
par une fonction linéaire sous-jacente de nos entrées,
corrompue par un bruit gaussien 
avec une moyenne de zéro et un écart type de 0,01.
À des fins d'illustration, 
nous pouvons accentuer les effets de l'ajustement excessif,
en augmentant la dimensionnalité de notre problème à $d = 200$
et en travaillant avec un petit ensemble d'apprentissage de seulement 20 exemples.

```{.python .input  n=5}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Implémentation à partir de zéro

Maintenant, essayons d'implémenter la décroissance de poids à partir de zéro.
Puisque la descente de gradient stochastique en minibatch
est notre optimiseur,
il nous suffit d'ajouter la pénalité $\ell_2$ au carré
à la fonction de perte originale.

### (**Defining $\ell_2$ Norm Penalty**)

La façon la plus pratique d'implémenter cette pénalité
est peut-être de mettre au carré tous les termes en place et de les additionner.

```{.python .input  n=6}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Définition du modèle

Dans le modèle final,
la régression linéaire et la perte au carré n'ont pas changé depuis :numref:`sec_linear_scratch`,
; nous nous contenterons donc de définir une sous-classe de `d2l.LinearRegressionScratch`. Le seul changement ici est que notre perte inclut désormais le terme de pénalité.

```{.python .input  n=7}
%%tab all
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.lambd * l2_penalty(self.w)        
```

Le code suivant ajuste notre modèle sur l'ensemble d'apprentissage avec 20 exemples et l'évalue sur l'ensemble de validation avec 100 exemples.

```{.python .input  n=8}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))
```

### [**Training without Regularization**]

Nous exécutons maintenant ce code avec `lambd = 0`,
en désactivant la décroissance du poids.
Notez que nous surajustons fortement,
diminuant l'erreur d'apprentissage mais pas l'erreur de validation
- un cas d'école de surajustement.

```{.python .input  n=9}
%%tab all
train_scratch(0)
```

### [**Using Weight Decay**]

Ci-dessous, nous fonctionnons avec une dégradation substantielle du poids.
Notez que l'erreur d'apprentissage augmente
mais que l'erreur de validation diminue.
C'est précisément l'effet
que nous attendons de la régularisation.

```{.python .input  n=10}
%%tab all
train_scratch(3)
```

## [**Mise en œuvre concise**]

La décroissance de poids étant omniprésente
dans l'optimisation des réseaux neuronaux,
le cadre d'apprentissage profond la rend particulièrement pratique,
intégrant la décroissance de poids dans l'algorithme d'optimisation lui-même
pour une utilisation facile en combinaison avec n'importe quelle fonction de perte.
En outre, cette intégration présente un avantage sur le plan du calcul,
permettant aux astuces de mise en œuvre d'ajouter la décroissance de poids à l'algorithme,
sans aucune surcharge de calcul supplémentaire.
Puisque la partie de la mise à jour concernant la décroissance du poids
ne dépend que de la valeur actuelle de chaque paramètre,
l'optimiseur doit de toute façon toucher chaque paramètre une fois.

:begin_tab:`mxnet`
Dans le code suivant, nous spécifions
l'hyperparamètre de décroissance de poids directement
à `wd` lors de l'instanciation de notre `Trainer`.
Par défaut, Gluon décompose simultanément les poids et les biais de
.
Notez que l'hyperparamètre `wd`
sera multiplié par `wd_mult`
lors de la mise à jour des paramètres du modèle.
Ainsi, si nous mettons `wd_mult` à zéro,
le paramètre de biais $b$ ne se décomposera pas.
:end_tab:

:begin_tab:`pytorch`
Dans le code suivant, nous spécifions
l'hyperparamètre de décroissance du poids directement
à `weight_decay` lors de l'instanciation de notre optimiseur.
Par défaut, PyTorch décompose simultanément les poids et les biais de
.
Ici, nous définissons uniquement `weight_decay` pour
le poids, donc le paramètre de biais $b$ ne se décomposera pas.
:end_tab:

:begin_tab:`tensorflow`
Dans le code suivant, nous créons un régularisateur $\ell_2$ avec
l'hyperparamètre de décroissance des poids `wd` et nous l'appliquons aux poids de la couche
via l'argument `kernel_regularizer`.
:end_tab:

```{.python .input  n=11}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input  n=12}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), 
                               lr=self.lr, weight_decay=self.wd)
```

```{.python .input  n=13}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

[**Le tracé est similaire à celui obtenu lorsque
nous avons implémenté la décroissance de poids à partir de zéro**].
Toutefois, cette version s'exécute plus rapidement
et est plus facile à mettre en œuvre,
avantages qui deviendront
plus prononcés à mesure que vous aborderez des problèmes plus importants
et que ce travail deviendra plus routinier.

```{.python .input  n=14}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

Jusqu'à présent, nous n'avons abordé qu'une seule notion de
ce qui constitue une fonction linéaire simple.
Par ailleurs, ce qui constitue une fonction non linéaire simple
peut être une question encore plus complexe.
Par exemple, [reproducing kernel Hilbert space (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
permet d'appliquer les outils introduits
pour les fonctions linéaires dans un contexte non linéaire.
Malheureusement, les algorithmes basés sur les RKHS
ont tendance à mal s'adapter aux données volumineuses et hautement dimensionnelles.
Dans ce livre, nous adopterons souvent l'heuristique commune
selon laquelle la décroissance des poids est appliquée
à toutes les couches d'un réseau profond.

## Résumé

* La régularisation est une méthode courante pour traiter l'overfitting. Les techniques de régularisation classiques ajoutent un terme de pénalité à la fonction de perte (lors de la formation) afin de réduire la complexité du modèle appris.
* Un choix particulier pour garder le modèle simple est l'utilisation d'une pénalité $\ell_2$. Cela conduit à une décroissance du poids dans les étapes de mise à jour de l'algorithme de descente de gradient stochastique en minibatchs.
* La fonctionnalité de décroissance du poids est fournie dans les optimiseurs des cadres d'apprentissage profond.
* Différents ensembles de paramètres peuvent avoir des comportements de mise à jour différents dans la même boucle d'apprentissage.



## Exercices

1. Expérimentez la valeur de $\lambda$ dans le problème d'estimation de cette section. Tracez la précision de l'apprentissage et de la validation en fonction de $\lambda$. Qu'observez-vous ?
1. Utilisez un ensemble de validation pour trouver la valeur optimale de $\lambda$. Est-ce vraiment la valeur optimale ? Cela a-t-il de l'importance ?
1. À quoi ressembleraient les équations de mise à jour si, au lieu de $\|\mathbf{w}\|^2$, nous utilisions $\sum_i |w_i|$ comme pénalité de choix (régularisation$\ell_1$ ) ?
1. Nous savons que $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Pouvez-vous trouver une équation similaire pour les matrices (voir la norme de Frobenius dans :numref:`subsec_lin-algebra-norms` ) ?
1. Examinez la relation entre l'erreur d'apprentissage et l'erreur de généralisation. Outre la décroissance des poids, l'augmentation de l'apprentissage et l'utilisation d'un modèle d'une complexité appropriée, quels autres moyens pouvez-vous imaginer pour lutter contre l'overfitting ?
1. Dans les statistiques bayésiennes, nous utilisons le produit de l'antériorité et de la vraisemblance pour obtenir une valeur postérieure via $P(w \mid x) \propto P(x \mid w) P(w)$. Comment pouvez-vous identifier $P(w)$ avec la régularisation ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
