```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Implémentation de la régression softmax à partir de zéro
:label:`sec_softmax_scratch` 

 La régression softmax étant si fondamentale,
nous pensons que vous devez savoir
comment l'implémenter vous-même.
Ici, nous nous limitons à définir les aspects spécifiques à
softmax du modèle
et réutilisons les autres composants
de notre section sur la régression linéaire,
y compris la boucle d'entraînement.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Le Softmax

Commençons par la partie la plus importante :
le mappage des scalaires en probabilités.
Pour vous rafraîchir la mémoire, rappelez-vous le fonctionnement de l'opérateur de somme
le long de dimensions spécifiques dans un tenseur,
comme nous l'avons vu dans :numref:`subsec_lin-alg-reduction` 
 et :numref:`subsec_lin-alg-non-reduction` .
[**Étant donné une matrice `X`, nous pouvons faire la somme de tous les éléments (par défaut) ou seulement
des éléments du même axe.**]
La variable `axis` nous permet de calculer les sommes de lignes et de colonnes :

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Le calcul de la softmax nécessite trois étapes :
(i) exponentiation de chaque terme ;
(ii) somme sur chaque ligne pour calculer la constante de normalisation pour chaque exemple ;
(iii) division de chaque ligne par sa constante de normalisation,
en s'assurant que la somme du résultat est égale à 1.

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$ 
**)

Le (logarithme du) dénominateur
est appelé la (log) *fonction de partition*.
Elle a été introduite dans [statistical physics](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
pour faire la somme de tous les états possibles dans un ensemble thermodynamique.
L'implémentation est simple :

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Pour toute entrée `X`, [**nous transformons chaque élément
en un nombre non négatif.
La somme de chaque ligne est égale à 1,**]
comme pour une probabilité. Attention : le code ci-dessus n'est *pas* robuste contre les arguments très grands ou très petits. Bien que cela soit suffisant pour illustrer ce qui se passe, vous ne devriez *pas* utiliser ce code mot à mot dans un but sérieux. Les cadres d'apprentissage profond ont de telles protections intégrées et nous utiliserons le softmax intégré à l'avenir.

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## Le modèle

Nous avons maintenant tout ce dont nous avons besoin
pour implémenter [**le modèle de régression softmax.**]
Comme dans notre exemple de régression linéaire,
chaque instance sera représentée
par un vecteur de longueur fixe.
Les données brutes étant ici constituées
d'images de pixels $28 \times 28$,
[**nous aplatissons chaque image,
en les traitant comme des vecteurs de longueur 784.**]
Dans les chapitres suivants, nous présenterons
les réseaux neuronaux convolutifs,
qui exploitent la structure spatiale
de manière plus satisfaisante.


Dans la régression softmax,
le nombre de sorties de notre réseau
doit être égal au nombre de classes.
(**Puisque notre ensemble de données comporte 10 classes,
notre réseau a une dimension de sortie de 10.**)
Par conséquent, nos poids constituent une matrice $784 \times 10$
 plus un vecteur ligne de dimension $1 \times 10$ pour les biais.
Comme pour la régression linéaire,
nous initialisons les poids `W`
 avec un bruit gaussien.
Les biais sont initialisés avec des zéros.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

Le code ci-dessous définit comment le réseau
fait correspondre chaque entrée à une sortie.
Notez que nous aplatissons chaque image de $28 \times 28$ pixels dans le lot
en un vecteur en utilisant `reshape`
 avant de faire passer les données dans notre modèle.

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    return softmax(d2l.matmul(d2l.reshape(
        X, (-1, self.W.shape[0])), self.W) + self.b)
```

## La perte d'entropie croisée

Nous devons ensuite implémenter la fonction de perte d'entropie croisée
(présentée dans :numref:`subsec_softmax-regression-loss-func` ).
Il s'agit peut-être de la fonction de perte la plus courante
dans tout l'apprentissage profond.
À l'heure actuelle, les applications de l'apprentissage profond
qui posent facilement des problèmes de classification
sont beaucoup plus nombreuses que celles qui sont mieux traitées comme des problèmes de régression.

Rappelons que l'entropie croisée prend la log-vraisemblance négative
de la probabilité prédite attribuée à l'étiquette réelle.
Pour des raisons d'efficacité, nous évitons les boucles for de Python et utilisons plutôt l'indexation.
En particulier, le codage à un coup dans $\mathbf{y}$
 nous permet de sélectionner les termes correspondants dans $\hat{\mathbf{y}}$.

Pour voir cela en action, nous [**créons des données échantillons `y_hat` avec 2 exemples de probabilités prédites sur 3 classes et leurs étiquettes correspondantes `y`.**]
Les étiquettes correctes sont respectivement $1$ et $2$.
[**En utilisant `y` comme indices des probabilités dans `y_hat`,**]
, nous pouvons sélectionner les termes efficacement.

```{.python .input}
%%tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Nous pouvons maintenant (**mettre en œuvre la fonction de perte d'entropie croisée**) en faisant la moyenne des logarithmes des probabilités sélectionnées.

```{.python .input}
%%tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.reduce_mean(d2l.log(y_hat[range(len(y_hat)), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return - tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

## Formation

Nous réutilisons la méthode `fit` définie dans :numref:`sec_linear_scratch` pour [**former le modèle avec 10 époques.**]
Notez que le nombre d'époques (`max_epochs`),
la taille des minibatchs (`batch_size`),
et le taux d'apprentissage (`lr`)
sont des hyperparamètres ajustables.
Cela signifie que, bien que ces valeurs ne soient pas apprises au cours de notre boucle d'apprentissage primaire (
),
elles influencent néanmoins les performances (
) de notre modèle, par rapport à l'apprentissage (
) et aux performances de généralisation.
Dans la pratique, vous voudrez choisir ces valeurs
en fonction de la répartition *validation* des données
et évaluer ensuite votre modèle final
sur la répartition *test*.
Comme nous l'avons vu à l'adresse :numref:`subsec_generalization-model-selection` ,
nous traiterons les données de test de Fashion-MNIST
comme l'ensemble de validation, et nous rapporterons donc
la perte de validation et la précision de validation
sur cette division.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Prédiction

Maintenant que l'entrainement est terminée,
notre modèle est prêt à [**classer certaines images.**]

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
preds = d2l.argmax(model(X), axis=1)
preds.shape
```

Nous sommes plus intéressés par les images que nous étiquetons *incorrectement*. Nous les visualisons en
en comparant leurs étiquettes réelles
(première ligne de sortie de texte)
avec les prédictions du modèle
(deuxième ligne de sortie de texte).

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Résumé

A présent, nous commençons à avoir un peu d'expérience
dans la résolution de problèmes de régression linéaire
et de classification.
Avec cela, nous avons atteint ce qui serait sans doute
l'état de l'art des années 1960-1970 de la modélisation statistique.
Dans la section suivante, nous vous montrerons comment tirer parti des cadres d'apprentissage profond
pour mettre en œuvre ce modèle
de manière beaucoup plus efficace.

## Exercices

1. Dans cette section, nous avons directement implémenté la fonction softmax en nous basant sur la définition mathématique de l'opération softmax. Comme nous l'avons vu dans :numref:`sec_softmax` , cela peut provoquer des instabilités numériques.
   1. Testez si `softmax` fonctionne toujours correctement si une entrée a une valeur de $100$?
 1. Testez si `softmax` fonctionne toujours correctement si la plus grande de toutes les entrées est inférieure à $-100$?
 1. Implémentez un correctif en regardant la valeur relative à la plus grande entrée dans l'argument.
1. Implémentez une fonction `cross_entropy` qui suit la définition de la fonction de perte d'entropie croisée $\sum_i y_i \log \hat{y}_i$.
   1. Essayez-la dans l'exemple de code ci-dessus.
   1. Pourquoi pensez-vous qu'elle s'exécute plus lentement ?
   1. Devriez-vous l'utiliser ? Dans quels cas cela aurait-il un sens ?
   1. À quoi devez-vous faire attention ? Conseil : considérez le domaine du logarithme.
1. Est-ce toujours une bonne idée de renvoyer l'étiquette la plus probable ? Par exemple, feriez-vous cela pour un diagnostic médical ? Comment essayeriez-vous de résoudre ce problème ?
1. Supposons que nous voulions utiliser la régression softmax pour prédire le mot suivant en fonction de certaines caractéristiques. Quels sont les problèmes que peut poser un grand vocabulaire ?
1. Expérimentez avec les hyperparamètres du code ci-dessus. En particulier :
   1. Tracez comment la perte de validation change quand vous changez le taux d'apprentissage.
   1. Les pertes de validation et d'apprentissage changent-elles lorsque vous modifiez la taille des minibatchs ? Quelle taille faut-il atteindre avant de voir un effet ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
