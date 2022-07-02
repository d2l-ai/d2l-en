# Naive Bayes
:label:`sec_naive_bayes` 

 Tout au long des sections précédentes, nous avons appris la théorie des probabilités et des variables aléatoires.  Pour mettre cette théorie en pratique, nous allons présenter le classificateur *Bayes naïf*.  Celui-ci n'utilise rien d'autre que les bases probabilistes pour nous permettre d'effectuer la classification de chiffres.

L'apprentissage consiste à faire des hypothèses. Si nous voulons classer un nouvel exemple de données que nous n'avons jamais vu auparavant, nous devons faire quelques hypothèses sur les exemples de données qui sont similaires les uns aux autres. Le classificateur Bayes naïf, un algorithme populaire et remarquablement clair, suppose que toutes les caractéristiques sont indépendantes les unes des autres afin de simplifier le calcul. Dans cette section, nous allons appliquer ce modèle pour reconnaître des caractères dans des images.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Reconnaissance optique de caractères

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` est l'un des jeux de données les plus utilisés. Il contient 60 000 images pour la formation et 10 000 images pour la validation. Chaque image contient un chiffre écrit à la main de 0 à 9. La tâche consiste à classer chaque image dans le chiffre correspondant.

Gluon fournit une classe `MNIST` dans le module `data.vision` pour
récupérer automatiquement l'ensemble de données sur Internet.
Par la suite, Gluon utilisera la copie locale déjà téléchargée.
Nous spécifions si nous demandons l'ensemble d'entraînement ou l'ensemble de test
en fixant la valeur du paramètre `train` à `True` ou `False`, respectivement.
Chaque image est une image en niveaux de gris avec une largeur et une hauteur de $28$ et une forme ($28$,$28$,$1$). Nous utilisons une transformation personnalisée pour supprimer la dernière dimension du canal. En outre, l'ensemble de données représente chaque pixel par un entier non signé $8$-bit.  Nous les quantifions en caractéristiques binaires pour simplifier le problème.

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

Nous pouvons accéder à un exemple particulier, qui contient l'image et le label correspondant.

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Notre exemple, stocké ici dans la variable `image`, correspond à une image d'une hauteur et d'une largeur de $28$ pixels.

```{.python .input}
#@tab all
image.shape, image.dtype
```

Notre code stocke le label de chaque image comme un scalaire. Son type est un entier de $32$-bit.

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

Nous pouvons également accéder à plusieurs exemples en même temps.

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Visualisons ces exemples.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## Le modèle probabiliste pour la classification

Dans une tâche de classification, nous faisons correspondre un exemple à une catégorie. Ici, un exemple est une image en niveaux de gris $28\times 28$, et une catégorie est un chiffre. (Reportez-vous à :numref:`sec_softmax` pour une explication plus détaillée.)
Une façon naturelle d'exprimer la tâche de classification est de poser la question probabiliste suivante : quelle est l'étiquette la plus probable étant donné les caractéristiques (c'est-à-dire les pixels de l'image) ? On désigne par $\mathbf x\in\mathbb R^d$ les caractéristiques de l'exemple et par $y\in\mathbb R$ l'étiquette. Les caractéristiques sont ici des pixels d'image, où nous pouvons transformer une image à $2$ dimensions en un vecteur de sorte que $d=28^2=784$, et les étiquettes sont des chiffres.
La probabilité de l'étiquette compte tenu des caractéristiques est $p(y  \mid  \mathbf{x})$. Si nous sommes en mesure de calculer ces probabilités, qui sont $p(y  \mid  \mathbf{x})$ pour $y=0, \ldots,9$ dans notre exemple, alors le classificateur produira la prédiction $\hat{y}$ donnée par l'expression :

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$ 

 Malheureusement, cela exige que nous estimions $p(y  \mid  \mathbf{x})$ pour chaque valeur de $\mathbf{x} = x_1, ..., x_d$. Imaginez que chaque caractéristique puisse prendre l'une des valeurs $2$. Par exemple, la caractéristique $x_1 = 1$ pourrait signifier que le mot pomme apparaît dans un document donné et $x_1 = 0$ signifierait qu'il n'apparaît pas. Si nous disposons de $30$ caractéristiques binaires de ce type, cela signifie que nous devons être prêts à classer n'importe laquelle des $2^{30}$ (plus d'un milliard !) valeurs possibles du vecteur d'entrée $\mathbf{x}$.

En outre, où est l'apprentissage ? Si nous devons voir chaque exemple possible afin de prédire l'étiquette correspondante, nous n'apprenons pas vraiment un modèle, mais nous nous contentons de mémoriser l'ensemble des données.

## Le classificateur Naive Bayes

Heureusement, en faisant quelques hypothèses sur l'indépendance conditionnelle, nous pouvons introduire un biais inductif et construire un modèle capable de généraliser à partir d'une sélection relativement modeste d'exemples d'apprentissage. Pour commencer, utilisons le théorème de Bayes, pour exprimer le classificateur comme suit :

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$ 

 Notez que le dénominateur est le terme de normalisation $p(\mathbf{x})$ qui ne dépend pas de la valeur de l'étiquette $y$. Par conséquent, nous devons uniquement nous préoccuper de la comparaison du numérateur entre différentes valeurs de $y$. Même si le calcul du dénominateur s'avérait difficile, nous pourrions l'ignorer, tant que nous pouvons évaluer le numérateur. Heureusement, même si nous voulions récupérer la constante de normalisation, nous le pourrions.  Nous pouvons toujours récupérer le terme de normalisation puisque $\sum_y p(y  \mid  \mathbf{x}) = 1$.

Maintenant, concentrons-nous sur $p( \mathbf{x}  \mid  y)$. En utilisant la règle de probabilité en chaîne, nous pouvons exprimer le terme $p( \mathbf{x}  \mid  y)$ comme suit :

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$ 

 En soi, cette expression ne nous fait pas avancer. Nous devons toujours estimer approximativement les paramètres de $2^d$. Cependant, si nous supposons que * les caractéristiques sont conditionnellement indépendantes les unes des autres, étant donné l'étiquette *, nous sommes soudainement en bien meilleure posture, car ce terme se simplifie en $\prod_i p(x_i  \mid  y)$, ce qui nous donne le prédicteur

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$ 

 Si nous pouvons estimer $p(x_i=1  \mid  y)$ pour chaque $i$ et $y$, et enregistrer sa valeur dans $P_{xy}[i, y]$, où $P_{xy}$ est une matrice $d\times n$ avec $n$ étant le nombre de classes et $y\in\{1, \ldots, n\}$, alors nous pouvons également l'utiliser pour estimer $p(x_i = 0 \mid y)$, c'est-à-dire ,

$$ 
p(x_i = t_i \mid y) = 
\begin{cases}
    P_{xy}[i, y] & \text{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \text{for } t_i = 0 .
\end{cases}
$$

En outre, nous estimons $p(y)$ pour chaque $y$ et l'enregistrons dans $P_y[y]$, $P_y$ étant un vecteur de longueur $n$. Ensuite, pour tout nouvel exemple $\mathbf t = (t_1, t_2, \ldots, t_d)$, nous pouvons calculer

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$ 
 :eqlabel:`eq_naive_bayes_estimation` 

 pour tout $y$. Ainsi, notre hypothèse d'indépendance conditionnelle a fait passer la complexité de notre modèle d'une dépendance exponentielle du nombre de caractéristiques $\mathcal{O}(2^dn)$ à une dépendance linéaire, qui est $\mathcal{O}(dn)$.


## Formation

Le problème maintenant est que nous ne connaissons pas $P_{xy}$ et $P_y$. Nous devons donc d'abord estimer leurs valeurs en fonction de certaines données d'entraînement. Il s'agit de l'*entraînement* du modèle. L'estimation de $P_y$ n'est pas trop difficile. Puisque nous ne traitons que des classes $10$, nous pouvons compter le nombre d'occurrences $n_y$ pour chacun des chiffres et le diviser par la quantité totale de données $n$. Par exemple, si le chiffre 8 apparaît $n_8 = 5,800$ fois et que nous avons un total de $n = 60,000$ images, l'estimation de la probabilité est $p(y=8) = 0.0967$.

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Passons maintenant à des choses un peu plus difficiles $P_{xy}$. Puisque nous avons choisi des images en noir et blanc, $p(x_i  \mid  y)$ désigne la probabilité que le pixel $i$ soit allumé pour la classe $y$. Comme précédemment, nous pouvons compter le nombre d'occurrences de $n_{iy}$ pour lesquelles un événement se produit et le diviser par le nombre total d'occurrences de $y$, c'est-à-dire $n_y$. Mais il y a quelque chose de légèrement troublant : certains pixels peuvent ne jamais être noirs (par exemple, pour les images bien recadrées, les pixels des coins peuvent toujours être blancs). Un moyen pratique pour les statisticiens de traiter ce problème consiste à ajouter des pseudo-comptes à toutes les occurrences. Ainsi, au lieu de $n_{iy}$, on utilise $n_{iy}+1$ et au lieu de $n_y$, on utilise $n_{y}+2$ (puisque le pixel $i$ peut prendre deux valeurs possibles - il peut être noir ou blanc). Cette méthode est également appelée *lissage de Laplace*.  Il peut sembler ad hoc, mais il peut être motivé d'un point de vue bayésien par un modèle bêta-binomial.

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

En visualisant ces probabilités $10\times 28\times 28$ (pour chaque pixel de chaque classe), nous pouvons obtenir des chiffres qui ressemblent à des moyennes.

Nous pouvons maintenant utiliser :eqref:`eq_naive_bayes_estimation` pour prédire une nouvelle image. Étant donné $\mathbf x$, les fonctions suivantes calculent $p(\mathbf x \mid y)p(y)$ pour chaque $y$.

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Cela s'est très mal passé ! Pour savoir pourquoi, examinons les probabilités par pixel. Il s'agit généralement de nombres compris entre $0.001$ et $1$. Nous les multiplions par $784$. À ce stade, il convient de préciser que nous calculons ces chiffres sur un ordinateur, donc avec une plage fixe pour l'exposant. Ce qui se passe, c'est que nous subissons un *débordement numérique *, c'est-à-dire que la multiplication de tous les petits nombres conduit à quelque chose d'encore plus petit jusqu'à ce qu'il soit arrondi à zéro.  Nous avons abordé ce problème d'un point de vue théorique sur :numref:`sec_maximum_likelihood` , mais nous constatons clairement ce phénomène dans la pratique.

Comme indiqué dans cette section, nous corrigeons ce problème en utilisant le fait que $\log a b = \log a + \log b$, c'est-à-dire que nous passons à la sommation des logarithmes.
Même si $a$ et $b$ sont de petits nombres, les valeurs logarithmiques devraient se situer dans une plage appropriée.

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Comme le logarithme est une fonction croissante, nous pouvons réécrire :eqref:`eq_naive_bayes_estimation` comme

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$ 

 Nous pouvons implémenter la version stable suivante :

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Nous pouvons maintenant vérifier si la prédiction est correcte.

```{.python .input}
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

Si nous prédisons maintenant quelques exemples de validation, nous pouvons constater que le classificateur de Bayes
fonctionne assez bien.

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() 
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Enfin, calculons la précision globale du classificateur.

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

Les réseaux profonds modernes atteignent des taux d'erreur inférieurs à $0.01$. Cette performance relativement faible est due aux hypothèses statistiques incorrectes que nous avons faites dans notre modèle : nous avons supposé que chaque pixel est généré *indépendamment*, en fonction uniquement de l'étiquette. Ce n'est clairement pas la façon dont les humains écrivent les chiffres, et cette hypothèse erronée a conduit à la chute de notre classificateur (Bayes) trop naïf.

## Résumé
* En utilisant la règle de Bayes, un classificateur peut être réalisé en supposant que toutes les caractéristiques observées sont indépendantes. 
* Ce classifieur peut être entraîné sur un ensemble de données en comptant le nombre d'occurrences de combinaisons d'étiquettes et de valeurs de pixels.
* Ce classificateur a été la référence pendant des décennies pour des tâches telles que la détection de spam.

## Exercices
1. Considérons l'ensemble de données $[[0,0], [0,1], [1,0], [1,1]]$ avec des étiquettes données par le XOR des deux éléments $[0,1,1,0]$.  Quelles sont les probabilités pour un classificateur Naive Bayes construit sur ce jeu de données.  Est-ce qu'il réussit à classer nos points ?  Si non, quelles hypothèses sont violées ?
1. Supposons que nous n'ayons pas utilisé le lissage de Laplace lors de l'estimation des probabilités et qu'un exemple de données contenant une valeur jamais observée lors de la formation arrive au moment du test.  Quel serait le résultat du modèle ?
1. Le classificateur de Bayes naïf est un exemple spécifique de réseau bayésien, où la dépendance des variables aléatoires est codée par une structure de graphe.  Bien que la théorie complète dépasse le cadre de cette section (voir :cite:`Koller.Friedman.2009` pour plus de détails), expliquez pourquoi le fait d'autoriser une dépendance explicite entre les deux variables d'entrée dans le modèle XOR permet de créer un classificateur efficace.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:
