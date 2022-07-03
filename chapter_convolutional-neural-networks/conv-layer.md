```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Convolutions pour les images
:label:`sec_conv_layer` 

Maintenant que nous comprenons comment les couches convolutionnelles fonctionnent en théorie,
nous sommes prêts à voir comment elles fonctionnent en pratique.
En nous appuyant sur notre motivation des réseaux de neurones convolutifs
comme architectures efficaces pour explorer la structure des données d'image,
nous nous en tenons aux images comme exemple courant.


## L'opération de corrélation croisée

Rappelons qu'à proprement parler, les couches convolutionnelles
sont mal nommées, puisque les opérations qu'elles expriment
sont plus précisément décrites comme des corrélations croisées.
D'après nos descriptions des couches convolutionnelles dans :numref:`sec_why-conv`,
dans une telle couche, un tenseur d'entrée
et un tenseur de noyau sont combinés
pour produire un tenseur de sortie par le biais d'une (**opération de corrélation croisée.**)

Ignorons les canaux pour l'instant et voyons comment cela fonctionne
avec des données bidimensionnelles et des représentations cachées.
Dans :numref:`fig_correlation`,
l'entrée est un tenseur bidimensionnel
avec une hauteur de 3 et une largeur de 3.
Nous marquons la forme du tenseur comme $3 \times 3$ ou ($3$, $3$).
La hauteur et la largeur du noyau sont toutes deux de 2.
La forme de la fenêtre du *noyau* (ou *fenêtre de convolution*)
est donnée par la hauteur et la largeur du noyau
(ici, c'est $2 \times 2$).

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

Dans l'opération de corrélation croisée bidimensionnelle,
nous commençons avec la fenêtre de convolution positionnée
au coin supérieur gauche du tenseur d'entrée
et la faisons glisser sur le tenseur d'entrée,
à la fois de gauche à droite et de haut en bas.
Lorsque la fenêtre de convolution glisse vers une certaine position,
le sous-tenseur d'entrée contenu dans cette fenêtre
et le tenseur du noyau sont multipliés par éléments
et le tenseur résultant est additionné,
ce qui donne une seule valeur scalaire.
Ce résultat donne la valeur du tenseur de sortie
à l'emplacement correspondant.
Ici, le tenseur de sortie a une hauteur de 2 et une largeur de 2
et les quatre éléments sont dérivés de
l'opération de corrélation croisée bidimensionnelle :

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Notez que le long de chaque axe, la taille de la sortie
est légèrement inférieure à celle de l'entrée.
Le noyau ayant une largeur et une hauteur supérieures à un,
nous ne pouvons calculer correctement la corrélation croisée
que pour les endroits où le noyau s'inscrit entièrement dans l'image,
la taille de sortie est donnée par la taille d'entrée $n_h \times n_w$
 moins la taille du noyau de convolution $k_h \times k_w$
via

$$(n_h-k_h+1) \times (n_w-k_w+1).$$ 

C'est le cas car nous avons besoin d'un espace suffisant
pour "déplacer" le noyau de convolution dans l'image.
Nous verrons plus tard comment garder la taille inchangée
en remplissant l'image de zéros autour de sa limite
afin qu'il y ait suffisamment d'espace pour déplacer le noyau.
Ensuite, nous implémentons ce processus dans la fonction `corr2d`,
qui accepte un tenseur d'entrée `X` et un tenseur de noyau `K`
et renvoie un tenseur de sortie `Y`.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab mxnet, pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

Nous pouvons construire le tenseur d'entrée `X` et le tenseur de noyau `K`
 à partir de :numref:`fig_correlation` 
pour [**valider la sortie de l'implémentation ci-dessus**]
de l'opération de corrélation croisée bidimensionnelle.

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Couches convolutives

Une couche convolutive effectue une corrélation croisée entre l'entrée et le noyau
et ajoute un biais scalaire pour produire une sortie.
Les deux paramètres d'une couche convolutive
sont le noyau et le biais scalaire.
Lors de l'entrainement de modèles basés sur des couches convolutionnelles,
nous initialisons généralement les noyaux de manière aléatoire,
comme nous le ferions avec une couche entièrement connectée.

Nous sommes maintenant prêts à [**implémenter une couche convolutionnelle bidimensionnelle**]
basée sur la fonction `corr2d` définie ci-dessus.
Dans la méthode du constructeur `__init__`,
nous déclarons `weight` et `bias` comme étant les deux paramètres du modèle.
La fonction de propagation avant
appelle la fonction `corr2d` et ajoute le biais.

```{.python .input}
%%tab mxnet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
%%tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

Dans
$h \times w$ convolution
ou un noyau de convolution $h \times w$,
la hauteur et la largeur du noyau de convolution sont respectivement $h$ et $w$.
Nous appelons également
une couche convolutive avec un noyau de convolution $h \times w$
simplement une couche convolutive $h \times w$.


## Détection des bords d'objets dans les images

Prenons un moment pour analyser [**une application simple d'une couche convolutionnelle :
détecter le bord d'un objet dans une image**]
en trouvant l'emplacement du changement de pixel.
Tout d'abord, nous construisons une "image" de $6\times 8$ pixels.
Les quatre colonnes du milieu sont noires (0) et les autres sont blanches (1).

```{.python .input}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
%%tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

Ensuite, nous construisons un noyau `K` avec une hauteur de 1 et une largeur de 2.
Lorsque nous effectuons l'opération de corrélation croisée avec l'entrée,
si les éléments horizontalement adjacents sont les mêmes,
la sortie est 0. Sinon, la sortie est non nulle.
Notez que ce noyau est un cas particulier d'un opérateur de différence finie. À l'emplacement $(i,j)$, il calcule $x_{i,j} - x_{(i+1),j}$, c'est-à-dire qu'il calcule la différence entre les valeurs des pixels horizontalement adjacents. Il s'agit d'une approximation discrète de la dérivée première dans la direction horizontale. Après tout, pour une fonction $f(i,j)$, sa dérivée est $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$. Voyons comment cela fonctionne en pratique.

```{.python .input}
%%tab all
K = d2l.tensor([[1.0, -1.0]])
```

Nous sommes prêts à effectuer l'opération de corrélation croisée
avec les arguments `X` (notre entrée) et `K` (notre noyau).
Comme vous pouvez le voir, [**nous détectons 1 pour le bord du blanc au noir
et -1 pour le bord du noir au blanc.**]
Toutes les autres sorties prennent la valeur 0.

```{.python .input}
%%tab all
Y = corr2d(X, K)
Y
```

Nous pouvons maintenant appliquer le noyau à l'image transposée.
Comme prévu, il disparaît. [**Le noyau `K` ne détecte que les bords verticaux.**]

```{.python .input}
%%tab all
corr2d(d2l.transpose(X), K)
```

## Apprendre un noyau

La conception d'un détecteur de bords par différences finies `[1, -1]` est intéressante
si nous savons que c'est précisément ce que nous recherchons.
Cependant, si l'on considère des noyaux plus grands,
et des couches successives de convolutions,
il peut être impossible de spécifier
précisément ce que chaque filtre doit faire manuellement.

Voyons maintenant si nous pouvons [**apprendre le noyau qui a généré `Y` à partir de `X`**]
en regardant uniquement les paires entrée-sortie.
Nous construisons d'abord une couche convolutionnelle
et initialisons son noyau comme un tenseur aléatoire.
Ensuite, à chaque itération, nous utiliserons l'erreur quadratique
pour comparer `Y` avec la sortie de la couche convolutive.
Nous pouvons alors calculer le gradient pour mettre à jour le noyau.
Pour des raisons de simplicité,
dans ce qui suit
nous utilisons la classe intégrée
pour les couches convolutionnelles bidimensionnelles
et ignorons le biais.

```{.python .input}
%%tab mxnet
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Learning rate

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
%%tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
%%tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Learning rate

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

Notez que l'erreur est tombée à une faible valeur après 10 itérations. Maintenant, nous allons [**jeter un coup d'œil au tenseur du noyau que nous avons appris.**]

```{.python .input}
%%tab mxnet
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
%%tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

En effet, le tenseur de noyau appris est remarquablement proche
du tenseur de noyau `K` que nous avons défini précédemment.

## Corrélation croisée et convolution

Rappelez-vous notre observation de :numref:`sec_why-conv` sur la correspondance
entre les opérations de corrélation croisée et de convolution.
Continuons ici à considérer les couches convolutionnelles bidimensionnelles.
Et si ces couches
effectuaient des opérations de convolution strictes
telles que définies dans :eqref:`eq_2d-conv-discrete` 
au lieu des corrélations croisées ?
Pour obtenir la sortie de l'opération de *convolution stricte*, il suffit de retourner le tenseur du noyau bidimensionnel à la fois horizontalement et verticalement, puis d'effectuer l'opération de *corrélation croisée* avec le tenseur d'entrée.

Il convient de noter que, puisque les noyaux sont appris à partir des données dans l'apprentissage profond,
les sorties des couches convolutionnelles ne sont pas affectées,
peu importe que ces couches
effectuent
les opérations de convolution stricte
ou les opérations de corrélation croisée.

Pour illustrer cela, supposons qu'une couche convolutionnelle effectue une *corrélation croisée* et apprend le noyau dans :numref:`fig_correlation`, qui est désigné ici comme la matrice $\mathbf{K}$.
En supposant que les autres conditions restent inchangées,
lorsque cette couche effectue une *convolution* stricte à la place,
le noyau appris $\mathbf{K}'$ sera le même que $\mathbf{K}$
après que $\mathbf{K}'$ soit
retourné à la fois horizontalement et verticalement.
En d'autres termes,
lorsque la couche convolutionnelle
effectue une *convolution* stricte
pour l'entrée dans :numref:`fig_correlation` 
et $\mathbf{K}'$,
la même sortie dans :numref:`fig_correlation` 
 (corrélation croisée de l'entrée et $\mathbf{K}$)
sera obtenue.

Conformément à la terminologie standard de la littérature sur l'apprentissage profond,
nous continuerons à désigner l'opération de corrélation croisée
comme une convolution, même si, au sens strict, elle est légèrement différente.
Par ailleurs,
nous utilisons le terme *élément* pour désigner
une entrée (ou composante) de tout tenseur représentant une représentation de couche ou un noyau de convolution.


## Carte de caractéristiques et champ récepteur

Comme décrit dans :numref:`subsec_why-conv-channels`,
la sortie de la couche de convolution dans
:numref:`fig_correlation` 
est parfois appelée une *carte de caractéristiques*,
car elle peut être considérée comme
les représentations apprises (caractéristiques)
dans les dimensions spatiales (par exemple, la largeur et la hauteur)
à la couche suivante.
Dans les CNN,
pour tout élément $x$ d'une couche,
son *champ réceptif* fait référence à
tous les éléments (de toutes les couches précédentes)
qui peuvent affecter le calcul de $x$
pendant la propagation vers l'avant.
Notez que le champ réceptif
peut être plus grand que la taille réelle de l'entrée.

Continuons à utiliser :numref:`fig_correlation` pour expliquer le champ réceptif.
Étant donné le noyau de convolution $2 \times 2$,
le champ réceptif de l'élément de sortie ombré (de valeur $19$)
est
les quatre éléments de la partie ombrée de l'entrée.
Désignons maintenant la sortie de $2 \times 2$
par $\mathbf{Y}$
et considérons un CNN plus profond
avec une couche de convolution supplémentaire $2 \times 2$ qui prend $\mathbf{Y}$
comme entrée et sort
un seul élément $z$.
Dans ce cas,
le champ réceptif de $z$
sur $\mathbf{Y}$ comprend les quatre éléments de $\mathbf{Y}$,
tandis que
le champ réceptif
sur l'entrée comprend les neuf éléments d'entrée.
Ainsi,
lorsqu'un élément d'une carte de caractéristiques
a besoin d'un champ réceptif plus grand
pour détecter les caractéristiques d'entrée sur une zone plus large,
nous pouvons construire un réseau plus profond.

Les champs réceptifs tirent leur nom de la neurophysiologie. Dans une série d'expériences :cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968` sur une série d'animaux 
et différents stimuli, Hubel et Wiesel ont exploré la réponse de ce qu'on appelle le cortex visuel 
auxdits stimuli. Dans l'ensemble, ils ont constaté que les niveaux inférieurs réagissent aux bords et aux formes connexes .
Plus tard, :cite:`Field.1987` ont illustré cet effet sur des images naturelles 
avec ce que l'on ne peut appeler que des noyaux convolutifs. 
Nous reproduisons une figure clé de :numref:`field_visual` pour illustrer les similitudes frappantes 

![Figure and caption taken from :cite:`Field.1987`: An example of coding with six different channels. (Left) Exemples des six types de capteurs associés à chaque canal. (Droite) Convolution de l'image en (milieu) avec les six capteurs montrés en (gauche). La réponse de chaque capteur est déterminée en échantillonnant ces images filtrées à une distance proportionnelle à la taille du capteur (représentée par des points). Ce diagramme montre la réponse des capteurs symétriques pairs uniquement.](../img/field-visual.png)
:label:`field_visual` 

Il s'avère que cette relation est également valable pour les caractéristiques calculées par des couches plus profondes de réseaux formés à des tâches de classification d'images, comme le montre par exemple 
dans :cite:`Kuzovkin.Vicente.Petton.ea.2018`. Il suffit de dire que les convolutions se sont avérées être un outil incroyablement puissant pour la vision par ordinateur, tant en biologie qu'en code. En tant que telles, il n'est pas surprenant (avec le recul) qu'elles aient annoncé les récents succès de l'apprentissage profond. 

## Résumé

Le calcul de base requis pour une couche convolutionnelle est une opération de corrélation croisée. Nous avons vu qu'une simple boucle for imbriquée est tout ce qui est nécessaire pour calculer sa valeur. Si nous avons plusieurs canaux d'entrée et plusieurs canaux de sortie, nous effectuons une opération matrice-matrice entre les canaux. Comme on peut le voir, le calcul est simple et, surtout, très *local*. Cela permet une optimisation matérielle significative et de nombreux résultats récents en vision par ordinateur ne sont possibles que grâce à cela. Après tout, cela signifie que les concepteurs de puces peuvent investir dans un calcul rapide plutôt que dans la mémoire, lorsqu'il s'agit d'optimiser les convolutions. Même si cela ne conduit pas nécessairement à des conceptions optimales pour d'autres applications, cela ouvre la voie à une vision par ordinateur omniprésente et abordable. 

En ce qui concerne les convolutions elles-mêmes, elles peuvent être utilisées à de nombreuses fins, par exemple pour détecter les bords et les lignes, pour brouiller les images ou pour les rendre plus nettes. Plus important encore, il n'est pas nécessaire que le statisticien (ou l'ingénieur) invente des filtres appropriés. Au lieu de cela, nous pouvons simplement les *apprendre* à partir des données. Les statistiques fondées sur des preuves remplacent ainsi les heuristiques de l'ingénierie des caractéristiques. Enfin, et c'est tout à fait charmant, ces filtres ne sont pas seulement avantageux pour la construction de réseaux profonds, mais ils correspondent également aux champs réceptifs et aux cartes de caractéristiques du cerveau. Cela nous donne la certitude que nous sommes sur la bonne voie. 

## Exercices

1. Construisez une image `X` avec des bords diagonaux.
   1. Que se passe-t-il si vous lui appliquez le noyau `K` de cette section ?
   1. Que se passe-t-il si vous transposez `X`?
   1. Que se passe-t-il si vous transposez `K`?
1. Concevez quelques noyaux manuellement.
   1. Étant donné un vecteur directionnel $\mathbf{v} = (v_1, v_2)$, dérivez un noyau de détection d'arêtes qui détecte les arêtes de 
orthogonales à $\mathbf{v}$, c'est-à-dire les arêtes dans la direction $(v_2, -v_1)$. 
    1. Déterminez un opérateur de différence finie pour la dérivée seconde. Quelle est la taille minimale 
du noyau convolutif qui lui est associé ? Quelles sont les structures des images qui y répondent le plus fortement ?
   1. Comment concevriez-vous un noyau de flou ? Pourquoi voudriez-vous utiliser un tel noyau ?
   1. Quelle est la taille minimale d'un noyau pour obtenir une dérivée d'ordre $d$?
1. Lorsque vous essayez de trouver automatiquement le gradient pour la classe `Conv2D` que nous avons créée, quel type de message d'erreur voyez-vous ?
1. Comment représenter une opération de corrélation croisée comme une multiplication matricielle en changeant les tenseurs d'entrée et de noyau ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
