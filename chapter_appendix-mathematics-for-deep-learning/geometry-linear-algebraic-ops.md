# Geometry and Linear Algebraic Operations
:label:`sec_geometry-linear-algebraic-ops`

Dans :numref:`sec_linear-algebra`, nous avons rencontré les bases de l'algèbre linéaire.
et vu comment elle pouvait être utilisée pour exprimer des opérations courantes de transformation de nos données.
L'algèbre linéaire est l'un des principaux piliers mathématiques
qui sous-tend une grande partie du travail que nous faisons dans l'apprentissage profond
et, plus largement, dans l'apprentissage automatique.
Alors que :numref:`sec_linear-algebra` contenait suffisamment de machines
pour communiquer les mécanismes des modèles modernes d'apprentissage profond,
le sujet est bien plus vaste.
Dans cette section, nous allons aller plus loin,
en mettant en évidence certaines interprétations géométriques des opérations d'algèbre linéaire,
et en introduisant quelques concepts fondamentaux, notamment les valeurs propres et les vecteurs propres.

## Représentation géométrique des vecteurs
Tout d'abord, nous devons discuter des deux interprétations géométriques courantes des vecteurs,
soit comme des points, soit comme des directions dans l'espace.
Fondamentalement, un vecteur est une liste de nombres, comme la liste Python ci-dessous.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

Les mathématiciens l'écrivent le plus souvent sous la forme d'un vecteur *colonne* ou *ligne*, c'est-à-dire soit sous forme de

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

ou

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

Ceux-ci ont souvent des interprétations différentes,
où les exemples de données sont des vecteurs colonnes
et les pondérations utilisées pour former des sommes pondérées sont des vecteurs de lignes.
Cependant, il peut être bénéfique d'être flexible.
Comme nous l'avons décrit dans :numref:`sec_linear-algebra`,
bien que l'orientation par défaut d'un vecteur simple soit un vecteur colonne,
pour toute matrice représentant un ensemble de données tabulaires,
traiter chaque exemple de données comme un vecteur ligne
dans la matrice
est plus conventionnel.

Étant donné un vecteur, la première interprétation
que l'on doit lui donner est celle d'un point dans l'espace.
En deux ou trois dimensions, nous pouvons visualiser ces points
en utilisant les composantes des vecteurs pour définir
l'emplacement des points dans l'espace par rapport
à une référence fixe appelée *origine*.  Ceci peut être vu dans :numref:`fig_grid`.

![Une illustration de la visualisation des vecteurs comme des points dans le plan.  La première composante du vecteur donne la coordonnée $x$, la deuxième composante donne la coordonnée $y$.  Les dimensions supérieures sont analogues, bien que beaucoup plus difficiles à visualiser.](../img/grid-points.svg)
:label:`fig_grid`

Ce point de vue géométrique nous permet de considérer le problème à un niveau plus abstrait.
Nous ne sommes plus confrontés à un problème apparemment insurmontable.
comme classer des images en tant que chats ou chiens,
nous pouvons commencer à considérer les tâches de manière abstraite
comme des collections de points dans l'espace et en imaginant la tâche
comme découvrir comment séparer deux groupes distincts de points.

En parallèle, il y a un deuxième point de vue
que les gens prennent souvent des vecteurs : comme des directions dans l'espace.
Non seulement on peut penser au vecteur $\mathbf{v} = [3,2]^\top$
comme la position à 3$ unités à droite et à 2$ unités vers le haut de l'origine,
on peut aussi le considérer comme la direction elle-même.
pour faire 3$ pas vers la droite et 2$ pas vers le haut.
De cette façon, nous considérons que tous les vecteurs de la figure :numref:`fig_arrow` sont identiques.

![Tout vecteur peut être visualisé comme une flèche dans le plan.  Dans ce cas, chaque vecteur dessiné est une représentation du vecteur $(3,2)^\top$.](../img/par-vec.svg)
:label:`fig_arrow`

L'un des avantages de ce changement est que
nous pouvons donner un sens visuel à l'acte d'addition de vecteurs.
En particulier, nous suivons les directions données par un vecteur,
puis on suit les directions données par l'autre, comme on le voit dans :numref:`fig_add-vec`.

![We can visualize vector addition by first following one vector, and then another.](../img/vec-add.svg)
:label:`fig_add-vec`

La soustraction vectorielle a une interprétation similaire.
En considérant l'identité que $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$,
nous voyons que le vecteur $\mathbf{u}-\mathbf{v}$ est la direction
qui nous mène du point $\mathbf{v}$ au point $\mathbf{u}$.


## Dot Products and Angles
Comme nous l'avons vu dans :numref:`sec_linear-algebra`,
si l'on prend deux vecteurs colonnes $\mathbf{u}$ et $\mathbf{v}$,
nous pouvons former leur produit scalaire en calculant :

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

Puisque :eqref:`eq_dot_def` est symétrique, nous refléterons la notation
de la multiplication classique et nous écrirons

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

pour souligner le fait qu'en échangeant l'ordre des vecteurs, on obtient la même réponse.

Le produit scalaire :eqref:`eq_dot_def` admet également une interprétation géométrique : il est étroitement lié à l'angle entre deux vecteurs.  Considérons l'angle représenté dans :numref:`fig_angle`.

![Entre deux vecteurs quelconques dans le plan, il existe un angle bien défini $\theta$.  Nous verrons que cet angle est intimement lié au produit scalaire.](../img/vec-angle.svg)
:label:`fig_angle`

Pour commencer, considérons deux vecteurs particuliers :

$$
\mathbf{v} = (r,0) \; \text{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

Le vecteur $\mathbf{v}$ est de longueur $r$ et est parallèle à l'axe $x$,
et le vecteur $\mathbf{w}$ est de longueur $s$ et fait un angle $\theta$ avec l'axe $x$.
Si l'on calcule le produit scalaire de ces deux vecteurs, on constate que

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

Avec quelques manipulations algébriques simples, nous pouvons réarranger les termes pour obtenir

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

En bref, pour ces deux vecteurs spécifiques,
le produit scalaire combiné aux normes nous donne l'angle entre les deux vecteurs. Ce même fait est vrai en général. Cependant, nous ne dériverons pas l'expression ici,
si nous envisageons d'écrire $\|\mathbf{v} - \mathbf{w}\|^2$ de deux façons :
l'une avec le produit scalaire, et l'autre géométriquement en utilisant la loi des cosinus,
nous pouvons obtenir la relation complète.
En effet, pour deux vecteurs quelconques $\mathbf{v}$ et $\mathbf{w}$,
l'angle entre ces deux vecteurs est

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

C'est un bon résultat puisque rien dans le calcul ne fait référence aux deux dimensions.
En effet, nous pouvons l'utiliser en trois ou trois millions de dimensions sans problème.

A titre d'exemple simple, voyons comment calculer l'angle entre une paire de vecteurs :

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

Nous ne l'utiliserons pas pour l'instant, mais il est utile de savoir
que nous ferons référence à des vecteurs pour lesquels l'angle est de $\pi/2$.
(ou équivalent à $90^{\circ}$) comme étant *orthogonaux*.
En examinant l'équation ci-dessus, nous voyons que cela se produit lorsque $\theta = \pi/2$,
ce qui revient à dire que $\cos(\theta) = 0$.
Le seul moyen pour que cela se produise est que le produit scalaire lui-même soit nul,
et deux vecteurs sont orthogonaux si et seulement si $\mathbf{v}\cdot\mathbf{w} = 0$.
Cette formule s'avérera utile pour comprendre les objets sur le plan géométrique.

Il est raisonnable de demander : pourquoi le calcul de l'angle est-il utile ?
La réponse se trouve dans le type d'invariance que nous attendons des données.
Considérons une image, et une image dupliquée,
où chaque valeur de pixel est la même, mais 10 % de la luminosité.
Les valeurs des pixels individuels sont en général très éloignées des valeurs d'origine.
Ainsi, si l'on calcule la distance entre l'image originale et l'image plus sombre,
la distance peut être grande.
Cependant, pour la plupart des applications ML, le *contenu* est le même - il s'agit toujours de
l'image d'un chat pour un classificateur chat/chien.
Cependant, si nous considérons l'angle, il n'est pas difficile de voir
que pour tout vecteur $\mathbf{v}$, l'angle
entre $\mathbf{v}$ et $0.1\cdot\mathbf{v}$ est nul.
Cela correspond au fait que la mise à l'échelle des vecteurs
conserve la même direction et ne modifie que la longueur.
L'angle considère que l'image la plus sombre est identique.

Des exemples comme celui-ci sont partout.
Dans un texte, nous pourrions vouloir que le sujet discuté
ne change pas si nous écrivons un document deux fois plus long qui dit la même chose.
Pour certains codages (comme le comptage du nombre d'occurrences de mots dans un certain vocabulaire), cela correspond à un doublement du vecteur codant le document,
donc encore une fois, on peut utiliser l'angle.

### Similitude Cosinus
Dans les contextes ML où l'angle est employé
pour mesurer la proximité de deux vecteurs,
les praticiens adoptent le terme de *similarité cosinus*.
pour faire référence à la partie
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

Le cosinus prend une valeur maximale de 1$$.
lorsque les deux vecteurs pointent dans la même direction,
une valeur minimale de $1$ lorsqu'ils pointent dans des directions opposées,
et une valeur de $0$ lorsque les deux vecteurs sont orthogonaux.
Notez que si les composantes des vecteurs à haute dimension
sont échantillonnées aléatoirement avec une moyenne de $0$,
leur cosinus sera presque toujours proche de 0 $.


## Hyperplans

Outre le fait de travailler avec des vecteurs, un autre objet clé
que vous devez comprendre pour aller loin en algèbre linéaire
est l'*hyperplan*, une généralisation à des dimensions supérieures
d'une ligne (deux dimensions) ou d'un plan (trois dimensions).
Dans un espace vectoriel de $d$ dimensions, un hyperplan a $d-1$ dimensions
et divise l'espace en deux demi-espaces.

Commençons par un exemple.
Supposons que nous ayons un vecteur colonne $\mathbf{w}=[2,1]^\top$. Nous voulons savoir, "quels sont les points $\mathbf{v}$ avec $\mathbf{w}\cdot\mathbf{v} = 1$ ?"
En rappelant la connexion entre les produits scalaires et les angles ci-dessus :eqref:`eq_angle_forumla`,
nous pouvons voir que ceci est équivalent à
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![En rappelant la trigonométrie, on voit que la formule ${\mathbf{v}{\i1}cos(\theta)} est la longueur de la projection du vecteur ${\mathbf{v}$ sur la direction de ${\mathbf{w}$.](../img/proj-vec.svg)
:label:`fig_vector-project`

Si nous considérons la signification géométrique de cette expression,
nous voyons que cela revient à dire
que la longueur de la projection de ${{mathbf{v}}
sur la direction de $\mathbf{w}$ est exactement $1/\|\mathbf{w}\|$, comme le montre :numref:`fig_vector-project`.
L'ensemble de tous les points où ceci est vrai est une ligne
à angle droit avec le vecteur $\mathbf{w}$.
Si on le souhaite, on peut trouver l'équation de cette droite
et voir qu'elle est $2x + y = 1$ ou encore $y = 1 - 2x$.

Si nous regardons maintenant ce qui se passe lorsque nous posons la question de l'ensemble des points avec
$\mathbf{w}\cdot\mathbf{v} > 1$ ou $\mathbf{w}\cdot\mathbf{v} < 1$,
on peut voir que ce sont des cas où les projections
sont plus longues ou plus courtes que $1/\|\mathbf{w}\|$, respectivement.
Ainsi, ces deux inégalités définissent chaque côté de la ligne.
De cette façon, nous avons trouvé un moyen de couper notre espace en deux moitiés,
où tous les points d'un côté ont un produit scalaire inférieur à un seuil,
et l'autre côté au-dessus comme nous le voyons dans :numref:`fig_space-division`.

!Si l'on considère maintenant la version inégalitaire de l'expression, on voit que notre hyperplan (dans ce cas : juste une ligne) sépare l'espace en deux moitiés](../img/space-division.svg)
:label:`fig_space-division`

L'histoire en dimension supérieure est à peu près la même.
Si l'on prend maintenant $\mathbf{w} = [1,2,3]^\top$
et que nous nous interrogeons sur les points en trois dimensions avec $\mathbf{w}\cdot\mathbf{v} = 1$,
nous obtenons un plan à angle droit par rapport au vecteur donné $\mathbf{w}$.
Les deux inégalités définissent à nouveau les deux côtés du plan comme le montre :numref:`fig_higher-division`.

!Les hyperplans de n'importe quelle dimension séparent l'espace en deux moitiés.](../img/space-division-3d.svg)
:label:`fig_higher-division`

Bien que notre capacité de visualisation s'épuise à ce stade,
rien ne nous empêche de le faire dans des dizaines, des centaines ou des milliards de dimensions.
Cela se produit souvent lorsque l'on pense à des modèles appris par machine.
Par exemple, nous pouvons comprendre les modèles de classification linéaire
comme ceux de :numref:`sec_softmax`,
comme des méthodes permettant de trouver des hyperplans qui séparent les différentes classes cibles.
Dans ce contexte, de tels hyperplans sont souvent appelés *plans de décision*.
La majorité des modèles de classification appris en profondeur se terminent
avec une couche linéaire alimentée par un softmax,
On peut donc interpréter le rôle du réseau neuronal profond
de trouver un encastrement non-linéaire tel que les classes cibles puissent être
puissent être séparées proprement par des hyperplans.

Pour donner un exemple concret, remarquez que nous pouvons produire un modèle raisonnable
pour classer de petites images de t-shirts et de pantalons provenant de l'ensemble de données Fashion-MNIST (:numref:`sec_fashion_mnist`)
en prenant simplement le vecteur entre leurs moyennes pour définir le plan de décision
et fixer à vue un seuil grossier.  Tout d'abord, nous allons charger les données et calculer les moyennes.

```{.python .input}
#@tab mxnet
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

Il peut être instructif d'examiner ces moyennes en détail, alors traçons ce à quoi elles ressemblent.  Dans ce cas, nous voyons que la moyenne ressemble effectivement à l'image floue d'un t-shirt.

```{.python .input}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Dans le deuxième cas, nous constatons à nouveau que la moyenne ressemble à une image floue d'un pantalon.

```{.python .input}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Dans une solution entièrement automatisée, nous apprendrions le seuil à partir de l'ensemble de données.  Dans ce cas, j'ai simplement fixé à la main un seuil qui semblait bon sur les données d'entraînement.

```{.python .input}
#@tab mxnet
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Print test set accuracy with eyeballed threshold
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## Géométrie des transformations linéaires

Grâce à :numref:`sec_linear-algebra` et aux discussions ci-dessus,
nous avons une solide compréhension de la géométrie des vecteurs, des longueurs et des angles.
Cependant, il y a un objet important que nous avons omis de discuter,
et c'est la compréhension géométrique des transformations linéaires représentées par des matrices.  L'intériorisation complète de ce que les matrices peuvent faire pour transformer des données
entre deux espaces de haute dimension potentiellement différents demande une pratique importante,
et dépasse le cadre de cette annexe.
Cependant, nous pouvons commencer à développer une intuition en deux dimensions.

Supposons que nous ayons une certaine matrice :

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

Si nous voulons l'appliquer à un vecteur arbitraire
$\mathbf{v} = [x, y]^\top$ ,
nous multiplions et voyons que

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

Cela peut sembler être un calcul étrange,
où quelque chose de clair est devenu quelque peu impénétrable.
Cependant, il nous indique que nous pouvons écrire la manière dont
une matrice transforme *n'importe quel* vecteur
en fonction de la manière dont elle transforme *deux vecteurs spécifiques* :
$[1,0]^\top$ et $[0,1]^\top$.
Cela vaut la peine de s'y attarder un instant.
Nous avons essentiellement réduit un problème infini
(ce qui arrive à toute paire de nombres réels)
à un problème fini (ce qui arrive à ces vecteurs spécifiques).
Ces vecteurs sont un exemple de *base*,
où nous pouvons écrire tout vecteur dans notre espace
comme une somme pondérée de ces *vecteurs de base*.

Dessinons ce qui se passe lorsque nous utilisons la matrice spécifique

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

Si nous regardons le vecteur spécifique $\mathbf{v} = [2, -1]^\top$,
nous voyons qu'il s'agit de $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$,
et donc nous savons que la matrice $A$ enverra ceci à
$2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
Si nous suivons cette logique avec attention,
par exemple en considérant la grille de toutes les paires de points entiers,
nous constatons que la multiplication matricielle
peut incliner, faire pivoter et mettre à l'échelle la grille,
mais la structure de la grille doit rester telle que vous la voyez sur :numref:`fig_grid-transform` .

![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.](../img/grid-transform.svg)
:label:`fig_grid-transform`

C'est le point intuitif le plus important
à intérioriser à propos des transformations linéaires représentées par des matrices.
Les matrices sont incapables de déformer certaines parties de l'espace différemment des autres.
Tout ce qu'elles peuvent faire, c'est prendre les coordonnées originales de notre espace
et les incliner, les faire pivoter et les mettre à l'échelle.

Certaines distorsions peuvent être graves.  Par exemple, la matrice

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

comprime l'ensemble du plan bidimensionnel en une seule ligne.
L'identification et l'utilisation de telles transformations font l'objet d'une section ultérieure,
mais, d'un point de vue géométrique, nous pouvons constater que ces transformations sont fondamentalement différentes
des types de transformations que nous avons vus précédemment.
Par exemple, le résultat de la matrice $\mathbf{A}$ peut être "replié" sur la grille d'origine.  Les résultats de la matrice $\mathbf{B}$ ne peuvent pas l'être
car nous ne saurons jamais d'où vient le vecteur $[1,2]^\top$ - est-ce que
était $[1,1]^\top$ ou $[0, -1]^\top$?

Bien que cette image concerne une matrice $2\times2$,
rien ne nous empêche de transposer les leçons apprises à des dimensions supérieures.
Si nous prenons des vecteurs de base similaires comme $[1,0, \ldots,0]$
 et que nous voyons où notre matrice les envoie,
nous pouvons commencer à avoir une idée de la façon dont la multiplication matricielle
déforme l'espace entier dans n'importe quel espace de dimension avec lequel nous travaillons.

## Dépendance linéaire

Considérons à nouveau la matrice

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

Cela comprime le plan entier pour qu'il vive sur la seule ligne $y = 2x$.
La question qui se pose maintenant est la suivante : existe-t-il un moyen de détecter ce phénomène
simplement en regardant la matrice elle-même ?
La réponse est qu'en effet, nous le pouvons.
Prenons $\mathbf{b}_1 = [2,4]^\top$ et $\mathbf{b}_2 = [-1, -2]^\top$
 comme les deux colonnes de $\mathbf{B}$.
Rappelez-vous que nous pouvons écrire tout ce qui est transformé par la matrice $\mathbf{B}$
 comme une somme pondérée des colonnes de la matrice :
comme $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
Nous appelons cela une *combinaison linéaire*.
Le fait que $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$
 signifie que nous pouvons écrire toute combinaison linéaire de ces deux colonnes
entièrement en termes de disons $\mathbf{b}_2$ puisque

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

Cela signifie que l'une des colonnes est, en un sens, redondante
car elle ne définit pas une direction unique dans l'espace.
Cela ne devrait pas trop nous surprendre
puisque nous avons déjà vu que cette matrice
réduit le plan entier à une seule ligne.
De plus, nous voyons que la dépendance linéaire
$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ rend compte de cela.
Pour rendre cela plus symétrique entre les deux vecteurs, nous l'écrirons comme suit

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

En général, nous dirons qu'un ensemble de vecteurs
$\mathbf{v}_1, \ldots, \mathbf{v}_k$ sont *linéairement dépendants*
s'il existe des coefficients $a_1, \ldots, a_k$ *non égaux à zéro* de sorte que

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

Dans ce cas, nous pouvons résoudre l'un des vecteurs
en fonction d'une combinaison des autres,
et le rendre effectivement redondant.
Ainsi, une dépendance linéaire dans les colonnes d'une matrice
témoigne du fait que notre matrice
comprime l'espace à une dimension inférieure.
S'il n'y a pas de dépendance linéaire, on dit que les vecteurs sont *linéairement indépendants*.
Si les colonnes d'une matrice sont linéairement indépendantes,
aucune compression ne se produit et l'opération peut être annulée.

## Rang

Si nous avons une matrice générale $n\times m$,
il est raisonnable de se demander dans quel espace de dimension la matrice s'applique.
Un concept connu sous le nom de *rank* sera notre réponse.
Dans la section précédente, nous avons noté qu'une dépendance linéaire
témoigne de la compression de l'espace dans une dimension inférieure
et nous pourrons donc utiliser cela pour définir la notion de rang.
En particulier, le rang d'une matrice $\mathbf{A}$
 est le plus grand nombre de colonnes linéairement indépendantes
parmi tous les sous-ensembles de colonnes. Par exemple, la matrice

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

a $\mathrm{rank}(B)=1$, puisque les deux colonnes sont linéairement dépendantes,
mais aucune des deux colonnes n'est linéairement dépendante.
Pour un exemple plus difficile, nous pouvons considérer

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

et montrer que $\mathbf{C}$ a le rang deux puisque, par exemple,
les deux premières colonnes sont linéairement indépendantes,
mais n'importe laquelle des quatre collections de trois colonnes est dépendante.

Cette procédure, telle qu'elle est décrite, est très inefficace.
Elle nécessite de regarder chaque sous-ensemble de colonnes de notre matrice donnée,
et est donc potentiellement exponentielle en nombre de colonnes.
Nous verrons plus tard une méthode plus efficace
pour calculer le rang d'une matrice, mais pour l'instant,
ceci est suffisant pour voir que le concept
est bien défini et comprendre sa signification.

## Invertibilité

Nous avons vu ci-dessus que la multiplication par une matrice avec des colonnes linéairement dépendantes
ne peut pas être annulée, c'est-à-dire qu'il n'y a pas d'opération inverse qui puisse toujours récupérer l'entrée.  Cependant, la multiplication par une matrice de rang complet
(c'est-à-dire une certaine $\mathbf{A}$ qui est $n \times n$ matrice de rang $n$),
devrait toujours pouvoir être annulée.  Considérons la matrice

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

qui est la matrice avec des uns le long de la diagonale et des zéros ailleurs.
Nous l'appelons la matrice *identité*.
C'est la matrice qui laisse nos données inchangées lorsqu'elle est appliquée.
Pour trouver une matrice qui défait ce que notre matrice $\mathbf{A}$ a fait,
nous voulons trouver une matrice $\mathbf{A}^{-1}$ telle que

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

Si nous considérons ceci comme un système, nous avons $n \times n$ inconnues
(les entrées de $\mathbf{A}^{-1}$) et $n \times n$ équations
(l'égalité qui doit exister entre chaque entrée du produit $\mathbf{A}^{-1}\mathbf{A}$ et chaque entrée de $\mathbf{I}$)
donc nous devrions généralement nous attendre à ce qu'une solution existe.
En effet, dans la section suivante, nous verrons une quantité appelée le *déterminant*,
qui a la propriété que tant que le déterminant n'est pas nul, nous pouvons trouver une solution.  Nous appelons une telle matrice $\mathbf{A}^{-1}$ la matrice *inverse*.
A titre d'exemple, si $\mathbf{A}$ est la matrice générale $2 \times 2$

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

alors nous pouvons voir que l'inverse est

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

Nous pouvons tester cela en voyant que la multiplication de
par l'inverse donné par la formule ci-dessus fonctionne en pratique.

```{.python .input}
#@tab mxnet
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### Problèmes numériques
Bien que l'inverse d'une matrice soit utile en théorie,
nous devons dire que la plupart du temps nous ne souhaitons pas
*utiliser* l'inverse de la matrice pour résoudre un problème en pratique.
En général, il existe des algorithmes beaucoup plus stables numériquement
pour résoudre des équations linéaires telles que

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

que de calculer l'inverse et de multiplier pour obtenir

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

De même que la division par un petit nombre peut conduire à une instabilité numérique,
peut également conduire à l'inversion d'une matrice qui est proche d'avoir un rang bas.

De plus, il est courant que la matrice $\mathbf{A}$ soit *sparse*,
c'est-à-dire qu'elle ne contienne qu'un petit nombre de valeurs non nulles.
Si nous devions explorer des exemples, nous verrions
que cela ne signifie pas que l'inverse est clairsemé.
Même si $\mathbf{A}$ était une matrice $1$ million par $1$ million
avec seulement $5$ million d'entrées non nulles
(et que nous n'avions donc besoin de stocker que ces $5$ millions),
l'inverse aura typiquement presque chaque entrée non négative,
nous obligeant à stocker toutes les $1\text{M}^2$ entrées--soit $1$ trillion d'entrées !

Bien que nous n'ayons pas le temps de nous plonger dans les problèmes numériques épineux
que l'on rencontre fréquemment lorsqu'on travaille avec l'algèbre linéaire,
nous voulons vous donner une idée de quand il faut procéder avec prudence,
et en général, éviter l'inversion en pratique est une bonne règle empirique.

## Déterminant
La vision géométrique de l'algèbre linéaire donne une manière intuitive
d'interpréter une quantité fondamentale connue sous le nom de *déterminant*.
Reprenez l'image de la grille précédente, mais avec une région en surbrillance (:numref:`fig_grid-filled` ).

![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

Regardez le carré en surbrillance.  Il s'agit d'un carré dont les bords sont donnés à
par $(0, 1)$ et $(1, 0)$ et qui a donc une aire de 1.
Après que $\mathbf{A}$ ait transformé ce carré,
nous voyons qu'il devient un parallélogramme.
Il n'y a aucune raison pour que ce parallélogramme ait la même aire
que celle de départ, et en effet, dans le cas spécifique montré ici de

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

c'est un exercice de géométrie des coordonnées pour calculer
l'aire de ce parallélogramme et obtenir que l'aire est $5$.

En général, si nous avons une matrice

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

nous pouvons voir, après quelques calculs, que l'aire
du parallélogramme résultant est $ad-bc$.
Cette aire est appelée le *déterminant*.

Vérifions cela rapidement à l'aide d'un exemple de code.

```{.python .input}
#@tab mxnet
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

Les plus attentifs d'entre nous remarqueront
que cette expression peut être nulle ou même négative.
Pour le terme négatif, il s'agit d'une question de convention
prise généralement en mathématiques :
si la matrice renverse la figure,
nous disons que l'aire est niée.
Voyons maintenant que lorsque le déterminant est nul, nous en apprenons davantage.

Considérons

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

Si nous calculons le déterminant de cette matrice,
nous obtenons $2\cdot(-2 ) - 4\cdot(-1) = 0$.
Étant donné ce que nous avons compris ci-dessus, cela a du sens.
$\mathbf{B}$ compresse le carré de l'image originale
en un segment de ligne, qui a une surface nulle.
Et en effet, être compressé dans un espace de dimension inférieure
est la seule façon d'avoir une surface nulle après la transformation.
Nous voyons donc que le résultat suivant est vrai :
une matrice $A$ est inversible si et seulement si
le déterminant n'est pas égal à zéro.

Pour terminer, imaginons que nous ayons une figure quelconque dessinée sur le plan.
En pensant comme des informaticiens, nous pouvons décomposer
cette figure en une collection de petits carrés
de sorte que l'aire de la figure est essentiellement
juste le nombre de carrés dans la décomposition.
Si nous transformons maintenant cette figure par une matrice,
nous transformons chacun de ces carrés en parallélogrammes,
dont l'aire est donnée par le déterminant.
Nous voyons que pour toute figure, le déterminant donne le nombre (signé)
qu'une matrice met à l'échelle l'aire de toute figure.

Le calcul des déterminants pour des matrices plus grandes peut être laborieux,
mais l'intuition est la même.
Le déterminant reste le facteur
que $n\times n$ les matrices mettent à l'échelle $n$-dimensions.

## Tenseurs et opérations courantes d'algèbre linéaire

Dans :numref:`sec_linear-algebra` , nous avons introduit le concept de tenseurs.
Dans cette section, nous allons nous plonger plus profondément dans les contractions tensorielles
(l'équivalent tensoriel de la multiplication matricielle),
et voir comment elles peuvent fournir une vue unifiée
sur un certain nombre d'opérations matricielles et vectorielles.

Avec les matrices et les vecteurs, nous savions comment les multiplier pour transformer les données.
Nous devons avoir une définition similaire pour les tenseurs si nous voulons qu'ils nous soient utiles.
Pensez à la multiplication matricielle :

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

ou de manière équivalente

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$ 

 Nous pouvons répéter ce schéma pour les tenseurs.
Pour les tenseurs, il n'y a pas de cas unique sur lequel
additionner qui puisse être choisi universellement,
. Nous devons donc spécifier exactement les indices sur lesquels nous voulons additionner.
Par exemple, nous pourrions considérer

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

Une telle transformation est appelée une *contraction de tenseurs*.
Elle peut représenter une famille de transformations bien plus flexible
que la seule multiplication matricielle.

Pour simplifier la notation,
nous pouvons remarquer que la somme porte exactement sur les indices
qui apparaissent plus d'une fois dans l'expression,
. Les gens travaillent donc souvent avec la *notation d'Einstein*,
où la somme porte implicitement sur tous les indices répétés.
On obtient ainsi l'expression compacte :

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Exemples courants d'algèbre linéaire

Voyons combien de définitions d'algèbre linéaire
que nous avons vues auparavant peuvent être exprimées dans cette notation tensorielle compacte :

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
 * $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
 * $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
 * $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
 * $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

 De cette façon, nous pouvons remplacer une myriade de notations spécialisées par des expressions tensorielles courtes.

### Expression dans le code
Les tenseurs peuvent également être utilisés de manière flexible dans le code.
Comme nous l'avons vu dans :numref:`sec_linear-algebra` ,
nous pouvons créer des tenseurs comme indiqué ci-dessous.

```{.python .input}
#@tab mxnet
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

La sommation d'Einstein a été implémentée directement.
Les indices qui se produisent dans la sommation d'Einstein peuvent être passés comme une chaîne,
suivie des tenseurs sur lesquels on agit.
Par exemple, pour implémenter la multiplication de matrices,
nous pouvons considérer la sommation d'Einstein vue ci-dessus
($\mathbf{A}\mathbf{v} = a_{ij}v_j$)
et supprimer les indices eux-mêmes pour obtenir l'implémentation :

```{.python .input}
#@tab mxnet
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

Il s'agit d'une notation très flexible.
Par exemple, si nous voulons calculer
, ce qui s'écrit traditionnellement comme suit

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

il peut être implémenté via la sommation d'Einstein comme :

```{.python .input}
#@tab mxnet
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

Cette notation est lisible et efficace pour les humains,
mais encombrante si, pour une raison quelconque,
nous devons générer une contraction tensorielle par programme.
Pour cette raison, `einsum` propose une notation alternative
en fournissant des indices entiers pour chaque tenseur.
Par exemple, la même contraction tensorielle peut aussi être écrite comme suit :

```{.python .input}
#@tab mxnet
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't support this type of notation.
```

```{.python .input}
#@tab tensorflow
# TensorFlow doesn't support this type of notation.
```

L'une ou l'autre notation permet une représentation concise et efficace des contractions tensorielles dans le code.

## Résumé
* Les vecteurs peuvent être interprétés géométriquement comme des points ou des directions dans l'espace.
* Les produits scalaires définissent la notion d'angle dans des espaces de haute dimension.
* Les hyperplans sont des généralisations à haute dimension des lignes et des plans.  Ils peuvent être utilisés pour définir des plans de décision qui sont souvent utilisés comme dernière étape d'une tâche de classification.
* La multiplication matricielle peut être interprétée géométriquement comme une déformation uniforme des coordonnées sous-jacentes. Elles représentent un moyen très restreint, mais mathématiquement propre, de transformer des vecteurs.
* La dépendance linéaire est un moyen de savoir si une collection de vecteurs se trouve dans un espace de dimension inférieure à celle à laquelle on pourrait s'attendre (disons que vous avez $3$ vecteurs vivant dans un espace à $2$-dimensions). Le rang d'une matrice est la taille du plus grand sous-ensemble de ses colonnes qui sont linéairement indépendantes.
* Lorsque l'inverse d'une matrice est défini, l'inversion de matrice nous permet de trouver une autre matrice qui annule l'action de la première. L'inversion matricielle est utile en théorie, mais nécessite de la prudence en pratique en raison de l'instabilité numérique.
* Les déterminants nous permettent de mesurer à quel point une matrice étend ou contracte un espace. Un déterminant non nul implique une matrice inversible (non singulière) et un déterminant nul signifie que la matrice est non inversible (singulière).
* Les contractions tensorielles et la sommation d'Einstein fournissent une notation propre et nette pour exprimer un grand nombre des calculs que l'on voit dans l'apprentissage automatique.

## Exercices
1. Quel est l'angle entre
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. Vrai ou faux : $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ et $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ sont inverses l'un de l'autre ?
3. Supposons que nous dessinions une figure dans le plan dont l'aire est $100\mathrm{m}^2$.  Quelle est l'aire après transformation de la figure par la matrice
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Lesquels des ensembles de vecteurs suivants sont linéairement indépendants ?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
 5. Supposons que vous ayez une matrice écrite sous la forme $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ pour un certain choix de valeurs $a, b, c$, et $d$.  Vrai ou faux : le déterminant d'une telle matrice est toujours $0$?
6. Les vecteurs $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ et $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ sont orthogonaux.  Quelle est la condition sur une matrice $A$ pour que $Ae_1$ et $Ae_2$ soient orthogonaux ?
7. Comment pouvez-vous écrire $\mathrm{tr}(\mathbf{A}^4)$ en notation d'Einstein pour une matrice arbitraire $A$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab:
