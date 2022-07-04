```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Linear Algebra
:label:`sec_linear-algebra` 

Nous savons maintenant charger des ensembles de données dans des tenseurs
et manipuler ces tenseurs 
avec des opérations mathématiques de base.
Pour commencer à construire des modèles sophistiqués,
nous aurons également besoin de quelques outils d'algèbre linéaire. 
Cette section offre une introduction douce 
aux concepts les plus essentiels,
en commençant par l'arithmétique scalaire
et en progressant vers la multiplication matricielle.



## Scalars


La plupart des mathématiques quotidiennes
consistent à manipuler 
des nombres, un par un.
Formellement, nous appelons ces valeurs des *scalaires*.
Par exemple, la température à Palo Alto 
est un doux $72$ degrés Fahrenheit.
Si vous vouliez convertir la température en degrés Celsius,
vous évalueriez l'expression 
$c = \frac{5}{9}(f - 32)$ , en fixant $f$ à $72$.
Dans cette équation, les valeurs 
$5$ , $9$ et $32$ sont des scalaires.
Les variables $c$ et $f$ 
représentent des scalaires inconnus.

Nous désignons les scalaires
par les lettres minuscules ordinaires 
(par exemple, $x$, $y$, et $z$)
et l'espace de tous les scalaires (continus) 
*à valeur réelle* par $\mathbb{R}$.
Pour des raisons de commodité, nous passerons sur les définitions rigoureuses des *espaces* de
.
Retenez simplement que l'expression $x \in \mathbb{R}$
est une façon formelle de dire que $x$ est un scalaire à valeurs réelles.
Le symbole $\in$ (prononcé "in")
dénote l'appartenance à un ensemble.
Par exemple, $x, y \in \{0, 1\}$
indique que $x$ et $y$ sont des variables
qui ne peuvent prendre que les valeurs $0$ ou $1$.

(**Les scalaires sont implémentés comme des tenseurs qui ne contiennent qu'un seul élément.**)
Ci-dessous, nous affectons deux scalaires
et effectuons les opérations familières d'addition, de multiplication, de division
et d'exponentiation.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## Vecteurs

Pour nos besoins, [**vous pouvez considérer les vecteurs comme des tableaux de scalaires de longueur fixe.**]
Comme pour leurs équivalents en code,
nous appelons ces valeurs les *éléments* du vecteur
(les synonymes incluent les *entrées* et les *composants*).
Lorsque les vecteurs représentent des exemples issus d'ensembles de données du monde réel,
leurs valeurs ont une certaine importance dans le monde réel.
Par exemple, si nous formons un modèle pour prédire
le risque de défaut de remboursement d'un prêt,
nous pourrions associer à chaque demandeur un vecteur
dont les composantes correspondent à des quantités
comme leur revenu, la durée de leur emploi, 
ou le nombre de défauts de remboursement précédents.
Si nous étudions le risque de crise cardiaque,
chaque vecteur pourrait représenter un patient
et ses composantes pourraient correspondre à
leurs derniers signes vitaux, leur taux de cholestérol, 
les minutes d'exercice par jour, etc.
Nous désignons les vecteurs par des lettres minuscules en gras, 
(par exemple, $\mathbf{x}$, $\mathbf{y}$, et $\mathbf{z}$).

Les vecteurs sont implémentés en tant que tenseurs d'ordre $1^{\mathrm{st}}$.
En général, ces tenseurs peuvent avoir des longueurs arbitraires,
sous réserve des limitations de mémoire. Attention : en Python, comme dans la plupart des langages de programmation, les indices des vecteurs commencent à $0$, également connu sous le nom d'indexation *à base zéro*, alors qu'en algèbre linéaire, les indices commencent à $1$ (indexation à base un).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

Nous pouvons nous référer à un élément d'un vecteur en utilisant un indice.
Par exemple, $x_2$ désigne le deuxième élément de $\mathbf{x}$. 
Comme $x_2$ est un scalaire, nous ne le mettons pas en gras.
Par défaut, nous visualisons les vecteurs 
en empilant leurs éléments verticalement.

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

Ici, $x_1, \ldots, x_n$ sont des éléments du vecteur.
Plus tard, nous ferons la distinction entre de tels *vecteurs colonnes*
et des *vecteurs lignes* dont les éléments sont empilés horizontalement.
Rappelons que [**on accède aux éléments d'un tenseur par indexation.**]

```{.python .input}
%%tab mxnet
x[2]
```

```{.python .input}
%%tab pytorch
x[2]
```

```{.python .input}
%%tab tensorflow
x[2]
```

Pour indiquer qu'un vecteur contient $n$ éléments,
nous écrivons $\mathbf{x} \in \mathbb{R}^n$.
Formellement, nous appelons $n$ la *dimensionnalité* du vecteur.
[**En code, cela correspond à la longueur du tenseur**],
accessible via la fonction intégrée de Python `len`.

```{.python .input}
%%tab mxnet
len(x)
```

```{.python .input}
%%tab pytorch
len(x)
```

```{.python .input}
%%tab tensorflow
len(x)
```

Nous pouvons également accéder à la longueur via l'attribut `shape`.
La forme est un tuple qui indique la longueur d'un tenseur le long de chaque axe.
(**Les tenseurs avec un seul axe ont des formes avec un seul élément.**)

```{.python .input}
%%tab mxnet
x.shape
```

```{.python .input}
%%tab pytorch
x.shape
```

```{.python .input}
%%tab tensorflow
x.shape
```

Souvent, le mot "dimension" est surchargé
pour signifier à la fois le nombre d'axes 
et la longueur le long d'un axe articulaire.
Pour éviter cette confusion, 
nous utilisons *ordre* pour faire référence au nombre d'axes
et *dimensionnalité* exclusivement pour faire référence 
au nombre de composants.


## Matrices

Tout comme les scalaires sont des tenseurs d'ordre $0^{\mathrm{th}}$
et les vecteurs des tenseurs d'ordre $1^{\mathrm{st}}$, les matrices
sont des tenseurs d'ordre $2^{\mathrm{nd}}$.
Nous désignons les matrices par des lettres majuscules en gras
(par exemple, $\mathbf{X}$, $\mathbf{Y}$, et $\mathbf{Z}$),
et les représentons en code par des tenseurs à deux axes.
L'expression $\mathbf{A} \in \mathbb{R}^{m \times n}$
indique qu'une matrice $\mathbf{A}$ 
contient $m \times n$ des scalaires à valeurs réelles,
disposés en $m$ lignes et $n$ colonnes.
Lorsque $m = n$, on dit qu'une matrice est *carrée*.
Visuellement, nous pouvons illustrer toute matrice comme un tableau.
Pour faire référence à un élément individuel,
nous mettons en indice à la fois la ligne et la colonne, par exemple,
$a_{ij}$ est la valeur qui appartient à la ligne
$i^{\mathrm{th}}$ et à la colonne $j^{\mathrm{th}}$ de $\mathbf{A}$:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$ 
:eqlabel:`eq_matrix_def` 

 
En code, nous représentons une matrice $\mathbf{A} \in \mathbb{R}^{m \times n}$
par un tenseur d'ordre $2^{\mathrm{nd}}$ de forme ($m$, $n$).
[**Nous pouvons convertir tout tenseur $m \times n$ de taille appropriée en une matrice $m \times n$**] 
en passant la forme souhaitée à `reshape`:

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

Parfois, nous voulons inverser les axes.
Lorsque nous échangeons les lignes et les colonnes d'une matrice,
le résultat est appelé sa *transposée*.
Formellement, nous désignons la transposition de la matrice $\mathbf{A}$ 
par $\mathbf{A}^\top$ et si $\mathbf{B} = \mathbf{A}^\top$, 
alors $b_{ij} = a_{ji}$ pour tous $i$ et $j$.
Ainsi, la transposée d'une matrice $m \times n$ 
est une matrice $n \times m$:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

En code, nous pouvons accéder à la **transposée** d'une matrice quelconque (*A*) comme suit :

```{.python .input}
%%tab mxnet
A.T
```

```{.python .input}
%%tab pytorch
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**Les matrices symétriques sont le sous-ensemble des matrices carrées qui sont égales à leurs propres transposées : $\mathbf{A} = \mathbf{A}^\top$ .**]
La matrice suivante est symétrique :

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

Les matrices sont utiles pour représenter des ensembles de données. 
Généralement, les lignes correspondent à des enregistrements individuels
et les colonnes correspondent à des attributs distincts.



## Tenseurs

Bien que vous puissiez aller loin dans votre apprentissage automatique
avec seulement des scalaires, des vecteurs et des matrices,
vous aurez peut-être besoin de travailler avec 
des [**tenseurs**] d'ordre supérieur.
Les tenseurs nous donnent un moyen générique de décrire 
les extensions des tableaux d'ordre $n^{\mathrm{th}}$.
Nous appelons les objets logiciels de la classe des *tenseurs* "tenseurs"
précisément parce qu'ils peuvent eux aussi avoir un nombre arbitraire d'axes.
Bien qu'il puisse être déroutant d'utiliser le mot
*tenseur* à la fois pour l'objet mathématique
et sa réalisation en code,
notre signification devrait généralement être claire à partir du contexte.
Nous désignons les tenseurs généraux par des lettres majuscules 
avec une police de caractères spéciale
(par exemple, $\mathsf{X}$, $\mathsf{Y}$ et $\mathsf{Z}$)
et leur mécanisme d'indexation 
(par exemple, $x_{ijk}$ et $[\mathsf{X}]_{1, 2i-1, 3}$) 
suit naturellement celui des matrices.

Les tenseurs deviendront plus importants 
lorsque nous commencerons à travailler avec des images.
Chaque image se présente comme un tenseur d'ordre $3^{\mathrm{rd}}$
dont les axes correspondent à la hauteur, à la largeur et au *canal*.
À chaque emplacement spatial, les intensités 
de chaque couleur (rouge, vert et bleu)
sont empilées le long du canal. 
De plus, une collection d'images est représentée 
en code par un tenseur d'ordre $4^{\mathrm{th}}$,
où les images distinctes sont indexées
le long du premier axe.
Les tenseurs d'ordre supérieur sont construits de manière analogue 
aux vecteurs et aux matrices,
en augmentant le nombre de composantes de forme.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

## Propriétés de base de l'arithmétique des tenseurs

Les scalaires, les vecteurs, les matrices, 
et les tenseurs d'ordre supérieur
ont tous des propriétés pratiques. 
Par exemple, les opérations par éléments
produisent des sorties qui ont 
la même forme que leurs opérandes.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

Le [**produit par éléments de deux matrices est appelé leur produit de Hadamard**] (noté $\odot$).
Ci-dessous, nous exposons les entrées 
du produit de Hadamard de deux matrices 
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ :



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab mxnet
A * B
```

```{.python .input}
%%tab pytorch
A * B
```

```{.python .input}
%%tab tensorflow
A * B
```

[**L'addition ou la multiplication d'un scalaire et d'un tenseur**] produit un résultat
ayant la même forme que le tenseur original.
Ici, chaque élément du tenseur est ajouté à (ou multiplié par) le scalaire.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## Réduction
:label:`subsec_lin-alg-reduction` 

Souvent, nous souhaitons calculer [**la somme des éléments d'un tenseur.**]
Pour exprimer la somme des éléments d'un vecteur $\mathbf{x}$ de longueur $n$,
nous écrivons $\sum_{i=1}^n x_i$. Il existe une fonction simple pour cela :

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

Pour exprimer [**sommes sur les éléments de tenseurs de forme arbitraire**],
nous faisons simplement la somme sur tous ses axes. 
Par exemple, la somme des éléments 
d'une matrice $m \times n$ $\mathbf{A}$ 
peut s'écrire $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
%%tab mxnet
A.shape, A.sum()
```

```{.python .input}
%%tab pytorch
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

Par défaut, l'invocation de la fonction somme
*réduit* un tenseur sur tous ses axes,
produisant finalement un scalaire.
Nos bibliothèques nous permettent également de [**spécifier les axes le long desquels le tenseur doit être réduit.**]
Pour faire la somme de tous les éléments le long des lignes (axe 0),
nous spécifions `axis=0` dans `sum`.
Puisque la matrice d'entrée se réduit le long de l'axe 0
pour générer le vecteur de sortie,
cet axe est absent de la forme de la sortie.

```{.python .input}
%%tab mxnet
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab pytorch
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

La spécification de `axis=1` réduira la dimension des colonnes (axe 1) en additionnant les éléments de toutes les colonnes.

```{.python .input}
%%tab mxnet
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab pytorch
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Réduire une matrice à la fois le long des lignes et des colonnes via la sommation
est équivalent à additionner tous les éléments de la matrice.

```{.python .input}
%%tab mxnet
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`
```

```{.python .input}
%%tab pytorch
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A) # Same as `tf.reduce_sum(A)`
```

[**Une quantité apparentée est la *moyenne*, également appelée *average*.**]
Nous calculons la moyenne en divisant la somme 
par le nombre total d'éléments.
Comme le calcul de la moyenne est très courant,
il bénéficie d'une fonction de bibliothèque dédiée 
qui fonctionne de manière analogue à `sum`.

```{.python .input}
%%tab mxnet
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

De même, la fonction de calcul de la moyenne 
peut également réduire un tenseur selon des axes spécifiques.

```{.python .input}
%%tab mxnet
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Somme sans réduction
:label:`subsec_lin-alg-non-reduction` 

Parfois, il peut être utile de [**garder le nombre d'axes inchangé**]
lorsque l'on invoque la fonction de calcul de la somme ou de la moyenne. 
C'est important lorsque l'on veut utiliser le mécanisme de diffusion.

```{.python .input}
%%tab mxnet
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Par exemple, puisque `sum_A` conserve ses deux axes après avoir additionné chaque ligne,
nous pouvons (**diviser `A` par `sum_A` avec diffusion**) 
pour créer une matrice dont la somme de chaque ligne est égale à $1$.

```{.python .input}
%%tab mxnet
A / sum_A
```

```{.python .input}
%%tab pytorch
A / sum_A
```

```{.python .input}
%%tab tensorflow
A / sum_A
```

Si nous voulons calculer [**la somme cumulative des éléments de `A` le long d'un certain axe**],
disons `axis=0` (ligne par ligne), nous pouvons appeler la fonction `cumsum`.
Par conception, cette fonction ne réduit pas le tenseur d'entrée le long d'un axe quelconque.

```{.python .input}
%%tab mxnet
A.cumsum(axis=0)
```

```{.python .input}
%%tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Produits scalaires

Jusqu'à présent, nous n'avons effectué que des opérations par éléments, des sommes et des moyennes. 
Et si c'était tout ce que nous pouvions faire, l'algèbre linéaire 
ne mériterait pas sa propre section.
Heureusement, c'est là que les choses deviennent plus intéressantes.
L'une des opérations les plus fondamentales est le produit scalaire.
Étant donné deux vecteurs $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
leur *produit scalaire* $\mathbf{x}^\top \mathbf{y}$ (ou $\langle \mathbf{x}, \mathbf{y}  \rangle$) 
est une somme sur les produits des éléments à la même position : 
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$ .

Le *produit scalaire* de deux vecteurs est une somme des produits des éléments à la même position

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

De manière équivalente, (**on peut calculer le produit scalaire de deux vecteurs en effectuant une multiplication par élément suivie d'une somme :**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

Les produits scalaires sont utiles dans un grand nombre de contextes.
Par exemple, pour un ensemble de valeurs,
désigné par un vecteur $\mathbf{x}  \in \mathbb{R}^n$
et un ensemble de poids désignés par $\mathbf{w} \in \mathbb{R}^n$,
la somme pondérée des valeurs de $\mathbf{x}$
en fonction des poids $\mathbf{w}$
peut être exprimée par le produit scalaire $\mathbf{x}^\top \mathbf{w}$.
Lorsque les poids sont non négatifs
et que leur somme est égale à un, c'est-à-dire $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
le produit scalaire exprime une *moyenne pondérée*.
Après avoir normalisé deux vecteurs pour qu'ils aient une longueur unitaire,
les produits scalaires expriment le cosinus de l'angle entre eux.
Plus loin dans cette section, nous introduirons formellement cette notion de *longueur*.


## Produits matrice-vecteur

Maintenant que nous savons comment calculer les produits scalaires,
nous pouvons commencer à comprendre le *produit*
entre une matrice $m \times n$ $\mathbf{A}$ 
et un vecteur $n$-dimensionnel $\mathbf{x}$.
Pour commencer, nous visualisons notre matrice
en fonction de ses vecteurs de lignes

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

où chaque $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
est un vecteur ligne représentant la ligne $i^\mathrm{th}$ 
de la matrice $\mathbf{A}$.

[**Le produit matrice-vecteur $\mathbf{A}\mathbf{x}$ est simplement un vecteur colonne de longueur $m$, dont l'élément $i^\mathrm{th}$ est le produit scalaire $\mathbf{a}^\top_i \mathbf{x}$ :**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Nous pouvons considérer la multiplication avec une matrice
$\mathbf{A}\in \mathbb{R}^{m \times n}$ 
comme une transformation qui projette les vecteurs
de $\mathbb{R}^{n}$ à $\mathbb{R}^{m}$.
Ces transformations sont remarquablement utiles.
Par exemple, nous pouvons représenter les rotations
comme des multiplications par certaines matrices carrées.
Les produits matrice-vecteur décrivent également 
le calcul clé impliqué dans le calcul
des sorties de chaque couche d'un réseau neuronal
compte tenu des sorties de la couche précédente.

:begin_tab:`mxnet`
Pour exprimer un produit matrice-vecteur en code,
nous utilisons la même fonction `dot`.
L'opération est déduite 
sur la base du type des arguments.
Notez que la dimension de la colonne de `A` 
 (sa longueur selon l'axe 1)
doit être la même que la dimension de `x` (sa longueur).
:end_tab:

:begin_tab:`pytorch`
Pour exprimer un produit matrice-vecteur en code,
nous utilisons la fonction `mv`. 
Notez que la dimension de la colonne de `A` 
 (sa longueur le long de l'axe 1)
doit être la même que la dimension de `x` (sa longueur). 
PyTorch possède un opérateur de commodité `@` 
qui peut exécuter à la fois des produits matrice-vecteur
et matrice-matrice
(en fonction de ses arguments). 
Ainsi, nous pouvons écrire `A@x`.
:end_tab:

:begin_tab:`tensorflow`
Pour exprimer un produit matrice-vecteur en code,
nous utilisons la fonction `matvec`. 
Notez que la dimension de la colonne de `A` 
 (sa longueur le long de l'axe 1)
doit être la même que la dimension de `x` (sa longueur).
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## Multiplication matrice-matrice

Si vous avez appris à utiliser les produits scalaires et les produits matrice-vecteur,
la multiplication *matrice-matrice* devrait être simple.

Disons que nous avons deux matrices 
$\mathbf{A} \in \mathbb{R}^{n \times k}$ 
et $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1k} \\
a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Désignons par $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ 
le vecteur ligne représentant la ligne $i^\mathrm{th}$ 
de la matrice $\mathbf{A}$
et par $\mathbf{b}_{j} \in \mathbb{R}^k$ 
le vecteur colonne de la colonne $j^\mathrm{th}$ 
de la matrice $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


Pour former le produit matriciel $\mathbf{C} \in \mathbb{R}^{n \times m}$,
nous calculons simplement chaque élément $c_{ij}$
comme le produit scalaire entre 
la ligne $i^{\mathrm{th}}$ de $\mathbf{A}$
et la ligne $j^{\mathrm{th}}$ de $\mathbf{B}$,
c'est-à-dire $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

 
[**On peut considérer que la multiplication matrice-matrice $\mathbf{AB}$ consiste à effectuer des produits matrice-vecteur $m$ ou des produits scalaires $m \times n$  et à assembler les résultats  pour former une matrice $n \times m$.**]
Dans l'extrait suivant, 
nous effectuons une multiplication matricielle sur `A` et `B`.
Ici, `A` est une matrice à 2 lignes et 3 colonnes,
et `B` est une matrice à 3 lignes et 4 colonnes.
Après la multiplication, nous obtenons une matrice de 2 lignes et 4 colonnes.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

Le terme *multiplication matricielle* est 
souvent simplifié en *multiplication matricielle*,
et ne doit pas être confondu avec le produit Hadamard.


## Normes
:label:`subsec_lin-algebra-norms` 

Certains des opérateurs les plus utiles en algèbre linéaire sont les *normes*.
De manière informelle, la norme d'un vecteur nous indique sa *taille*. 
Par exemple, la norme $\\ell_2$ mesure
la longueur (euclidienne) d'un vecteur.
Ici, nous employons une notion de *taille* qui concerne la magnitude des composantes d'un vecteur
(et non sa dimensionnalité). 

Une norme est une fonction $\| \cdot \|$ qui fait correspondre un vecteur
à un scalaire et qui satisfait aux trois propriétés suivantes :

1. Étant donné un vecteur quelconque $\mathbf{x}$, si nous mettons à l'échelle (tous les éléments) du vecteur 
par un scalaire $\alpha \in \mathbb{R}$, sa norme est mise à l'échelle en conséquence :
$$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Pour tout vecteur $\mathbf{x}$ et $\mathbf{y}$:
les normes satisfont l'inégalité triangulaire :
$$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$ 
 3. La norme d'un vecteur est non négative et ne disparaît que si le vecteur est nul :
$$\|\mathbf{x}\| > 0 \text{ for all } \mathbf{x} \neq 0.$$

De nombreuses fonctions sont des normes valides et différentes normes 
codent différentes notions de taille. 
La norme euclidienne que nous avons tous apprise en géométrie à l'école primaire
en calculant l'hypoténuse d'un triangle rectangle
est la racine carrée de la somme des carrés des éléments d'un vecteur.
Formellement, cela s'appelle [**la $\ell_2$ *norme***] et s'exprime comme suit :

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

La méthode `norm` calcule la norme $\ell_2$.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

La norme $\ell_1$ est également populaire 
et la métrique associée est appelée la distance de Manhattan. 
Par définition, la norme $\ell_1$ additionne 
les valeurs absolues des éléments d'un vecteur :

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

Par rapport à la norme $\ell_2$, elle est moins sensible aux valeurs aberrantes.
Pour calculer la norme $\ell_1$, 
on compose la valeur absolue
avec l'opération de somme.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

Les normes $\ell_2$ et $\ell_1$ sont toutes deux des cas particuliers
des normes plus générales $\ell_p$ *norms* :

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$ 

Dans le cas des matrices, les choses sont plus compliquées. 
Après tout, les matrices peuvent être considérées à la fois comme des collections d'entrées individuelles 
*et* comme des objets qui opèrent sur des vecteurs et les transforment en d'autres vecteurs. 
Par exemple, nous pouvons nous demander de combien de temps 
le produit matrice-vecteur $\mathbf{X} \mathbf{v}$ 
pourrait être par rapport à $\mathbf{v}$. 
Ce raisonnement conduit à une norme appelée norme *spectrale*. 
Pour l'instant, nous introduisons [**la *norme de Frobenius*, qui est beaucoup plus facile à calculer**] et définie comme
la racine carrée de la somme des carrés 
des éléments d'une matrice :

[**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

La norme de Frobenius se comporte comme s'il s'agissait 
d'une norme $\ell_2$ d'un vecteur matriciel.
L'invocation de la fonction suivante permet de calculer 
la norme de Frobenius d'une matrice.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

Bien que nous ne voulions pas trop nous avancer,
nous pouvons déjà planter quelques intuitions sur l'utilité de ces concepts.
En apprentissage profond, nous essayons souvent de résoudre des problèmes d'optimisation :
*maximiser* la probabilité attribuée aux données observées ;
*maximiser* le revenu associé à un modèle de recommandation ; 
*minimiser* la distance entre les prédictions
et les observations de terrain ; 
*minimiser* la distance entre les représentations 
de photos de la même personne 
tout en *maximiser* la distance entre les représentations 
de photos de personnes différentes. 
Ces distances, qui constituent 
les objectifs des algorithmes d'apprentissage profond, 
sont souvent exprimées sous forme de normes. 


## Discussion

Dans cette section, nous avons passé en revue toute l'algèbre linéaire
dont vous aurez besoin pour comprendre
une partie remarquable de l'apprentissage profond moderne.
L'algèbre linéaire ne s'arrête pas là
et une grande partie d'entre elle est utile à l'apprentissage automatique.
Par exemple, les matrices peuvent être décomposées en facteurs,
et ces décompositions peuvent révéler
une structure à faible dimension dans les ensembles de données du monde réel.
Il existe des sous-domaines entiers de l'apprentissage automatique
qui se concentrent sur l'utilisation des décompositions de matrices
et leurs généralisations aux tenseurs d'ordre élevé
pour découvrir la structure des ensembles de données 
et résoudre les problèmes de prédiction.
Mais ce livre se concentre sur l'apprentissage profond.
Et nous pensons que vous serez plus enclin 
à apprendre davantage de mathématiques
une fois que vous aurez mis la main à la pâte
en appliquant l'apprentissage automatique à des ensembles de données réels.
Ainsi, bien que nous nous réservions le droit 
d'introduire plus de mathématiques ultérieurement,
nous concluons cette section ici.

Si vous êtes impatient d'en savoir plus sur l'algèbre linéaire,
il existe de nombreux livres et ressources en ligne excellents.
Pour un cours accéléré plus avancé, vous pouvez consulter
:cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.

Pour récapituler :

* Les scalaires, les vecteurs, les matrices et les tenseurs sont 
les objets mathématiques de base utilisés en algèbre linéaire 
et ont respectivement zéro, un, deux et un nombre arbitraire d'axes.
* Les tenseurs peuvent être découpés ou réduits le long d'axes spécifiques 
via l'indexation, ou des opérations telles que `sum` et `mean`, respectivement.
* Les produits par éléments sont appelés produits de Hadamard. 
  En revanche, les produits par points, les produits matrice-vecteur et les produits matrice-matrice 
ne sont pas des opérations par éléments et renvoient en général des objets 
dont la forme est différente de celle des opérandes. 
* Par rapport aux produits Hadamard, les produits matrice-matrice 
sont considérablement plus longs à calculer (temps cubique plutôt que quadratique).
* Les normes capturent diverses notions de la magnitude d'un vecteur, 
et sont généralement appliquées à la différence de deux vecteurs 
pour mesurer leur distance.
* Les normes vectorielles courantes comprennent les normes $\ell_1$ et $\ell_2$, 
et les normes matricielles courantes comprennent les normes *spectrales* et *de Frobenius*.


## Exercices

1. Prouvez que la transposée de la transposée d'une matrice est la matrice elle-même : $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Étant donné deux matrices $\mathbf{A}$ et $\mathbf{B}$, montrez que la somme et la transposition commutent : $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Pour toute matrice carrée $\mathbf{A}$, $\mathbf{A} + \mathbf{A}^\top$ est-elle toujours symétrique ? Pouvez-vous prouver le résultat en utilisant uniquement le résultat des deux exercices précédents ?
1. Nous avons défini le tenseur `X` de forme (2, 3, 4) dans cette section. Quel est le résultat de `len(X)`? Ecrivez votre réponse sans implémenter de code, puis vérifiez votre réponse en utilisant le code. 
1. Pour un tenseur `X` de forme arbitraire, est-ce que `len(X)` correspond toujours à la longueur d'un certain axe de `X`? Quel est cet axe ?
1. Exécutez `A / A.sum(axis=1)` et voyez ce qui se passe. Pouvez-vous en analyser la raison ?
1. Lorsque vous voyagez entre deux points du centre de Manhattan, quelle est la distance que vous devez parcourir en termes de coordonnées, c'est-à-dire en termes d'avenues et de rues ? Pouvez-vous voyager en diagonale ?
1. Considérons un tenseur de forme (2, 3, 4). Quelles sont les formes des sorties de la sommation le long des axes 0, 1 et 2 ?
1. Introduisez un tenseur avec 3 axes ou plus dans la fonction `linalg.norm` et observez sa sortie. Que calcule cette fonction pour des tenseurs de forme arbitraire ?
1. Définissez trois grandes matrices, disons $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ et $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, par exemple initialisées avec des variables aléatoires gaussiennes. Vous voulez calculer le produit $\mathbf{A} \mathbf{B} \mathbf{C}$. Y a-t-il une différence en termes d'empreinte mémoire et de vitesse selon que vous calculez $(\mathbf{A} \mathbf{B}) \mathbf{C}$ ou $\mathbf{A} (\mathbf{B} \mathbf{C})$? Pourquoi ?
1. Définissez trois grandes matrices, par exemple $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ et $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Y a-t-il une différence en termes de vitesse selon que vous calculez $\mathbf{A} \mathbf{B}$ ou $\mathbf{A} \mathbf{C}^\top$? Pourquoi ? Qu'est-ce qui change si vous initialisez $\mathbf{C} = \mathbf{B}^\top$ sans cloner la mémoire ? Pourquoi ?
1. Définissez trois matrices, disons $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Constituez un tenseur à 3 axes en empilant $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. Quelle est la dimensionnalité ? Découpez la deuxième coordonnée du troisième axe pour récupérer $\mathbf{B}$. Vérifiez que votre réponse est correcte.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
