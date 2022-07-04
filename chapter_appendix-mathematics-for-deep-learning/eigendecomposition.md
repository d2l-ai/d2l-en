# Décomposition spectrale
:label:`sec_eigendecompositions` 

Les valeurs propres sont souvent l'une des notions les plus utiles 
que nous rencontrons lorsque nous étudions l'algèbre linéaire, 
cependant, en tant que débutant, il est facile de négliger leur importance.
Ci-dessous, nous présentons la décomposition des valeurs propres et 
essayons de faire comprendre pourquoi elle est si importante. 

Supposons que nous ayons une matrice $A$ avec les entrées suivantes :

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

Si nous appliquons $A$ à n'importe quel vecteur $\mathbf{v} = [x, y]^\top$, 
nous obtenons un vecteur $\mathbf{A}\mathbf{v} = [2x, -y]^\top$.
L'interprétation est intuitive :
étire le vecteur pour qu'il soit deux fois plus large dans la direction $x$,
puis le retourne dans la direction $y$.

Cependant, il existe *certains* vecteurs pour lesquels quelque chose reste inchangé.
A savoir $[1, 0]^\top$ est envoyé à $[2, 0]^\top$
et $[0, 1]^\top$ est envoyé à $[0, -1]^\top$.
Ces vecteurs sont toujours sur la même ligne,
et la seule modification est que la matrice les étire
par un facteur de $2$ et $-1$ respectivement.
Nous appelons ces vecteurs *vecteurs propres*
et le facteur par lequel ils sont étirés *valeurs propres*.

En général, si nous pouvons trouver un nombre $\lambda$ 
et un vecteur $\mathbf{v}$ tels que 

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

Nous disons que $\mathbf{v}$ est un vecteur propre de $A$ et que $\lambda$ est une valeur propre.

## Trouver les valeurs propres
Voyons comment les trouver. :eqref: En soustrayant $\lambda \mathbf{v}$ des deux côtés,
puis en factorisant le vecteur,
nous voyons que ce qui précède est équivalent à :

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$ 
:eqlabel:`eq_eigvalue_der` 

Pour que _COPY9`eq_eigvalue_der` se produise, nous voyons que $(\mathbf{A} - \lambda \mathbf{I})$ 
doit comprimer une certaine direction jusqu'à zéro, 
; il n'est donc pas inversible, et son déterminant est donc nul.
Ainsi, nous pouvons trouver les *valeurs propres* 
en trouvant pour quoi $\lambda$ est $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$.
Une fois que nous avons trouvé les valeurs propres, nous pouvons résoudre 
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ 
pour trouver le(s) *vecteur(s) propre(s)* associé(s).

### Un exemple
Voyons cela avec une matrice plus difficile

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$

Si nous considérons $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$, 
nous voyons que cela est équivalent à l'équation polynomiale
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$ .
Ainsi, les deux valeurs propres sont $4$ et $1$.
Pour trouver les vecteurs associés, nous devons résoudre le problème suivant

$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

Nous pouvons résoudre cette équation avec les vecteurs $[1, -1]^\top$ et $[1, 2]^\top$ respectivement.

Nous pouvons vérifier cela dans le code en utilisant la routine intégrée `numpy.linalg.eig`.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64),
          eigenvectors=True)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

Notez que `numpy` normalise les vecteurs propres pour qu'ils soient de longueur un,
alors que nous avons pris les nôtres pour être de longueur arbitraire.
De plus, le choix du signe est arbitraire.
Cependant, les vecteurs calculés sont parallèles 
à ceux que nous avons trouvés à la main avec les mêmes valeurs propres.

## Décomposition de matrices
Poursuivons l'exemple précédent un peu plus loin.  Soit

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

la matrice dont les colonnes sont les vecteurs propres de la matrice $\mathbf{A}$. Soit

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

la matrice avec les valeurs propres associées sur la diagonale.
La définition des valeurs propres et des vecteurs propres nous dit alors que

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

La matrice $W$ est inversible, nous pouvons donc multiplier les deux côtés par $W^{-1}$ à droite,
nous voyons que nous pouvons écrire

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$ 
:eqlabel:`eq_eig_decomp` 

Dans la section suivante, nous verrons quelques belles conséquences de ceci,
mais pour l'instant nous avons seulement besoin de savoir qu'une telle décomposition 
existera tant que nous pourrons trouver une collection complète 
de vecteurs propres linéairement indépendants (de sorte que $W$ soit inversible).

## Opérations sur les décompositions eigènes
Une chose intéressante à propos des décompositions eigènes :eqref:`eq_eig_decomp` est que 
nous pouvons écrire de nombreuses opérations que nous rencontrons habituellement de manière propre 
en termes de décomposition eigène. Comme premier exemple, considérons :

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Cela nous indique que pour toute puissance positive d'une matrice,
la décomposition effective est obtenue en élevant simplement les valeurs propres à la même puissance.
La même chose peut être montrée pour les puissances négatives.
Ainsi, si nous voulons inverser une matrice, il suffit de considérer

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

ou, en d'autres termes, inverser simplement chaque valeur propre.
Cela fonctionne tant que chaque valeur propre n'est pas nulle.
Nous voyons donc que l'inversion est synonyme d'absence de valeurs propres nulles. 

En effet, des travaux supplémentaires peuvent montrer que si $\lambda_1, \ldots, \lambda_n$ 
sont les valeurs propres d'une matrice, alors le déterminant de cette matrice est

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

ou le produit de toutes les valeurs propres.
Cela a un sens intuitif, car quel que soit l'étirement effectué par $\mathbf{W}$, 
$W^{-1}$ le défait, de sorte qu'au final, le seul étirement qui se produit est 
par multiplication par la matrice diagonale $\boldsymbol{\Sigma}$, 
qui étire les volumes par le produit des éléments diagonaux.

Enfin, rappelez-vous que le rang était le nombre maximum 
de colonnes linéairement indépendantes de votre matrice.
En examinant de près la décomposition en fragments,
nous pouvons voir que le rang est le même 
que le nombre de valeurs propres non nulles de $\mathbf{A}$.

Les exemples pourraient continuer, mais nous espérons que le point est clair :
la décomposition en fragments peut simplifier de nombreux calculs d'algèbre linéaire
et est une opération fondamentale qui sous-tend de nombreux algorithmes numériques
et une grande partie de l'analyse que nous faisons en algèbre linéaire. 

## Valeurs propres de matrices symétriques
Il n'est pas toujours possible de trouver suffisamment de vecteurs propres linéairement indépendants 
pour que le processus ci-dessus fonctionne. Par exemple, la matrice

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

ne possède qu'un seul vecteur propre, à savoir $(1, 0)^\top$. 
Pour traiter de telles matrices, nous avons besoin de techniques plus avancées 
que celles que nous pouvons couvrir (comme la forme normale de Jordan ou la décomposition en valeurs singulières).
Nous devrons souvent limiter notre attention aux matrices 
pour lesquelles nous pouvons garantir l'existence d'un ensemble complet de vecteurs propres.

La famille la plus couramment rencontrée est celle des matrices *symétriques*,
qui sont les matrices où $\mathbf{A} = \mathbf{A}^\top$. 
Dans ce cas, nous pouvons considérer que $W$ est une *matrice orthogonale* - une matrice dont les colonnes sont toutes des vecteurs de longueur 1 qui sont à angle droit les uns par rapport aux autres, où 
$\mathbf{W}^\top = \mathbf{W}^{-1}$ - et toutes les valeurs propres seront réelles. 
Ainsi, dans ce cas particulier, nous pouvons écrire :eqref:`eq_eig_decomp` comme suit

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Théorème du cercle de Gershgorin
Il est souvent difficile de raisonner intuitivement avec les valeurs propres.
Si l'on présente une matrice arbitraire, on ne peut pas dire grand-chose à
sur ce que sont les valeurs propres sans les calculer.
Il existe cependant un théorème qui permet de faire facilement une bonne approximation 
si les valeurs les plus grandes sont sur la diagonale.

Soit $\mathbf{A} = (a_{ij})$ une matrice carrée quelconque ($n\times n$).
Nous allons définir $r_i = \sum_{j \neq i} |a_{ij}|$.
Soit $\mathcal{D}_i$ représentant le disque dans le plan complexe 
avec le centre $a_{ii}$ et le rayon $r_i$.
Alors, chaque valeur propre de $\mathbf{A}$ est contenue dans l'une des matrices $\mathcal{D}_i$.

Cela peut être un peu difficile à comprendre, alors regardons un exemple. 
Considérons la matrice :

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Nous avons $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ et $r_4 = 0.9$.
La matrice est symétrique, donc toutes les valeurs propres sont réelles.
Cela signifie que toutes nos valeurs propres se situeront dans l'une des plages de 

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$ 

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$ 

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$ 

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$ 

 
Le calcul numérique montre à 
que les valeurs propres sont approximativement $0.99$, $2.97$, $4.95$, $9.08$,
toutes confortablement situées dans les plages fournies.

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

De cette façon, les valeurs propres peuvent être approximées, 
et les approximations seront assez précises 
dans le cas où la diagonale est 
significativement plus grande que tous les autres éléments. 

Il s'agit d'une petite chose, mais avec un sujet complexe 
et subtil comme la décomposition des eigènes, 
il est bon d'obtenir toute compréhension intuitive que nous pouvons.

## Une application utile : La croissance des cartes itérées

Maintenant que nous comprenons ce que sont les vecteurs propres en principe,
voyons comment ils peuvent être utilisés pour fournir une compréhension approfondie 
d'un problème central au comportement des réseaux neuronaux : l'initialisation correcte des poids. 

### Les vecteurs propres en tant que comportement à long terme

L'étude mathématique complète de l'initialisation 
des réseaux neuronaux profonds dépasse le cadre de ce texte, 
mais nous pouvons en voir une version jouet ici pour comprendre
comment les valeurs propres peuvent nous aider à voir comment ces modèles fonctionnent.
Comme nous le savons, les réseaux neuronaux fonctionnent en intercalant des couches 
de transformations linéaires avec des opérations non linéaires.
Pour simplifier, nous supposerons ici qu'il n'y a pas de non-linéarité,
et que la transformation est une opération matricielle unique répétée $A$,
de sorte que la sortie de notre modèle est

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

Lorsque ces modèles sont initialisés, $A$ est considéré comme étant 
une matrice aléatoire avec des entrées gaussiennes, alors créons-en une. 
Pour être concret, nous commençons par une matrice de moyenne zéro et de variance un distribuée de manière gaussienne $5 \times 5$.

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Comportement sur des données aléatoires
Pour simplifier notre modèle-jouet,
nous supposerons que le vecteur de données que nous introduisons dans $\mathbf{v}_{in}$ 
est un vecteur gaussien aléatoire à cinq dimensions.
Réfléchissons à ce que nous voulons qu'il se passe.
Pour le contexte, pensons à un problème ML générique,
où nous essayons de transformer des données d'entrée, comme une image, en une prédiction, 
comme la probabilité que l'image soit la photo d'un chat.
Si l'application répétée de $\mathbf{A}$ 
 étire un vecteur aléatoire jusqu'à ce qu'il soit très long, 
alors de petits changements en entrée seront amplifiés 
en de grands changements en sortie - d'infimes modifications de l'image d'entrée
conduiraient à des prédictions très différentes.
Cela ne semble pas correct !

D'un autre côté, si $\mathbf{A}$ réduit des vecteurs aléatoires pour qu'ils soient plus courts,
alors, après avoir traversé de nombreuses couches, le vecteur sera essentiellement réduit à néant, 
et la sortie ne dépendra pas de l'entrée. Ce n'est clairement pas correct non plus !

Nous devons marcher sur la ligne étroite entre la croissance et la décroissance 
pour nous assurer que notre sortie change en fonction de notre entrée, mais pas beaucoup !

Voyons ce qui se passe lorsque nous multiplions de manière répétée notre matrice $\mathbf{A}$ 
contre un vecteur d'entrée aléatoire, et que nous gardons trace de la norme.

```{.python .input}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

La norme augmente de façon incontrôlable ! 
En effet, si nous prenons la liste des quotients, nous verrons une tendance.

```{.python .input}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

Si nous regardons la dernière partie du calcul ci-dessus, 
nous voyons que le vecteur aléatoire est étiré par un facteur de `1.974459321485[...]`,
où la partie à la fin se déplace un peu, 
mais le facteur d'étirement est stable. 

### Retour aux vecteurs propres

Nous avons vu que les vecteurs propres et les valeurs propres correspondent 
à la quantité d'étirement de quelque chose, 
mais cela concernait des vecteurs spécifiques et des étirements spécifiques.
Voyons ce qu'il en est pour $\mathbf{A}$.
Une petite mise en garde s'impose : il s'avère que pour les voir toutes,
nous devrons passer aux nombres complexes.
Vous pouvez les considérer comme des étirements et des rotations.
En prenant la norme du nombre complexe
(racine carrée des sommes des carrés des parties réelles et imaginaires)
nous pouvons mesurer ce facteur d'étirement. Faisons également un tri.

```{.python .input}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.eig(A)[0][:,0].tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### Une observation

Nous voyons quelque chose d'un peu inattendu se produire ici : 
ce nombre que nous avons identifié précédemment pour l'étirement à long terme 
de notre matrice $\mathbf{A}$ 
appliquée à un vecteur aléatoire est *exactement* 
(précis à treize décimales près !) 
la plus grande valeur propre de $\mathbf{A}$.
Ce n'est clairement pas une coïncidence !

Mais, si nous réfléchissons maintenant à ce qui se passe d'un point de vue géométrique,
cela commence à avoir un sens. Considérons un vecteur aléatoire. 
Ce vecteur aléatoire pointe un peu dans toutes les directions.
En particulier, il pointe au moins un peu 
dans la même direction que le vecteur propre de $\mathbf{A}$
associé à la plus grande valeur propre.
Ce point est si important qu'il est appelé 
la *valeur propre de principe* et le *vecteur propre de principe*.
Après avoir appliqué $\mathbf{A}$, notre vecteur aléatoire 
est étiré dans toutes les directions possibles,
car il est associé à tous les vecteurs propres possibles,
mais il est surtout étiré dans la direction 
associée à ce vecteur propre principal.
Cela signifie qu'après avoir appliqué $A$, 
notre vecteur aléatoire est plus long et pointe dans une direction 
plus proche de l'alignement avec le vecteur propre principal.
Après avoir appliqué la matrice de nombreuses fois, 
l'alignement avec le vecteur propre principal devient de plus en plus proche jusqu'à ce que, 
à toutes fins utiles, notre vecteur aléatoire ait été transformé 
en vecteur propre principal !
En effet, cet algorithme est la base 
de ce que l'on appelle l'itération *puissance*
pour trouver la plus grande valeur propre et le plus grand vecteur propre d'une matrice. Pour plus de détails, voir, par exemple, :cite:`Van-Loan.Golub.1983`.

### Correction de la normalisation

Les discussions ci-dessus nous ont permis de conclure 
que nous ne voulons pas du tout qu'un vecteur aléatoire soit étiré ou écrasé,
et que nous souhaitons que les vecteurs aléatoires conservent à peu près la même taille tout au long du processus.
Pour ce faire, nous redimensionnons maintenant notre matrice par cette valeur propre principale 
de sorte que la plus grande valeur propre ne soit plus qu'une seule.
Voyons ce qui se passe dans ce cas.

```{.python .input}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Nous pouvons également tracer le rapport entre les normes consécutives comme précédemment et voir qu'effectivement il se stabilise.

```{.python .input}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Discussion

Nous voyons maintenant exactement ce que nous espérions !
Après avoir normalisé les matrices par la valeur propre principale,
nous constatons que les données aléatoires n'explosent pas comme avant,
mais finissent par s'équilibrer à une valeur spécifique.
Ce serait bien de pouvoir faire ces choses à partir des premiers principes,
et il s'avère que si nous examinons en profondeur les mathématiques,
nous pouvons voir que la plus grande valeur propre 
d'une grande matrice aléatoire avec des entrées indépendantes de moyenne zéro,
de variance une gaussienne est en moyenne d'environ $\sqrt{n}$,
ou dans notre cas $\sqrt{5} \approx 2.2$,
en raison d'un fait fascinant connu sous le nom de *loi circulaire* :cite:`Ginibre.1965`.
La relation entre les valeurs propres (et un objet connexe appelé valeurs singulières) des matrices aléatoires s'est avérée avoir des liens profonds avec l'initialisation appropriée des réseaux neuronaux, comme cela a été discuté dans :cite:`Pennington.Schoenholz.Ganguli.2017` et les travaux ultérieurs.

## Résumé
* Les vecteurs propres sont des vecteurs qui sont étirés par une matrice sans changer de direction.
* Les valeurs propres sont la quantité de vecteurs propres qui sont étirés par l'application de la matrice.
* La décomposition propre d'une matrice peut permettre de réduire de nombreuses opérations à des opérations sur les valeurs propres.
* Le théorème du cercle de Gershgorin peut fournir des valeurs approximatives pour les valeurs propres d'une matrice.
* Le comportement des puissances de matrices itérées dépend principalement de la taille de la plus grande valeur propre.  Cette compréhension a de nombreuses applications dans la théorie de l'initialisation des réseaux de neurones.

## Exercices
1. Quelles sont les valeurs propres et les vecteurs propres de
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  Quelles sont les valeurs propres et les vecteurs propres de la matrice suivante, et qu'est-ce qui est étrange dans cet exemple par rapport au précédent ?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Sans calculer les valeurs propres, est-il possible que la plus petite valeur propre de la matrice suivante soit inférieure à $0.5$? *Note* : ce problème peut être fait de tête.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:
