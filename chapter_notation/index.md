# Notation
:label:`chap_notation` 

 Tout au long de cet ouvrage, nous respectons les conventions de notation suivantes.
Notez que certains de ces symboles sont des caractères de substitution,
tandis que d'autres font référence à des objets spécifiques.
En règle générale, 
l'article indéfini " a " indique
que le symbole est un caractère générique
et que des symboles de format similaire
peuvent désigner d'autres objets du même type.
Par exemple, "$x$: un scalaire " signifie 
que les lettres minuscules représentent généralement
des valeurs scalaires.



## Objets numériques

* $x$: un scalaire
* $\mathbf{x}$: un vecteur
* $\mathbf{X}$: une matrice
* $\mathsf{X}$: un tenseur général
* $\mathbf{I}$: une matrice d'identité---carrée, avec $1$ sur toutes les entrées diagonales et $0$ sur toutes les entrées hors-diagonales
* $x_i$, $[\mathbf{x}]_i$: l'élément $i^\mathrm{th}$ du vecteur $\mathbf{x}$
 * $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: l'élément de la matrice $\mathbf{X}$ à la ligne $i$ et à la colonne $j$.



## Théorie des ensembles


 * $\mathcal{X}$: un ensemble
* $\mathbb{Z}$: l'ensemble des entiers
* $\mathbb{Z}^+$: l'ensemble des entiers positifs
* $\mathbb{R}$: l'ensemble des nombres réels
* $\mathbb{R}^n$: l'ensemble des vecteurs $n$-dimensionnels des nombres réels
* $\mathbb{R}^{a\times b}$: L'ensemble des matrices de nombres réels avec $a$ lignes et $b$ colonnes
* $|\mathcal{X}|$: cardinalité (nombre d'éléments) de l'ensemble $\mathcal{X}$
 * $\mathcal{A}\cup\mathcal{B}$: union des ensembles $\mathcal{A}$ et $\mathcal{B}$
 * $\mathcal{A}\cap\mathcal{B}$: intersection des ensembles $\mathcal{A}$ et $\mathcal{B}$
 * $\mathcal{A}\setminus\mathcal{B}$: soustraction de l'ensemble $\mathcal{B}$ de $\mathcal{A}$ (ne contient que les éléments de $\mathcal{A}$ qui n'appartiennent pas à $\mathcal{B}$)



 ## Fonctions et opérateurs


 * $f(\cdot)$: une fonction
* $\log(\cdot)$: le logarithme naturel (base $e$)
* $\log_2(\cdot)$: logarithme avec base $2$
 * $\exp(\cdot)$: la fonction exponentielle
* $\mathbf{1}(\cdot)$: la fonction indicateur, évalue à $1$ si l'argument booléen est vrai et à $0$ sinon
* $\mathbf{1}_{\mathcal{X}}(z)$: la fonction indicateur d'appartenance à un ensemble, évalue à $1$ si l'élément $z$ appartient à l'ensemble $\mathcal{X}$ et $0$ sinon
* $\mathbf{(\cdot)}^\top$: transposée d'un vecteur ou d'une matrice
* $\mathbf{X}^{-1}$: inverse de la matrice $\mathbf{X}$
 * $\odot$: Produit de Hadamard (par éléments)
* $[\cdot, \cdot]$: concaténation
* $\|\cdot\|_p$: $\ell_p$ norm
* $\|\cdot\|$: $\ell_2$ norm
* $\langle \mathbf{x}, \mathbf{y} \rangle$: produit scalaire de vecteurs $\mathbf{x}$ et $\mathbf{y}$
 * $\sum$: somme sur une collection d'éléments
* $\prod$: produit sur une collection d'éléments
* $\stackrel{\mathrm{def}}{=}$: une égalité affirmée comme définition du symbole du côté gauche



 ## Calcul

* $\frac{dy}{dx}$: dérivée de $y$ par rapport à $x$
 * $\frac{\partial y}{\partial x}$: dérivée partielle de $y$ par rapport à $x$
 * $\nabla_{\mathbf{x}} y$: gradient de $y$ par rapport à $\mathbf{x}$
 * $\int_a^b f(x) \;dx$: intégrale définie de $f$ de $a$ à $b$ par rapport à $x$
 * $\int f(x) \;dx$: intégrale indéfinie de $f$ par rapport à $x$

 

 ## Probabilité et théorie de l'information

* $X$: une variable aléatoire
* $P$: une distribution de probabilité
* $X \sim P$: la variable aléatoire $X$ a la distribution $P$
 * $P(X=x)$: la probabilité attribuée à l'événement où la variable aléatoire $X$ prend la valeur $x$
 * $P(X \mid Y)$: la distribution de probabilité conditionnelle de $X$ étant donné $Y$
 * $p(\cdot)$: une fonction de densité de probabilité (PDF) associée à la distribution P
* ${E}[X]$: espérance d'une variable aléatoire $X$
 * $X \perp Y$: les variables aléatoires $X$ et $Y$ sont indépendantes
* $X \perp Y \mid Z$: les variables aléatoires $X$ et $Y$ sont conditionnellement indépendantes étant donné $Z$
 * $\sigma_X$: l'écart-type de la variable aléatoire $X$
 * $\mathrm{Var}(X)$: la variance de la variable aléatoire $X$, égale à $\sigma^2_X$
 * $\mathrm{Cov}(X, Y)$: la covariance des variables aléatoires $X$ et $Y$
 * $\rho(X, Y)$: le coefficient de corrélation de Pearson entre $X$ et $Y$, égal à $\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
 * $H(X)$: entropie de la variable aléatoire $X$
 * $D_{\mathrm{KL}}(P\|Q)$: la divergence KL (ou entropie relative) de la distribution $Q$ à la distribution $P$



[Discussions](https://discuss.d2l.ai/t/25)
