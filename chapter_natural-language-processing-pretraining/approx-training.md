# Approximate Training
:label:`sec_approx_train` 

Rappelez-vous nos discussions dans :numref:`sec_word2vec`.
L'idée principale du modèle de saut de programme est
l'utilisation d'opérations softmax pour calculer
la probabilité conditionnelle de
générer un mot de contexte $w_o$
basé sur le mot central donné $w_c$
dans :eqref:`eq_skip-gram-softmax`,
dont la perte logarithmique correspondante est donnée par
le contraire de :eqref:`eq_skip-gram-log`.



En raison de la nature de l'opération softmax,
étant donné qu'un mot contextuel peut être n'importe qui dans le 
dictionnaire \mathcal{V},
l'opposé de :eqref:`eq_skip-gram-log` 
contient la somme
d'éléments aussi nombreux que la taille totale du vocabulaire.
Par conséquent,
le calcul du gradient
pour le modèle de saut de programme
dans :eqref:`eq_skip-gram-grad` 
et celui
pour le modèle de sac de mots continu
dans :eqref:`eq_cbow-gradient` 
contiennent tous deux
la sommation.
Malheureusement,
le coût de calcul
pour de tels gradients
qui font la somme sur
un grand dictionnaire
(souvent avec
des centaines de milliers ou des millions de mots)
est énorme !

Afin de réduire la complexité de calcul susmentionnée, cette section présente deux méthodes d'apprentissage approximatives :
*negative sampling* et *hierarchical softmax*.
En raison de la similitude
entre le modèle de saut de programme et
le modèle de sac de mots continu,
nous prendrons simplement le modèle de saut de programme comme exemple
pour décrire ces deux méthodes d'apprentissage approximatives.

### Echantillonnage négatif
:label:`subsec_negative-sampling` 

 
L'échantillonnage négatif modifie la fonction objectif originale.
Étant donné la fenêtre de contexte d'un mot central $w_c$,
le fait que tout mot (de contexte) $w_o$
provienne de cette fenêtre de contexte
est considéré comme un événement avec la probabilité
modélisée par


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

où \sigma utilise la définition de la fonction d'activation sigmoïde :

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

Commençons par
maximiser la probabilité conjointe de
tous les événements de ce type dans les séquences de texte
pour former les embeddings des mots.
Plus précisément,
étant donné une séquence de texte de longueur $T$,
désignant par $w^{(t)}$ le mot au pas de temps $t$
et la taille de la fenêtre de contexte étant $m$,
considérons la maximisation de la probabilité conjointe

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`


Cependant,
:eqref:`eq-negative-sample-pos` 
ne prend en compte que les événements
qui impliquent des exemples positifs.
Par conséquent,
la probabilité conjointe dans
:eqref:`eq-negative-sample-pos` 
est maximisée en 1
seulement si tous les vecteurs de mots sont égaux à l'infini.
Bien entendu,
de tels résultats sont dénués de sens.
Pour rendre la fonction objectif
plus significative,
*échantillonnage négatif*
ajoute des exemples négatifs échantillonnés
à partir d'une distribution prédéfinie.

Dénotez par $S$
l'événement selon lequel
un mot contextuel $w_o$ provient de
la fenêtre contextuelle d'un mot central $w_c$.
Pour cet événement impliquant $w_o$,
à partir d'une distribution prédéfinie $P(w)$
échantillonnent $K$ mots de *bruit*
qui ne proviennent pas de cette fenêtre de contexte.
Désignez par $N_k$
l'événement selon lequel
un mot de bruit $w_k$ ($k=1, \ldots, K$)
ne provient pas de
la fenêtre de contexte de $w_c$.
Supposons que
ces événements impliquant
l'exemple positif et les exemples négatifs
$S, N_1, \ldots, N_K$ sont mutuellement indépendants.
L'échantillonnage négatif
réécrit la probabilité conjointe (impliquant uniquement des exemples positifs)
en :eqref:`eq-negative-sample-pos`
comme

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

où la probabilité conditionnelle est approximée par
les événements $S, N_1, \ldots, N_K$:

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

Dénotez par
$i_t$ et $h_k$
les indices de
un mot $w^{(t)}$ au pas de temps $t$
d'une séquence de texte
et un mot de bruit $w_k$,
respectivement.
La perte logarithmique par rapport aux probabilités conditionnelles dans :eqref:`eq-negative-sample-conditional-prob` est

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

 
Nous pouvons voir que
maintenant le coût de calcul des gradients
à chaque étape de formation
n'a rien à voir avec la taille du dictionnaire,
mais dépend linéairement de $K$.
En fixant l'hyperparamètre $K$
à une valeur plus petite,
le coût de calcul des gradients
à chaque étape de entrainement avec un échantillonnage négatif
est plus petit.




## Softmax hiérarchique

Comme autre méthode de entrainement approximative,
*hierarchical softmax*
utilise l'arbre binaire,
une structure de données
illustrée dans :numref:`fig_hi_softmax`,
où chaque nœud feuille
de l'arbre représente
un mot du dictionnaire \mathcal{V}.

![Softmax hiérarchique pour l'apprentissage approximatif, où chaque nœud feuille de l'arbre représente un mot du dictionnaire.](../img/hi-softmax.svg)
:label:`fig_hi_softmax` 

Soit $L(w)$
le nombre de nœuds (y compris les deux extrémités)
sur le chemin
du nœud racine au nœud feuille représentant le mot $w$
dans l'arbre binaire.
Soit $n(w,j)$ le $j^\mathrm{ième}$ nœud sur ce chemin,
avec son vecteur de mot contextuel étant
$\mathbf{u}_{n(w, j)}$.
Par exemple,
$L(w_3) = 4$ dans :numref:`fig_hi_softmax`.
La méthode hiérarchique softmax approxime la probabilité conditionnelle dans :eqref:`eq_skip-gram-softmax` comme suit :


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

où la fonction $\sigma$
est définie dans :eqref:`eq_sigma-f`,
et \text{leftChild} $(n)$ est le nœud enfant gauche du nœud $n$: si $x$ est vrai, $[\![x]\!] = 1$; sinon $[\![x]\!] = -1$.

Pour illustrer,
calculons
la probabilité conditionnelle
de générer le mot $w_3$
étant donné le mot $w_c$ dans :numref:`fig_hi_softmax`.
Cela nécessite des produits scalaires
entre le vecteur de mot
$\mathbf{v}_c$ de $w_c$
et
les vecteurs de nœuds non feu
sur le chemin (le chemin en gras dans :numref:`fig_hi_softmax`) de la racine à $w_3$,
qui est parcouru à gauche, à droite, puis à gauche :


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$ 

Puisque $\sigma(x)+\sigma(-x) = 1$,
il s'avère que
les probabilités conditionnelles de
générer tous les mots du dictionnaire
$\mathcal{V}$ 
sur la base de n'importe quel mot $w_c$
sont égales à un :

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

Heureusement, puisque $L(w_o)-1$ est de l'ordre de $\mathcal{O}(\text{log}_2|\mathcal{V}|)$ en raison de la structure d'arbre binaire,
lorsque la taille du dictionnaire $\mathcal{V}$ est énorme,
le coût de calcul pour chaque étape de entrainement utilisant la méthode softmax hiérarchique
est considérablement réduit par rapport à celui de
sans entrainement approximative.

## Résumé

* L'échantillonnage négatif construit la fonction de perte en considérant des événements mutuellement indépendants qui impliquent des exemples positifs et négatifs. Le coût de calcul pour l'apprentissage dépend linéairement du nombre de mots de bruit à chaque étape.
* Le softmax hiérarchique construit la fonction de perte en utilisant le chemin du nœud racine au nœud feuille dans l'arbre binaire. Le coût de calcul pour l'apprentissage dépend du logarithme de la taille du dictionnaire à chaque étape.

## Exercices

1. Comment échantillonner les mots de bruit dans l'échantillonnage négatif ?
1. Vérifiez que :eqref:`eq_hi-softmax-sum-one` est valable.
1. Comment entraîner le modèle de sac de mots continu en utilisant l'échantillonnage négatif et la méthode softmax hiérarchique, respectivement ?

[Discussions](https://discuss.d2l.ai/t/382)
