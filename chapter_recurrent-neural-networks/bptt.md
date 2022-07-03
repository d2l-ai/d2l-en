# Backpropagation à travers le temps
:label:`sec_bptt` 

Jusqu'à présent, nous avons fait plusieurs fois allusion à des choses comme
*gradients explosifs*,
*gradients évanouissants*,
et la nécessité de
*détacher le gradient* pour les RNN.
Par exemple, dans :numref:`sec_rnn-scratch` 
nous avons invoqué la fonction `detach` sur la séquence.
Rien de tout cela n'a été vraiment expliqué en détail,
dans l'intérêt de pouvoir construire un modèle rapidement et
de voir comment il fonctionne.
Dans cette section,
nous allons nous plonger un peu plus profondément
dans les détails de la rétropropagation pour les modèles de séquence et pourquoi (et comment) les mathématiques fonctionnent.

Nous avons rencontré certains des effets de l'explosion du gradient lorsque nous avons implémenté pour la première fois les RNN à l'adresse
(:numref:`sec_rnn-scratch` ).
En particulier,

si vous avez résolu les exercices,
vous aurez vu
que l'écrêtage du gradient est vital pour assurer une convergence correcte
.
Pour mieux comprendre ce problème, cette section
examinera comment les gradients sont calculés pour les modèles de séquence.
Notez
qu'il n'y a rien de conceptuellement nouveau dans la façon dont cela fonctionne. Après tout, nous ne faisons qu'appliquer la règle de la chaîne pour calculer les gradients. Néanmoins, il est utile de revoir la rétropropagation (:numref:`sec_backprop` ) sur
.


Nous avons décrit les propagations avant et arrière
et les graphes de calcul
dans les MLP dans :numref:`sec_backprop`.
La propagation avant dans un RNN est relativement simple
.
*La rétropropagation dans le temps* est en fait une application spécifique
de la rétropropagation
dans les RNN :cite:`Werbos.1990`.
Elle
nous oblige à étendre le graphe de calcul 
d'un RNN
un pas de temps à la fois pour
obtenir les dépendances
entre les variables et les paramètres du modèle.
Ensuite,
sur la base de la règle de la chaîne,
nous appliquons la rétropropagation pour calculer et
stocker les gradients.
Comme les séquences peuvent être assez longues, les dépendances peuvent être assez longues.
Par exemple, pour une séquence de 1000 caractères, 
le premier jeton pourrait potentiellement avoir une influence significative sur le jeton en position finale.
Ce n'est pas vraiment faisable sur le plan informatique
(cela prend trop de temps et nécessite trop de mémoire) et il faut plus de 1000 produits matriciels avant d'arriver à ce gradient si insaisissable.
Il s'agit d'un processus entaché d'incertitude informatique et statistique.
Dans ce qui suit, nous allons élucider ce qui se passe
et comment aborder ce problème dans la pratique.

## Analyse des gradients dans les RNN
:label:`subsec_bptt_analysis` 

Nous commençons par un modèle simplifié du fonctionnement d'un RNN.
Ce modèle ignore les détails concernant les spécificités de l'état caché et la façon dont il est mis à jour.
La notation mathématique utilisée ici
ne distingue pas explicitement
les scalaires, les vecteurs et les matrices comme elle le faisait auparavant.
Ces détails sont sans importance pour l'analyse
et ne serviraient qu'à encombrer la notation
dans cette sous-section.

Dans ce modèle simplifié,
nous désignons $h_t$ comme l'état caché,
$x_t$ comme entrée, et $o_t$ comme sortie
au pas de temps $t$.
Rappelez-vous nos discussions dans
:numref:`subsec_rnn_w_hidden_states` 
que l'entrée et l'état caché
peuvent être concaténés pour
être multipliés par une variable de poids dans la couche cachée.
Ainsi, nous utilisons $w_h$ et $w_o$ à
pour indiquer les poids de la couche cachée et de la couche de sortie, respectivement.
Par conséquent, les états cachés et les sorties à chaque pas de temps peuvent être expliqués comme suit :

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$ 
:eqlabel:`eq_bptt_ht_ot` 

où $f$ et $g$ sont des transformations
de la couche cachée et de la couche de sortie, respectivement.
Nous avons donc une chaîne de valeurs $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ qui dépendent les unes des autres par le biais du calcul récurrent.
La propagation vers l'avant est assez simple.
Tout ce dont nous avons besoin est de boucler à travers les triples $(x_t, h_t, o_t)$, un pas de temps à la fois.
L'écart entre la sortie $o_t$ et la cible souhaitée $y_t$ est ensuite évalué par une fonction objectif
sur tous les pas de temps $T$
comme

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$ 

 

Pour la rétropropagation, les choses sont un peu plus délicates, notamment lorsque nous calculons les gradients par rapport aux paramètres $w_h$ de la fonction objectif $L$. Pour être précis, selon la règle de la chaîne,

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$ 
:eqlabel:`eq_bptt_partial_L_wh` 

Le premier et le deuxième facteurs du produit
dans :eqref:`eq_bptt_partial_L_wh` 
sont faciles à calculer.
C'est pour le troisième facteur $\partial h_t/\partial w_h$ que les choses se compliquent, car nous devons calculer de manière récurrente l'effet du paramètre $w_h$ sur $h_t$.
Selon le calcul récurrent
dans :eqref:`eq_bptt_ht_ot`,
$h_t$ dépend à la fois de $h_{t-1}$ et $w_h$,
où le calcul de $h_{t-1}$
dépend également de $w_h$.
Ainsi, l'évaluation du dérivé total de $h_t$ par rapport à $w_h$
en utilisant la règle de la chaîne donne

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$ 
:eqlabel:`eq_bptt_partial_ht_wh_recur` 

 
Pour dériver le gradient ci-dessus, supposons que nous avons trois séquences $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ satisfaisant
$a_{0}=0$ et $a_{t}=b_{t}+c_{t}a_{t-1}$ pour $t=1, 2,\ldots$.
Pour $t\geq 1$, il est facile de montrer que

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$ 
:eqlabel:`eq_bptt_at` 

En substituant $a_t$, $b_t$, et $c_t$
conformément à

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

le calcul du gradient dans :eqref:`eq_bptt_partial_ht_wh_recur` satisfait à
$a_{t}=b_{t}+c_{t}a_{t-1}$ .
Ainsi,
par :eqref:`eq_bptt_at`,
nous pouvons supprimer le calcul récurrent dans :eqref:`eq_bptt_partial_ht_wh_recur` 
avec

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$ 
:eqlabel:`eq_bptt_partial_ht_wh_gen` 

Bien que nous puissions utiliser la règle de la chaîne pour calculer $\partial h_t/\partial w_h$ de manière récursive, cette chaîne peut devenir très longue lorsque $t$ est grand. Examinons un certain nombre de stratégies pour résoudre ce problème.

### Calcul complet### 

Évidemment,
nous pouvons simplement calculer la somme complète dans
:eqref:`eq_bptt_partial_ht_wh_gen`.
Cependant,
c'est très lent et les gradients peuvent exploser,
puisque des changements subtils dans les conditions initiales peuvent potentiellement affecter beaucoup le résultat.
En d'autres termes, nous pourrions assister à des phénomènes similaires à l'effet papillon, où des changements minimes dans les conditions initiales entraînent des changements disproportionnés dans le résultat.
Ceci est en fait assez indésirable en termes du modèle que nous voulons estimer.
Après tout, nous recherchons des estimateurs robustes qui se généralisent bien. Cette stratégie n'est donc presque jamais utilisée dans la pratique.

### Tronquer les pas de temps###

Alternativement,
nous pouvons tronquer la somme en
:eqref:`eq_bptt_partial_ht_wh_gen` 
après $\tau$ pas. 
C'est ce dont nous avons discuté jusqu'à présent,
comme lorsque nous avons détaché les gradients dans :numref:`sec_rnn-scratch`. 
Cela conduit à une *approximation* du vrai gradient, simplement en terminant la somme à 
$\partial h_{t-\tau}/\partial w_h$ . 
En pratique, cela fonctionne assez bien. C'est ce que l'on appelle communément la rétropropagation tronquée dans le temps :cite:`Jaeger.2002`.
L'une des conséquences de cette méthode est que le modèle se concentre principalement sur l'influence à court terme plutôt que sur les conséquences à long terme. C'est en fait *souhaitable*, car cela oriente l'estimation vers des modèles plus simples et plus stables.

### Troncature aléatoire### 

Enfin, nous pouvons remplacer $\partial h_t/\partial w_h$
par une variable aléatoire qui est correcte dans l'attente mais qui tronque la séquence.
Pour ce faire, nous utilisons une séquence de $\xi_t$
avec des valeurs prédéfinies $0 \leq \pi_t \leq 1$,
où $P(\xi_t = 0) = 1-\pi_t$ et $P(\xi_t = \pi_t^{-1}) = \pi_t$, donc $E[\xi_t] = 1$.
Nous utilisons ceci pour remplacer le gradient
$\partial h_t/\partial w_h$ 
dans :eqref:`eq_bptt_partial_ht_wh_recur` 
par

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$ 

 
Il découle de la définition de $\xi_t$ que $E[z_t] = \partial h_t/\partial w_h$.
Chaque fois que $\xi_t = 0$ le calcul récurrent
se termine à ce pas de temps $t$.
Cela conduit à une somme pondérée de séquences de différentes longueurs, où les longues séquences sont rares mais correctement surpondérées. 
Cette idée a été proposée par Tallec et Ollivier
:cite:`Tallec.Ollivier.2017`.

### Comparaison des stratégies

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg) 
:label:`fig_truncated_bptt` 

 
:numref:`fig_truncated_bptt` illustre les trois stratégies lors de l'analyse des premiers caractères du livre *The Time Machine* en utilisant la rétropropagation dans le temps pour les RNN :

* La première ligne est la troncature aléatoire qui partitionne le texte en segments de longueurs variables.
* La deuxième ligne est la troncature régulière qui divise le texte en sous-séquences de même longueur. C'est ce que nous avons fait dans les expériences RNN.
* La troisième ligne est la rétropropagation complète dans le temps qui conduit à une expression infaisable sur le plan informatique.


Malheureusement, bien que séduisante en théorie, la troncature aléatoire ne fonctionne pas beaucoup mieux que la troncature régulière, très probablement en raison d'un certain nombre de facteurs.
Premièrement, l'effet d'une observation après un certain nombre d'étapes de rétropropagation dans le passé est tout à fait suffisant pour capturer les dépendances en pratique. 
Deuxièmement, l'augmentation de la variance contrebalance le fait que le gradient est plus précis avec plus d'étapes. 
Troisièmement, nous *voulons* en fait des modèles qui n'ont qu'une courte gamme d'interactions. Par conséquent, la rétropropagation par le temps régulièrement tronquée a un léger effet de régularisation qui peut être souhaitable.

## Backpropagation par le temps en détail

Après avoir discuté du principe général,
discutons de la rétro-propagation par le temps en détail.
Contrairement à l'analyse présentée dans le document
:numref:`subsec_bptt_analysis`,
dans la suite de ce document
nous allons montrer
comment calculer
les gradients de la fonction objectif
par rapport à tous les paramètres du modèle décomposé.
Pour garder les choses simples, nous considérons 
un RNN sans paramètres de biais,
dont la fonction d'activation 

dans la couche cachée
utilise la correspondance d'identité ($\phi(x)=x$).
Pour le pas de temps $t$,
l'entrée et la cible de l'exemple unique sont respectivement
$\mathbf{x}_t \in \mathbb{R}^d$ et $y_t$. 
L'état caché $\mathbf{h}_t \in \mathbb{R}^h$ 
et la sortie $\mathbf{o}_t \in \mathbb{R}^q$
sont calculés comme suit

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

où $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, et
$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ 
sont les paramètres de poids.
On désigne par $l(\mathbf{o}_t, y_t)$
la perte au pas de temps $t$. 
Notre fonction objective,
la perte sur $T$ pas de temps
depuis le début de la séquence
est donc

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$ 

 
Afin de visualiser les dépendances entre
variables et paramètres du modèle pendant le calcul
du RNN,
nous pouvons dessiner un graphe de calcul pour le modèle,
comme indiqué dans :numref:`fig_rnn_bptt`.
Par exemple, le calcul des états cachés du pas de temps 3, $\mathbf{h}_3$, dépend des paramètres du modèle $\mathbf{W}_{hx}$ et $\mathbf{W}_{hh}$,
de l'état caché du dernier pas de temps $\mathbf{h}_2$,
et de l'entrée du pas de temps actuel $\mathbf{x}_3$.

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) ou les paramètres (grisés) et les cercles représentent les opérateurs](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt` 

Comme nous venons de le mentionner, les paramètres du modèle dans :numref:`fig_rnn_bptt` sont $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$ et $\mathbf{W}_{qh}$. 
En général, l'apprentissage de ce modèle
par
nécessite 
le calcul du gradient par rapport à ces paramètres
$\partial L/\partial \mathbf{W}_{hx}$ , $\partial L/\partial \mathbf{W}_{hh}$ et $\partial L/\partial \mathbf{W}_{qh}$.
Selon les dépendances dans :numref:`fig_rnn_bptt`,
nous pouvons parcourir 
dans la direction opposée des flèches
pour calculer et stocker les gradients à tour de rôle.
Pour exprimer de manière flexible la multiplication
de matrices, de vecteurs et de scalaires de différentes formes
dans la règle de la chaîne,
nous continuons à utiliser 
l'opérateur 
$\text{prod}$ comme décrit dans
:numref:`sec_backprop`.


Tout d'abord,
différencier la fonction objectif
par rapport à la sortie du modèle
à n'importe quel pas de temps $t$
est assez simple :

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$ 
:eqlabel:`eq_bptt_partial_L_ot` 

Maintenant, nous pouvons calculer le gradient de la fonction objectif
par rapport à
le paramètre $\mathbf{W}_{qh}$
dans la couche de sortie :
$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ . Sur la base de :numref:`fig_rnn_bptt`, 
la fonction objectif
$L$ dépend de $\mathbf{W}_{qh}$ via $\mathbf{o}_1, \ldots, \mathbf{o}_T$. L'utilisation de la règle de la chaîne donne

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

où $\partial L/\partial \mathbf{o}_t$
est donné par :eqref:`eq_bptt_partial_L_ot`.

Ensuite, comme le montre :numref:`fig_rnn_bptt`,
au dernier pas de temps $T$
la fonction objectif
$L$ dépend de l'état caché $\mathbf{h}_T$ uniquement via $\mathbf{o}_T$.
Par conséquent, nous pouvons facilement trouver
le gradient 
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$ 
en utilisant la règle de la chaîne :

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$ 
:eqlabel:`eq_bptt_partial_L_hT_final_step` 

Cela devient plus délicat pour tout pas de temps $t < T$,
où la fonction objectif $L$ dépend de $\mathbf{h}_t$ via $\mathbf{h}_{t+1}$ et $\mathbf{o}_t$.
Selon la règle de la chaîne,
le gradient de l'état caché
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$ 
 à tout pas de temps $t < T$ peut être calculé de manière récurrente comme suit :


$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`


Pour l'analyse,
en étendant le calcul récurrent
pour tout pas de temps $1 \leq t \leq T$
donne

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$ 
:eqlabel:`eq_bptt_partial_L_ht` 

Nous pouvons voir à partir de :eqref:`eq_bptt_partial_L_ht` que cet exemple linéaire simple
présente déjà certains problèmes clés des modèles à longue séquence : il implique des puissances potentiellement très grandes de $\mathbf{W}_{hh}^\top$.
Dans ce modèle, les valeurs propres inférieures à 1 disparaissent
et les valeurs propres supérieures à 1 divergent.
Cette situation est numériquement instable,
ce qui se manifeste sous la forme de gradients qui disparaissent 
et explosent.
Une façon de résoudre ce problème est de tronquer les pas de temps
à une taille pratique du point de vue informatique, comme discuté dans :numref:`subsec_bptt_analysis`. 
En pratique, cette troncature est effectuée en détachant le gradient après un nombre donné de pas de temps.
Nous verrons plus loin 
comment des modèles de séquence plus sophistiqués, tels que la mémoire à long terme, peuvent atténuer ce problème. 

Enfin,
:numref:`fig_rnn_bptt` montre que
la fonction objective
$L$ dépend des paramètres du modèle
$\mathbf{W}_{hx}$ et $\mathbf{W}_{hh}$
dans la couche cachée
via les états cachés
$\mathbf{h}_1, \ldots, \mathbf{h}_T$ .
Pour calculer les gradients
par rapport à ces paramètres
$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ et $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$,
nous appliquons la règle de la chaîne qui donne

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

où
$\partial L/\partial \mathbf{h}_t$ 
qui est calculé de manière récurrente par
:eqref:`eq_bptt_partial_L_hT_final_step` 
et
:eqref:`eq_bptt_partial_L_ht_recur` 
est la quantité clé
qui affecte la stabilité numérique.



Étant donné que la rétropropagation dans le temps
est l'application de la rétropropagation dans les RNN,
comme nous l'avons expliqué dans :numref:`sec_backprop`,
l'entrainement des RNN
alterne la propagation vers l'avant avec
la rétropropagation dans le temps.

En outre, la rétropropagation dans le temps
calcule et stocke les gradients ci-dessus
à tour de rôle.
Plus précisément, les valeurs intermédiaires stockées sur

sont réutilisées sur
pour éviter les calculs en double,
comme le stockage de 
$\partial L/\partial \mathbf{h}_t$ 
pour être utilisées dans le calcul de $\partial L / \partial \mathbf{W}_{hx}$ et $\partial L / \partial \mathbf{W}_{hh}$.


## Résumé

* La rétropropagation dans le temps est simplement une application de la rétropropagation aux modèles de séquence avec un état caché.
* La troncature est nécessaire pour la commodité du calcul et la stabilité numérique, comme la troncature régulière et la troncature aléatoire.
* Les puissances élevées des matrices peuvent conduire à des valeurs propres divergentes ou évanouissantes. Cela se manifeste sous la forme de gradients qui explosent ou disparaissent.
* Pour un calcul efficace, les valeurs intermédiaires sont mises en cache pendant la rétropropagation dans le temps.



## Exercices

1. Supposons que nous ayons une matrice symétrique $\mathbf{M} \in \mathbb{R}^{n \times n}$ avec des valeurs propres $\lambda_i$ dont les vecteurs propres correspondants sont $\mathbf{v}_i$ ($i = 1, \ldots, n$). Sans perte de généralité, supposez qu'ils sont ordonnés dans l'ordre $|\lambda_i| \geq |\lambda_{i+1}|$. 
 1. Montrez que $\mathbf{M}^k$ a des valeurs propres $\lambda_i^k$.
  1. Prouvez que pour un vecteur aléatoire $\mathbf{x} \in \mathbb{R}^n$, avec une forte probabilité, $\mathbf{M}^k \mathbf{x}$ sera très aligné avec le vecteur propre $\mathbf{v}_1$ 
de $\mathbf{M}$. Formalisez cette affirmation.
  1. Que signifie le résultat ci-dessus pour les gradients dans les RNN ?
1. Outre l'écrêtage du gradient, connaissez-vous d'autres méthodes pour faire face à l'explosion du gradient dans les réseaux de neurones récurrents ?

[Discussions](https://discuss.d2l.ai/t/334)
