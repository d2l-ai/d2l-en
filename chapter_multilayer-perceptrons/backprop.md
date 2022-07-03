# Propagation vers l'avant, propagation vers l'arrière et graphes computationnels
:label:`sec_backprop` 

 Jusqu'à présent, nous avons entraîné nos modèles
avec la descente de gradient stochastique en minibatch.
Cependant, lorsque nous avons implémenté l'algorithme,
nous nous sommes uniquement préoccupés des calculs impliqués
dans la *propagation vers l'avant* à travers le modèle.
Lorsqu'il s'agissait de calculer les gradients,
nous avons simplement invoqué la fonction de rétropropagation fournie par le cadre d'apprentissage profond.

Le calcul automatique des gradients (différenciation automatique) simplifie profondément
la mise en œuvre des algorithmes d'apprentissage profond.
Avant la différenciation automatique,
même de petites modifications apportées à des modèles compliqués nécessitaient
de recalculer à la main des dérivées compliquées.
De manière surprenante, les articles académiques devaient consacrer
de nombreuses pages à la dérivation des règles de mise à jour.
Bien que nous devions continuer à nous appuyer sur la différenciation automatique
pour pouvoir nous concentrer sur les parties intéressantes,
vous devez savoir comment ces gradients
sont calculés sous le capot
si vous voulez aller au-delà d'une compréhension superficielle
de l'apprentissage profond.

Dans cette section, nous plongeons en profondeur
dans les détails de la *propagation en arrière*
(plus communément appelée *rétropropagation*).
Pour donner un aperçu des techniques
et de leurs implémentations,
nous nous appuyons sur des mathématiques de base et des graphes de calcul.
Pour commencer, nous concentrons notre exposé sur
un MLP à une couche cachée
avec décroissance du poids ($\ell_2$ régularisation, qui sera décrite dans les chapitres suivants).

## Propagation vers l'avant

*La propagation vers l'avant* (ou *forward pass*) fait référence au calcul et au stockage
de variables intermédiaires (y compris les sorties)
pour un réseau neuronal afin
de passer de la couche d'entrée à la couche de sortie.
Nous allons maintenant étudier, étape par étape, la mécanique
d'un réseau neuronal à une couche cachée.
Cela peut sembler fastidieux, mais selon les mots éternels
du virtuose du funk James Brown,
il faut " payer le prix pour être le patron ".


Par souci de simplicité, supposons
que l'exemple d'entrée est $\mathbf{x}\in \mathbb{R}^d$
 et que notre couche cachée ne comporte pas de terme de biais.
Ici, la variable intermédiaire est :

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$ 

 où $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
 est le paramètre de poids de la couche cachée.
Après avoir fait passer la variable intermédiaire
$\mathbf{z}\in \mathbb{R}^h$ par la fonction d'activation
 $\phi$ 
 , nous obtenons notre vecteur d'activation caché de longueur $h$,

$$\mathbf{h}= \phi (\mathbf{z}).$$ 

 La sortie de la couche cachée $\mathbf{h}$
 est également une variable intermédiaire.
En supposant que les paramètres de la couche de sortie
possèdent uniquement un poids de
$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ ,
nous pouvons obtenir une variable de couche de sortie
avec un vecteur de longueur $q$:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

En supposant que la fonction de perte est $l$
 et que l'étiquette de l'exemple est $y$,
nous pouvons alors calculer le terme de perte
pour un seul exemple de données,

$$L = l(\mathbf{o}, y).$$ 

 Selon la définition de la régularisation $\ell_2$
 que nous introduirons plus tard,
étant donné l'hyperparamètre $\lambda$,
le terme de régularisation est

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$ 
 :eqlabel:`eq_forward-s` 

 où la norme de Frobenius de la matrice
est simplement la norme $\ell_2$ appliquée
après avoir aplati la matrice en un vecteur.
Enfin, la perte régularisée du modèle
sur un exemple de données donné est :

$$J = L + s.$$ 

 Nous faisons référence à $J$ comme à la *fonction objective*
dans la discussion suivante.


## Graphique de calcul de la propagation vers l'avant

Le tracé des *graphiques de calcul* nous aide à visualiser
les dépendances des opérateurs
et des variables au sein du calcul.
:numref:`fig_forward` contient le graphique associé
au réseau simple décrit ci-dessus,
où les carrés représentent les variables et les cercles les opérateurs.
Le coin inférieur gauche représente l'entrée
et le coin supérieur droit représente la sortie.
Remarquez que les directions des flèches
(qui illustrent le flux de données)
sont principalement vers la droite et vers le haut.

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## Backpropagation

*Backpropagation* fait référence à la méthode de calcul
du gradient des paramètres du réseau neuronal.
En bref, la méthode traverse le réseau en ordre inverse,
de la couche de sortie à la couche d'entrée,
selon la *règle de la chaîne* du calcul.
L'algorithme stocke toutes les variables intermédiaires
(dérivées partielles)
requises lors du calcul du gradient
par rapport à certains paramètres.
Supposons que nous ayons des fonctions
$\mathsf{Y}=f(\mathsf{X})$ 
 et $\mathsf{Z}=g(\mathsf{Y})$,
dans lesquelles l'entrée et la sortie
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ 
 sont des tenseurs de formes arbitraires.
En utilisant la règle de la chaîne,
nous pouvons calculer la dérivée
de $\mathsf{Z}$ par rapport à $\mathsf{X}$ via

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$ 

 Ici, nous utilisons l'opérateur $\text{prod}$
 pour multiplier ses arguments
après avoir effectué les opérations nécessaires,
telles que la transposition et la permutation des positions d'entrée,
.
Pour les vecteurs, c'est simple :
il s'agit simplement d'une multiplication matrice-matrice.
Pour les tenseurs de dimension supérieure,
nous utilisons la contrepartie appropriée.
L'opérateur $\text{prod}$ masque toute la surcharge de notation.

Rappelons que
les paramètres du réseau simple à une couche cachée,
dont le graphe de calcul est dans :numref:`fig_forward` ,
sont $\mathbf{W}^{(1)}$ et $\mathbf{W}^{(2)}$.
L'objectif de la rétropropagation est de
calculer les gradients $\partial J/\partial \mathbf{W}^{(1)}$
 et $\partial J/\partial \mathbf{W}^{(2)}$.
Pour ce faire, nous appliquons la règle de la chaîne
et calculons, à tour de rôle, le gradient de
chaque variable et paramètre intermédiaire.
L'ordre des calculs est inversé
par rapport à ceux effectués en propagation directe,
puisque nous devons commencer par le résultat du graphe de calcul
et progresser vers les paramètres.
La première étape consiste à calculer les gradients
de la fonction objectif $J=L+s$
 par rapport au terme de perte $L$
 et au terme de régularisation $s$.

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

Ensuite, nous calculons le gradient de la fonction objectif
par rapport à la variable de la couche de sortie $\mathbf{o}$
 selon la règle de la chaîne :

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Ensuite, nous calculons les gradients
du terme de régularisation
par rapport aux deux paramètres :

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Nous sommes maintenant en mesure de calculer le gradient
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ 
 des paramètres du modèle les plus proches de la couche de sortie.
L'utilisation de la règle de la chaîne donne :

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$ 
 :eqlabel:`eq_backprop-J-h` 

 Pour obtenir le gradient par rapport à $\mathbf{W}^{(1)}$
 , nous devons poursuivre la rétropropagation
le long de la couche de sortie vers la couche cachée.
Le gradient par rapport à la sortie de la couche cachée
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ est donné par


$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Comme la fonction d'activation $\phi$ s'applique par éléments,
le calcul du gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
 de la variable intermédiaire $\mathbf{z}$
 nécessite l'utilisation de l'opérateur de multiplication par éléments,
que nous désignons par $\odot$:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Enfin, nous pouvons obtenir le gradient
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ 
 des paramètres du modèle les plus proches de la couche d'entrée.
Selon la règle de la chaîne, nous obtenons

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## entrainement des réseaux neuronaux

Lors de l'entrainement des réseaux neuronaux,
la propagation avant et arrière dépendent l'une de l'autre.
En particulier, pour la propagation en avant,
nous traversons le graphe de calcul dans le sens des dépendances
et calculons toutes les variables sur son chemin.
Celles-ci sont ensuite utilisées pour la rétropropagation
où l'ordre de calcul sur le graphe est inversé.

Prenons l'exemple du réseau simple susmentionné pour l'illustrer.
D'une part,
le calcul du terme de régularisation :eqref:`eq_forward-s` 
 pendant la propagation en avant
dépend des valeurs actuelles des paramètres du modèle $\mathbf{W}^{(1)}$ et $\mathbf{W}^{(2)}$.
Elles sont données par l'algorithme d'optimisation selon la rétropropagation dans la dernière itération.
D'autre part,
le calcul du gradient pour le paramètre
:eqref:`eq_backprop-J-h` pendant la rétropropagation
dépend de la valeur actuelle de la sortie de la couche cachée $\mathbf{h}$,
qui est donnée par la propagation vers l'avant.


Par conséquent, lors de l'entrainement des réseaux de neurones, après l'initialisation des paramètres du modèle,
nous alternons la propagation directe avec la rétropropagation,
mettant à jour les paramètres du modèle à l'aide des gradients donnés par la rétropropagation.
Notez que la rétropropagation réutilise les valeurs intermédiaires stockées de la propagation directe pour éviter les calculs en double.
L'une des conséquences est que nous devons conserver
les valeurs intermédiaires jusqu'à ce que la rétropropagation soit terminée.
C'est également l'une des raisons pour lesquelles l'entraînement de
nécessite beaucoup plus de mémoire que la prédiction simple.
En outre, la taille de ces valeurs intermédiaires est à peu près
proportionnelle au nombre de couches du réseau et à la taille du lot.
Ainsi,
l'entraînement de réseaux plus profonds en utilisant des tailles de lots plus importantes
conduit plus facilement à des erreurs *out of memory*.


## Résumé

* La propagation directe calcule et stocke séquentiellement les variables intermédiaires dans le graphe de calcul défini par le réseau neuronal. Elle procède de la couche d'entrée à la couche de sortie.
* La propagation inverse calcule et stocke séquentiellement les gradients des variables et paramètres intermédiaires au sein du réseau neuronal dans l'ordre inverse.
* Lors de l'entrainement de modèles d'apprentissage profond, la propagation en avant et la propagation en arrière sont interdépendantes.
* L'entraînement nécessite beaucoup plus de mémoire que la prédiction.


## Exercices

1. Supposons que les entrées $\mathbf{X}$ d'une certaine fonction scalaire $f$ sont des matrices $n \times m$. Quelle est la dimensionnalité du gradient de $f$ par rapport à $\mathbf{X}$?
1. Ajoutez un biais à la couche cachée du modèle décrit dans cette section (vous n'avez pas besoin d'inclure le biais dans le terme de régularisation).
   1. Dessinez le graphe de calcul correspondant.
   1. Dérivez les équations de propagation avant et arrière.
1. Calculez l'empreinte mémoire pour l'entrainement et la prédiction dans le modèle décrit dans cette section.
1. Supposons que vous souhaitiez calculer les dérivées secondes. Qu'arrive-t-il au graphe de calcul ? Combien de temps pensez-vous que le calcul prendra ?
1. Supposons que le graphe de calcul soit trop grand pour votre GPU.
   1. Pouvez-vous le partitionner sur plus d'un GPU ?
   1. Quels sont les avantages et les inconvénients de l'entraînement sur un plus petit minilot ?

[Discussions](https://discuss.d2l.ai/t/102)
