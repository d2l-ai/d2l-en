# Annexe : Mathématiques pour l'apprentissage profond
:label:`chap_appendix_math` 

 **Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), et auteurs de ce livre


 L'un des aspects merveilleux de l'apprentissage profond moderne est le fait qu'une grande partie de celui-ci peut être comprise et utilisée sans une compréhension totale des mathématiques qui le sous-tendent.  C'est le signe que le domaine arrive à maturité.  De même que la plupart des développeurs de logiciels n'ont plus besoin de se préoccuper de la théorie des fonctions calculables, les praticiens de l'apprentissage profond ne devraient pas non plus avoir à se préoccuper des fondements théoriques de l'apprentissage par probabilité maximale.

Mais nous ne sommes pas encore tout à fait au point.

En pratique, vous aurez parfois besoin de comprendre comment les choix architecturaux influencent le flux de gradient, ou les hypothèses implicites que vous faites en vous entraînant avec une certaine fonction de perte.  Vous aurez peut-être besoin de savoir ce que mesure l'entropie et comment elle peut vous aider à comprendre la signification exacte des bits par caractère dans votre modèle.  Toutes ces questions nécessitent une compréhension mathématique plus approfondie.

Cette annexe vise à vous fournir le contexte mathématique dont vous avez besoin pour comprendre la théorie fondamentale de l'apprentissage profond moderne, mais elle n'est pas exhaustive.  Nous commencerons par examiner l'algèbre linéaire de manière plus approfondie.  Nous développons une compréhension géométrique de tous les objets et opérations courants de l'algèbre linéaire qui nous permettront de visualiser les effets de diverses transformations sur nos données.  Un élément clé est le développement des bases des décompositions propres.

Nous développons ensuite la théorie du calcul différentiel jusqu'à ce que nous puissions comprendre pourquoi le gradient est la direction de la descente la plus raide, et pourquoi la rétro-propagation prend la forme qu'elle prend.  Le calcul intégral est ensuite abordé dans la mesure nécessaire pour soutenir notre sujet suivant, la théorie des probabilités.

Les problèmes rencontrés dans la pratique ne sont souvent pas certains, et nous avons donc besoin d'un langage pour parler des choses incertaines.  Nous passons en revue la théorie des variables aléatoires et les distributions les plus courantes afin de pouvoir discuter des modèles de manière probabiliste.  Ceci constitue la base du classificateur naïf de Bayes, une technique de classification probabiliste.

L'étude des statistiques est étroitement liée à la théorie des probabilités.  Bien que la statistique soit un domaine bien trop vaste pour qu'on puisse lui rendre justice dans une courte section, nous présenterons des concepts fondamentaux que tous les praticiens de l'apprentissage automatique devraient connaître, en particulier : l'évaluation et la comparaison des estimateurs, la réalisation de tests d'hypothèse et la construction d'intervalles de confiance.

Enfin, nous aborderons la théorie de l'information, qui est l'étude mathématique du stockage et de la transmission de l'information.  Elle fournit le langage de base grâce auquel nous pouvons discuter quantitativement de la quantité d'informations qu'un modèle détient sur un domaine du discours.

L'ensemble de ces éléments constitue le noyau des concepts mathématiques nécessaires pour s'engager sur la voie d'une compréhension approfondie de l'apprentissage profond.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```

