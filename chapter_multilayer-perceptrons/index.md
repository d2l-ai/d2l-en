# Perceptrons multicouches
:label:`chap_perceptrons` 

Dans ce chapitre, nous allons présenter votre premier réseau véritablement *profond*.
Les réseaux profonds les plus simples sont appelés *perceptrons multicouches*.
Ils sont constitués de plusieurs couches de neurones,
chacune étant entièrement connectée à celles de la couche inférieure
(dont elle reçoit les entrées)
et à celles de la couche supérieure (qu'elle influence à son tour).
Bien que la différenciation automatique
simplifie considérablement la mise en œuvre des algorithmes d'apprentissage profond,
nous allons nous plonger dans la manière dont ces gradients
sont calculés dans les réseaux profonds.

Nous serons alors prêts à
discuter des questions relatives à la stabilité numérique et à l'initialisation des paramètres
qui sont essentielles pour réussir l'entraînement des réseaux profonds.
Lorsque nous formons des modèles à haute capacité, nous courons le risque d'un surajustement (overfitting). Ainsi, nous allons
revisiter la régularisation et la généralisation
pour les réseaux profonds.
Tout au long de ce chapitre, notre objectif est de vous donner une bonne maîtrise non seulement des concepts mais aussi de la pratique de l'utilisation des réseaux profonds 
.
À la fin de ce chapitre, nous appliquons ce que nous avons introduit jusqu'à présent à un cas réel : la prédiction du prix des maisons.
Les questions relatives aux performances de calcul, à l'évolutivité et à l'efficacité
de nos modèles seront traitées dans les chapitres suivants.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```

