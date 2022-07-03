# Réseaux neuronaux linéaires
:label:`chap_linear` 

 Avant de nous soucier de rendre nos réseaux neuronaux profonds,
il sera utile d'implémenter quelques réseaux neuronaux peu profonds,
pour lesquels les entrées se connectent directement aux sorties.
Cela s'avère important pour plusieurs raisons.
Premièrement, plutôt que de se laisser distraire par des architectures compliquées,
nous pouvons nous concentrer sur les bases de l'entrainement des réseaux neuronaux,
y compris le paramétrage de la couche de sortie, la manipulation des données,
la spécification d'une fonction de perte et l'entrainement du modèle.
Deuxièmement, il se trouve que cette classe de réseaux peu profonds
comprend l'ensemble des modèles linéaires,
qui englobe de nombreuses méthodes classiques de prédiction statistique,
dont la régression linéaire et softmax.
Il est essentiel de comprendre ces outils classiques
car ils sont largement utilisés dans de nombreux contextes
et nous aurons souvent besoin de les utiliser comme référence
pour justifier l'utilisation d'architectures plus sophistiquées.
Ce chapitre se concentre sur la régression linéaire
et le chapitre suivant élargira notre répertoire de modélisation
en développant des réseaux de neurones linéaires pour la classification.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

