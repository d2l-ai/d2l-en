# Algorithmes d'optimisation
:label:`chap_optimization` 

Si vous avez lu le livre dans l'ordre jusqu'à ce point, vous avez déjà utilisé un certain nombre d'algorithmes d'optimisation pour former des modèles d'apprentissage profond.
Ce sont les outils qui nous ont permis de continuer à mettre à jour les paramètres du modèle et de minimiser la valeur de la fonction de perte, telle qu'évaluée sur l'ensemble d'entraînement. En effet, quiconque se contente de traiter l'optimisation comme une boîte noire permettant de minimiser des fonctions objectives dans un cadre simple peut se contenter de savoir qu'il existe une panoplie d'incantations d'une telle procédure (avec des noms tels que "SGD" et "Adam").

Pour bien faire, cependant, des connaissances plus approfondies sont nécessaires.
Les algorithmes d'optimisation sont importants pour l'apprentissage profond.
D'une part, l'entrainement d'un modèle d'apprentissage profond complexe peut prendre des heures, des jours, voire des semaines.
La performance de l'algorithme d'optimisation affecte directement l'efficacité de l'apprentissage du modèle.
D'autre part, comprendre les principes des différents algorithmes d'optimisation et le rôle de leurs hyperparamètres
nous permettra de régler les hyperparamètres de manière ciblée pour améliorer les performances des modèles d'apprentissage profond.

Dans ce chapitre, nous explorons en profondeur les algorithmes d'optimisation d'apprentissage profond les plus courants.
Presque tous les problèmes d'optimisation qui se posent en apprentissage profond sont *non convexes*.
Néanmoins, la conception et l'analyse d'algorithmes dans le contexte de problèmes *convexes* se sont avérées très instructives.
C'est pour cette raison que ce chapitre comprend une introduction à l'optimisation convexe et la preuve d'un algorithme très simple de descente de gradient stochastique sur une fonction objectif convexe.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```

