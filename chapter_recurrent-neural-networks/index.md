# Réseaux neuronaux récurrents
:label:`chap_rnn` 

 Jusqu'à présent, nous nous sommes principalement concentrés sur les données de longueur fixe.
Lors de l'introduction de la régression linéaire et logistique 
dans :numref:`chap_linear` et :numref:`chap_classification` 
 et des perceptrons multicouches dans :numref:`chap_perceptrons` ,
nous nous sommes contentés de supposer que chaque vecteur de caractéristiques $\mathbf{x}_i$
 était constitué d'un nombre fixe de composantes $x_1, \dots, x_d$, 
où chaque caractéristique numérique $x_j$
 correspondait à un attribut particulier. 
Ces ensembles de données sont parfois appelés *tabulaires*,
car ils peuvent être organisés en tableaux, 
où chaque exemple $i$ a sa propre ligne,
et chaque attribut sa propre colonne. 
Il est important de noter qu'avec les données tabulaires, nous supposons rarement 
une structure particulière sur les colonnes. 

Par la suite, dans :numref:`chap_cnn` , 
nous sommes passés aux données d'image, où les entrées sont constituées 
des valeurs brutes des pixels à chaque coordonnée dans une image. 
Les données d'image ne correspondent pas à l'image 
d'un ensemble de données tabulaires typique. 
Nous avons dû faire appel à des réseaux de neurones convolutifs (CNN)
pour gérer la structure hiérarchique et les invariances.
Cependant, nos données étaient toujours de longueur fixe.
Chaque image Fashion-MNIST est représentée 
comme une grille $28 \times 28$ de valeurs de pixels.
De plus, notre objectif était de développer un modèle
qui n'examine qu'une seule image et qui produit ensuite une seule prédiction 
. 
Mais que faire face à une séquence d'images 
, comme dans une vidéo, 
ou lorsqu'il s'agit de produire 
une prédiction structurée séquentiellement,
comme dans le cas du sous-titrage d'images ? 

D'innombrables tâches d'apprentissage nécessitent de traiter des données séquentielles. 
Le sous-titrage d'images, la synthèse vocale et la génération de musique 
exigent tous que les modèles produisent des sorties constituées de séquences. 
Dans d'autres domaines, tels que la prédiction de séries temporelles, l'analyse vidéo 
et la recherche d'informations musicales, 
un modèle doit apprendre à partir d'entrées qui sont des séquences. 
Ces exigences sont souvent simultanées :
des tâches telles que la traduction de passages de texte
d'une langue naturelle à une autre, 
l'engagement d'un dialogue ou le contrôle d'un robot, 
exigent que les modèles ingèrent et produisent
des données structurées de manière séquentielle. 


Les réseaux neuronaux récurrents (RNN) sont des modèles d'apprentissage profond 
qui capturent la dynamique des séquences via des connexions 
*récurrentes*, que l'on peut considérer comme
des cycles dans le réseau de nœuds.
Cela peut sembler contre-intuitif au premier abord.
Après tout, c'est la nature feedforward des réseaux neuronaux
qui rend l'ordre de calcul sans ambiguïté.
Cependant, les arêtes récurrentes sont définies d'une manière précise
qui garantit qu'aucune ambiguïté de ce type ne peut survenir.
Les réseaux neuronaux récurrents sont *déroulés* à travers les étapes de la séquence,
avec les *mêmes* paramètres sous-jacents appliqués à chaque étape.
Alors que les connexions standard sont appliquées de manière *synchrone*
pour propager les activations de chaque couche 
à la couche suivante *au même pas de temps*,
les connexions récurrentes sont *dynamiques*,
transmettant des informations à travers des pas de temps adjacents. 
Comme le révèle la vue dépliée de :numref:`fig_unfolded-rnn` ,
les RNN peuvent être considérés comme des réseaux neuronaux à anticipation
où les paramètres de chaque couche (conventionnels et récurrents)
sont partagés entre les pas de temps. 


![On the left recurrent connections are depicted via cyclic edges. On the right, we unfold the RNN over sequence steps. Here, recurrent edges span adjacent sequence steps, while conventional connections are computed synchronously.](../img/unfolded-rnn.svg) 
:label:`fig_unfolded-rnn`


À l'instar des réseaux neuronaux au sens large,
, les RNN ont une longue histoire qui s'étend sur plusieurs disciplines,
ayant pour origine les modèles du cerveau popularisés
par les spécialistes des sciences cognitives et adoptés par la suite
comme outils de modélisation pratiques employés 
par la communauté de l'apprentissage automatique. 
Comme pour l'apprentissage profond au sens large,
ce livre adopte la perspective de l'apprentissage automatique,
en se concentrant sur les RNN en tant qu'outils pratiques qui ont gagné 
en popularité dans les années 2010 grâce à 
des résultats révolutionnaires sur des tâches aussi diverses que 
la reconnaissance de l'écriture manuscrite :cite:`graves2008novel` ,
la traduction automatique :cite:`Sutskever.Vinyals.Le.2014` ,
et la reconnaissance de diagnostics médicaux :cite:`Lipton.Kale.2016` . 
Nous indiquons au lecteur qui souhaite obtenir plus d'informations sur le sujet 
qu'il peut consulter une revue complète :cite:`Lipton.Berkowitz.Elkan.2015` accessible au public
.
Nous notons également que la séquentialité n'est pas propre aux RNN.
Par exemple, les CNN que nous avons déjà présentés
peuvent être adaptés pour traiter des données de longueur variable,
par exemple, des images de résolution variable.
De plus, les RNN ont récemment cédé une part de marché considérable
aux modèles transformateurs, 
qui seront abordés dans :numref:`chap_attention` .
Cependant, les RNN se sont imposés comme les modèles par défaut
pour le traitement de la structure séquentielle complexe dans l'apprentissage profond,
et restent à ce jour les modèles de base de la modélisation séquentielle.
Les histoires des RNN et de la modélisation séquentielle
sont inextricablement liées, et ce chapitre est autant 
un chapitre sur l'ABC des problèmes de modélisation séquentielle 
qu'un chapitre sur les RNN. 


Une idée clé a ouvert la voie à une révolution dans la modélisation des séquences.
Si les entrées et les cibles de nombreuses tâches fondamentales de l'apprentissage automatique 
ne peuvent pas être facilement représentées sous forme de vecteurs de longueur fixe, 
elles peuvent néanmoins souvent être représentées comme 
des séquences de longueur variable de vecteurs de longueur fixe. 
Par exemple, les documents peuvent être représentés comme des séquences de mots.
Les dossiers médicaux peuvent souvent être représentés comme des séquences d'événements 
(rencontres, médicaments, procédures, tests de laboratoire, diagnostics).
Les vidéos peuvent être représentées comme des séquences d'images fixes de longueur variable.


Bien que les modèles de séquences soient apparus dans d'innombrables domaines d'application,
la recherche fondamentale dans ce domaine a été principalement motivée 
par les progrès réalisés dans les tâches de base du traitement du langage naturel (NLP).
Ainsi, tout au long de ce chapitre, nous concentrerons 
notre exposé et nos exemples sur les données textuelles.
Si vous vous habituez à ces exemples, 
, l'application de ces modèles à d'autres modalités de données 
devrait être relativement simple. 
Dans les sections suivantes, nous introduisons la notation de base
pour les séquences et certaines mesures d'évaluation 
pour évaluer la qualité des sorties de modèles structurés de manière séquentielle. 
Ensuite, nous abordons les concepts de base d'un modèle de langage 
et utilisons cette discussion pour motiver nos premiers modèles RNN.
Enfin, nous décrivons la méthode de calcul des gradients 
lors de la rétropropagation à travers les RNN et explorons certains défis
souvent rencontrés lors de l'entraînement de ces réseaux,
motivant les architectures RNN modernes qui suivront 
dans :numref:`chap_modern_rnn` .

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
rnn-scratch
rnn-concise
bptt
```

