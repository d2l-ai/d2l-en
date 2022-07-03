# Généralisation dans l'apprentissage profond


 Dans :numref:`chap_linear` et :numref:`chap_classification` ,
nous avons abordé les problèmes de régression et de classification
en adaptant des modèles linéaires aux données d'apprentissage.
Dans les deux cas, nous avons fourni des algorithmes pratiques
pour trouver les paramètres qui maximisent
la vraisemblance des étiquettes d'apprentissage observées.
Puis, vers la fin de chaque chapitre,
nous avons rappelé que l'ajustement des données d'apprentissage
n'était qu'un objectif intermédiaire.
Notre véritable quête a toujours été de découvrir des modèles *généraux*
sur la base desquels nous pouvons faire des prédictions précises
même sur de nouveaux exemples tirés de la même population sous-jacente.
Les chercheurs en apprentissage automatique sont des *consommateurs* d'algorithmes d'optimisation.
Parfois, nous devons même développer de nouveaux algorithmes d'optimisation.
Mais en fin de compte, l'optimisation n'est qu'un moyen pour atteindre une fin.
À la base, l'apprentissage automatique est une discipline statistique
et nous ne souhaitons optimiser la perte de entrainement que dans la mesure où
un certain principe statistique (connu ou inconnu)
conduit les modèles résultants à généraliser au-delà de l'ensemble de formation.


Du côté positif, il s'avère que les réseaux neuronaux profonds
formés par descente de gradient stochastique se généralisent remarquablement bien
dans une myriade de problèmes de prédiction, couvrant la vision par ordinateur,
le traitement du langage naturel, les données de séries temporelles, les systèmes de recommandation,
les dossiers médicaux électroniques, le repliement des protéines,
l'approximation de la fonction de valeur dans les jeux vidéo
et les jeux de société, et d'innombrables autres domaines.
En revanche, si vous cherchez
un compte rendu direct
de l'histoire de l'optimisation
(pourquoi nous pouvons les adapter aux données d'apprentissage)
ou de l'histoire de la généralisation
(pourquoi les modèles résultants se généralisent à des exemples non vus),
alors vous devriez vous servir un verre.
Alors que nos procédures d'optimisation des modèles linéaires
et les propriétés statistiques des solutions
sont toutes deux bien décrites par un ensemble complet de théories,
notre compréhension de l'apprentissage profond
ressemble encore au Far West sur ces deux fronts.

La théorie et la pratique de l'apprentissage profond
évoluent rapidement sur ces deux fronts,
les théoriciens adoptant de nouvelles stratégies
pour expliquer ce qui se passe,
alors même que les praticiens continuent
d'innover à un rythme effréné,
construisant des arsenaux d'heuristiques pour l'entrainement des réseaux profonds
et un ensemble d'intuitions et de connaissances populaires
qui fournissent des conseils pour décider
des techniques à appliquer dans telle ou telle situation.

En résumé, la théorie de l'apprentissage profond
a donné lieu à des lignes d'attaque prometteuses et à des résultats fascinants épars,
mais elle est encore loin d'expliquer de manière exhaustive
à la fois (i) pourquoi nous sommes capables d'optimiser les réseaux neuronaux
et (ii) comment les modèles appris par descente de gradient
parviennent à se généraliser aussi bien, même pour des tâches à haute dimension.
Cependant, dans la pratique, le point (i) est rarement un problème
(nous pouvons toujours trouver des paramètres qui conviennent à toutes nos données d'apprentissage)
et, par conséquent, la compréhension de la généralisation est de loin le plus gros problème.
D'autre part, même en l'absence du confort d'une théorie scientifique cohérente,
les praticiens ont développé une large collection de techniques
qui peuvent vous aider à produire des modèles qui généralisent bien dans la pratique.
Bien qu'aucun résumé lapidaire ne puisse rendre justice
au vaste sujet de la généralisation dans l'apprentissage profond,
et bien que l'état général de la recherche soit loin d'être résolu,
nous espérons, dans cette section, présenter un large aperçu
de l'état de la recherche et de la pratique.


## Réexamen de l'overfitting et de la régularisation

Rappelons que notre approche de l'entrainement des modèles d'apprentissage automatique
se compose généralement de deux phases : (i) ajustement des données de entrainement ;
et (ii) estimation de l'erreur de *généralisation*
(l'erreur réelle sur la population sous-jacente)
en évaluant le modèle sur des données d'attente.
La différence entre notre ajustement sur les données d'apprentissage
et notre ajustement sur les données de test est appelée *écart de généralisation*
et lorsque l'écart de généralisation est important,
nous disons que nos modèles sont *surajustés* aux données d'apprentissage.
Dans les cas extrêmes d'ajustement excessif,
nous pourrions ajuster exactement les données d'apprentissage,
même si l'erreur de test reste importante.
Et dans la vision classique,
l'interprétation est que nos modèles sont trop complexes,
ce qui exige que nous réduisions soit le nombre de caractéristiques,
le nombre de paramètres non nuls appris,
ou la taille des paramètres tels que quantifiés.
Rappelez-vous le graphique de la complexité du modèle en fonction de la perte
(:numref:`fig_capacity_vs_error` )
de :numref:`sec_generalization_basics` .


 Cependant, l'apprentissage profond complique ce tableau de manière contre-intuitive.
Tout d'abord, pour les problèmes de classification,
nos modèles sont généralement assez expressifs
pour s'adapter parfaitement à chaque exemple d'apprentissage,
même dans des ensembles de données composés de millions de
:cite:`zhang2021understanding` .
Dans l'image classique, nous pourrions penser
que ce paramètre se situe à l'extrême droite
de l'axe de complexité du modèle,
et que toute amélioration de l'erreur de généralisation
doit passer par une régularisation,
soit en réduisant la complexité de la classe du modèle,
soit en appliquant une pénalité, limitant sévèrement
l'ensemble des valeurs que nos paramètres peuvent prendre.
Mais c'est là que les choses commencent à devenir étranges.

Étrangement, pour de nombreuses tâches d'apprentissage profond
(par exemple, la reconnaissance d'images et la classification de textes)
, nous choisissons généralement parmi des architectures de modèles,
qui peuvent toutes atteindre une perte d'apprentissage arbitrairement faible
(et une erreur d'apprentissage nulle).
Puisque tous les modèles considérés atteignent une erreur d'apprentissage nulle,
*la seule possibilité de gains supplémentaires est de réduire l'overfitting*.
Plus étrange encore, il arrive souvent que
bien que les données d'apprentissage soient parfaitement adaptées,
nous pouvons en fait *réduire l'erreur de généralisation*
en rendant le modèle *encore plus expressif*,
par exemple en ajoutant des couches, des nœuds ou en effectuant un apprentissage
pour un plus grand nombre d'époques.
Plus étrange encore, le modèle reliant l'écart de généralisation
à la *complexité* du modèle (telle que représentée, par exemple,
dans la profondeur ou la largeur des réseaux)
peut être non monotone,
avec une plus grande complexité qui nuit au début
mais qui aide ensuite dans un modèle dit de "double-descente"
:cite:`nakkiran2021deep` .
Ainsi, le praticien de l'apprentissage profond dispose d'un ensemble d'astuces,
dont certaines semblent restreindre le modèle d'une certaine manière
et d'autres qui semblent le rendre encore plus expressif,
et qui, dans un certain sens, sont toutes appliquées pour atténuer le surajustement.

Pour compliquer encore les choses,
alors que les garanties fournies par la théorie classique de l'apprentissage
peuvent être conservatrices même pour les modèles classiques,
elles semblent impuissantes à expliquer pourquoi c'est
que les réseaux de neurones profonds généralisent en premier lieu.
Étant donné que les réseaux neuronaux profonds sont capables de s'adapter à
des étiquettes arbitraires, même pour de grands ensembles de données,
et malgré l'utilisation de méthodes familières comme $\ell_2$ la régularisation,
les limites de généralisation traditionnelles basées sur la complexité,
par exemple, celles basées sur la dimension VC
ou la complexité de Rademacher d'une classe d'hypothèses
ne peuvent pas expliquer pourquoi les réseaux neuronaux généralisent.

## Inspiration de la non-paramétrie

En abordant l'apprentissage profond pour la première fois,
il est tentant de les considérer comme des modèles paramétriques.
Après tout, les modèles *font* des millions de paramètres.
Lorsque nous mettons à jour les modèles, nous mettons à jour leurs paramètres.
Lorsque nous sauvegardons les modèles, nous écrivons leurs paramètres sur le disque.
Cependant, les mathématiques et l'informatique sont truffées
de changements de perspective contre-intuitifs,
et de surprenants isomorphismes entre des problèmes apparemment différents.
Bien que les réseaux neuronaux aient clairement des *paramètres,
d'une certaine manière, il peut être plus fructueux
de penser qu'ils se comportent comme des modèles non paramétriques.
Qu'est-ce qui fait précisément qu'un modèle est non paramétrique ?
Bien que le nom couvre un ensemble d'approches diverses,
un thème commun est que les méthodes non paramétriques
ont tendance à avoir un niveau de complexité qui augmente
avec la quantité de données disponibles.

L'exemple le plus simple de modèle non paramétrique
est l'algorithme du plus proche voisin $k$(nous aborderons plus tard d'autres modèles non paramétriques, notamment dans :numref:`sec_nadaraya-watson` ).
Ici, au moment de la formation,
l'apprenant mémorise simplement l'ensemble de données.
Puis, au moment de la prédiction,
lorsqu'il est confronté à un nouveau point $\mathbf{x}$,
l'apprenant recherche les $k$ plus proches voisins
(les $k$ points $\mathbf{x}_i'$ qui minimisent
une certaine distance $d(\mathbf{x}, \mathbf{x}_i')$).
Lorsque $k=1$, cet algorithme est appelé 1-nearest neighbors,
et l'algorithme atteindra toujours une erreur d'apprentissage de zéro.
Toutefois, cela ne signifie pas que l'algorithme ne généralisera pas.
En fait, il s'avère que sous certaines conditions légères,
l'algorithme des 1 plus proches voisins est cohérent
(il converge finalement vers le prédicteur optimal).


Notez que l'algorithme du 1 plus proche voisin exige que nous spécifiions
une certaine fonction de distance $d$, ou, de manière équivalente,
une certaine fonction de base à valeur vectorielle $\phi(\mathbf{x})$
 pour caractériser nos données.
Quel que soit le choix de la métrique de distance,
nous obtiendrons une erreur d'apprentissage nulle
et finirons par atteindre un prédicteur optimal,
mais différentes métriques de distance $d$
 codent différents biais inductifs
et avec une quantité finie de données disponibles
produiront différents prédicteurs.
Les différents choix de la métrique de distance $d$
 représentent différentes hypothèses sur les modèles sous-jacents
et les performances des différents prédicteurs
dépendront de la compatibilité des hypothèses
avec les données observées.

Dans un sens, comme les réseaux neuronaux sont sur-paramétrés,
possédant beaucoup plus de paramètres que nécessaire pour s'adapter aux données de formation,
ils ont tendance à *interpoler* les données de entrainement (en les adaptant parfaitement)
et se comportent donc, d'une certaine manière, davantage comme des modèles non paramétriques.
Des recherches théoriques plus récentes ont établi
un lien profond entre les grands réseaux neuronaux
et les méthodes non paramétriques, notamment les méthodes à noyau.
En particulier, :cite:`Jacot.Grabriel.Hongler.2018` 
 ont démontré qu'à la limite, lorsque les perceptrons multicouches
avec des poids initialisés de manière aléatoire croissent à l'infini,
ils deviennent équivalents aux méthodes à noyau (non paramétriques)
pour un choix spécifique de la fonction de noyau
(essentiellement, une fonction de distance),
qu'ils appellent le noyau tangent neuronal.
Bien que les modèles actuels de noyau tangent neuronal n'expliquent peut-être pas entièrement
le comportement des réseaux profonds modernes,
leur succès en tant qu'outil analytique
souligne l'utilité de la modélisation non paramétrique
pour comprendre le comportement des réseaux profonds surparamétrés.


## Apprentissage précoce et arrêt précoce

Alors que les réseaux neuronaux profonds sont capables de s'adapter à des étiquettes arbitraires,
même lorsque les étiquettes sont attribuées de manière incorrecte ou aléatoire
(:cite:`zhang2021understanding` ),
cette capacité n'apparaît qu'après de nombreuses itérations d'apprentissage.
Une nouvelle série de travaux (:cite:`Rolnick.Veit.Belongie.Shavit.2017` )
a révélé que dans le cadre d'un bruit d'étiquette,
les réseaux neuronaux ont tendance à s'adapter d'abord aux données proprement étiquetées
et seulement ensuite à interpoler les données mal étiquetées.
De plus, il a été établi que ce phénomène
se traduit directement par une garantie de généralisation :
lorsqu'un modèle a ajusté les données proprement étiquetées
mais pas les exemples étiquetés au hasard inclus dans l'ensemble d'apprentissage,
il a en fait généralisé (:cite:`Garg.Balakrishnan.Kolter.Lipton.2021` ).

Ensemble, ces résultats contribuent à motiver l'arrêt *précoce*,
une technique classique de régularisation des réseaux neuronaux profonds.
Ici, plutôt que de contraindre directement les valeurs des poids,
on contraint le nombre d'époques d'apprentissage.
La méthode la plus courante pour déterminer le critère d'arrêt
consiste à surveiller l'erreur de validation tout au long de la formation
(généralement en vérifiant une fois après chaque époque)
et à interrompre l'entrainement lorsque l'erreur de validation
n'a pas diminué de plus d'une petite quantité $\epsilon$
 pendant un certain nombre d'époques.
Ceci est parfois appelé un critère de *patience*.
Outre le fait qu'il peut conduire à une meilleure généralisation,
dans le cadre d'étiquettes bruyantes,
un autre avantage de l'arrêt précoce est le gain de temps.
Une fois que le critère de patience est atteint, on peut mettre fin à la formation.
Pour les grands modèles qui peuvent nécessiter des jours d'apprentissage
simultanément sur 8 GPU ou plus,
un arrêt précoce bien réglé peut faire gagner des jours aux chercheurs
et peut faire économiser plusieurs milliers de dollars à leurs employeurs.

Notamment, lorsqu'il n'y a pas de bruit d'étiquette et que les ensembles de données sont *réalisables*
(les classes sont vraiment séparables, par exemple, distinguer les chats des chiens),
l'arrêt précoce a tendance à ne pas entraîner d'améliorations significatives de la généralisation.
En revanche, lorsqu'il y a du bruit dans l'étiquette,
ou une variabilité intrinsèque de l'étiquette
(par exemple, la prédiction de la mortalité chez les patients),
l'arrêt précoce est crucial.
Former des modèles jusqu'à ce qu'ils interpolent des données bruyantes est généralement une mauvaise idée.


## Méthodes de régularisation classiques pour les réseaux profonds

Dans :numref:`chap_linear` , nous avons décrit
plusieurs techniques de régularisation classiques
pour limiter la complexité de nos modèles.
En particulier, :numref:`sec_weight_decay` 
 a présenté une méthode appelée weight decay,
qui consiste à ajouter un terme de régularisation à la fonction de perte
pour pénaliser les grandes valeurs des poids.
Selon la norme de poids pénalisée
, cette technique est connue sous le nom de régularisation ridge (pour la pénalité $\ell_2$ )
ou de régularisation lasso (pour une pénalité $\ell_1$ ).
Dans l'analyse classique de ces régularisateurs,
on considère qu'ils restreignent les valeurs
que les poids peuvent prendre suffisamment
pour empêcher le modèle de s'adapter à des étiquettes arbitraires.

Dans les implémentations d'apprentissage profond,
la décroissance des poids reste un outil populaire.
Cependant, les chercheurs ont remarqué
que les forces typiques de la régularisation $\ell_2$
 sont insuffisantes pour empêcher les réseaux
d'interpoler les données
(:cite:`zhang2021understanding` )
et donc les avantages s'ils sont interprétés
comme une régularisation pourraient n'avoir de sens
qu'en combinaison avec les critères d'arrêt précoce.
En l'absence d'arrêt précoce, il est possible
que, tout comme le nombre de couches
ou le nombre de nœuds (dans l'apprentissage profond)
ou la métrique de distance (dans le voisin le plus proche),
ces méthodes puissent conduire à une meilleure généralisation
non pas parce qu'elles limitent de manière significative
la puissance du réseau neuronal
mais plutôt parce qu'elles encodent d'une manière ou d'une autre des biais inductifs
qui sont mieux compatibles avec les modèles
trouvés dans les ensembles de données d'intérêt.
Ainsi, les régularisateurs classiques restent populaires
dans les implémentations d'apprentissage profond,
même si la justification théorique
de leur efficacité peut être radicalement différente.

Notamment, les chercheurs en apprentissage profond se sont également appuyés sur
sur des techniques d'abord popularisées
dans des contextes de régularisation classique,
comme l'ajout de bruit aux entrées du modèle.
Dans la section suivante, nous présenterons
la fameuse technique d'abandon
(inventée par :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` ),
qui est devenue un pilier de l'apprentissage profond,
même si la base théorique de son efficacité
reste tout aussi mystérieuse.


## Résumé

Contrairement aux modèles linéaires classiques,
qui ont tendance à avoir moins de paramètres que d'exemples,
les réseaux profonds ont tendance à être sur-paramétrés,
et pour la plupart des tâches, ils sont capables
de s'adapter parfaitement à l'ensemble d'apprentissage.
Ce *régime d'interpolation* remet en question
de nombreuses intuitions tenaces.
Sur le plan fonctionnel, les réseaux neuronaux ressemblent à des modèles paramétriques.
Mais les considérer comme des modèles non paramétriques
peut parfois être une source d'intuition plus fiable.
Étant donné que tous les réseaux profonds considérés
sont souvent capables de s'adapter à toutes les étiquettes d'entraînement,
presque tous les gains doivent provenir de l'atténuation de l'overfitting
(combler le *gap de généralisation*).
Paradoxalement, les interventions
qui réduisent l'écart de généralisation
semblent parfois augmenter la complexité du modèle
et à d'autres moments, elles semblent diminuer la complexité.
Cependant, ces méthodes diminuent rarement la complexité
suffisamment pour que la théorie classique
puisse expliquer la généralisation des réseaux profonds,
et *pourquoi certains choix conduisent à une meilleure généralisation*
reste pour la plupart une question ouverte massive
malgré les efforts concertés de nombreux chercheurs brillants.


## Exercices

1. Dans quel sens les mesures traditionnelles basées sur la complexité ne parviennent-elles pas à rendre compte de la généralisation des réseaux neuronaux profonds ?
1. Pourquoi l'arrêt *précoce* peut-il être considéré comme une technique de régularisation ?
1. Comment les chercheurs déterminent-ils généralement les critères d'arrêt ?
1. Quel est le facteur important qui semble différencier les cas où l'arrêt précoce entraîne une amélioration importante de la généralisation ?
1. Au-delà de la généralisation, décrivez un autre avantage de l'arrêt précoce.

[Discussions](https://discuss.d2l.ai/t/7473)
