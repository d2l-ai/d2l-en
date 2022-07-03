
# Généralisation
:label:`sec_generalization_basics` 

 Prenons l'exemple de deux étudiants de l'université qui se préparent assidûment à leur examen final.
En général, cette préparation consiste à
s'entraîner et à tester leurs capacités
en passant des examens administrés les années précédentes.
Néanmoins, le fait d'avoir bien réussi les examens précédents ne garantit pas
qu'ils excelleront au moment crucial.
Imaginons par exemple un étudiant, Elephantine Ellie,
dont la préparation consiste entièrement
à mémoriser les réponses
aux questions des examens des années précédentes.
Même si Ellie était dotée
d'une mémoire éléphantesque,
et pouvait donc se souvenir parfaitement de la réponse
à n'importe quelle question *précédemment vue*,
elle pourrait néanmoins se figer
face à une nouvelle question (*précédemment non vue*).
À titre de comparaison, imaginez une autre étudiante,
Inductive Irene, dont les capacités de mémorisation sont comparativement faibles
,
mais qui a un don pour repérer les modèles.
Notez que si l'examen se composait réellement de
questions recyclées d'une année précédente,
Ellie surpasserait largement Irène.
Même si les modèles déduits par Irène
donnaient des prédictions exactes à 90 %,
ils ne pourraient jamais rivaliser avec le rappel de 100 % d'Ellie
.
Cependant, même si l'examen se composait
entièrement de nouvelles questions,
Irène pourrait maintenir sa moyenne de 90 %.

En tant que scientifiques de l'apprentissage automatique,
notre objectif est de découvrir des *modèles*.
Mais comment pouvons-nous être sûrs que nous avons
vraiment découvert un modèle *général*
et pas simplement mémorisé nos données ?
La plupart du temps, nos prédictions ne sont utiles
que si notre modèle découvre un tel modèle.
Nous ne voulons pas prédire le cours des actions d'hier, mais celui de demain.
Nous n'avons pas besoin de reconnaître
des maladies déjà diagnostiquées
pour des patients vus précédemment,
mais plutôt des maladies non diagnostiquées
chez des patients non vus précédemment.
Ce problème - comment découvrir des modèles qui *généralisent*--- est
le problème fondamental de l'apprentissage automatique,
et sans doute de toutes les statistiques.
Nous pourrions considérer ce problème comme l'une des facettes
d'une question bien plus vaste
qui englobe toute la science :
quand sommes-nous justifiés
de passer d'observations particulières
à des déclarations plus générales :cite:`popper2005logic` ?


Dans la vie réelle, nous devons ajuster des modèles
en utilisant une collection finie de données.
Les échelles typiques de ces données
varient énormément d'un domaine à l'autre.
Pour de nombreux problèmes médicaux importants,
nous ne pouvons accéder qu'à quelques milliers de points de données.
Pour l'étude des maladies rares,
nous pouvons avoir la chance d'accéder à des centaines de points.
En revanche, les plus grands ensembles de données publiques
composés de photographies étiquetées
(par exemple, ImageNet :cite:`Deng.Dong.Socher.ea.2009` ),
contiennent des millions d'images.
Et certaines collections d'images non étiquetées
telles que l'ensemble de données Flickr YFC100M
peuvent être encore plus grandes, contenant
plus de 100 millions d'images :cite:`thomee2016yfcc100m` .
Cependant, même à cette échelle extrême,
le nombre de points de données disponibles
reste infiniment petit
par rapport à l'espace de toutes les images possibles
à une résolution de 1 mégapixel.
Chaque fois que nous travaillons avec des échantillons finis,
nous devons garder à l'esprit le risque
d'ajuster nos données de formation,
pour découvrir ensuite que nous n'avons pas réussi
à découvrir un modèle généralisable.

Le phénomène d'ajustement plus proche de nos données d'apprentissage
que de la distribution sous-jacente est appelé *overfitting*,
et les techniques de lutte contre l'overfitting
sont souvent appelées *méthodes de régularisation*.
Bien que rien ne remplace une bonne introduction
à la théorie de l'apprentissage statistique (voir :cite:`Vapnik98,boucheron2005theory` ),
nous vous donnerons juste assez d'intuition pour commencer.
Nous reviendrons sur la généralisation dans de nombreux chapitres
tout au long du livre,
en explorant à la fois ce que l'on sait sur
les principes qui sous-tendent la généralisation
dans divers modèles,
et également les techniques heuristiques
qui se sont avérées (empiriquement)
produire une généralisation améliorée
sur des tâches d'intérêt pratique.



## Erreur de entrainement et erreur de généralisation


 Dans le cadre standard de l'apprentissage supervisé,
nous supposons que les données de entrainement et les données de test
sont tirées *indépendamment* de distributions *identiques*.
C'est ce qu'on appelle communément l'hypothèse *IID*.
Bien que cette hypothèse soit forte,
il convient de noter qu'en l'absence d'une telle hypothèse
, nous serions dans le pétrin.
Pourquoi devrions-nous croire que les données de formation
échantillonnées à partir de la distribution $P(X,Y)$
 devraient nous indiquer comment faire des prédictions sur
des données de test générées par une *distribution différente* $Q(X,Y)$?
Il s'avère que pour faire de tels sauts, il faut
des hypothèses fortes sur la façon dont $P$ et $Q$ sont liés.
Nous aborderons plus loin certaines hypothèses
qui permettent des changements de distribution
mais nous devons d'abord comprendre le cas IID,
où $P(\cdot) = Q(\cdot)$.

Pour commencer, nous devons faire la différence entre
l'erreur *d'apprentissage* $R_\text{emp}$,
qui est une *statistique*
calculée sur l'ensemble de données d'apprentissage,
et l'erreur *de généralisation* $R$,
qui est une *espérance* prise
par rapport à la distribution sous-jacente.
Vous pouvez considérer l'erreur de généralisation comme
ce que vous verriez si vous appliquiez votre modèle
à un flux infini d'exemples de données supplémentaires
tirés de la même distribution de données sous-jacente.
Formellement, l'erreur d'apprentissage est exprimée sous forme de *somme* (avec la même notation dans :numref:`sec_linear_regression` ) :

$$R_\text{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$ 

 
 tandis que l'erreur de généralisation est exprimée sous forme d'intégrale :

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] =
\int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Le problème est que nous ne pouvons jamais calculer
l'erreur de généralisation $R$ exactement.
Personne ne nous dit jamais la forme précise
de la fonction de densité $p(\mathbf{x}, y)$.
De plus, nous ne pouvons pas échantillonner un flux infini de points de données.
En pratique, nous devons donc *estimer* l'erreur de généralisation
en appliquant notre modèle à un ensemble de test indépendant
constitué d'une sélection aléatoire d'exemples
$\mathbf{X}'$ et d'étiquettes $\mathbf{y}'$
 qui ont été retenus dans notre ensemble d'apprentissage.
Cela consiste à appliquer la même formule
que pour le calcul de l'erreur d'apprentissage empirique
mais à un ensemble de test $\mathbf{X}', \mathbf{y}'$.


Il est important de noter que lorsque nous évaluons notre classificateur sur l'ensemble de test,
nous travaillons avec un classificateur *fixe*
(il ne dépend pas de l'échantillon de l'ensemble de test),
et donc l'estimation de son erreur
est simplement le problème de l'estimation de la moyenne.
Cependant, on ne peut pas en dire autant de
pour l'ensemble d'apprentissage.
Notez que le modèle que nous obtenons
dépend explicitement de la sélection de l'ensemble d'apprentissage
et donc l'erreur d'apprentissage sera en général
une estimation biaisée de l'erreur réelle
sur la population sous-jacente.
La question centrale de la généralisation
est donc de savoir quand nous devons nous attendre à ce que notre erreur d'apprentissage
soit proche de l'erreur de la population
(et donc de l'erreur de généralisation).

### Complexité du modèle

Dans la théorie classique, lorsque nous disposons de
modèles simples et de données abondantes,
les erreurs de entrainement et de généralisation ont tendance à être proches.
Cependant, lorsque nous travaillons avec
des modèles plus complexes et/ou moins d'exemples,
nous nous attendons à ce que l'erreur de entrainement diminue
mais que l'écart de généralisation augmente.
Cela ne devrait pas être surprenant.
Imaginez une classe de modèles si expressive que
pour n'importe quel ensemble de données d'exemples $n$,
nous pouvons trouver un ensemble de paramètres
qui peut parfaitement correspondre à des étiquettes arbitraires,
même si elles sont attribuées de manière aléatoire.
Dans ce cas, même si nous ajustons parfaitement nos données d'apprentissage,
comment pouvons-nous conclure quoi que ce soit sur l'erreur de généralisation ?
Pour autant que nous le sachions, notre erreur de généralisation
pourrait ne pas être meilleure qu'une supposition aléatoire.

En général, en l'absence de toute restriction sur la classe de notre modèle,
nous ne pouvons pas conclure, sur la base de l'ajustement des données d'apprentissage uniquement
, que notre modèle a découvert un modèle généralisable :cite:`vapnik1994measuring` .
D'autre part, si notre classe de modèle
n'était pas capable de s'adapter à des étiquettes arbitraires,
alors elle doit avoir découvert un modèle.
Les idées de la théorie de l'apprentissage sur la complexité des modèles
se sont inspirées des idées
de Karl Popper, un philosophe des sciences influent,
qui a formalisé le critère de falsifiabilité.
Selon Popper, une théorie
qui peut expliquer toutes les observations
n'est pas du tout une théorie scientifique !
Après tout, que nous a-t-elle dit sur le monde
si elle n'a pas exclu toute possibilité ?
En bref, ce que nous voulons, c'est une hypothèse
qui ne pourrait pas expliquer les observations
que nous pourrions faire
et qui serait néanmoins compatible
avec les observations que nous faisons effectivement.

Maintenant, ce qui constitue précisément une notion appropriée
de la complexité du modèle est une question complexe.
Souvent, les modèles comportant plus de paramètres
sont capables de s'adapter à un plus grand nombre
d'étiquettes attribuées arbitrairement.
Cependant, cela n'est pas nécessairement vrai.
Par exemple, les méthodes à noyau fonctionnent dans des espaces
avec un nombre infini de paramètres,
mais leur complexité est contrôlée
par d'autres moyens :cite:`scholkopf2002learning` .
Une notion de complexité qui s'avère souvent utile
est la gamme de valeurs que peuvent prendre les paramètres.
Dans ce cas, un modèle dont les paramètres sont autorisés
à prendre des valeurs arbitraires
serait plus complexe.
Nous reviendrons sur cette idée dans la section suivante,
lorsque nous introduirons la *décroissance de poids*,
votre première technique de régularisation pratique.
Notamment, il peut être difficile de comparer la complexité de
entre les membres de classes de modèles sensiblement différentes
(par exemple, les arbres de décision par rapport aux réseaux neuronaux).


À ce stade, nous devons souligner un autre point important
sur lequel nous reviendrons lors de la présentation des réseaux neuronaux profonds.
Lorsqu'un modèle est capable de s'adapter à des étiquettes arbitraires,
une faible erreur d'apprentissage n'implique pas nécessairement
une faible erreur de généralisation.
*Cependant, cela n'implique pas nécessairement
une erreur de généralisation élevée non plus !*
Tout ce que nous pouvons dire avec certitude, c'est que
une faible erreur d'apprentissage ne suffit pas
pour certifier une faible erreur de généralisation.
Les réseaux neuronaux profonds s'avèrent être de tels modèles :
alors qu'ils généralisent bien dans la pratique,
ils sont trop puissants pour nous permettre de conclure
sur la base de la seule erreur d'apprentissage.
Dans ce cas, nous devons nous appuyer davantage sur
nos données d'attente pour certifier la généralisation
après coup.
L'erreur sur les données d'attente, c'est-à-dire l'ensemble de validation,
est appelée l'erreur de validation *.

## Sous-adaptation ou sur-adaptation ?

Lorsque nous comparons les erreurs d'apprentissage et de validation,
nous devons être attentifs à deux situations courantes.
Tout d'abord, nous voulons faire attention aux cas
où l'erreur d'apprentissage et l'erreur de validation sont toutes deux substantielles
mais où il y a un petit écart entre elles.
Si le modèle n'est pas en mesure de réduire l'erreur d'apprentissage,
cela pourrait signifier que notre modèle est trop simple
(c'est-à-dire insuffisamment expressif)
pour capturer le modèle que nous essayons de modéliser.
En outre, étant donné que l'écart de *généralisation* ($R_\text{emp} - R$)
entre nos erreurs d'apprentissage et de généralisation est faible,
nous avons des raisons de penser que nous pourrions nous en sortir avec un modèle plus complexe.
Ce phénomène est connu sous le nom de *underfitting*.

D'autre part, comme nous l'avons vu plus haut,
nous voulons faire attention aux cas
où notre erreur d'apprentissage est significativement inférieure
à notre erreur de validation, ce qui indique un *overfitting* grave.
Notez que l'overfitting n'est pas toujours une mauvaise chose.
Dans l'apprentissage profond en particulier,
les meilleurs modèles prédictifs ont souvent des performances
bien meilleures sur les données d'entraînement que sur les données de validation.
En fin de compte, nous nous intéressons généralement à
pour réduire l'erreur de généralisation,
et ne nous préoccupons de l'écart que dans la mesure où
devient un obstacle à cette fin.
Notez que si l'erreur de entrainement est nulle,
l'écart de généralisation est précisément égal à l'erreur de généralisation
et nous ne pouvons progresser qu'en réduisant l'écart.

### Ajustement de courbes polynomiales
:label:`subsec_polynomial-curve-fitting` 

 Pour illustrer certaines intuitions classiques
sur l'overfitting et la complexité des modèles,
considérons ce qui suit :
étant donné des données d'apprentissage composées d'une seule caractéristique $x$
 et d'une étiquette correspondante à valeur réelle $y$,
nous essayons de trouver le polynôme de degré $d$

 $$\hat{y}= \sum_{i=0}^d x^i w_i$$ 

 pour estimer l'étiquette $y$.
Il s'agit simplement d'un problème de régression linéaire
où nos caractéristiques sont données par les puissances de $x$,
; les poids du modèle sont donnés par $w_i$,
et le biais est donné par $w_0$ depuis $x^0 = 1$ pour tout $x$.
Comme il s'agit simplement d'un problème de régression linéaire,
nous pouvons utiliser l'erreur quadratique comme fonction de perte.


Une fonction polynomiale d'ordre supérieur est plus complexe
qu'une fonction polynomiale d'ordre inférieur,
car le polynôme d'ordre supérieur a plus de paramètres
et la plage de sélection de la fonction modèle est plus large.
En fixant l'ensemble de données d'apprentissage,
les fonctions polynomiales d'ordre supérieur devraient toujours
atteindre une erreur d'apprentissage inférieure (au pire, égale)
par rapport aux polynômes de degré inférieur.
En fait, lorsque les exemples de données
ont chacun une valeur distincte de $x$,
une fonction polynomiale de degré
égal au nombre d'exemples de données
peut s'adapter parfaitement à l'ensemble d'apprentissage.
Nous visualisons la relation entre le degré polynomial (complexité du modèle)
et l'ajustement insuffisant par rapport à l'ajustement excessif dans :numref:`fig_capacity_vs_error` .

![Influence of model complexity on underfitting and overfitting](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`


### Taille de l'ensemble de données

Comme l'indique déjà la limite ci-dessus,
la taille de l'ensemble de données est une autre considération importante
à prendre en compte.
En fixant notre modèle, moins nous avons d'échantillons
dans l'ensemble de données d'apprentissage,
plus nous sommes susceptibles (et plus sévèrement)
de rencontrer un surajustement.
Lorsque nous augmentons la quantité de données d'apprentissage,
l'erreur de généralisation diminue généralement.
De plus, en général, un plus grand nombre de données ne fait jamais de mal.
Pour une tâche et une distribution de données fixes,
la complexité du modèle ne devrait pas augmenter
plus rapidement que la quantité de données.
Si l'on dispose de plus de données, on peut essayer
d'ajuster un modèle plus complexe.
En l'absence de données suffisantes, les modèles plus simples
peuvent être plus difficiles à battre.
Pour de nombreuses tâches, l'apprentissage profond
ne surpasse les modèles linéaires
que lorsque plusieurs milliers d'exemples d'apprentissage sont disponibles.
Le succès actuel de l'apprentissage profond
est en partie dû à l'abondance de jeux de données massifs
provenant des sociétés Internet, du stockage bon marché, des appareils connectés
et de la numérisation généralisée de l'économie.

## Sélection de modèles
:label:`subsec_generalization-model-selection` 

 Généralement, nous ne sélectionnons notre modèle final
qu'après avoir évalué plusieurs modèles
qui diffèrent de diverses manières
(différentes architectures, objectifs de formation,
caractéristiques sélectionnées, prétraitement des données,
taux d'apprentissage, etc.)
Le choix entre plusieurs modèles est judicieusement appelé
*sélection de modèles*.

En principe, nous ne devrions pas toucher à notre jeu de test
avant d'avoir choisi tous nos hyperparamètres.
Si nous utilisons les données de test dans le processus de sélection du modèle,
nous risquons de les surajuster.
Nous aurions alors de sérieux problèmes.
Si nous adaptons trop nos données d'apprentissage,
il y a toujours l'évaluation sur les données de test pour nous garder honnêtes.
Mais si nous adaptons trop les données de test, comment le saurions-nous ?
Voir :cite:`ong2005learning` pour voir comment
cela peut conduire à des résultats absurdes, même pour les modèles dont la complexité
peut être étroitement contrôlée.

Par conséquent, nous ne devrions jamais nous fier aux données de test pour sélectionner un modèle.
Et pourtant, nous ne pouvons pas non plus nous appuyer uniquement sur les données d'apprentissage
pour la sélection du modèle, car
nous ne pouvons pas estimer l'erreur de généralisation
sur les données mêmes que nous utilisons pour entraîner le modèle.


Dans les applications pratiques, les choses se compliquent.
Alors que l'idéal serait de ne toucher les données de test qu'une seule fois,
pour évaluer le meilleur modèle ou pour comparer
un petit nombre de modèles entre eux,
les données de test du monde réel sont rarement jetées après une seule utilisation.
Nous pouvons rarement nous permettre de créer un nouvel ensemble de tests pour chaque série d'expériences.

En fait, le recyclage des données de référence pendant des décennies
peut avoir un impact significatif sur le développement des algorithmes,
par exemple pour [image classification](https://paperswithcode.com/sota/image-classification-on-imagenet)
 et [optical character recognition](https://paperswithcode.com/sota/image-classification-on-mnist).

La pratique courante pour résoudre le problème de la *formation sur l'ensemble de test*
consiste à diviser nos données en trois,
en incorporant un *ensemble de validation*
en plus des ensembles de données de entrainement et de test.
Le résultat est une pratique obscure où les limites
entre les données de validation et de test sont d'une ambiguïté inquiétante.
Sauf indication contraire explicite, dans les expériences présentées dans ce livre
, nous travaillons en réalité avec ce qu'il convient d'appeler des données de entrainement et des données de validation
, sans véritables ensembles de test.
Par conséquent, la précision rapportée dans chaque expérience du livre est en réalité
la précision de validation et non une véritable précision d'ensemble de test.

### Validation croisée

Lorsque les données d'apprentissage sont rares,
il se peut que nous ne puissions même pas nous permettre de conserver
suffisamment de données pour constituer un ensemble de validation adéquat.
Une solution populaire à ce problème consiste à employer
$K$ *-fold cross-validation*.
Dans ce cas, les données de entrainement originales sont divisées en sous-ensembles non chevauchants $K$.
Ensuite, l'entrainement et la validation du modèle sont exécutées $K$ fois,
chaque fois en formant sur $K-1$ sous-ensembles et en validant
sur un sous-ensemble différent (celui qui n'a pas été utilisé pour l'entrainement dans ce tour).
Enfin, les erreurs de entrainement et de validation sont estimées
en faisant la moyenne des résultats des expériences $K$.



## Résumé

Cette section a exploré certains des fondements
de la généralisation dans l'apprentissage automatique.
Certaines de ces idées se compliquent
et deviennent contre-intuitives lorsque nous arrivons à des modèles plus profonds,
là, les modèles sont capables de sur-ajuster les données de manière inadéquate,
et les notions pertinentes de complexité
peuvent être à la fois implicites et contre-intuitives
(par exemple, des architectures plus grandes avec plus de paramètres
généralisant mieux).
Nous vous laissons avec quelques règles de base :

1. Utilisez des ensembles de validation (ou $K$*-fold cross-validation*) pour la sélection des modèles ;
1. Les modèles plus complexes nécessitent souvent plus de données ;
1. Les notions pertinentes de complexité comprennent à la fois le nombre de paramètres et la gamme de valeurs qu'ils sont autorisés à prendre ;
1. Toutes choses égales par ailleurs, plus de données conduit presque toujours à une meilleure généralisation ;
1. Toute cette discussion sur la généralisation est fondée sur l'hypothèse IID. Si nous assouplissons cette hypothèse, en permettant aux distributions de se déplacer entre les périodes de entrainement et de test, nous ne pouvons rien dire sur la généralisation sans une autre hypothèse (peut-être plus légère).


## Exercices

1. Quand pouvez-vous résoudre exactement le problème de la régression polynomiale ?
1. Donnez au moins cinq exemples où les variables aléatoires dépendantes rendent déconseillé de traiter le problème comme des données IID.
1. Pouvez-vous espérer voir un jour une erreur d'apprentissage nulle ? Dans quelles circonstances verriez-vous une erreur de généralisation nulle ?
1. Pourquoi la validation croisée $K$ est-elle très coûteuse à calculer ?
1. Pourquoi l'estimation de l'erreur par validation croisée de $K$ est-elle biaisée ?
1. La dimension VC est définie comme le nombre maximum de points qui peuvent être classés avec des étiquettes arbitraires $\{\pm 1\}$ par une fonction d'une classe de fonctions. Pourquoi ne serait-ce pas une bonne idée de mesurer la complexité de la classe de fonctions ? Indice : qu'en est-il de la magnitude des fonctions ?
1. Votre responsable vous donne un ensemble de données difficile sur lequel votre algorithme actuel n'est pas très performant. Comment pouvez-vous lui expliquer que vous avez besoin de plus de données ? Indice : vous ne pouvez pas augmenter les données, mais vous pouvez les diminuer.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
