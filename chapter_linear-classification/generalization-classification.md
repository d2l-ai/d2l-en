# Généralisation dans la classification

:label:`chap_classification_generalization` 

 

 Jusqu'à présent, nous nous sommes concentrés sur la manière d'aborder les problèmes de classification multiclasse
en formant des réseaux neuronaux (linéaires) à sorties multiples et des fonctions softmax.
En interprétant les sorties de notre modèle comme des prédictions probabilistes,
nous avons motivé et dérivé la fonction de perte d'entropie croisée,
qui calcule la probabilité logarithmique négative
que notre modèle (pour un ensemble fixe de paramètres)
attribue aux étiquettes réelles.
Enfin, nous mettons ces outils en pratique
en adaptant notre modèle à l'ensemble d'apprentissage.
Cependant, comme toujours, notre objectif est d'apprendre des modèles *généraux*,
tels qu'évalués empiriquement sur des données inédites (l'ensemble de test).
Une précision élevée sur l'ensemble d'apprentissage ne signifie rien.
Si chacune de nos entrées est unique
(et c'est effectivement vrai pour la plupart des ensembles de données à haute dimension),
nous pouvons atteindre une précision parfaite sur l'ensemble d'apprentissage
en mémorisant simplement l'ensemble de données à la première époque d'apprentissage,
et en recherchant ensuite l'étiquette chaque fois que nous voyons une nouvelle image.
Pourtant, la mémorisation des étiquettes exactes
associées aux exemples d'apprentissage exacts
ne nous indique pas comment classer les nouveaux exemples.
En l'absence d'indications supplémentaires, nous devrons peut-être nous rabattre sur
et deviner au hasard chaque fois que nous rencontrerons de nouveaux exemples.

Un certain nombre de questions brûlantes exigent une attention immédiate :
1. De combien d'exemples de test avons-nous besoin pour estimer précisément
 la précision de nos classificateurs sur la population sous-jacente ?
1. Que se passe-t-il si nous continuons à évaluer les modèles sur le même test à plusieurs reprises ?
1. Pourquoi devrions-nous nous attendre à ce que l'adaptation de nos modèles linéaires à l'ensemble d'apprentissage
 donne de meilleurs résultats que notre schéma de mémorisation naïve ?


Alors que :numref:`sec_generalization_basics` a présenté
les principes de base de l'overfitting et de la généralisation
dans le contexte de la régression linéaire,
ce chapitre va aller un peu plus loin,
présentant certaines des idées fondamentales
de la théorie de l'apprentissage statistique.
Il s'avère que nous pouvons souvent garantir la généralisation *a priori* :
pour de nombreux modèles,
et pour toute limite supérieure souhaitée
sur l'écart de généralisation $\epsilon$,
nous pouvons souvent déterminer un certain nombre d'échantillons requis $n$
 de sorte que si notre ensemble d'apprentissage contient au moins $n$
 échantillons, alors notre erreur empirique se situera
dans les limites de $\epsilon$ de l'erreur réelle,
*pour toute distribution générant des données*.
Malheureusement, il s'avère également que
, alors que ces types de garanties fournissent
un ensemble profond de blocs de construction intellectuelle,
ils sont d'une utilité pratique limitée
pour le praticien de l'apprentissage profond.
En bref, ces garanties suggèrent
qu'assurer la généralisation
des réseaux neuronaux profonds *a priori*
nécessite un nombre absurde d'exemples 
(peut-être des trillions ou plus),
même lorsque nous constatons que sur les tâches qui nous intéressent
que les réseaux neuronaux profonds généralisent généralement
remarquablement bien avec beaucoup moins d'exemples (des milliers).
C'est pourquoi les praticiens de l'apprentissage profond renoncent souvent à toute garantie a priori
,
employant plutôt des méthodes sur la base 
qu'elles ont bien généralisé
sur des problèmes similaires dans le passé,
et certifiant la généralisation *post hoc*
par des évaluations empiriques.
Lorsque nous arriverons à :numref:`chap_perceptrons` ,
nous reviendrons sur la généralisation
et fournirons une introduction légère
à la vaste littérature scientifique
qui a vu le jour pour tenter
d'expliquer pourquoi les réseaux neuronaux profonds se généralisent dans la pratique.

## L'ensemble de test

Puisque nous avons déjà commencé à utiliser les ensembles de test
comme méthode de référence
pour évaluer l'erreur de généralisation,
commençons par discuter
des propriétés de ces estimations d'erreur.
Concentrons-nous sur un classificateur fixe $f$,
sans nous soucier de la manière dont il a été obtenu.
Supposons en outre que nous possédions
un ensemble de données *fraîches* d'exemples $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$
 qui n'ont pas été utilisés pour entraîner le classificateur $f$.
L'erreur *empirique* de notre classificateur $f$ sur $\mathcal{D}$
 est simplement la fraction d'instances
pour lesquelles la prédiction $f(\mathbf{x}^{(i)})$
 est en désaccord avec l'étiquette réelle $y^{(i)}$,
et est donnée par l'expression suivante :

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

En revanche, l'erreur *de population*
est la fraction *attendue*
d'exemples dans la population sous-jacente
(une certaine distribution $P(X,Y)$ caractérisée
par une fonction de densité de probabilité $p(\mathbf{x},y)$
 pour laquelle notre classificateur est en désaccord
avec la véritable étiquette :

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Bien que $\epsilon(f)$ soit la quantité qui nous intéresse réellement,
nous ne pouvons pas l'observer directement,
tout comme nous ne pouvons pas observer directement
la taille moyenne dans une grande population
sans mesurer chaque personne.
Nous ne pouvons qu'estimer cette quantité en nous basant sur des échantillons.
Comme notre ensemble de test $\mathcal{D}$
 est statistiquement représentatif
de la population sous-jacente,
nous pouvons considérer $\epsilon_\mathcal{D}(f)$ comme un estimateur statistique
de l'erreur de population $\epsilon(f)$.
De plus, étant donné que notre quantité d'intérêt $\epsilon(f)$
 est une espérance (de la variable aléatoire $\mathbf{1}(f(X) \neq Y)$)
et que l'estimateur correspondant $\epsilon_\mathcal{D}(f)$
 est la moyenne de l'échantillon, 
l'estimation de l'erreur de population 
est simplement le problème classique de l'estimation de la moyenne,
dont vous vous souvenez peut-être de :numref:`sec_prob` .

Un important résultat classique de la théorie des probabilités
appelé le *théorème central limite* garantit
que lorsque nous possédons $n$ des échantillons aléatoires $a_1, ..., a_n$
 tirés d'une distribution quelconque avec une moyenne $\mu$ et un écart type $\sigma$,
lorsque le nombre d'échantillons $n$ s'approche de l'infini,
la moyenne de l'échantillon $\hat{\mu}$ tend approximativement
vers une distribution normale centrée
sur la vraie moyenne et avec un écart type $\sigma/\sqrt{n}$.
Cela nous apprend déjà quelque chose d'important :
lorsque le nombre d'exemples augmente,
notre erreur de test $\epsilon_\mathcal{D}(f)$
 devrait se rapprocher de l'erreur réelle $\epsilon(f)$
 à un rythme de $\mathcal{O}(1/\sqrt{n})$.
Ainsi, pour estimer notre erreur de test avec deux fois plus de précision,
nous devons collecter un ensemble de tests quatre fois plus grand.
Pour réduire notre erreur de test d'un facteur cent,
, nous devons collecter un ensemble de tests dix mille fois plus grand.
En général, un tel taux de $\mathcal{O}(1/\sqrt{n})$ 
 est souvent le meilleur que l'on puisse espérer en statistique.

Maintenant que nous connaissons le taux asymptotique
auquel notre erreur de test $\epsilon_\mathcal{D}(f)$ converge vers l'erreur réelle $\epsilon(f)$,
nous pouvons nous concentrer sur certains détails importants.
Rappelons que la variable aléatoire qui nous intéresse
$\mathbf{1}(f(X) \neq Y)$ 
 ne peut prendre que les valeurs $0$ et $1$
 et qu'il s'agit donc d'une variable aléatoire de Bernoulli,
caractérisée par un paramètre
indiquant la probabilité qu'elle prenne la valeur $1$.
Ici, $1$ signifie que notre classificateur a commis une erreur,
donc le paramètre de notre variable aléatoire
est en fait le taux d'erreur réel $\epsilon(f)$.
La variance $\sigma^2$ d'une variable de Bernoulli
dépend de son paramètre (ici, $\epsilon(f)$)
selon l'expression $\epsilon(f)(1-\epsilon(f))$.
Bien que $\epsilon(f)$ soit initialement inconnu,
nous savons qu'il ne peut pas être supérieur à $1$.
Une petite étude de cette fonction
révèle que notre variance est la plus élevée
lorsque le taux d'erreur réel est proche de $0.5$
 et peut être bien plus faible lorsqu'il est
proche de $0$ ou proche de $1$.
Cela nous indique que l'écart type asymptotique
de notre estimation $\epsilon_\mathcal{D}(f)$ de l'erreur $\epsilon(f)$
 (sur le choix des échantillons de test $n$ )
ne peut pas être supérieur à $\sqrt{0.25/n}$.

Si nous ignorons le fait que ce taux caractérise le comportement de
lorsque la taille de l'ensemble de test s'approche de l'infini
plutôt que lorsque nous possédons des échantillons finis,
cela nous indique que si nous voulons que notre erreur de test $\epsilon_\mathcal{D}(f)$
 se rapproche de l'erreur de la population $\epsilon(f)$
 de telle sorte qu'un écart-type corresponde
à un intervalle de $\pm 0.01$,
alors nous devons collecter environ 2500 échantillons.
Si nous voulons ajuster deux écarts-types
dans cet intervalle et être ainsi 95%
que $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$,
alors nous aurons besoin de 10000 échantillons !

Il s'avère que c'est la taille des ensembles de test
pour de nombreux repères populaires en apprentissage automatique.
Vous serez peut-être surpris d'apprendre que des milliers
d'articles sur l'apprentissage profond appliqué sont publiés chaque année
et que les améliorations du taux d'erreur de $0.01$ ou moins font l'objet d'une grande attention.
Bien sûr, lorsque les taux d'erreur sont beaucoup plus proches de $0$,
, une amélioration de $0.01$ peut effectivement être un gros problème.


L'une des caractéristiques de notre analyse jusqu'à présent
est qu'elle ne nous renseigne que sur l'asymptotique,
c'est-à-dire sur la façon dont la relation entre $\epsilon_\mathcal{D}$ et $\epsilon$
 évolue lorsque la taille de notre échantillon atteint l'infini.
Heureusement, comme notre variable aléatoire est bornée,
nous pouvons obtenir des limites valides pour les échantillons finis
en appliquant une inégalité due à Hoeffding (1963) :

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Résoudre la plus petite taille d'ensemble de données
qui nous permettrait de conclure
avec une confiance de 95 % que la distance $t$
 entre notre estimation $\epsilon_\mathcal{D}(f)$
 et le taux d'erreur réel $\epsilon(f)$
 ne dépasse pas $0.01$,
vous constaterez qu'environ $15000$ exemples sont nécessaires
par rapport aux $10000$ exemples suggérés
par l'analyse asymptotique ci-dessus.
Si vous approfondissez les statistiques
, vous constaterez que cette tendance est généralement valable.
Les garanties qui tiennent même dans des échantillons finis
sont généralement légèrement plus conservatrices.
Notez que dans l'ordre des choses,
ces chiffres ne sont pas si éloignés les uns des autres,
reflétant l'utilité générale
de l'analyse asymptotique pour nous donner
des chiffres approximatifs, même s'il ne s'agit pas de
garanties que nous pouvons présenter au tribunal.

## Réutilisation des jeux d'essai

Dans un certain sens, vous êtes maintenant prêt à réussir
à mener des recherches empiriques sur l'apprentissage automatique.
Presque tous les modèles pratiques sont développés
et validés sur la base des performances de l'ensemble de tests
et vous êtes maintenant un maître de l'ensemble de tests.
Pour tout classificateur fixe $f$,
vous savez évaluer son erreur de test $\epsilon_\mathcal{D}(f)$,
et savez précisément ce qui peut (et ne peut pas)
être dit sur son erreur de population $\epsilon(f)$.

Supposons que vous utilisiez ces connaissances
et que vous vous prépariez à former votre premier modèle $f_1$.
Sachant à quel point vous devez être confiant
dans la performance du taux d'erreur de votre classificateur
vous appliquez notre analyse ci-dessus pour déterminer
un nombre approprié d'exemples
à mettre de côté pour l'ensemble de test.
De plus, supposons que vous ayez pris à cœur les leçons de
:numref:`sec_generalization_basics` 
 et que vous ayez veillé à préserver le caractère sacré de l'ensemble de test
en effectuant toute votre analyse préliminaire,
le réglage des hyperparamètres et même la sélection
parmi plusieurs architectures de modèle concurrentes
sur un ensemble de validation.
Enfin, vous évaluez votre modèle $f_1$
 sur l'ensemble de test et présentez une estimation sans biais
de l'erreur de population
avec un intervalle de confiance associé.

Jusqu'à présent, tout semble bien se passer.
Cependant, cette nuit-là, vous vous réveillez à 3 heures du matin
avec une idée brillante pour une nouvelle approche de modélisation.
Le lendemain, vous codez votre nouveau modèle,
ajustez ses hyperparamètres sur l'ensemble de validation
et non seulement votre nouveau modèle $f_2$ fonctionne
mais son taux d'erreur semble être bien inférieur à celui de $f_1$.
Cependant, l'excitation de la découverte s'estompe soudainement
alors que vous vous préparez à l'évaluation finale.
Vous n'avez pas de jeu de test !

Même si le jeu de test original $\mathcal{D}$
 se trouve toujours sur votre serveur,
vous êtes maintenant confronté à deux problèmes majeurs.
Tout d'abord, lorsque vous avez collecté votre ensemble de test,
vous avez déterminé le niveau de précision requis
en partant du principe que vous évaluiez
un seul classificateur $f$.

Cependant, si vous vous lancez dans l'évaluation de plusieurs classificateurs $f_1, ..., f_k$
 sur le même ensemble de test,
vous devez prendre en compte le problème de la fausse découverte.
Auparavant, vous pouviez être sûr à 95%
que $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$
 pour un seul classificateur $f$
 et donc la probabilité d'un résultat erroné
était de seulement 5%.
Avec $k$ classificateurs dans le mélange,
il peut être difficile de garantir
qu'il n'y en a même pas un parmi
dont la performance du jeu de test est trompeuse.
Avec 20 classificateurs à l'étude,
, il se peut que vous n'ayez aucune puissance
pour exclure la possibilité
qu'au moins l'un d'entre eux
ait reçu un score trompeur.
Ce problème est lié aux tests d'hypothèses multiples,
qui, en dépit d'une vaste littérature en statistiques,
reste un problème persistant qui gangrène la recherche scientifique.


Si cela ne suffit pas à vous inquiéter,
il existe une raison particulière de se méfier
des résultats obtenus lors des évaluations ultérieures.
Rappelez-vous que notre analyse des performances de l'ensemble de test
reposait sur l'hypothèse que le classificateur
était choisi sans aucun contact avec l'ensemble de test
et que nous pouvions donc considérer l'ensemble de test
comme tiré au hasard de la population sous-jacente.
Ici, non seulement vous testez plusieurs fonctions, mais
la fonction suivante $f_2$ a été choisie
après que vous ayez observé les performances du jeu de test $f_1$.
Une fois que des informations du jeu de test ont été divulguées au modélisateur,
il ne peut plus jamais être un véritable jeu de test au sens strict.
Ce problème est appelé *overfitting adaptatif* et a récemment émergé
comme un sujet d'intérêt intense pour les théoriciens de l'apprentissage et les statisticiens
:cite:`dwork2015preserving` .
Heureusement, bien qu'il soit possible
de faire fuir toutes les informations d'un ensemble de retenue,
et que les pires scénarios théoriques soient sombres,
ces analyses peuvent être trop conservatrices.
Dans la pratique, veillez à créer de véritables ensembles de test,
à les consulter aussi rarement que possible,
à tenir compte des tests d'hypothèses multiples
lors de la communication des intervalles de confiance,
et à redoubler de vigilance
lorsque les enjeux sont importants et que la taille de votre ensemble de données est réduite.
Lors de l'exécution d'une série de défis de référence,
, il est souvent judicieux de conserver
plusieurs ensembles de test afin qu'après chaque tour,
l'ancien ensemble de test puisse être rétrogradé en ensemble de validation.





## Théorie de l'apprentissage statistique

D'emblée, les *jeux d'essai sont tout ce que nous avons vraiment*,
et pourtant ce fait semble étrangement insatisfaisant.
Tout d'abord, il est rare que nous possédions un *véritable ensemble de test*--sauf si
c'est nous qui créons l'ensemble de données,
quelqu'un d'autre a probablement déjà évalué
son propre classificateur sur notre prétendu "ensemble de test".
Et même lorsque nous avons la primeur,
nous nous retrouvons vite frustrés, souhaitant pouvoir
évaluer nos tentatives de modélisation ultérieures
sans le sentiment tenace
que nous ne pouvons pas faire confiance à nos chiffres.
De plus, même un véritable ensemble de test ne peut que nous dire *post hoc*
si un classificateur a en fait généralisé à la population,
et non si nous avons une raison de nous attendre *a priori*
à ce qu'il généralise.

Avec ces doutes à l'esprit,
vous êtes peut-être maintenant suffisamment préparé
pour comprendre l'attrait de la *théorie de l'apprentissage statistique*,
le sous-domaine mathématique de l'apprentissage automatique
dont les praticiens visent à élucider les
principes fondamentaux qui expliquent
pourquoi/quand les modèles formés sur des données empiriques
peuvent/seront généralisés à des données non vues.
Depuis plusieurs décennies, l'un des principaux objectifs
des chercheurs en apprentissage statistique
est de combler l'écart de généralisation,
reliant les propriétés de la classe de modèles,
le nombre d'échantillons dans l'ensemble de données.

Les théoriciens de l'apprentissage visent à limiter la différence
entre l'erreur *empirique* $\epsilon_\mathcal{S}(f_\mathcal{S})$
 d'un classificateur appris $f_\mathcal{S}$,
à la fois formé et évalué
sur l'ensemble de formation $\mathcal{S}$,
et l'erreur réelle $\epsilon(f_\mathcal{S})$
 de ce même classificateur sur la population sous-jacente.
Cela peut sembler similaire au problème d'évaluation
que nous venons de traiter, mais il y a une différence majeure.
Auparavant, le classificateur $f$ était fixe
et nous n'avions besoin d'un ensemble de données
qu'à des fins d'évaluation.
Et en effet, tout classifieur fixe se généralise :
son erreur sur un ensemble de données (non vues auparavant)
est une estimation non biaisée de l'erreur de la population.
Mais que pouvons-nous dire lorsqu'un classificateur
est formé et évalué sur le même ensemble de données ?
Pouvons-nous jamais être sûrs que l'erreur d'apprentissage
sera proche de l'erreur de test ?


Supposons que notre classificateur appris $f_\mathcal{S}$ doive être choisi
parmi un ensemble pré-spécifié de fonctions $\mathcal{F}$.
Rappelez-vous de notre discussion sur les ensembles de test
que s'il est facile d'estimer
l'erreur d'un seul classificateur,
les choses se compliquent lorsque nous commençons
à considérer des collections de classificateurs.
Même si l'erreur empirique
d'un classificateur (fixe)
sera proche de son erreur réelle
avec une forte probabilité,
une fois que nous considérons une collection de classificateurs,
nous devons nous inquiéter de la possibilité
que *juste un* classificateur dans l'ensemble
recevra une erreur mal estimée.
Le problème est que si un seul classificateur
de notre collection reçoit
une erreur faussement faible
, nous pourrions le choisir
et ainsi sous-estimer grossièrement
l'erreur de la population.
De plus, même pour les modèles linéaires,
parce que leurs paramètres sont évalués de manière continue,
nous choisissons généralement parmi
une classe infinie de fonctions ($|\mathcal{F}| = \infty$).

Une solution ambitieuse au problème
consiste à développer des outils analytiques
pour prouver la convergence uniforme, c'est-à-dire
qu'avec une forte probabilité,
le taux d'erreur empirique de chaque classificateur de la classe $f\in\mathcal{F}$
 convergera *simultanément* vers son taux d'erreur réel.
En d'autres termes, nous cherchons un principe théorique
qui nous permettrait d'affirmer que
avec une probabilité d'au moins $1-\delta$
 (pour une petite $\delta$)
aucun taux d'erreur de classificateur $\epsilon(f)$
 (parmi tous les classificateurs de la classe $\mathcal{F}$)
ne sera mal estimé par plus de
qu'une petite quantité $\alpha$.
Il est clair que nous ne pouvons pas faire de telles déclarations
pour toutes les classes de modèles $\mathcal{F}$.
Rappelez-vous la classe des machines à mémoriser
qui atteignent toujours l'erreur empirique $0$
 mais ne surpassent jamais la devinette aléatoire
sur la population sous-jacente.

En un sens, la classe des machines à mémoriser est trop flexible.
Un tel résultat de convergence uniforme ne pourrait pas exister.
D'autre part, un classificateur fixe est inutile - il
généralise parfaitement, mais ne s'adapte ni aux données d'apprentissage ni aux données de test
.
La question centrale de l'apprentissage
a donc été historiquement formulée comme un compromis
entre des classes de modèles plus flexibles (variance plus élevée)
qui s'adaptent mieux aux données d'apprentissage mais risquent d'être surajustées,
et des classes de modèles plus rigides (biais plus élevé)
qui généralisent bien mais risquent d'être sous-ajustées.
Une question centrale dans la théorie de l'apprentissage
a été de développer l'analyse mathématique
appropriée pour quantifier
où un modèle se situe le long de ce spectre,
et de fournir les garanties associées.

Dans une série d'articles fondamentaux,
Vapnik et Chervonenkis ont étendu
la théorie de la convergence
des fréquences relatives
à des classes plus générales de fonctions
:cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91` .
L'une des principales contributions de cette ligne de travail
est la dimension Vapnik-Chervonenkis (VC),
qui mesure (une notion de)
la complexité (flexibilité) d'une classe de modèles.
De plus, l'un de leurs principaux résultats limite
la différence entre l'erreur empirique
et l'erreur de population en fonction
de la dimension VC et du nombre d'échantillons :

$$P\left(R[p, f] - R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta
\ \text{ for }\ \alpha \geq c \sqrt{(\mathrm{VC} - \log \delta)/n}.$$

Ici, $\delta > 0$ est la probabilité que la limite soit violée,
$\alpha$ est la limite supérieure de l'écart de généralisation,
et $n$ est la taille de l'ensemble de données.
Enfin, $c > 0$ est une constante qui ne dépend
que de l'ampleur de la perte qui peut être encourue.
Une utilisation de la limite pourrait consister à introduire les valeurs souhaitées de
 $\delta$ et $\alpha$
 pour déterminer le nombre d'échantillons à collecter.
La dimension VC quantifie le plus grand
nombre de points de données pour lesquels nous pouvons attribuer
un étiquetage (binaire) arbitraire
et trouver pour chacun un modèle $f$ dans la classe
qui correspond à cet étiquetage.
Par exemple, les modèles linéaires sur des entrées $d$-dimensionnelles
ont une dimension VC $d+1$.
Il est facile de voir qu'une ligne peut attribuer
tout étiquetage possible à trois points en deux dimensions,
mais pas à quatre.
Malheureusement, la théorie a tendance à être
trop pessimiste pour les modèles plus complexes
et l'obtention de cette garantie nécessite généralement
beaucoup plus d'exemples que ce qui est réellement nécessaire
pour atteindre le taux d'erreur souhaité.
Notez également qu'en fixant la classe du modèle et $\delta$,
notre taux d'erreur décroît à nouveau
avec le taux habituel $\mathcal{O}(1/\sqrt{n})$.
Il semble peu probable que nous puissions faire mieux en termes de $n$.
Cependant, lorsque nous faisons varier la classe de modèle,
la dimension VC peut présenter
une image pessimiste
de l'écart de généralisation.





## Résumé

La façon la plus directe d'évaluer un modèle
est de consulter un ensemble de test composé de données non vues précédemment.
Les évaluations de l'ensemble de test fournissent une estimation non biaisée de l'erreur réelle
et convergent au taux souhaité $\mathcal{O}(1/\sqrt{n})$ à mesure que l'ensemble de test augmente.
Nous pouvons fournir des intervalles de confiance approximatifs
basés sur des distributions asymptotiques exactes
ou des intervalles de confiance valides d'échantillons finis
basés sur des garanties d'échantillons finis (plus conservatrices).
En effet, l'évaluation des ensembles de tests est le fondement
de la recherche moderne en apprentissage automatique.
Cependant, les jeux d'essai sont rarement de véritables jeux d'essai
(utilisés par plusieurs chercheurs, encore et encore).
Une fois que le même ensemble de tests est utilisé
pour évaluer plusieurs modèles,
le contrôle des fausses découvertes peut être difficile.
Cela peut poser d'énormes problèmes en théorie.
En pratique, l'importance du problème
dépend de la taille des ensembles de retenue en question
et du fait qu'ils sont simplement utilisés pour choisir les hyperparamètres
ou qu'ils laissent échapper des informations plus directement.
Quoi qu'il en soit, une bonne pratique consiste à créer de véritables ensembles de test (ou plusieurs)
et à être aussi conservateur que possible quant à la fréquence de leur utilisation.


Dans l'espoir de fournir une solution plus satisfaisante, les théoriciens de l'apprentissage statistique
ont développé des méthodes
pour garantir une convergence uniforme sur une classe de modèles.
Si, en effet, l'erreur empirique de chaque modèle
converge simultanément vers son erreur réelle,
alors nous sommes libres de choisir le modèle qui donne les meilleurs résultats
, en minimisant l'erreur d'apprentissage,
en sachant qu'il donnera également de bons résultats
sur les données retenues.
Il est essentiel que chacun de ces résultats dépende
d'une propriété de la classe de modèles.
Vladimir Vapnik et Alexey Chernovenkis
ont introduit la dimension VC,
présentant des résultats de convergence uniformes
qui sont valables pour tous les modèles d'une classe VC.
Les erreurs d'apprentissage de tous les modèles de la classe
sont (simultanément) garanties
pour être proches de leurs erreurs réelles,
et garanties pour se rapprocher
à des taux $\mathcal{O}(1/\sqrt{n})$.
Suite à la découverte révolutionnaire de la dimension VC,
de nombreuses mesures de complexité alternatives ont été proposées,
chacune facilitant une garantie de généralisation analogue.
Voir :cite:`boucheron2005theory` pour une discussion détaillée
de plusieurs manières avancées de mesurer la complexité des fonctions.
Malheureusement, alors que ces mesures de complexité
sont devenues des outils largement utiles en théorie statistique,
elles s'avèrent impuissantes
(dans leur application directe)
pour expliquer la généralisation des réseaux neuronaux profonds.
Les réseaux neuronaux profonds ont souvent des millions de paramètres (ou plus),
et peuvent facilement attribuer des étiquettes aléatoires à de grandes collections de points.
Néanmoins, ils se généralisent bien sur des problèmes pratiques
et, étonnamment, ils se généralisent souvent mieux,
lorsqu'ils sont plus grands et plus profonds,
malgré des dimensions VC plus importantes.
Dans le prochain chapitre, nous revisiterons la généralisation
dans le contexte de l'apprentissage profond.

## Exercices

1. Si nous souhaitons estimer l'erreur d'un modèle fixe $f$
 à l'intérieur de $0.0001$ avec une probabilité supérieure à 99,9 %,
 de combien d'échantillons avons-nous besoin ?
1. Supposons que quelqu'un d'autre possède un ensemble de tests étiquetés
 $\mathcal{D}$ et ne met à disposition que les entrées non étiquetées (caractéristiques).
  Supposons maintenant que vous ne puissiez accéder aux étiquettes de l'ensemble de test
 qu'en exécutant un modèle $f$ (aucune restriction sur la classe du modèle)
 sur chacune des entrées non étiquetées
 et en recevant l'erreur correspondante $\epsilon_\mathcal{D}(f)$.
  Combien de modèles devrez-vous évaluer
 avant d'avoir accès à l'ensemble de l'ensemble de test
 et de pouvoir ainsi donner l'impression d'avoir une erreur $0$,
 indépendamment de votre erreur réelle ?
1. Quelle est la dimension VC de la classe des polynômes d'ordre $5^\mathrm{th}$?
1. Quelle est la dimension VC des rectangles alignés sur l'axe des données bidimensionnelles ?

[Discussions](https://discuss.d2l.ai/t/6829)
