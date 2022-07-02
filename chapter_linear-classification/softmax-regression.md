# Softmax Regression
:label:`sec_softmax` 

 Dans :numref:`sec_linear_regression` , nous avons présenté la régression linéaire,
en travaillant sur des implémentations à partir de zéro dans :numref:`sec_linear_scratch` 
 et à nouveau en utilisant les API de haut niveau d'un cadre d'apprentissage profond
dans :numref:`sec_linear_concise` pour faire le gros du travail.

La régression est le marteau que nous utilisons lorsque
nous voulons répondre aux questions *combien ?* ou *combien ?*.
Si vous voulez prédire le nombre de dollars (prix)
auquel une maison sera vendue,
ou le nombre de victoires d'une équipe de baseball,
ou le nombre de jours d'hospitalisation d'un patient
avant sa sortie,
alors vous recherchez probablement un modèle de régression.
Cependant, même au sein des modèles de régression,
, il existe des distinctions importantes.
Par exemple, le prix d'une maison
ne sera jamais négatif et les changements peuvent souvent être *relatifs* à son prix de base.
À ce titre, il peut être plus efficace de régresser
sur le logarithme du prix.
De même, le nombre de jours qu'un patient passe à l'hôpital
est une variable aléatoire *discrète et non négative*.
En tant que telle, la méthode des moindres carrés moyens n'est peut-être pas non plus une approche idéale.
Ce type de modélisation temporelle
s'accompagne d'une foule d'autres complications qui sont traitées
dans un sous-domaine spécialisé appelé *modélisation de survie*.


L'objectif n'est pas de vous submerger, mais simplement de vous faire comprendre, à l'adresse
, que l'estimation ne se limite pas à la simple minimisation des erreurs quadratiques.
Et plus largement, l'apprentissage supervisé ne se limite pas à la régression.
Dans cette section, nous nous concentrons sur les problèmes de *classification*
où nous mettons de côté les questions *combien ?*
et nous nous concentrons plutôt sur les questions *quelle catégorie ?*.



* Cet email doit-il être placé dans le dossier spam ou dans la boîte de réception ?
* Ce client est-il plus susceptible de s'inscrire
 ou de ne pas s'inscrire à un service d'abonnement ?
* Cette image représente-t-elle un âne, un chien, un chat ou un coq ?
* Quel film Aston est-il le plus susceptible de regarder ensuite ?
* Quelle section du livre allez-vous lire ensuite ?

De manière familière, les praticiens de l'apprentissage automatique
surchargent le mot *classification*
pour décrire deux problèmes subtilement différents :
(i) ceux où nous ne sommes intéressés que par
des affectations dures d'exemples à des catégories (classes) ;
et (ii) ceux où nous souhaitons effectuer des affectations douces,
c'est-à-dire évaluer la probabilité que chaque catégorie s'applique.
La distinction a tendance à s'estomper, en partie,
parce que souvent, même lorsque nous ne nous intéressons qu'aux affectations fermes,
nous utilisons toujours des modèles qui effectuent des affectations souples.

Qui plus est, il existe des cas où plus d'une étiquette peut être vraie.
Par exemple, un article d'actualité peut couvrir simultanément
les thèmes du divertissement, des affaires et du vol spatial,
mais pas ceux de la médecine ou du sport.
Par conséquent, le classer dans l'une des catégories ci-dessus
ne serait pas très utile.
Ce problème est communément appelé [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification).
Voir :cite:`Tsoumakas.Katakis.2007` pour un aperçu
et :cite:`Huang.Xu.Yu.2015` 
 pour un algorithme efficace lors du balisage des images.

## Classification
:label:`subsec_classification-problem` 

 Pour nous mettre dans le bain, commençons par
un problème simple de classification d'images.
Ici, chaque entrée consiste en une image en niveaux de gris $2\times2$.
Nous pouvons représenter chaque valeur de pixel par un seul scalaire,
, ce qui nous donne quatre caractéristiques $x_1, x_2, x_3, x_4$.
De plus, supposons que chaque image appartient à une
des catégories "chat", "poulet" et "chien".

Ensuite, nous devons choisir comment représenter les étiquettes.
Deux choix évidents s'offrent à nous.
L'impulsion la plus naturelle serait peut-être
de choisir $y \in \{1, 2, 3\}$,
où les entiers représentent respectivement
$\{\text{dog}, \text{cat}, \text{chicken}\}$ .
C'est un excellent moyen de *stocker* de telles informations sur un ordinateur.
Si les catégories avaient un ordre naturel entre elles,
, par exemple si nous essayions de prédire
$\{\text{baby}, \text{toddler}, \text{adolescent}, \text{young adult}, \text{adult}, \text{geriatric}\}$ ,
, il pourrait même être logique de les présenter comme
un problème [ordinal regression](https://en.wikipedia.org/wiki/Ordinal_regression)
 et de conserver les étiquettes dans ce format.
Voir :cite:`Moon.Smola.Chang.ea.2010` pour un aperçu
des différents types de fonctions de perte de classement
et :cite:`Beutel.Murray.Faloutsos.ea.2014` pour une approche bayésienne
qui traite les réponses avec plus d'un mode.

En général, les problèmes de classification ne s'accompagnent pas
d'un ordre naturel entre les classes.
Heureusement, les statisticiens ont inventé il y a longtemps un moyen simple
de représenter les données catégorielles : le *codage à un coup*.
Un codage à un coup est un vecteur
avec autant de composantes que de catégories.
La composante correspondant à la catégorie d'une instance particulière est fixée à 1
et toutes les autres composantes sont fixées à 0.
Dans notre cas, une étiquette $y$ serait un vecteur tridimensionnel,
avec $(1, 0, 0)$ correspondant à "chat", $(0, 1, 0)$ à "poulet",
et $(0, 0, 1)$ à "chien" :

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$ 

 ### Modèle linéaire

Afin d'estimer les probabilités conditionnelles
associées à toutes les classes possibles,
nous avons besoin d'un modèle à sorties multiples, une par classe.
Pour aborder la classification avec des modèles linéaires,
nous aurons besoin d'autant de fonctions affines que de sorties.
Strictement parlant, nous n'en avons besoin que d'une seule,
puisque la dernière catégorie doit être la différence
entre $1$ et la somme des autres catégories
mais pour des raisons de symétrie
nous utilisons une paramétrisation légèrement redondante.
Chaque sortie correspond à sa propre fonction affine.
Dans notre cas, puisque nous avons 4 caractéristiques et 3 catégories de sortie possibles,
nous avons besoin de 12 scalaires pour représenter les poids ($w$ avec les indices),
et 3 scalaires pour représenter les biais ($b$ avec les indices). Cela donne :

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Le diagramme de réseau neuronal correspondant
est présenté dans :numref:`fig_softmaxreg` .
Comme pour la régression linéaire,
nous utilisons un réseau neuronal à une seule couche.
Et comme le calcul de chaque sortie, $o_1, o_2$, et $o_3$,
dépend de toutes les entrées, $x_1$, $x_2$, $x_3$, et $x_4$,
, la couche de sortie peut également être décrite comme une couche *entièrement connectée*.

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Pour une notation plus concise, nous utilisons des vecteurs et des matrices :
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$ est
beaucoup mieux adapté aux mathématiques et au code.
Notez que nous avons rassemblé tous nos poids dans une matrice $3 \times 4$ et tous les biais
$\mathbf{b} \in \mathbb{R}^3$ dans un vecteur.

### La Softmax
:label:`subsec_softmax_operation` 

 En supposant une fonction de perte appropriée,
nous pourrions essayer, directement, de minimiser la différence
entre $\mathbf{o}$ et les étiquettes $\mathbf{y}$.
S'il s'avère que le traitement de la classification
comme un problème de régression à valeurs vectorielles fonctionne étonnamment bien,
il présente néanmoins les lacunes suivantes :

* Il n'y a aucune garantie que les sorties $o_i$ s'additionnent à $1$ de la manière dont nous nous attendons à ce que les probabilités se comportent.
* Il n'y a aucune garantie que les sorties $o_i$ soient même non négatives, même si leur somme est égale à $1$, ou qu'elles ne dépassent pas $1$.

Ces deux aspects rendent le problème d'estimation difficile à résoudre
et la solution très fragile aux valeurs aberrantes.
Par exemple, si nous supposons qu'il existe
une dépendance linéaire positive
entre le nombre de chambres et la probabilité
qu'une personne achète une maison,
la probabilité pourrait dépasser $1$
 lorsqu'il s'agit d'acheter un manoir !
Nous avons donc besoin d'un mécanisme pour "écraser" les sorties.

Il existe de nombreuses façons d'atteindre cet objectif.
Par exemple, nous pouvons supposer que les sorties
$\mathbf{o}$ sont des versions corrompues de $\mathbf{y}$,
où la corruption se produit par l'ajout de bruit $\mathbf{\epsilon}$
 tiré d'une distribution normale.
En d'autres termes, $\mathbf{y} = \mathbf{o} + \mathbf{\epsilon}$,
où $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
Il s'agit de ce que l'on appelle [probit model](https://en.wikipedia.org/wiki/Probit_model),
, introduit pour la première fois par :cite:`Fechner.1860` .
Bien que séduisante, elle ne fonctionne pas aussi bien
et ne conduit pas à un problème d'optimisation particulièrement intéressant,
par rapport à la méthode softmax.

Une autre façon d'atteindre cet objectif
(et de garantir la non-négativité) consiste à utiliser
une fonction exponentielle $P(y = i) \propto \exp o_i$.
Cette fonction satisfait effectivement à la condition
selon laquelle la probabilité conditionnelle de classe
augmente avec $o_i$, elle est monotone,
et toutes les probabilités sont non négatives.
Nous pouvons ensuite transformer ces valeurs de manière à ce que leur somme soit égale à $1$
 en divisant chacune d'elles par leur somme.
Ce processus est appelé *normalisation*.
En assemblant ces deux éléments
, nous obtenons la fonction *softmax* :

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \text{where}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$ 
 :eqlabel:`eq_softmax_y_and_o` 

 Notez que la plus grande coordonnée de $\mathbf{o}$
 correspond à la classe la plus probable selon $\hat{\mathbf{y}}$.
De plus, comme l'opération softmax
préserve l'ordre entre ses arguments,
nous n'avons pas besoin de calculer la softmax
pour déterminer quelle classe s'est vue attribuer la plus forte probabilité.

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$


L'idée d'une softmax remonte à Gibbs,
qui a adapté des idées de la physique :cite:`Gibbs.1902` .
Encore plus loin, Boltzmann,
le père de la thermodynamique moderne,
a utilisé cette astuce pour modéliser une distribution
sur les états énergétiques des molécules de gaz.
Il a notamment découvert que la prévalence
d'un état d'énergie dans un ensemble thermodynamique,
tel que les molécules d'un gaz,
est proportionnelle à $\exp(-E/kT)$.
Ici, $E$ est l'énergie d'un état,
$T$ est la température et $k$ est la constante de Boltzmann.
Lorsque les statisticiens parlent d'augmenter ou de diminuer
la "température" d'un système statistique,
ils font référence à la modification de $T$
 afin de favoriser des états d'énergie inférieure ou supérieure.
Suivant l'idée de Gibbs, l'énergie équivaut à l'erreur.
Les modèles basés sur l'énergie :cite:`Ranzato.Boureau.Chopra.ea.2007` 
 utilisent ce point de vue pour décrire
des problèmes d'apprentissage profond.

### Vectorisation
:label:`subsec_softmax_vectorization` 

 Pour améliorer l'efficacité des calculs,
nous vectorisons les calculs dans des minibatchs de données.
Supposons que l'on nous donne un minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$
 de $n$ caractéristiques avec une dimensionnalité (nombre d'entrées) $d$.
De plus, supposons que nous avons $q$ catégories en sortie.
Alors les poids satisfont $\mathbf{W} \in \mathbb{R}^{d \times q}$
 et le biais satisfait $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Cela accélère l'opération dominante en
un produit matrice-matrice $\mathbf{X} \mathbf{W}$.
De plus, puisque chaque ligne de $\mathbf{X}$ représente un exemple de données,
l'opération softmax elle-même peut être calculée *rowwise* :
pour chaque ligne de $\mathbf{O}$, exponentiez toutes les entrées
puis normalisez-les par la somme.
Notez cependant qu'il faut faire attention
à ne pas exponentiser et prendre les logarithmes de grands nombres,
car cela peut provoquer un dépassement ou un sous-dépassement numérique.
Les cadres d'apprentissage profond s'en chargent automatiquement.

## Fonction de perte
:label:`subsec_softmax-regression-loss-func` 

 Maintenant que nous avons une correspondance entre les caractéristiques $\mathbf{x}$
 et les probabilités $\mathbf{\hat{y}}$,
, nous devons trouver un moyen d'optimiser la précision de cette correspondance.
Nous nous appuierons sur l'estimation du maximum de vraisemblance,
le même concept que nous avons rencontré
lorsque nous avons fourni une justification probabiliste
pour la perte de l'erreur quadratique moyenne dans
:numref:`subsec_normal_distribution_and_squared_loss` .

### Log-Vraisemblance

La fonction softmax nous donne un vecteur $\hat{\mathbf{y}}$,
que nous pouvons interpréter comme des probabilités conditionnelles (estimées)
de chaque classe, étant donné toute entrée $\mathbf{x}$,
telle que $\hat{y}_1$ = $P(y=\text{cat} | \mathbf{x})$.
Dans ce qui suit, nous supposons que pour un ensemble de données
avec des caractéristiques $\mathbf{X}$, les étiquettes $\mathbf{Y}$
 sont représentées à l'aide d'un vecteur d'étiquettes à codage à un coup.
Nous pouvons comparer les estimations avec la réalité
en vérifiant la probabilité que les classes réelles soient
selon notre modèle, compte tenu des caractéristiques :

$$
P(\mathbf{Y} | \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} | \mathbf{x}^{(i)}).
$$

Nous sommes autorisés à utiliser la factorisation
puisque nous supposons que chaque étiquette est tirée indépendamment
de sa distribution respective $P(\mathbf{y}|\mathbf{x}^{(i)})$.
Comme la maximisation du produit des termes est maladroite,
nous prenons le logarithme négatif pour obtenir le problème équivalent
de minimisation de la log-vraisemblance négative :

$$
-\log P(\mathbf{Y} | \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} | \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

où pour toute paire d'étiquette $\mathbf{y}$
 et de prédiction de modèle $\hat{\mathbf{y}}$
 sur $q$ classes, la fonction de perte $l$ est

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$ 
 :eqlabel:`eq_l_cross_entropy` 

 Pour des raisons expliquées plus loin,
la fonction de perte dans :eqref:`eq_l_cross_entropy` 
 est communément appelée la *perte d'entropie croisée*.
Puisque $\mathbf{y}$ est un vecteur à un coup de longueur $q$,
la somme de toutes ses coordonnées $j$ disparaît pour tous les termes sauf un.
Notez que la perte $l(\mathbf{y}, \hat{\mathbf{y}})$
 est limitée par le bas par $0$
 chaque fois que $\hat{y}$ est un vecteur de probabilité :
aucune entrée n'est plus grande que $1$,
donc leur logarithme négatif ne peut être inférieur à $0$;
$l(\mathbf{y}, \hat{\mathbf{y}}) = 0$ seulement si nous prédisons
l'étiquette réelle avec *certitude*.
Cela ne peut jamais se produire pour tout réglage fini des poids
car prendre une sortie softmax vers $1$
 nécessite de prendre l'entrée correspondante $o_i$ à l'infini
(ou toutes les autres sorties $o_j$ pour $j \neq i$ à l'infini négatif).
Même si notre modèle pouvait attribuer une probabilité de sortie de $0$,
, toute erreur commise lors de l'attribution d'une confiance aussi élevée à
entraînerait une perte infinie ($-\log 0 = \infty$).


### Softmax et perte d'entropie croisée
:label:`subsec_softmax_and_derivatives` 

 La fonction softmax
et la perte d'entropie croisée correspondante étant si courantes,
il est utile de comprendre un peu mieux comment elles sont calculées.
En plaçant :eqref:`eq_softmax_y_and_o` dans la définition de la perte
dans :eqref:`eq_l_cross_entropy` 
 et en utilisant la définition de la fonction softmax, on obtient :

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

Pour comprendre un peu mieux ce qui se passe,
considérons la dérivée par rapport à tout logit $o_j$. Nous obtenons

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

En d'autres termes, la dérivée est la différence
entre la probabilité attribuée par notre modèle,
telle qu'exprimée par l'opération softmax,
et ce qui s'est réellement passé, tel qu'exprimé
par les éléments du vecteur d'étiquettes à un coup.
En ce sens, c'est très similaire
à ce que nous avons vu dans la régression,
où le gradient était la différence
entre l'observation $y$ et l'estimation $\hat{y}$.
Ce n'est pas une coïncidence.
Dans tout modèle de la famille exponentielle,
les gradients de la log-vraisemblance sont précisément donnés par ce terme.
Ce fait rend le calcul des gradients facile dans la pratique.

Considérons maintenant le cas où nous n'observons pas seulement un résultat unique
mais une distribution entière de résultats.
Nous pouvons utiliser la même représentation que précédemment pour l'étiquette $\mathbf{y}$.
La seule différence est qu'au lieu d'un vecteur contenant uniquement des entrées binaires (
),
ou $(0, 0, 1)$, nous avons maintenant un vecteur de probabilité générique (
ou $(0.1, 0.2, 0.7)$).
Les mathématiques que nous avons utilisées précédemment pour définir la perte $l$
 dans :eqref:`eq_l_cross_entropy` 
 fonctionnent toujours bien,
mais l'interprétation est légèrement plus générale.
Il s'agit de la valeur attendue de la perte pour une distribution sur les étiquettes.
Cette perte s'appelle la *perte d'entropie croisée* et c'est
l'une des pertes les plus couramment utilisées pour les problèmes de classification.
Nous pouvons démystifier son nom en introduisant simplement les bases de la théorie de l'information.
En un mot, elle mesure le nombre de bits nécessaires pour coder ce que nous voyons $\mathbf{y}$
 par rapport à ce que nous prédisons qui devrait se produire $\hat{\mathbf{y}}$.
Nous fournissons une explication très élémentaire dans ce qui suit. Pour plus de détails
sur la théorie de l'information, voir 
:cite:`Cover.Thomas.1999` ou :cite:`mackay2003information` .



## Bases de la théorie de l'information
:label:`subsec_info_theory_basics` 

 De nombreux articles sur l'apprentissage profond utilisent des intuitions et des termes issus de la théorie de l'information.
Pour leur donner un sens, nous avons besoin d'un langage commun.
Voici un guide de survie.
*La théorie de l'information* traite du problème
du codage, du décodage, de la transmission,
et de la manipulation des informations (également appelées données).

### Entropie

L'idée centrale de la théorie de l'information est de quantifier la
quantité d'information contenue dans les données.
Cela impose une limite à notre capacité à compresser les données.
Pour une distribution $P$, son *entropie* est définie comme suit :

$$H[P] = \sum_j - P(j) \log P(j).$$ 
 :eqlabel:`eq_softmax_reg_entropy` 

 L'un des théorèmes fondamentaux de la théorie de l'information stipule
que pour coder des données tirées au hasard de la distribution $P$,
il faut au moins $H[P]$ "nats" pour les coder :cite:`Shannon.1948` .
Si vous vous demandez ce qu'est un "nat", c'est l'équivalent d'un bit
mais en utilisant un code de base $e$ plutôt qu'un code de base 2.
Ainsi, un nat est un bit $\frac{1}{\log(2)} \approx 1.44$.


### Surprisal

Vous vous demandez peut-être ce que la compression a à voir avec la prédiction.
Imaginez que nous ayons un flux de données que nous voulons compresser.
S'il est toujours facile pour nous de prédire le prochain jeton,
alors ces données sont faciles à compresser.
Prenez l'exemple extrême où chaque jeton du flux
prend toujours la même valeur.
C'est un flux de données très ennuyeux !
Et non seulement il est ennuyeux, mais il est également facile à prédire.
Parce qu'ils sont toujours les mêmes,
nous n'avons pas besoin de transmettre d'informations
pour communiquer le contenu du flux.
Facile à prédire, facile à compresser.

Cependant, si nous ne pouvons pas prédire parfaitement chaque événement,
alors nous pouvons parfois être surpris.
Notre surprise est d'autant plus grande que nous avons attribué une probabilité plus faible à un événement.
Claude Shannon a choisi $\log \frac{1}{P(j)} = -\log P(j)$
 pour quantifier la *surprise* d'une personne qui observe un événement $j$
 après lui avoir attribué une probabilité (subjective) $P(j)$.
L'entropie définie dans :eqref:`eq_softmax_reg_entropy` 
 est alors la *surprise* attendue
lorsque l'on attribue les probabilités correctes
qui correspondent réellement au processus de génération des données.


### L'entropie croisée revisitée

Donc, si l'entropie est le niveau de surprise éprouvé
par quelqu'un qui connaît la vraie probabilité,
alors vous pouvez vous demander ce qu'est l'entropie croisée ?
L'entropie croisée *de* $P$ *à* $Q$, notée $H(P, Q)$,
est la surprise attendue d'un observateur ayant des probabilités subjectives $Q$
 en voyant des données qui ont été réellement générées selon les probabilités $P$.
Elle est donnée par $H(P, Q) \stackrel{\mathrm{def}}{=} \sum_j - P(j) \log Q(j)$.
L'entropie croisée la plus faible possible est atteinte lorsque $P=Q$.
Dans ce cas, l'entropie croisée de $P$ à $Q$ est $H(P, P)= H(P)$.

En résumé, nous pouvons considérer l'objectif de classification de l'entropie croisée
de deux manières : (i) comme maximisant la vraisemblance des données observées ;
et (ii) comme minimisant notre surprise (et donc le nombre de bits)
nécessaire pour communiquer les étiquettes.

## Résumé et discussion

Dans cette section, nous avons rencontré la première fonction de perte non triviale,
qui nous permet d'optimiser sur des espaces de sortie *discrets*.
La clé de sa conception est que nous avons adopté une approche probabiliste,
en traitant les catégories discrètes comme des instances de tirages d'une distribution de probabilité.
Comme effet secondaire, nous avons rencontré le softmax,
une fonction d'activation pratique qui transforme
les sorties d'une couche de réseau neuronal ordinaire
en distributions de probabilité discrètes valides.
Nous avons vu que la dérivée de la perte d'entropie croisée
, lorsqu'elle est combinée à la softmax
, se comporte de manière très similaire
à la dérivée de l'erreur quadratique,
à savoir en prenant la différence entre
le comportement attendu et sa prédiction.
Et, bien que nous n'ayons pu qu'effleurer
,
nous avons rencontré des liens passionnants
avec la physique statistique et la théorie de l'information.

Bien que cela suffise à vous mettre sur la voie,
et, espérons-le, à vous mettre en appétit,
nous n'avons guère plongé en profondeur ici.
Entre autres choses, nous avons fait l'impasse sur des considérations informatiques.
Plus précisément, pour toute couche entièrement connectée avec $d$ entrées et $q$ sorties,
le paramétrage et le coût de calcul sont $\mathcal{O}(dq)$,
ce qui peut être prohibitif en pratique.
Heureusement, ce coût de transformation des entrées $d$ en sorties $q$
 peut être réduit par approximation et compression.
Par exemple, Deep Fried Convnets :cite:`Yang.Moczulski.Denil.ea.2015` 
 utilise une combinaison de permutations, de transformées de Fourier
et de mise à l'échelle
pour réduire le coût de quadratique à log-linéaire.
Des techniques similaires fonctionnent pour des approximations de matrices structurelles plus avancées
 :cite:`sindhwani2015structured` .
Enfin, nous pouvons utiliser des décompositions de type quaternion
pour réduire le coût à $\mathcal{O}(\frac{dq}{n})$,
encore une fois si nous sommes prêts à échanger une petite quantité de précision
contre un coût de calcul et de stockage :cite:`Zhang.Tay.Zhang.ea.2021` 
 basé sur un facteur de compression $n$.
Il s'agit d'un domaine de recherche actif.
Le défi réside dans le fait que
nous ne cherchons pas nécessairement
la représentation la plus compacte
ou le plus petit nombre d'opérations en virgule flottante
mais plutôt la solution
qui peut être exécutée le plus efficacement sur les GPU modernes.

## Exercices

1. Nous pouvons explorer le lien entre les familles exponentielles et le softmax de manière plus approfondie.
   1. Calculez la dérivée seconde de la perte d'entropie croisée $l(\mathbf{y},\hat{\mathbf{y}})$ pour la softmax.
   1. Calculez la variance de la distribution donnée par $\mathrm{softmax}(\mathbf{o})$ et montrez qu'elle correspond à la dérivée seconde calculée ci-dessus.
1. Supposons que nous ayons trois classes qui se produisent avec la même probabilité, c'est-à-dire que le vecteur de probabilité est $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
 1. Quel est le problème si nous essayons de concevoir un code binaire pour cette classe ?
   1. Pouvez-vous concevoir un meilleur code ? Indice : que se passe-t-il si nous essayons de coder deux observations indépendantes ? Que se passe-t-il si nous codons conjointement $n$ observations ?
1. Lorsqu'ils codent des signaux transmis sur un fil physique, les ingénieurs n'utilisent pas toujours des codes binaires. Par exemple, [PAM-3](https://en.wikipedia.org/wiki/Ternary_signal) utilise trois niveaux de signal $\{-1, 0, 1\}$ par opposition à deux niveaux $\{0, 1\}$. De combien d'unités ternaires avez-vous besoin pour transmettre un nombre entier dans l'intervalle $\{0, \ldots, 7\}$? Pourquoi cela serait-il une meilleure idée en termes d'électronique ?
1. Le site [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) utilise
un modèle logistique pour saisir les préférences. Pour qu'un utilisateur puisse choisir entre des pommes et des oranges,
suppose des scores $o_{\mathrm{apple}}$ et $o_{\mathrm{orange}}$. Nos exigences sont les suivantes : des scores plus élevés doivent conduire à une plus grande probabilité de choisir l'élément associé et
l'élément avec le score le plus élevé est le plus susceptible d'être choisi :cite:`Bradley.Terry.1952` .
   1. Prouvez que le softmax satisfait à cette exigence.
   1. Que se passe-t-il si l'on veut permettre une option par défaut consistant à ne choisir ni les pommes ni les oranges ? Indice : l'utilisateur a maintenant trois choix.
1. Softmax tire son nom de la correspondance suivante : $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
   1. Prouvez que $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
   1. Jusqu'à quel point pouvez-vous réduire la différence entre les deux fonctions ? Indice : sans perdre
 de sa généralité, vous pouvez définir $b = 0$ et $a \geq b$.
 1. Prouvez que ceci est valable pour $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, à condition que $\lambda > 0$.
   1. Montrez que pour $\lambda \to \infty$ nous avons $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
   1. A quoi ressemble le soft-min ?
   1. Étendez ce calcul à plus de deux nombres.
1. La fonction $g(\mathbf{x}) \stackrel{\mathrm{def}}{=} \log \sum_i \exp x_i$ est parfois appelée aussi [log-partition function](https://en.wikipedia.org/wiki/Partition_function_(mathematics)).
   1. Prouvez que cette fonction est convexe. Conseil : pour ce faire, utilisez le fait que la première dérivée correspond aux probabilités de la fonction softmax et montrez que la seconde dérivée est la variance.
   1. Montrez que $g$ est invariant en translation, c'est-à-dire $g(\mathbf{x} + b) = g(\mathbf{x})$.
   1. Que se passe-t-il si certaines des coordonnées $x_i$ sont très grandes ? Que se passe-t-il si elles sont toutes très petites ?
   1. Montrez que si nous choisissons $b = \mathrm{max}_i x_i$, nous obtenons une implémentation numériquement stable.
1. Supposons que nous ayons une certaine distribution de probabilité $P$. Supposons que nous choisissions une autre distribution $Q$ avec $Q(i) \propto P(i)^\alpha$ pour $\alpha > 0$.
 1. Quel choix de $\alpha$ correspond à un doublement de la température ? Quel choix correspond à une réduction de moitié de la température ?
   1. Que se passe-t-il si nous laissons la température converger vers $0$?
   1. Que se passe-t-il si nous laissons la température converger vers $\infty$?

[Discussions](https://discuss.d2l.ai/t/46)
