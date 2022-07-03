[# Des couches entièrement connectées aux convolutions
:label:`sec_why-conv` 

A ce jour,
les modèles que nous avons abordés jusqu'à présent
restent des options appropriées
lorsque nous traitons des données tabulaires.
Par tableau, nous entendons que les données sont constituées
de lignes correspondant à des exemples
et de colonnes correspondant à des caractéristiques.
Avec des données tabulaires, nous pouvons anticiper
que les modèles que nous recherchons pourraient impliquer
des interactions entre les caractéristiques,
mais nous ne supposons aucune structure *a priori*
concernant la façon dont les caractéristiques interagissent.

Parfois, nous manquons vraiment de connaissances pour guider
la construction d'architectures plus sophistiquées.
Dans ces cas, un MLP
peut être le meilleur que nous puissions faire.
Cependant, pour les données perceptives à haute dimension,
de tels réseaux sans structure peuvent devenir difficiles à manier.

Par exemple, revenons à notre exemple courant
qui consiste à distinguer les chats des chiens.
Supposons que nous fassions un travail minutieux de collecte de données,
en recueillant un ensemble de données annotées de photographies d'un mégapixel.
Cela signifie que chaque entrée du réseau a un million de dimensions.
Même une réduction agressive à mille dimensions cachées
nécessiterait une couche entièrement connectée
caractérisée par $10^6 \times 10^3 = 10^9$ paramètres.
À moins d'avoir beaucoup de GPU, un talent
pour l'optimisation distribuée,
et une quantité extraordinaire de patience,
l'apprentissage des paramètres de ce réseau
peut s'avérer infaisable.

Un lecteur attentif pourrait objecter à cet argument
en arguant qu'une résolution d'un mégapixel n'est peut-être pas nécessaire.
Cependant, même si nous pouvons
nous en sortir avec cent mille pixels,
notre couche cachée de taille 1000 sous-estime largement
le nombre d'unités cachées qu'il faut
pour apprendre de bonnes représentations d'images,
de sorte qu'un système pratique nécessitera encore des milliards de paramètres.
De plus, l'apprentissage d'un classifieur en ajustant autant de paramètres
pourrait nécessiter la collecte d'un énorme ensemble de données.
Pourtant, aujourd'hui, tant les humains que les ordinateurs sont capables
de distinguer assez bien les chats des chiens,
ce qui semble contredire ces intuitions.
La raison en est que les images présentent une structure riche
qui peut être exploitée aussi bien par les humains
que par les modèles d'apprentissage automatique.
Les réseaux de neurones convolutifs (CNN) sont un moyen créatif
que l'apprentissage automatique a adopté pour exploiter
une partie de la structure connue des images naturelles.


## Invariance

Imaginez que nous voulions détecter un objet dans une image.
Il semble raisonnable que la méthode
que nous utilisons pour reconnaître les objets ne se préoccupe pas trop
de l'emplacement précis de l'objet dans l'image.
Idéalement, notre système devrait exploiter cette connaissance.
Les cochons ne volent généralement pas et les avions ne nagent généralement pas.
Néanmoins, nous devrions reconnaître
un cochon s'il apparaissait en haut de l'image.
Nous pouvons nous inspirer ici
du jeu pour enfants "Where's Waldo"
(illustré sur :numref:`img_waldo` ).
Le jeu consiste en un certain nombre de scènes chaotiques
débordant d'activités.
Waldo apparaît quelque part dans chacune d'elles,
se cachant généralement dans un endroit improbable.
Le but du lecteur est de le localiser.
Malgré sa tenue caractéristique,
cela peut être étonnamment difficile,
en raison du grand nombre de distractions.
Cependant, *ce à quoi ressemble Waldo*
ne dépend pas de *l'endroit où se trouve Waldo*.
Nous pourrions balayer l'image avec un détecteur de Waldo
qui pourrait attribuer un score à chaque patch,
indiquant la probabilité que le patch contienne Waldo. 
En fait, de nombreux algorithmes de détection et de segmentation d'objets 
sont basés sur cette approche :cite:)[`Long.Shelhamer.Darrell.2015`. 
Les CNN systématisent cette idée d'*invariance spatiale*,
en l'exploitant pour apprendre des représentations utiles
avec moins de paramètres.

![Une image du jeu " Where's Waldo "](../img/where-wally-walker-books.jpg)
:width:`400px` 
:label:`img_waldo` 

Nous pouvons maintenant rendre ces intuitions plus concrètes 
en énumérant quelques desiderata pour guider notre conception
d'une architecture de réseau de neurones adaptée à la vision par ordinateur :

1. Dans les premières couches, notre réseau
doit répondre de manière similaire à la même tache,
indépendamment de l'endroit où elle apparaît dans l'image. Ce principe est appelé *invariance de translation* (ou *équivariance de translation*).
1. Les premières couches du réseau doivent se concentrer sur les régions locales,
sans tenir compte du contenu de l'image dans les régions éloignées. C'est le principe de *localité*.
  Finalement, ces représentations locales peuvent être agrégées
pour faire des prédictions au niveau de l'image entière.
1. Au fur et à mesure, des couches plus profondes devraient être capables de capturer des caractéristiques à plus grande portée de l'image ,
d'une manière similaire à la vision de plus haut niveau dans la nature. 

Voyons comment cela se traduit en mathématiques.


## Contrainte du MLP

Pour commencer, nous pouvons considérer un MLP
avec des images bidimensionnelles \mathbf{X} comme entrées
et leurs représentations cachées immédiates
$\mathbf{H}$ représentées de manière similaire sous forme de matrices (ce sont des tenseurs bidimensionnels en code), où \mathbf{X} et \mathbf{H} ont la même forme.
Laissez-vous convaincre.
Nous concevons maintenant que non seulement les entrées mais aussi
les représentations cachées possèdent une structure spatiale.

Que $[\mathbf{X}]_{i, j}$ et $[\mathbf{H}]_{i, j}$ désignent le pixel
à l'emplacement $(i,j)$
dans l'image d'entrée et la représentation cachée, respectivement.
Par conséquent, pour que chacune des unités cachées
reçoive des entrées de chacun des pixels d'entrée,
nous passerions de l'utilisation de matrices de poids
(comme nous l'avons fait précédemment dans les MLP)
à la représentation de nos paramètres
sous forme de tenseurs de poids du quatrième ordre \mathsf{W}.
Supposons que \mathbf{U} contienne des biais,
nous pourrions formellement exprimer la couche entièrement connectée comme

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

Le passage de \mathsf{W} à \mathsf{V} est entièrement cosmétique pour le moment
puisqu'il existe une correspondance biunivoque
entre les coefficients des deux tenseurs d'ordre 4.
Nous réindexons simplement les indices $(k, l)$
de sorte que $k = i+a$ et $l = j+b$.
En d'autres termes, nous définissons $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
Les indices $a$ et $b$ s'étendent sur les décalages positifs et négatifs,
couvrant la totalité de l'image.
Pour tout emplacement donné ($i$, $j$) dans la représentation cachée $[\mathbf{H}]_{i, j}$,
nous calculons sa valeur en additionnant les pixels en $x$,
centrés autour de $(i, j)$ et pondérés par $[\mathsf{V}]_{i, j, a, b}$. Avant de poursuivre, considérons le nombre total de paramètres requis pour une *seule* couche dans cette paramétrisation : une image de $1000 \times 1000$ (1 mégapixel) est mise en correspondance avec une représentation cachée de $1000 \times 1000$. Cela nécessite $10^{12}$ paramètres, bien au-delà de ce que les ordinateurs peuvent actuellement gérer. 

#### Invariance de la traduction

Invoquons maintenant le premier principe
établi ci-dessus : l'invariance de la traduction :cite:)[`Zhang.ea.1988`.
Cela implique qu'un changement dans l'entrée \mathbf{X}
devrait simplement conduire à un changement dans la représentation cachée \mathbf{H}.
Ceci n'est possible que si \mathsf{V} et \mathbf{U} ne dépendent pas réellement de $(i, j)$. Ainsi,
nous avons $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ et \mathbf{U} est une constante, disons $u$.
Par conséquent, nous pouvons simplifier la définition de \mathbf{H} :

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}$$ [$$\mathbf{X}]_{i+a, j+b}.$$


Il s'agit d'une *convolution* !
Nous pondérons effectivement les pixels à $(i+a, j+b)$
à proximité de l'emplacement $(i, j)$ avec les coefficients $[\mathbf{V}]_{a, b}$
pour obtenir la valeur $[\mathbf{H}]_{i, j}$.
Notez que $[\mathbf{V}]_{a, b}$ nécessite beaucoup moins de coefficients que $[\mathsf{V}]_{i, j, a, b}$ puisqu'il
ne dépend plus de l'emplacement dans l'image. Par conséquent, le nombre de paramètres requis n'est plus $10^{12}$ mais un nombre beaucoup plus raisonnable de $4 \cdot 10^6$: nous avons toujours la dépendance sur $a, b \in (-1000, 1000)$. En bref, nous avons fait des progrès significatifs. Les réseaux de neurones à retardement (TDNN) sont parmi les premiers exemples à exploiter cette idée :cite:)[`Waibel.Hanazawa.Hinton.ea.1989`.

### Localité

Invoquons maintenant le deuxième principe : la localité.
Comme nous l'avons motivé ci-dessus, nous pensons que nous ne devrions pas avoir à
regarder très loin de l'emplacement $(i, j)$
afin de glaner des informations pertinentes
pour évaluer ce qui se passe à $[\mathbf{H}]_{i, j}$.
Cela signifie qu'en dehors d'une certaine plage $|a|&gt; \Delta$ ou $|b| &gt; \Delta$,
nous devons définir $[\mathbf{V}]_{a, b} = 0$.
De manière équivalente, nous pouvons réécrire $[\mathbf{H}]_{i, j}$ comme suit :

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b} \mathbf{X} []_{i+a, j+b}.)$$
:eqlabel:`eq_conv-layer`

Cela réduit le nombre de paramètres de $4 \cdot 10^6$ à $4 \Delta^2$, où \Delta est généralement inférieur à $10$. Ainsi, nous avons réduit le nombre de paramètres de 4 autres ordres de grandeur. Notez que :eqref:`eq_conv-layer`, en un mot, est ce qu'on appelle une *couche convolutive*. 
*Convolutional neural network* (CNNs)
sont une famille particulière de réseaux neuronaux qui contiennent des couches convolutionnelles.
Dans la communauté des chercheurs en apprentissage profond,
$\mathbf{V}$ est appelé un *noyau de convolution*,
un *filtre*, ou simplement les *poids* de la couche qui sont des paramètres apprenables.

Alors qu'auparavant, nous aurions pu avoir besoin de milliards de paramètres
pour représenter une seule couche dans un réseau de traitement d'images,
nous n'en avons plus besoin que de quelques centaines, sans
altérer la dimensionnalité des entrées ou des représentations cachées
.
Le prix à payer pour cette réduction drastique des paramètres
est que nos caractéristiques sont désormais invariantes par rapport à la traduction
et que notre couche ne peut incorporer que des informations locales,
lors de la détermination de la valeur de chaque activation cachée.
Tout apprentissage dépend de l'imposition d'un biais inductif.
Lorsque ce biais correspond à la réalité,
nous obtenons des modèles efficaces en termes d'échantillonnage
qui se généralisent bien aux données non vues.
Mais bien sûr, si ces biais ne correspondent pas à la réalité,
par exemple si les images s'avéraient ne pas être invariantes par rapport à la traduction,
nos modèles pourraient même avoir du mal à s'adapter à nos données d'apprentissage.

Cette réduction spectaculaire des paramètres nous amène à notre dernier desideratum,
à savoir que les couches plus profondes doivent représenter des aspects plus importants et plus complexes 
d'une image. Ceci peut être réalisé en entrelaçant les non-linéarités et les couches convolutives 
de manière répétée. 

## Convolutions

Rappelons brièvement pourquoi :eqref:`eq_conv-layer` est appelé une convolution. 
En mathématiques, la *convolution* entre deux fonctions :cite:`Rudin.1973`,
dis $f, g : \mathbb{R}^d \to \mathbb{R}$ est définie comme

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$ 

C'est-à-dire que nous mesurons le chevauchement entre $f$ et $g$
lorsqu'une fonction est "retournée" et décalée par $\mathbf{x}$.
Lorsque nous avons des objets discrets, l'intégrale se transforme en une somme.
Par exemple, pour les vecteurs de,
l'ensemble des vecteurs infinis de dimension infinie sommables au carré
avec un indice courant sur $\mathbb{Z}$, nous obtenons la définition suivante :

$$(f * g)(i) = \sum_a f(a) g(i-a).$$ 

Pour les tenseurs bidimensionnels, nous avons une somme correspondante
avec des indices $(a, b)$ pour $f$ et $(i-a, j-b)$ pour $g$, respectivement :

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$ 
:eqlabel:`eq_2d-conv-discrete` 

Cela ressemble à :eqref:`eq_conv-layer`, avec une différence majeure.
Au lieu d'utiliser $(i+a, j+b)$, nous utilisons la différence.
Notez cependant que cette distinction est surtout cosmétique
puisque nous pouvons toujours faire correspondre la notation entre
:eqref:`eq_conv-layer` et :eqref:`eq_2d-conv-discrete`.
Notre définition originale dans :eqref:`eq_conv-layer` décrit plus correctement
une *corrélation croisée*.
Nous y reviendrons dans la section suivante.


## Canaux
:label:`subsec_why-conv-channels` 

Revenons à notre détecteur Waldo, voyons à quoi cela ressemble.
La couche convolutive choisit des fenêtres d'une taille donnée
et pondère les intensités en fonction du filtre $\mathsf{V}$, comme le montre :numref:`fig_waldo_mask`.
Nous pourrions chercher à apprendre un modèle de telle sorte que
partout où la "waldoness" est la plus élevée,
nous devrions trouver un pic dans les représentations de la couche cachée.

![Détecter Waldo.](../img/waldo-mask.jpg)
:width:`400px` 
:label:`fig_waldo_mask` 

Cette approche pose un seul problème.
Jusqu'à présent, nous avons ignoré béatement que les images se composent
de 3 canaux : rouge, vert et bleu. 
En somme, les images ne sont pas des objets bidimensionnels
mais plutôt des tenseurs du troisième ordre,
caractérisés par une hauteur, une largeur et un canal,
par exemple, avec une forme $1024 \times 1024 \times 3$ pixels. 
Alors que les deux premiers axes concernent les relations spatiales,
le troisième peut être considéré comme attribuant
une représentation multidimensionnelle à chaque emplacement de pixel.
Nous indexons donc $\mathsf{X}$ comme $[\mathsf{X}]_{i, j, k}$.
Le filtre convolutif doit s'adapter en conséquence.
Au lieu de $[\mathbf{V}]_{a,b}$, nous avons maintenant $[\mathsf{V}]_{a,b,c}$.

De plus, tout comme notre entrée consiste en un tenseur du troisième ordre,
il s'avère être une bonne idée de formuler de la même manière
nos représentations cachées comme des tenseurs du troisième ordre $\mathsf{H}$.
En d'autres termes, plutôt que d'avoir une seule représentation cachée
correspondant à chaque emplacement spatial,
nous voulons un vecteur entier de représentations cachées
correspondant à chaque emplacement spatial.
On peut considérer que les représentations cachées comprennent
un certain nombre de grilles bidimensionnelles empilées les unes sur les autres.
Comme dans les entrées, on les appelle parfois des *canaux*.
Elles sont aussi parfois appelées *cartes de caractéristiques*,
car chacune fournit un ensemble spatialisé
de caractéristiques apprises à la couche suivante.
Intuitivement, on peut imaginer qu'au niveau des couches inférieures, plus proches des entrées,
certains canaux pourraient être spécialisés dans la reconnaissance des bords tandis que
d'autres pourraient reconnaître les textures.

Pour prendre en charge plusieurs canaux à la fois dans les entrées ($\mathsf{X}$) et les représentations cachées ($\mathsf{H}$),
nous pouvons ajouter une quatrième coordonnée à $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$.
En mettant tout ensemble, nous avons :

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$ 
:eqlabel:`eq_conv-layer-channels` 

où $d$ indexe les canaux de sortie dans les représentations cachées $\mathsf{H}$. La couche convolutive suivante prendra en entrée un tenseur de troisième ordre, $\mathsf{H}$.
De manière plus générale,
:eqref:`eq_conv-layer-channels` est
la définition d'une couche convolutive pour des canaux multiples, où $\mathsf{V}$ est un noyau ou un filtre de la couche.

Il reste encore de nombreuses opérations à traiter.
Par exemple, nous devons trouver comment combiner toutes les représentations cachées
en une seule sortie, par exemple pour savoir s'il y a un Waldo *n'importe où* dans l'image.
Nous devons également décider comment calculer les choses efficacement,
comment combiner plusieurs couches,
des fonctions d'activation appropriées,
et comment faire des choix de conception raisonnables
pour produire des réseaux efficaces en pratique.
Nous aborderons ces questions dans le reste du chapitre.

## Résumé et discussion

Dans cette section, nous avons dérivé la structure des réseaux de neurones convolutifs à partir des premiers principes. Bien qu'il ne soit pas clair si c'est ce qui a conduit à l'invention des CNN, il est satisfaisant de savoir qu'ils constituent le *bon* choix lorsqu'on applique des principes raisonnables à la manière dont les algorithmes de traitement d'images et de vision par ordinateur devraient fonctionner, au moins aux niveaux inférieurs. En particulier, l'invariance de translation dans les images implique que tous les patchs d'une image seront traités de la même manière. La localité signifie que seul un petit voisinage de pixels sera utilisé pour calculer les représentations cachées correspondantes. Certaines des premières références aux CNN se trouvent sous la forme du Neocognitron :cite:`Fukushima.1982`. 

Un deuxième principe que nous avons rencontré dans notre raisonnement est de savoir comment réduire le nombre de paramètres dans une classe de fonctions sans limiter son pouvoir expressif, du moins, lorsque certaines hypothèses sur le modèle se vérifient. Nous avons constaté une réduction spectaculaire de la complexité grâce à cette restriction, transformant des problèmes infaisables sur le plan informatique et statistique en modèles traitables. 

L'ajout de canaux nous a permis de récupérer une partie de la complexité perdue en raison des restrictions imposées au noyau convolutif par la localité et l'invariance de traduction. Notez que les canaux sont un ajout tout à fait naturel au-delà du rouge, du vert et du bleu. De nombreuses images satellites ,
 notamment pour l'agriculture et la météorologie, comportent des dizaines voire des centaines de canaux, 
générant plutôt des images hyperspectrales. Elles rapportent des données sur de nombreuses longueurs d'onde différentes. Dans ce qui suit, nous verrons comment utiliser efficacement les convolutions pour manipuler la dimensionnalité des images sur lesquelles elles opèrent, comment passer de représentations basées sur l'emplacement à des représentations basées sur les canaux et comment traiter efficacement un grand nombre de catégories. 

## Exercices

1. Supposons que la taille du noyau de convolution soit $\Delta = 0$.
Montrez que dans ce cas, le noyau de convolution
 met en œuvre un MLP indépendamment pour chaque ensemble de canaux. Cela conduit au réseau en réseau 
architectures :cite:`Lin.Chen.Yan.2013`. 
1. Les données audio sont souvent représentées comme une séquence unidimensionnelle. 
    1. Quand voudriez-vous imposer la localité et l'invariance de translation pour l'audio ? 
 1. Dérivez les opérations de convolution pour l'audio.
  1. Pouvez-vous traiter l'audio en utilisant les mêmes outils que la vision par ordinateur ? Indice : utilisez le spectrogramme.
1. Pourquoi l'invariance de traduction n'est-elle pas une bonne idée après tout ? Donnez un exemple. 
1. Pensez-vous que les couches convolutionnelles pourraient également être applicables aux données textuelles ?
Quels problèmes pouvez-vous rencontrer avec la langue ?
1. Que se passe-t-il avec les convolutions lorsqu'un objet se trouve à la limite d'une image. 
1. Prouvez que la convolution est symétrique, c'est-à-dire $f * g = g * f$.
1. Prouvez le théorème de convolution, c'est-à-dire $f * g = \mathcal{F}^{-1}\left[\mathcal{F}[f] \cdot \mathcal{F}[g]\right]$. 
   Pouvez-vous l'utiliser pour accélérer les convolutions ? 

[Discussions](https://discuss.d2l.ai/t/64)
