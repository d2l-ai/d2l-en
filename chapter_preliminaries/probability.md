```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Probabilité et statistiques
:label:`sec_prob` 

 D'une manière ou d'une autre, 
l'apprentissage automatique concerne l'incertitude.
Dans l'apprentissage supervisé, nous voulons prédire 
quelque chose d'inconnu (la *cible*)
à partir de quelque chose de connu (les *caractéristiques*). 
En fonction de notre objectif, 
nous pouvons tenter de prédire 
la valeur la plus probable de la cible.
Ou nous pouvons prédire la valeur dont la distance attendue par rapport à la cible est la plus faible
.
Et parfois, nous souhaitons non seulement
prédire une valeur spécifique
mais aussi *quantifier notre incertitude*.
Par exemple, étant donné certaines caractéristiques 
décrivant un patient,
nous pourrions vouloir savoir *quelle est la probabilité* que ce patient
subisse une crise cardiaque l'année suivante. 
Dans l'apprentissage non supervisé, 
nous nous intéressons souvent à l'incertitude. 
Pour déterminer si un ensemble de mesures est anormal,
il est utile de savoir quelle est la probabilité 
d'observer des valeurs dans une population d'intérêt.
De plus, dans l'apprentissage par renforcement, 
nous souhaitons développer des agents
qui agissent intelligemment dans divers environnements.
Pour ce faire, il est nécessaire de raisonner sur 
la manière dont on peut s'attendre à ce qu'un environnement change
et sur les récompenses que l'on peut s'attendre à rencontrer
en réponse à chacune des actions disponibles. 

*La probabilité* est le domaine mathématique
concerné par le raisonnement dans l'incertitude.
Étant donné un modèle probabiliste d'un certain processus, 
nous pouvons raisonner sur la probabilité de divers événements.
L'utilisation des probabilités pour décrire 
les fréquences d'événements répétables 
(comme les tirages à pile ou face)
est assez peu controversée. 
En fait, les chercheurs *fréquentistes* adhèrent 
à une interprétation de la probabilité
qui s'applique *uniquement* à de tels événements répétables.
En revanche, les spécialistes des probabilités *bayésiennes* 
utilisent le langage des probabilités de manière plus large 
pour formaliser notre raisonnement dans l'incertitude.
La probabilité bayésienne se caractérise 
par deux caractéristiques uniques :
(i) l'attribution de degrés de croyance 
à des événements non répétables,
par exemple, quelle est la *probabilité* 
que la lune soit faite de fromage ?
et (ii) la subjectivité -- alors que la probabilité bayésienne
fournit des règles non ambiguës
sur la façon dont on doit mettre à jour ses croyances 
à la lumière de nouvelles preuves,
elle permet à différents individus 
de commencer avec différentes *croyances préalables*.
 les *statistiques* nous aident à raisonner à rebours,
en commençant par la collecte et l'organisation des données
et en remontant jusqu'aux inférences 
que nous pourrions tirer du processus 
qui a généré les données.
Chaque fois que nous analysons un ensemble de données, à la recherche de modèles
qui, nous l'espérons, pourraient caractériser une population plus large,
nous employons la pensée statistique.
La plupart des cours, majeures, thèses, carrières, départements,
entreprises et institutions ont été consacrés 
à l'étude des probabilités et des statistiques. 
Bien que cette section ne fasse qu'effleurer la surface,
nous vous fournirons les bases
dont vous avez besoin pour commencer à construire des modèles.



## Un exemple simple : Lancer de pièces de monnaie

Imaginez que nous prévoyons de lancer une pièce de monnaie
et que nous voulons quantifier la probabilité
de voir apparaître face (par rapport à pile).
Si la pièce est *juste*, 
, les deux résultats 
(pile et face), 
sont également probables.
De plus, si nous prévoyons de lancer la pièce $n$ plusieurs fois
, la fraction de face 
que nous *espérons* voir
devrait correspondre exactement à
à la fraction de pile *attendue*.
Une façon intuitive de voir cela
est la symétrie :
pour chaque résultat possible
avec $n_h$ face et $n_t = (n - n_h)$ pile,
il existe un résultat tout aussi probable
avec $n_t$ face et $n_h$ pile.
 
Notez que cela n'est possible que si, en moyenne, nous nous attendons à ce que
$1/2$ des lancers aboutissent à face 
et $1/2$ à pile.
Bien sûr, si vous réalisez cette expérience 
de nombreuses fois avec $n=1000000$ lancers chacun,
vous ne verrez peut-être jamais un essai
où $n_h = n_t$ exactement.


Formellement, la quantité $1/2$ s'appelle une *probabilité*
et elle représente ici la certitude avec laquelle 
un lancer donné sortira face.
Les probabilités attribuent des notes comprises entre $0$ et $1$
 à des résultats intéressants, appelés *événements*.
Ici, l'événement d'intérêt est $\textrm{heads}$
 et nous désignons la probabilité correspondante par $P(\textrm{heads})$.
Une probabilité de $1$ indique une certitude absolue 
(imaginez une pièce de monnaie truquée dont les deux côtés sont face)
et une probabilité de $0$ indique une impossibilité
(par exemple, si les deux côtés sont pile). 
Les fréquences $n_h/n$ et $n_t/n$ ne sont pas des probabilités
mais plutôt des *statistiques*.
Les probabilités sont des quantités *théoriques* 
qui sous-tendent le processus de génération des données.
Ici, la probabilité $1/2$ 
 est une propriété de la pièce elle-même.
En revanche, les statistiques sont des quantités *empiriques*
qui sont calculées en tant que fonctions des données observées.
Nos intérêts pour les quantités probabilistes et statistiques
sont inextricablement liés.
Nous concevons souvent des statistiques spéciales appelées *estimateurs*
qui, étant donné un ensemble de données, produisent des *estimations* 
de paramètres de modèles comme les probabilités.
De plus, lorsque ces estimateurs satisfont 
une propriété intéressante appelée *cohérence*,
nos estimations convergeront 
vers la probabilité correspondante.
À leur tour, ces probabilités déduites
nous renseignent sur les propriétés statistiques probables
des données de la même population
que nous pourrions rencontrer à l'avenir.

Supposons que nous tombions par hasard sur une pièce de monnaie réelle
pour laquelle nous ne connaissons pas 
le vrai $P(\textrm{heads})$.
Pour étudier cette quantité 
avec des méthodes statistiques,
nous devons (i) collecter des données ;
et (ii) concevoir un estimateur.
L'acquisition de données est ici facile ;
nous pouvons lancer la pièce de monnaie plusieurs fois
et enregistrer tous les résultats.
Formellement, le fait de tirer des réalisations 
d'un certain processus aléatoire sous-jacent 
est appelé *échantillonnage*.
Comme vous l'avez peut-être deviné, 
un estimateur naturel 
est la fraction entre
le nombre de *têtes* observées
par le nombre total de tirages.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

Supposons maintenant que la pièce soit en fait juste,
c'est-à-dire $P(\textrm{heads}) = 0.5$.
Pour simuler les tirages d'une pièce de monnaie équitable,
nous pouvons faire appel à n'importe quel générateur de nombres aléatoires.
Il existe des moyens simples de tirer des échantillons 
d'un événement avec une probabilité $0.5$.
Par exemple, le générateur de nombres aléatoires de Python `random.random`
 donne des nombres dans l'intervalle $[0,1]$
 où la probabilité de se trouver 
dans n'importe quel sous-intervalle $[a, b] \subset [0,1]$
 est égale à $b-a$.
Ainsi, nous pouvons extraire `0` et `1` avec la probabilité `0.5` chaque
en testant si le flottant retourné est supérieur à `0.5`

```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(100)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

Plus généralement, nous pouvons simuler plusieurs tirages 
à partir de n'importe quelle variable avec un nombre fini 
de résultats possibles
(comme le lancer d'une pièce de monnaie ou d'un dé)
en appelant la fonction multinomiale, 
en définissant le premier argument 
comme le nombre de tirages
et le second comme une liste de probabilités
associées à chacun des résultats possibles.
Pour simuler dix tirages d'une pièce de monnaie équitable, 
, nous attribuons le vecteur de probabilité `[0.5, 0.5]`,
en interprétant l'indice 0 comme étant pile
et l'indice 1 comme étant face.
La fonction renvoie un vecteur 
dont la longueur est égale au nombre 
de résultats possibles (ici, 2),
où la première composante nous indique 
le nombre d'occurrences de pile
et la deuxième composante nous indique 
le nombre d'occurrences de face.

```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5] 
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

Chaque fois que vous exécutez ce processus d'échantillonnage,
vous recevez une nouvelle valeur aléatoire 
qui peut être différente du résultat précédent. 
En divisant par le nombre de lancers
, nous obtenons la *fréquence* 
de chaque résultat dans nos données.
Notez que ces fréquences,
comme les probabilités 
qu'elles sont censées 
estimer, ont une somme égale à $1$.

```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

Ici, même si notre pièce de monnaie simulée est juste 
(nous avons défini nous-mêmes les probabilités `[0.5, 0.5]` ),
les nombres de pile et de face peuvent ne pas être identiques.
Cela s'explique par le fait que nous n'avons tiré qu'un nombre limité d'échantillons.
Si nous ne réalisions pas la simulation nous-mêmes,
et ne voyions que le résultat, 
comment saurions-nous si la pièce est légèrement injuste
ou si l'écart possible par rapport à $1/2$ est 
simplement un artefact de la petite taille de l'échantillon ?
Voyons ce qui se passe lorsque nous simulons `10000` des lancers de pièces.

```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

En général, pour les moyennes d'événements répétés (comme les lancers de pièces de monnaie),
plus le nombre de répétitions augmente, 
plus nos estimations sont garanties de converger
vers les véritables probabilités sous-jacentes. 
La preuve mathématique de ce phénomène
est appelée la *loi des grands nombres*
et le *théorème central limite*
nous dit que dans de nombreuses situations,
à mesure que la taille de l'échantillon $n$ augmente,
ces erreurs devraient diminuer 
à un taux de $(1/\sqrt{n})$.
Prenons un peu plus d'intuition en étudiant 
comment notre estimation évolue lorsque nous augmentons
le nombre de lancers de `1` à `10000`.

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab all
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Chaque courbe pleine correspond à l'une des deux valeurs de la pièce 
et donne notre estimation de la probabilité que la pièce donne cette valeur 
après chaque groupe d'expériences.
La ligne noire en pointillés donne la véritable probabilité sous-jacente.
Au fur et à mesure que nous obtenons plus de données en réalisant plus d'expériences, 
les courbes convergent vers la vraie probabilité.
Vous commencez peut-être déjà à entrevoir la forme 
de certaines des questions plus pointues
qui préoccupent les statisticiens :
À quelle vitesse cette convergence se produit-elle ?
Si nous avions déjà testé de nombreuses pièces
fabriquées dans la même usine, 
comment pourrions-nous intégrer cette information ?

## Un traitement plus formel

Nous sommes déjà allés assez loin : poser 
un modèle probabiliste,
générer des données synthétiques,
exécuter un estimateur statistique,
évaluer empiriquement la convergence,
et rendre compte des mesures d'erreur (vérifier la déviation). 
Cependant, pour aller beaucoup plus loin,
nous devrons être plus précis.


Lorsqu'il est question d'aléatoire, 
nous désignons l'ensemble des résultats possibles $\mathcal{S}$
 et l'appelons *espace d'échantillonnage* ou *espace des résultats*.
Ici, chaque élément est un *résultat possible distinct.
Dans le cas du lancer d'une seule pièce de monnaie,
$\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$ .
Dans le cas d'un seul dé, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
Si l'on tire à pile ou face sur deux pièces, il y a quatre résultats possibles :
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$ .
*Les événements* sont des sous-ensembles de l'espace d'échantillonnage.
Par exemple, l'événement "le premier lancer de pièce donne lieu à un pile ou face"
correspond à l'ensemble $\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$.
Lorsque le résultat $z$ d'une expérience aléatoire est conforme à
$z \in \mathcal{A}$ , l'événement $\mathcal{A}$ s'est produit.
Pour un seul lancer de dé, nous pourrions définir les événements 
"voir un $5$" ($\mathcal{A} = \{5\}$) 
et "voir un nombre impair" ($\mathcal{B} = \{1, 3, 5\}$).
Dans ce cas, si le dé tombe sur `5`,
nous dirons que $A$ et $B$ se sont produits.
En revanche, si $z = 3$, 
alors $\mathcal{A}$ n'a pas eu lieu 
mais $\mathcal{B}$ oui.


Une fonction de *probabilité* associe des événements 
à des valeurs réelles ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
La probabilité d'un événement $\mathcal{A}$ 
 dans l'espace d'échantillonnage donné $\mathcal{S}$,
noté $P(\mathcal{A})$,
satisfait aux propriétés suivantes :

* La probabilité de tout événement $\mathcal{A}$ est un nombre réel non négatif, c'est-à-dire $P(\mathcal{A}) \geq 0$;
* La probabilité de l'espace d'échantillonnage entier est $1$, c'est-à-dire , $P(\mathcal{S}) = 1$;
* Pour toute séquence dénombrable d'événements $\mathcal{A}_1, \mathcal{A}_2, \ldots$ qui sont *mutuellement exclusifs* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ pour tout $i \neq j$), la probabilité que l'un d'entre eux se produise est égale à la somme de leurs probabilités individuelles, c'est-à-dire $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Ces axiomes de la théorie des probabilités,
proposés par :cite:`Kolmogorov.1933` ,
peuvent être appliqués pour dériver rapidement un certain nombre de conséquences importantes.
Par exemple, il s'ensuit immédiatement
que la probabilité que tout événement $\mathcal{A}$
 *ou* son complément $\mathcal{A}'$ se produise est de 1 
(car $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Nous pouvons également prouver que $P(\emptyset) = 0$
 parce que $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Par conséquent, la probabilité que tout événement $\mathcal{A}$
 *et* son complément $\mathcal{A}'$ se produise simultanément 
est $P(\mathcal{A} \cap \mathcal{A}') = 0$.
De manière informelle, cela nous indique que les événements impossibles
ont une probabilité nulle de se produire. 
 


## Variables aléatoires

Lorsque nous avons parlé d'événements tels que le lancer d'un dé
qui donne une cote ou le premier lancer d'une pièce de monnaie qui donne face,
nous avons invoqué l'idée d'une *variable aléatoire*.
Formellement, les variables aléatoires sont des mappings
d'un espace d'échantillonnage sous-jacent 
vers un ensemble de valeurs (éventuellement nombreuses).
Vous pouvez vous demander en quoi une variable aléatoire 
est différente de l'espace d'échantillonnage
, puisque tous deux sont des collections de résultats. 
Il est important de noter que les variables aléatoires peuvent être beaucoup plus grossières 
que l'espace d'échantillonnage brut.
Nous pouvons définir une variable aléatoire binaire comme "supérieure à 0,5"
même lorsque l'espace d'échantillonnage sous-jacent est infini,
par exemple, le segment de droite entre $0$ et $1$.
En outre, plusieurs variables aléatoires
peuvent partager le même espace d'échantillonnage sous-jacent.
Par exemple, "si l'alarme de ma maison se déclenche"
et "si ma maison a été cambriolée"
sont deux variables aléatoires binaires 
qui partagent un espace d'échantillonnage sous-jacent. 
Par conséquent, le fait de connaître la valeur prise par une variable aléatoire 
peut nous renseigner sur la valeur probable d'une autre variable aléatoire.
Sachant que l'alarme s'est déclenchée, 
nous pouvons soupçonner que la maison a probablement été cambriolée.


Chaque valeur prise par une variable aléatoire correspond 
à un sous-ensemble de l'espace d'échantillonnage sous-jacent.
Ainsi, l'occurrence où la variable aléatoire $X$
 prend la valeur $v$, désignée par $X=v$, est un *événement*
et $P(X=v)$ désigne sa probabilité.
Parfois, cette notation peut devenir encombrante,
et nous pouvons abuser de la notation lorsque le contexte est clair.
Par exemple, nous pouvons utiliser $P(X)$ pour faire référence de manière générale à
à la *distribution* de $X$, c'est-à-dire 
la fonction qui nous indique la probabilité
que $X$ prenne une valeur donnée. 
D'autres fois, nous écrivons des expressions 
comme $P(X,Y) = P(X) P(Y)$,
comme raccourci pour exprimer une déclaration 
qui est vraie pour toutes les valeurs
que les variables aléatoires $X$ et $Y$ peuvent prendre, c'est-à-dire
pour toutes les $i,j$ il est vrai que $P(X=i \textrm{ and } Y=j) = P(X=i)P(Y=j)$.
D'autres fois, nous abusons de la notation en écrivant
$P(v)$ lorsque la variable aléatoire est claire d'après le contexte. 
Étant donné que, dans la théorie des probabilités, un événement est un ensemble d'issues de l'espace d'échantillonnage,
nous pouvons spécifier une plage de valeurs pour une variable aléatoire.
Par exemple, $P(1 \leq X \leq 3)$ désigne la probabilité de l'événement $\{1 \leq X \leq 3\}$.


Notez qu'il existe une différence subtile 
entre les variables aléatoires *discrètes*,
comme les tirages à pile ou face ou les lancers de dé, 
et les variables *continues*,
comme le poids et la taille d'une personne
échantillonnée au hasard dans la population.
Dans ce cas, il est rare que la taille exacte d'une personne nous intéresse vraiment 
. 
De plus, si nous prenions des mesures suffisamment précises,
nous constaterions que deux personnes sur la planète 
n'ont pas exactement la même taille. 
En fait, avec des mesures suffisamment fines, 
vous n'auriez jamais la même taille 
quand vous vous réveillez et quand vous vous couchez. 
Il ne sert pas à grand-chose de demander à 
la probabilité exacte que quelqu'un 
mesure 1,801392782910287192 mètres. 
En revanche, il s'agit plutôt de pouvoir dire
si la taille d'une personne se situe dans un intervalle donné, 
par exemple entre 1,79 et 1,81 mètre. 
Dans ce cas, nous travaillons avec des *densités de probabilité*. 
La taille d'exactement 1,80 mètre 
n'a pas de probabilité, mais une densité non nulle. 
Pour obtenir la probabilité attribuée à un intervalle,
nous devons prendre une *intégrale* de la densité 
sur cet intervalle. 




## Variables aléatoires multiples

Vous avez peut-être remarqué que nous ne pouvions pas dépasser la dernière section
sans
faire des déclarations impliquant des interactions 
entre plusieurs variables aléatoires
(rappelez-vous $P(X,Y) = P(X) P(Y)$).
La majeure partie de l'apprentissage automatique 
s'intéresse à ces relations.
Ici, l'espace d'échantillonnage serait 
la population d'intérêt,
par exemple les clients qui effectuent une transaction avec une entreprise,
les photographies sur Internet,
ou les protéines connues des biologistes.
Chaque variable aléatoire représenterait 
la valeur (inconnue) d'un attribut différent. 
Chaque fois que nous échantillonnons un individu de la population,
nous observons une réalisation de chacune des variables aléatoires.
Étant donné que les valeurs prises par les variables aléatoires 
correspondent à des sous-ensembles de l'espace d'échantillonnage 
qui peuvent se chevaucher, se chevaucher partiellement, 
ou être entièrement disjoints,
la connaissance de la valeur prise par une variable aléatoire
peut nous amener à mettre à jour nos croyances 
sur les valeurs probables d'une autre variable aléatoire.
Si un patient entre dans un hôpital 
et que nous observons qu'il 
a des difficultés à respirer
et qu'il a perdu son odorat,
alors nous pensons qu'il est plus probable
qu'il soit atteint de la COVID-19 que nous le ferions 
s'il n'avait aucune difficulté à respirer
et un odorat parfaitement ordinaire.


Lorsque nous travaillons avec des variables aléatoires multiples,
nous pouvons construire des événements correspondant 
à chaque combinaison de valeurs 
que les variables peuvent prendre conjointement.
La fonction de probabilité qui attribue
des probabilités à chacune de ces combinaisons
(par exemple $A=a$ et $B=b$)
est appelée fonction de *probabilité conjointe*
et renvoie simplement la probabilité attribuée 
à l'intersection des sous-ensembles correspondants
de l'espace d'échantillonnage. 
La probabilité *conjointe* attribuée à l'événement 
où les variables aléatoires $A$ et $B$ 
 prennent les valeurs $a$ et $b$, respectivement,
est désignée par $P(A = a, B = b)$,
où la virgule indique "et". 
Notez que pour toute valeur de $a$ et de $b$,
il s'avère que
$P(A=a, B=b) \leq P(A=a)$ 
 et $P(A=a, B=b) \leq P(B = b)$,
puisque pour que $A=a$ et $B=b$ se produisent,
$A=a$ doit se produire *et* $B=b$ doit également se produire.
Il est intéressant de noter que la probabilité conjointe
nous dit tout ce que nous pouvons savoir sur ces
variables aléatoires au sens probabiliste,
et peut être utilisée pour dériver de nombreuses autres
quantités utiles, y compris pour récupérer les 
distributions individuelles $P(A)$ et $P(B)$.
Pour récupérer $P(A=a) $, il suffit de faire la somme de 
$P(A=a, B=v)$ sur toutes les valeurs $v$ 
 que peut prendre la variable aléatoire $B$:
$P(A=a) = \sum_v P(A=a, B=v)$ .


Le rapport $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
 s'avère être extrêmement important.
Il est appelé probabilité *conditionnelle*,
et est désigné par le symbole "|",
$P(B=b|A=a) = P(A=a,B=b)/P(A=a)$ .
Elle nous indique la nouvelle probabilité
associée à l'événement $B=b$,
une fois que nous avons conditionné le fait que $A=a$ a eu lieu.
Nous pouvons considérer cette probabilité conditionnelle
comme restreignant l'attention au seul sous-ensemble
de l'espace d'échantillonnage associé à $A=a$
 , puis en la renormalisant de sorte que
la somme de toutes les probabilités soit égale à 1.
Les probabilités conditionnelles 
sont en fait des probabilités
et respectent donc tous les axiomes,
tant que nous conditionnons tous les termes 
au même événement et donc que 
restreint l'attention au même espace d'échantillonnage. 
Par exemple, pour les événements disjoints 
$\mathcal{B}$ et $\mathcal{B}'$, nous avons que 
$P(\mathcal{B} \cup \mathcal{B}'|A = a) = P(\mathcal{B}|A = a) + P(\mathcal{B}'|A = a)$ . 


En utilisant la définition des probabilités conditionnelles, 
nous pouvons dériver le célèbre résultat appelé *théorème de Bayes*.
Par construction, nous avons que $P(A, B) = P(B|A) P(A)$ 
 et $P(A, B) = P(A|B) P(B)$. 
En combinant les deux équations, on obtient 
$P(B|A) P(A) = P(A|B) P(B)$ et donc 

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}.$$ 

 Cette simple équation a de profondes implications car
elle nous permet d'inverser l'ordre du conditionnement.
Si nous savons comment estimer $P(B|A)$, $P(A)$, et $P(B)$,
nous pouvons alors estimer $P(A|B)$. 
Il est souvent plus facile d'estimer directement un terme 
mais pas l'autre et le théorème de Bayes peut venir à la rescousse dans ce cas.
Par exemple, si nous connaissons la prévalence des symptômes d'une maladie donnée,
et les prévalences globales de la maladie et des symptômes, respectivement,
nous pouvons déterminer la probabilité qu'une personne 
soit atteinte de la maladie sur la base de ses symptômes.
Dans certains cas, nous n'avons pas d'accès direct à $P(B)$, 
comme la prévalence des symptômes. 
Dans ce cas, une version simplifiée du théorème de Bayes s'avère utile :

$$P(A|B) \propto P(B|A) P(A).$$ 

 Comme nous savons que $P(A|B)$ doit être normalisé à $1$, c'est-à-dire , $\sum_a P(A=a|B) = 1$
 nous pouvons l'utiliser pour calculer

$$P(A|B) = \frac{P(B|A) P(A)}{\sum_b P(B=b|A) P(A)}.$$ 

 Dans les statistiques bayésiennes, nous considérons qu'un observateur 
possède des croyances préalables (subjectives)
sur la plausibilité des hypothèses disponibles 
encodées dans le *préalable* $P(H)$,
et une *fonction de vraisemblance* qui indique la probabilité 
d'observer une valeur quelconque des preuves collectées 
pour chacune des hypothèses de la classe $P(E|H)$.
Le théorème de Bayes est alors interprété comme nous indiquant
comment mettre à jour le *prior* initial $P(H)$
 à la lumière des preuves disponibles $E$
 pour produire des croyances *posterior* 
$P(H|E) = \frac{P(E|H) P(H)}{P(E)}$ .
De manière informelle, cela peut être énoncé comme suit : 
"postérieur est égal à antérieur multiplié par la probabilité, divisé par la preuve".
Maintenant, comme la preuve $P(E)$ est la même pour toutes les hypothèses,
nous pouvons nous en sortir en normalisant simplement sur les hypothèses.

Notez que $\sum_a P(A=a|B) = 1$ nous permet également de *marginaliser* les variables aléatoires. En d'autres termes, nous pouvons supprimer les variables d'une distribution conjointe telle que $P(A, B)$. Après tout, nous savons que 

$$\sum_a P(A=a, B) = P(B) \sum_a P(A = a|B) = P(B).$$ 

 L'indépendance est un autre concept fondamental
qui constitue l'épine dorsale de 
nombreuses idées importantes en statistiques.
En bref, deux variables sont *indépendantes*
si le conditionnement de la valeur de $A$ n'entraîne pas
de modification de la distribution de probabilité
associée à $B$ et vice versa.
Plus formellement, l'indépendance, notée $A \perp B$, 
exige que $P(A|B) = P(A)$ et, par conséquent, 
que $P(A,B) = P(A|B) P(B) = P(A) P(B)$.
L'indépendance est souvent une hypothèse appropriée.
Par exemple, si la variable aléatoire $A$ 
 représente le résultat du lancer d'une pièce de monnaie équitable 
et la variable aléatoire $B$ 
 représente le résultat du lancer d'une autre pièce de monnaie,
alors le fait de savoir si $A$ est tombé sur face
ne devrait pas influencer la probabilité
que $B$ tombe sur face.


L'indépendance est particulièrement utile lorsqu'elle est valable pour les tirages successifs 
de nos données à partir d'une distribution sous-jacente 
(ce qui nous permet de tirer des conclusions statistiques solides)
ou lorsqu'elle est valable pour diverses variables de nos données,
ce qui nous permet de travailler avec des modèles plus simples
qui encodent cette structure d'indépendance.
D'autre part, l'estimation des dépendances 
entre les variables aléatoires est souvent le but même de l'apprentissage.
Nous nous soucions d'estimer la probabilité de la maladie compte tenu des symptômes
spécifiquement parce que nous pensons 
que les maladies et les symptômes ne sont *pas* indépendants. 


Notez que, comme les probabilités conditionnelles sont des probabilités propres,
les concepts d'indépendance et de dépendance s'appliquent également à elles. 
Deux variables aléatoires $A$ et $B$ sont *conditionnellement indépendantes* 
étant donné une troisième variable $C$ si et seulement si $P(A, B|C) = P(A|C)P(B|C)$.
Il est intéressant de noter que deux variables peuvent être indépendantes en général
mais devenir dépendantes lorsqu'elles sont conditionnées par une troisième. 
Cela se produit souvent lorsque les deux variables aléatoires $A$ et $B$
 correspondent aux causes d'une troisième variable $C$.
Par exemple, les os cassés et le cancer du poumon peuvent être indépendants 
dans la population générale, mais si nous conditionnons sur le fait d'être à l'hôpital
, nous pouvons constater que les os cassés sont négativement corrélés au cancer du poumon. 
Cela s'explique par le fait que l'os cassé * explique pourquoi une personne est à l'hôpital
et réduit donc la probabilité qu'elle ait un cancer du poumon. 


Et inversement, deux variables aléatoires dépendantes 
peuvent devenir indépendantes si elles sont conditionnées par une troisième. 
Cela se produit souvent lorsque deux événements sans rapport entre eux
ont une cause commune. 
La taille des chaussures et le niveau de lecture sont fortement corrélés 
parmi les élèves de l'école primaire,
mais cette corrélation disparaît si nous conditionnons sur l'âge. 



## Un exemple
:label:`subsec_probability_hiv_app` 

 Mettons nos compétences à l'épreuve. 
Supposons qu'un médecin administre un test de dépistage du VIH à un patient. 
Ce test est assez précis et n'échoue qu'avec une probabilité de 1 % 
si le patient est en bonne santé mais le signale comme malade. 
De plus, il n'échoue jamais à détecter le VIH si le patient en est effectivement atteint. 
Nous utilisons $D_1 \in \{0, 1\}$ pour indiquer le diagnostic 
($0$ si négatif et $1$ si positif)
et $H \in \{0, 1\}$ pour indiquer le statut VIH.

| Probabilité conditionnelle | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 | H)$ | 1 | 0,01 |
| $P(D_1 = 0 | H)$ | 0 | 0,99 |

Notez que les sommes des colonnes sont toutes égales à 1 (mais pas les sommes des lignes), 
puisqu'il s'agit de probabilités conditionnelles.
Calculons la probabilité que le patient ait le VIH 
si le test est positif, c'est-à-dire $P(H = 1|D_1 = 1)$. 
Intuitivement, cela va dépendre de la fréquence de la maladie,
puisque cela affecte le nombre de fausses alarmes. 
Supposons que la population soit en assez bonne santé, par exemple $P(H=1) = 0.0015$. 
Pour appliquer le théorème de Bayes, nous devons appliquer la marginalisation
afin de déterminer

$$\begin{aligned}
P(D_1 = 1) 
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1|H=0) P(H=0) + P(D_1=1|H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Cela nous conduit à 

$$P(H = 1|D_1 = 1) = \frac{P(D_1=1|H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$ 

 En d'autres termes, il n'y a que 13,06 % de chances 
que le patient soit réellement séropositif, 
malgré l'utilisation d'un test très précis.
Comme nous pouvons le constater, la probabilité peut être contre-intuitive.
Que devrait faire un patient en recevant une nouvelle aussi terrifiante ?
Il est probable qu'il demande au médecin 
de lui faire passer un autre test pour y voir plus clair. 
Le deuxième test a des caractéristiques différentes
et il n'est pas aussi bon que le premier.

| Probabilité conditionnelle | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1|H)$ | 0,98 | 0,03 |
| $P(D_2 = 0|H)$ | 0,02 | 0,97 |

Malheureusement, le deuxième test se révèle également positif.
Calculons les probabilités nécessaires pour invoquer le théorème de Bayes
en supposant une indépendance conditionnelle :

$$\begin{aligned}
P(D_1 = 1, D_2 = 1|H = 0) 
& = P(D_1 = 1|H = 0) P(D_2 = 1|H = 0)  
=& 0.0003, \\
P(D_1 = 1, D_2 = 1|H = 1) 
& = P(D_1 = 1|H = 1) P(D_2 = 1|H = 1)  
=& 0.98.
\end{aligned}
$$

Nous pouvons maintenant appliquer la marginalisation pour obtenir la probabilité 
que les deux tests reviennent positifs :

$$\begin{aligned}
P(D_1 = 1, D_2 = 1) 
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1| H = 0)P(H=0) + P(D_1 = 1, D_2 = 1|H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

Finalement, la probabilité que le patient soit séropositif étant donné que les deux tests sont positifs est de

$$P(H = 1| D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1|H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

En d'autres termes, le second test nous a permis d'acquérir une confiance bien plus grande dans le fait que tout ne va pas bien.
Bien que le second test soit considérablement moins précis que le premier, 
il a tout de même amélioré notre estimation de manière significative. 
L'hypothèse selon laquelle les deux tests sont conditionnellement indépendants l'un de l'autre 
a été cruciale pour notre capacité à générer une estimation plus précise.
Prenons le cas extrême où nous exécutons deux fois le même test. 
Dans cette situation, nous nous attendons à ce que le résultat soit le même dans les deux cas, 
. Il n'y a donc rien de plus à tirer de la répétition du même test. 
Le lecteur avisé aura peut-être remarqué que le diagnostic se comporte 
comme un classificateur caché 
où notre capacité à décider si un patient est en bonne santé 
augmente à mesure que nous obtenons plus de caractéristiques (résultats du test).


## Attentes

Souvent, pour prendre des décisions, il ne suffit pas d'examiner 
les probabilités attribuées aux événements individuels
, mais il faut les regrouper en agrégats utiles
qui peuvent nous guider.
Par exemple, lorsque les variables aléatoires prennent des valeurs scalaires continues,
nous nous intéressons souvent à la valeur à laquelle nous devons nous attendre *en moyenne*.
Cette quantité est formellement appelée une *espérance*.
Si nous faisons des investissements,
la première quantité d'intérêt
pourrait être le rendement que nous pouvons attendre, 
en faisant la moyenne de tous les résultats possibles
(et en les pondérant par les probabilités appropriées).
Par exemple, disons qu'avec une probabilité de 50 %, 
un investissement peut échouer complètement,
avec une probabilité de 40 % il peut fournir un rendement de 2$\times$,
et avec une probabilité de 10 % il peut fournir un rendement de 10$\times$ 10$\times$.
Pour calculer le rendement attendu,
nous additionnons tous les rendements, en multipliant chacun
par la probabilité qu'ils se produisent. 
On obtient ainsi l'espérance 
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$ . 
Le rendement attendu est donc de 1,8$\times$.


 En général, l'espérance * (ou moyenne)
de la variable aléatoire $X$ est définie comme suit :

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$ 

 De même, pour les densités, nous obtenons $E[X] = \int x \;dp(x)$. 
Parfois, nous nous intéressons à la valeur attendue
d'une certaine fonction de $x$.
Nous pouvons calculer ces attentes sous la forme

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \text{ and } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$ 

 pour les probabilités et les densités discrètes, respectivement. 
Pour en revenir à l'exemple d'investissement ci-dessus, 
$f$ pourrait être l'*utilité* (bonheur)
associée au rendement. 
Les économistes comportementaux ont remarqué depuis longtemps
que les gens associent une plus grande désutilité
à la perte d'argent que l'utilité gagnée
en gagnant un dollar par rapport à leur base de référence.
En outre, la valeur de l'argent a tendance à être sub-linéaire. 
Posséder 100 000 dollars contre zéro dollar
peut faire la différence entre payer le loyer,
bien manger et bénéficier de soins de santé de qualité 
et souffrir du sans-abrisme.
En revanche, les gains dus à la possession de
200k contre 100k sont moins spectaculaires.
Un tel raisonnement motive le cliché 
selon lequel "l'utilité de l'argent est logarithmique".


Si l'utilité associée à une perte totale était de -1,
et que les utilités associées à des rendements de 1, 2 et 10 
étaient respectivement de 1, 2 et 4, 
alors le bonheur attendu d'investir 
serait $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
 (une perte d'utilité attendue de 30%). 
Si telle était votre fonction d'utilité, 
il serait préférable de garder l'argent à la banque. 

Pour les décisions financières, 
nous pouvons également vouloir mesurer 
le degré de *risque* d'un investissement. 
Dans ce cas, nous ne nous intéressons pas seulement à la valeur attendue
, mais à la mesure dans laquelle les valeurs réelles ont tendance à *varier*
par rapport à cette valeur. 
Notez que nous ne pouvons pas simplement prendre 
l'espérance de la différence 
entre les valeurs réelles et attendues.
En effet, l'espérance d'une différence 
est la différence des espérances,
et donc $E[X - E[X]] = E[X] - E[E[X]] = 0$.
Cependant, nous pouvons examiner l'espérance 
de toute fonction non négative de cette différence.
La *variance* d'une variable aléatoire est calculée en examinant 
la valeur attendue des écarts *au carré* :

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$ 

 Ici, l'égalité découle de l'expansion de 
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$ 
 et de la prise en compte des attentes pour chaque terme. 
La racine carrée de la variance est une autre 
quantité utile appelée *écart-type*. 
Alors que la variance et l'écart-type
véhiculent la même information (l'un peut être calculé à partir de l'autre),
l'écart-type a la propriété intéressante 
d'être exprimé dans les mêmes unités 
que la quantité originale représentée
par la variable aléatoire.

Enfin, la variance d'une fonction 
d'une variable aléatoire 
est définie de manière analogue comme suit : 

$$\mathrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$ 

 Pour revenir à notre exemple d'investissement,
nous pouvons maintenant calculer la variance de l'investissement. 
Elle est donnée par $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$. 
À toutes fins utiles, il s'agit d'un investissement risqué. 
Notez que, par convention mathématique, la moyenne et la variance 
sont souvent désignées par $\mu$ et $\sigma^2$.
Cela est particulièrement courant lorsque nous les utilisons 
pour paramétrer une distribution gaussienne. 

De la même manière que nous avons introduit les attentes 
et la variance pour les variables aléatoires *scalaires*, 
nous pouvons le faire pour les variables vectorielles. 
Les espérances sont faciles, car nous pouvons les appliquer par éléments. 
Par exemple, $\boldsymbol{\mu} \stackrel{\mathrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$ 
 a pour coordonnées $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
Les covariances sont plus compliquées. 
Nous résolvons le problème en prenant les attentes du *produit extérieur* 
de la différence entre les variables aléatoires et leur moyenne. 

$$\boldsymbol{\Sigma} \stackrel{\mathrm{def}}{=} \mathrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

Cette matrice $\boldsymbol{\Sigma}$ est appelée matrice de covariance. 
Une façon simple de voir son effet est de considérer un vecteur $\mathbf{v}$ 
 de la même taille que $\mathbf{x}$. 
Il s'ensuit que 

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \mathrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$ 

 . Ainsi, $\boldsymbol{\Sigma}$ nous permet de calculer la variance 
pour toute fonction linéaire de $\mathbf{x}$
 par une simple multiplication matricielle. 
Les éléments hors diagonale nous indiquent dans quelle mesure les coordonnées sont corrélées :
une valeur de 0 signifie qu'il n'y a pas de corrélation, 
où une valeur positive plus grande 
signifie qu'elles sont plus fortement corrélées. 



## Discussion

Dans l'apprentissage automatique, les incertitudes sont nombreuses !
Nous pouvons être incertains de la valeur d'une étiquette donnée en entrée.
Nous pouvons être incertains de la valeur estimée d'un paramètre.
Nous pouvons même être incertains quant à savoir si les données arrivant au déploiement
proviennent de la même distribution que les données d'apprentissage.

Par incertitude *aléatoire*, nous désignons l'incertitude 
qui est intrinsèque au problème, 
et due à un véritable aléa 
non pris en compte par les variables observées.
Par *incertitude épistémique*, nous désignons l'incertitude
sur les paramètres d'un modèle, le type d'incertitude
que nous pouvons espérer réduire en collectant davantage de données.
Nous pouvons avoir une incertitude épistémique 
concernant la probabilité 
qu'une pièce de monnaie tombe sur face,
mais, même une fois que nous connaissons cette probabilité,
il nous reste une incertitude aléatoire 
sur le résultat de tout lancer futur.
Peu importe le temps que nous passons à regarder quelqu'un lancer une pièce de monnaie équitable, 
nous ne serons jamais plus ou moins sûrs à 50% 
que le prochain lancer sera face.
Ces termes sont issus de la littérature sur la modélisation mécanique,
(voir, par exemple, :cite:`Der-Kiureghian.Ditlevsen.2009` pour une revue de cet aspect de [uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
Il convient de noter que ces termes constituent un léger abus de langage.
Le terme *épistémique* fait référence à tout ce qui concerne la *connaissance*
et donc, au sens philosophique, toute incertitude est épistémique.


Nous avons vu que l'échantillonnage de données à partir d'une certaine distribution de probabilité inconnue 
peut nous fournir des informations qui peuvent être utilisées pour estimer
les paramètres de la distribution générant les données.
Cela dit, la vitesse à laquelle cela est possible peut être assez lente. 
Dans notre exemple de lancer de pièce de monnaie (et bien d'autres) 
, nous ne pouvons pas faire mieux que de concevoir des estimateurs
qui convergent à un taux de $1/\sqrt{n}$,
où $n$ est la taille de l'échantillon (par exemple, le nombre de lancers). 
Cela signifie qu'en passant de 10 à 1000 observations (une tâche généralement très réalisable) 
, nous constatons une réduction de l'incertitude par dix, 
alors que les 1000 observations suivantes sont comparativement peu utiles, 
offrant seulement une réduction de 1,41 fois. 
Il s'agit d'une caractéristique persistante de l'apprentissage automatique : 
alors que les gains sont souvent faciles, il faut une très grande quantité de données, 
et souvent une énorme quantité de calculs pour obtenir des gains encore plus importants. 
Pour un examen empirique de ce fait pour les modèles de langage à grande échelle, voir :cite:`Revels.Lubin.Papamarkou.2016` . 

Nous avons également affiné notre langage et nos outils de modélisation statistique. 
Ce faisant, nous avons appris à connaître les probabilités conditionnelles 
et l'une des équations les plus importantes en statistique, le théorème de Bayes. 
Il s'agit d'un outil efficace pour découpler les informations véhiculées par les données 
à travers un terme de vraisemblance $P(B|A)$ qui traite 
de la façon dont les observations $B$ correspondent à un choix de paramètres $A$,
et une probabilité antérieure $P(A)$ qui régit la plausibilité 
d'un choix particulier de $A$ en premier lieu.
En particulier, nous avons vu comment cette règle peut être appliquée
pour attribuer des probabilités aux diagnostics,
en fonction de l'efficacité du test *et* 
de la prévalence de la maladie elle-même (c'est-à-dire notre a priori).

Enfin, nous avons introduit une première série de questions non triviales 
sur l'effet d'une distribution de probabilité spécifique,
à savoir les attentes et les variances. 
Bien qu'il existe bien d'autres espérances que les espérances linéaires et quadratiques 
pour une distribution de probabilité, 
ces deux-là fournissent déjà une bonne partie des connaissances 
sur le comportement possible de la distribution. 
Par exemple, [Chebyshev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
 indique que $P(|X - \mu| \geq k \sigma) \leq 1/k^2$, 
où $\mu$ est l'espérance, $\sigma^2$ est la variance de la distribution,
et $k > 1$ est un paramètre de confiance de notre choix. 
Elle nous indique que les tirages d'une distribution se situent 
avec une probabilité d'au moins 50 % 
à l'intérieur d'un intervalle $[-\sqrt{2} \sigma, \sqrt{2} \sigma]$
 centré sur l'espérance.




## Exercices

1. Donnez un exemple où l'observation d'un plus grand nombre de données peut réduire le degré d'incertitude concernant le résultat à un niveau arbitrairement bas. 
1. Donnez un exemple où l'observation d'un plus grand nombre de données ne réduit le degré d'incertitude que jusqu'à un certain point, sans plus. Expliquez pourquoi c'est le cas et où vous pensez que ce point se situe.
1. Nous avons démontré empiriquement la convergence vers la moyenne pour le lancer d'une pièce de monnaie. Calculez la variance de l'estimation de la probabilité de voir une tête après avoir tiré $n$ échantillons. 
    1. Comment la variance varie-t-elle en fonction du nombre d'observations ? 
    1. Utilisez l'inégalité de Chebyshev pour limiter l'écart par rapport à l'espérance. 
    1. Quel est le lien avec le théorème central limite ?
1. Supposez que nous tirions $n$ échantillons $x_i$ d'une distribution de probabilité de moyenne nulle et de variance unitaire. Calculez les moyennes $z_m \stackrel{\mathrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Pouvons-nous appliquer l'inégalité de Chebyshev pour chaque $z_m$ indépendamment ? Pourquoi pas ?
1. Étant donné deux événements avec une probabilité de $P(\mathcal{A})$ et $P(\mathcal{B})$, calculez les limites supérieures et inférieures de $P(\mathcal{A} \cup \mathcal{B})$ et $P(\mathcal{A} \cap \mathcal{B})$. Conseil : représentez la situation par un graphique [Venn diagram](https://en.wikipedia.org/wiki/Venn_diagram).
1. Supposez que nous ayons une séquence de variables aléatoires, disons $A$, $B$ et $C$, où $B$ ne dépend que de $A$, et $C$ ne dépend que de $B$, pouvez-vous simplifier la probabilité conjointe $P(A, B, C)$? Indice : il s'agit d'un [Markov chain](https://en.wikipedia.org/wiki/Markov_chain).
1. Dans :numref:`subsec_probability_hiv_app` , supposez que les résultats des deux tests ne sont pas indépendants. En particulier, supposez que l'un ou l'autre des tests a, à lui seul, un taux de faux positifs de 10 % et un taux de faux négatifs de 1 %. Autrement dit, supposons que $P(D =1|H=0) = 0.1$ et que $P(D = 0|H=1) = 0.01$. En outre, supposons que pour $H = 1$ (infecté), les résultats des tests sont conditionnellement indépendants, c'est-à-dire que $P(D_1, D_2|H=1) = P(D_1|H=1) P(D_2|H=1)$ mais que pour les patients sains, les résultats sont couplés via $P(D_1 = D_2 = 1|H=0) = 0.02$. 
    1. Sur la base des informations dont vous disposez jusqu'à présent, élaborez la table de probabilité conjointe pour $D_1$ et $D_2$, étant donné $H=0$.
   1. Déterminez la probabilité que le patient soit positif ($H=1$) après qu'un test soit positif. Vous pouvez supposer la même probabilité de base $P(H=1) = 0.0015$ que précédemment. 
    1. Déterminez la probabilité que le patient soit positif ($H=1$) lorsque les deux tests sont positifs.
1. Supposons que vous soyez gestionnaire d'actifs pour une banque d'investissement et que vous ayez un choix d'actions $s_i$ dans lesquelles investir. Votre portefeuille doit totaliser $1$ avec des pondérations $\alpha_i$ pour chaque action. Les actions ont un rendement moyen $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ et une covariance $\boldsymbol{\Sigma} = \mathrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
   1. Calculez le rendement attendu d'un portefeuille donné $\boldsymbol{\alpha}$.
   1. Si vous vouliez maximiser le rendement du portefeuille, comment devriez-vous choisir vos investissements ?
   1. Calculez la *variance* du portefeuille. 
    1. Formulez un problème d'optimisation pour maximiser le rendement tout en maintenant la variance dans une limite supérieure. Il s'agit du prix Nobel [Markovitz portfolio](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013` . Pour le résoudre, vous aurez besoin d'un solveur de programmation quadratique, ce qui dépasse largement le cadre de ce livre.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
