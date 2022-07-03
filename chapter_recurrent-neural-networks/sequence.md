# Working with Sequences
:label:`sec_sequence` 



Jusqu'à présent, nous nous sommes concentrés sur les modèles dont les entrées 
consistaient en un seul vecteur de caractéristiques $\mathbf{x} \ dans \mathbb{R}^d$.
Le principal changement de perspective lors du développement de modèles
capables de traiter des séquences est que nous nous concentrons désormais 
sur des entrées qui consistent en une liste ordonnée 
de vecteurs de caractéristiques $\mathbf{x}_1, \dots, \mathbf{x}_T$,
où chaque vecteur de caractéristiques $x_t$
indexé par un pas de séquence $t \in \mathbb{Z}^+$
se trouve dans $\mathbb{R}^d$.

Certains ensembles de données sont constitués d'une seule séquence massive.
Considérons, par exemple, les flux extrêmement longs
de relevés de capteurs dont peuvent disposer les climatologues. 
Dans ce cas, nous pouvons créer des ensembles de données d'entraînement
en échantillonnant de manière aléatoire des sous-séquences d'une longueur prédéterminée.
Plus souvent, nos données arrivent sous la forme d'une collection de séquences.
Prenons les exemples suivants : 
(i) une collection de documents,
chacun étant représenté comme sa propre séquence de mots,
et chacun ayant sa propre longueur $T_i$;
(ii) une représentation séquentielle de 
séjours de patients à l'hôpital,
où chaque séjour est constitué d'un certain nombre d'événements
et la longueur de la séquence dépend approximativement 
de la durée du séjour.


Auparavant, lorsque nous traitions des entrées individuelles,
nous supposions qu'elles étaient échantillonnées indépendamment 
à partir de la même distribution sous-jacente $P(X)$.
Si nous supposons toujours que des séquences entières 
(par exemple, des documents entiers ou des trajectoires de patients)
sont échantillonnées indépendamment,
nous ne pouvons pas supposer que les données arrivant 
à chaque étape de la séquence sont indépendantes les unes des autres. 
Par exemple, les mots qui sont susceptibles d'apparaître plus tard dans un document
dépendent fortement des mots qui sont apparus plus tôt dans le document. 
Le médicament qu'un patient est susceptible de recevoir 
le 10e jour d'une visite à l'hôpital 
dépend fortement de ce qui s'est passé 
au cours des neuf jours précédents. 

Cela ne devrait pas être une surprise.
Si nous ne pensions pas que les éléments d'une séquence étaient liés,
nous n'aurions pas pris la peine de les modéliser sous forme de séquence. 
Considérez l'utilité des fonctions de remplissage automatique
qui sont populaires sur les outils de recherche et les clients de messagerie modernes.
Elles sont utiles précisément parce qu'il est souvent possible 
de prédire (imparfaitement, mais mieux qu'une supposition aléatoire)
quelles pourraient être les continuations probables d'une séquence,
étant donné un certain préfixe initial. 
Pour la plupart des modèles de séquence,
nous n'exigeons pas l'indépendance,
ou même la stationnarité, de nos séquences. 
Au lieu de cela, nous exigeons seulement que 
les séquences elles-mêmes soient échantillonnées 
à partir d'une distribution sous-jacente fixe 
sur les séquences entières. 

 
Cette approche flexible permet de prendre en compte des phénomènes tels que
(i) des documents dont l'aspect est sensiblement différent 
au début et à la fin,
ou (ii) l'état d'un patient évoluant soit vers la guérison, soit vers la mort 
au cours d'un séjour à l'hôpital ;
et (iii) les goûts des clients évoluant de manière prévisible
au cours d'une interaction continue avec un système de recommandation.


Nous souhaitons parfois prédire une cible fixe $y$
à partir d'une entrée structurée de manière séquentielle
(par exemple, la classification de sentiments basée sur une critique de film). 
À d'autres moments, nous souhaitons prédire une cible structurée séquentiellement
($y_1, \cdots, y_T$)
étant donné une entrée fixe (par exemple, le sous-titrage d'une image).
D'autres fois encore, notre objectif est de prédire des cibles structurées séquentiellement
à partir d'entrées structurées séquentiellement 
(par exemple, traduction automatique ou sous-titrage vidéo).
Ces tâches de séquence à séquence peuvent prendre deux formes :
(a) **aligned :** où l'entrée à chaque étape de la séquence
s'aligne sur une cible correspondante (par exemple, le marquage des parties du discours) ;
(b) **unaligned** où l'entrée et la cible 
ne présentent pas nécessairement une correspondance étape par étape
(par exemple, la traduction automatique). 

Mais avant de nous préoccuper de la manipulation de cibles de toute sorte,
nous pouvons nous attaquer au problème le plus simple : 
modélisation de densité non supervisée (également appelée *modélisation de séquence*).
Ici, étant donné une collection de séquences, 
notre objectif est d'estimer la fonction de masse de probabilité
qui nous indique la probabilité de voir une séquence donnée,
c'est-à-dire $p(\mathbf{x}_1, \cdots, \mathbf{x}_T)$.





## Outils de base

Avant de présenter les réseaux neuronaux spécialisés 
conçus pour traiter des données structurées de manière séquentielle,
examinons quelques données de séquence réelles
et mettons en place quelques intuitions et outils statistiques de base.
Nous nous concentrerons plus particulièrement sur les données de cours de bourse 
de l'indice FTSE 100 (:numref:`fig_ftse100` ).
À chaque *pas de temps* $t \in \mathbb{Z}^+$, nous observons 
le prix de l'indice à ce moment-là, noté $x_t$.


![Indice FTSE 100 sur environ 30 ans](../img/ftse100.png)
:width:`400px` 
:label:`fig_ftse100` 

 
Supposons maintenant qu'un trader souhaite effectuer des transactions à court terme,
en entrant ou en sortant stratégiquement de l'indice, 
selon qu'il pense 
qu'il va augmenter ou baisser
au cours du pas de temps suivant. 
En l'absence de toute autre caractéristique 
(actualités, données financières, etc.),
le seul signal disponible pour prédire
la valeur ultérieure est l'historique des prix à ce jour. 




### Modèles autorégressifs

Le trader souhaite donc connaître 
la distribution de probabilité 
$$P(x_t \mid x_{t-1}, \ldots, x_1).$$ 
sur les prix que l'indice pourrait prendre 
au cours du pas de temps suivant.
Bien que l'estimation de l'ensemble de la distribution 
sur une variable aléatoire à valeur continue 
puisse être difficile, le trader sera heureux
de se concentrer sur quelques statistiques clés de la distribution,
en particulier la valeur attendue et la variance.
Une stratégie simple pour estimer l'espérance conditionnelle

$$ \mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)],$$

consiste à appliquer un modèle de régression linéaire,
(rappel :numref:`sec_linear_concise` ).
De tels modèles qui font régresser la valeur d'un signal
sur les valeurs précédentes de ce même signal 
sont naturellement appelés *modèles autorégressifs*.
Il n'y a qu'un seul problème majeur : le nombre d'entrées, 
$x_{t-1}, \ldots, x_1$ varie en fonction de $t$.
Plus précisément, le nombre d'entrées augmente 
avec la quantité de données que nous rencontrons.
Ainsi, si nous voulons traiter nos données historiques 
comme un ensemble d'apprentissage, nous nous retrouvons avec le problème 
que chaque exemple possède un nombre différent de caractéristiques.
Une grande partie de ce qui suit dans ce chapitre 
tournera autour des techniques 
permettant de surmonter ces difficultés 
lorsque nous nous engageons dans de tels problèmes de modélisation *autorégressive*
où l'objet d'intérêt est 
$P(x_t \mid x_{t-1}, \ldots, x_1)$
ou une ou plusieurs statistiques de cette distribution. 

Quelques stratégies reviennent fréquemment. 
Tout d'abord, nous pouvons penser que, bien que de longues séquences
$x_{t-1}, \ldots, x_1$ soient disponibles,
il n'est peut-être pas nécessaire 
de remonter aussi loin dans l'histoire 
pour prédire le futur proche. 
Dans ce cas, nous pourrions nous contenter 
de conditionner sur une certaine fenêtre de longueur $\tau$ 
et n'utiliser que les observations $x_{t-1}, \ldots, x_{t-\tau}$. 
L'avantage immédiat est que maintenant le nombre d'arguments 
est toujours le même, au moins pour $t &gt; \tau$. 
Cela nous permet d'entraîner n'importe quel modèle linéaire ou réseau profond 
qui nécessite des vecteurs de longueur fixe comme entrées.
Deuxièmement, nous pourrions développer des modèles qui maintiennent
un certain résumé $h_t$ des observations passées
(voir :numref:`fig_sequence-model` )
et qui, en même temps, mettent à jour $h_t$ 
en plus de la prédiction $\hat{x}_t$.
Cela conduit à des modèles qui estiment $x_t$ 
avec $\hat{x}_t = P(x_t \mid h_{t})$ 
et en outre à des mises à jour de la forme 
$h_t = g(h_{t-1}, x_{t-1})$. 
Comme $h_t$ n'est jamais observé, 
ces modèles sont également appelés 
*modèles autorégressifs latents*.

![Un modèle autorégressif latent](../img/sequence-model.svg)
:label:`fig_sequence-model` 

Pour construire des données d'apprentissage à partir de données historiques, on 
crée généralement des exemples en échantillonnant des fenêtres de manière aléatoire.
En général, nous ne nous attendons pas à ce que le temps s'arrête. 
Cependant, nous supposons souvent que si 
les valeurs spécifiques de $x_t$ peuvent changer,
la dynamique selon laquelle chaque observation suivante 
est générée compte tenu des observations précédentes ne change pas. 
Les statisticiens appellent les dynamiques qui ne changent pas *stationnaires*.



### Modèles de langage


Parfois, en particulier lorsque l'on travaille avec le langage,
nous souhaitons estimer la probabilité conjointe 
d'une séquence entière.
Il s'agit d'une tâche courante lorsqu'on travaille avec des séquences
composées de tokens discrets, tels que des mots. 
En général, ces fonctions estimées sont appelées *modèles de séquence* 
et pour les données en langage naturel, elles sont appelées *modèles de langage*.
Le domaine de la modélisation de séquences a été tellement influencé par le TAL,
que nous décrivons souvent les modèles de séquences comme des "modèles de langage",
même lorsqu'il s'agit de données non linguistiques. 
Les modèles linguistiques s'avèrent utiles pour toutes sortes de raisons.
Parfois, nous voulons évaluer la vraisemblance des phrases.
Par exemple, nous pouvons souhaiter comparer 
le caractère naturel de deux sorties candidates 
générées par un système de traduction automatique
ou par un système de reconnaissance vocale.
Mais la modélisation du langage nous donne non seulement 
la capacité d'*évaluer* la vraisemblance,
mais aussi la capacité d'*échantillonner* des séquences,
et même d'optimiser pour les séquences les plus probables.

Bien que la modélisation du langage ne ressemble pas, à première vue,
à un problème autorégressif,
nous pouvons réduire la modélisation du langage à une prédiction autorégressive
en décomposant la densité conjointe d'une séquence $p(x_t| x_1, \ldots, x_T)$
en un produit de densités conditionnelles 
de gauche à droite
en appliquant la règle de la chaîne de probabilité :

$$P(x_1, \ldots, x_T) = P(x_1) * \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

Notez que si nous travaillons avec des signaux discrets tels que des mots,
alors le modèle autorégressif doit être un classifieur probabiliste,
produisant une distribution de probabilité complète
sur le vocabulaire pour savoir quel mot viendra ensuite,
étant donné le contexte de gauche.



### Modèles de Markov
:label:`subsec_markov-models` 

 
Supposons maintenant que nous souhaitions employer la stratégie mentionnée ci-dessus,
où nous ne conditionnons que les $\tau$ précédentes étapes de la séquence,
c'est-à-dire, $x_{t-1}, \ldots, x_{t-\tau}$, plutôt que 
l'historique complet de la séquence $x_{t-1}, \ldots, x_1$.
Chaque fois que nous pouvons nous débarrasser de l'historique 
au-delà des précieux $\tau$ pas 
sans aucune perte de pouvoir prédictif,
nous disons que la séquence satisfait une *condition de Markov*,
c'est-à-dire, *que le futur est conditionnellement indépendant du passé,
étant donné l'histoire récente*. 
Lorsque $\tau = 1$, on dit que les données sont caractérisées 
par un *modèle de Markov de premier ordre*,
et lorsque $\tau = k$, on dit que les données sont caractérisées
par un modèle de Markov de $kième$ ordre.
Car lorsque la condition de Markov du premier ordre est vérifiée$(\tau = 1$),
la factorisation de notre probabilité conjointe devient un produit
des probabilités de chaque mot étant donné le *mot* précédent :

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \text{ où } P(x_1 \mid x_0) = P(x_1).$$

Nous trouvons souvent utile de travailler avec des modèles qui procèdent 
comme si une condition de Markov était satisfaite,
même lorsque nous savons que ce n'est qu'*approximativement* vrai. 
Avec des documents textuels réels, nous continuons à gagner des informations
à mesure que nous incluons de plus en plus de contexte vers la gauche.
Mais ces gains diminuent rapidement.
C'est pourquoi nous faisons parfois des compromis, en évitant les difficultés informatiques et statistiques
par l'entrainement de modèles dont la validité dépend 
d'une condition de Markov d'ordre $k$.
Même les modèles de langage massifs d'aujourd'hui, basés sur les RNN et les transformateurs ,
intègrent rarement plus de mille mots de contexte.


Avec des données discrètes, un véritable modèle de Markov
compte simplement le nombre de fois 
où chaque mot est apparu dans chaque contexte, produisant 
l'estimation de la fréquence relative de $P(x_t \mid x_{t-1})$.
Lorsque les données ne prennent que des valeurs discrètes 
(comme dans le langage),
la séquence de mots la plus probable peut être calculée efficacement
en utilisant la programmation dynamique. 


### L'ordre de décodage

Vous vous demandez peut-être pourquoi nous avons dû représenter 
la factorisation d'une séquence de texte $P(x_1, \ldots, x_T)$
comme une chaîne de probabilités conditionnelles de gauche à droite.
Pourquoi pas de droite à gauche ou dans un autre ordre, apparemment aléatoire ?
En principe, il n'y a rien de mal à déplier 
$P(x_1, \ldots, x_T)$ dans l'ordre inverse. 
Le résultat est une factorisation valide :

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$


Cependant, il existe de nombreuses raisons pour lesquelles la factorisation du texte
dans le même sens que nous le lisons 
(de gauche à droite pour la plupart des langues,
mais de droite à gauche pour l'arabe et l'hébreu)
est préférable pour la tâche de modélisation du langage.
Tout d'abord, il s'agit simplement d'une direction plus naturelle pour nous.
Après tout, nous lisons tous du texte tous les jours, 
et ce processus est guidé par notre capacité
à anticiper les mots et les phrases 
susceptibles de suivre.
Pensez simplement au nombre de fois où vous avez complété la phrase de quelqu'un d'autre .

Ainsi, même si nous n'avions aucune autre raison de préférer de tels décodages dans l'ordre, 
ils seraient utiles ne serait-ce que parce que nous avons de meilleures intuitions 
de ce qui devrait être probable lorsque l'on prédit dans cet ordre. 

Deuxièmement, en factorisant dans l'ordre, 
nous pouvons attribuer des probabilités à des séquences arbitrairement longues
en utilisant le même modèle de langage. 
Pour convertir une probabilité sur les étapes $1$ à $t$ 
en une probabilité qui s'étend au mot $t+1$, il suffit de
multiplier par la probabilité conditionnelle 
$P(x_{t+1} \mid x_{t+1}, \ldots, x_T)$.

Troisièmement, nous disposons de modèles prédictifs plus puissants 
pour prédire les mots adjacents par rapport aux mots 
situés à d'autres emplacements arbitraires. 
Bien que tous les ordres de factorisation soient valables,
ils ne représentent pas nécessairement tous des problèmes de 
modélisation prédictive aussi faciles. 
Cela est vrai non seulement pour le langage,
mais aussi pour d'autres types de données,
par exemple, lorsque les données sont structurées de manière causale.
Par exemple, nous pensons que les événements futurs ne peuvent pas influencer le passé. 
Par conséquent, si nous modifions $x_t$, nous pouvons être en mesure d'influencer 
ce qui se passe pour $x_{t+1}$ dans le futur, mais pas l'inverse. 
Autrement dit, si nous changeons $x_t$, la distribution des événements passés ne changera pas. 
Dans certains contextes, il est donc plus facile de prédire $P(x_{t+1} \mid x_t)$ 
que de prédire $P(x_t \mid x_{t+1})$. 
Par exemple, dans certains cas, nous pouvons trouver $x_{t+1} = f(x_t) + \epsilon$ 
pour un certain bruit additif $\epsilon$, 
alors que l'inverse n'est pas vrai :cite:`Hoyer.Janzing.Mooij.ea.2009`. 
Il s'agit d'une excellente nouvelle, car c'est généralement la direction avant 
qui nous intéresse pour l'estimation.
Le livre de Peters et al. en dit plus sur ce sujet 
:cite:`Peters.Janzing.Scholkopf.2017`.
Nous ne faisons qu'effleurer le sujet.


## Formation

<!-- fix -->
Après avoir passé en revue de nombreux outils statistiques différents, essayons de les mettre en pratique.

```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Nous commençons par générer quelques données.
Pour garder les choses simples, nous 
(**générons nos données de séquence en utilisant une fonction sinus
avec un certain bruit additif pour les pas de temps $1, 2, \ldots, 1000$.**)

```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):    
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2

data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

Ensuite, nous devons transformer une telle séquence en caractéristiques et en étiquettes 
<!-- language -->
sur lesquelles notre modèle peut s'entraîner.
Avec l'hypothèse de Markov selon laquelle $x_t$ dépend uniquement 
des observations des $\tau$ derniers pas de temps,
nous [**construisons des exemples avec des étiquettes $y_t = x_t$ et des caractéristiques 
$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$.**]
Le lecteur avisé aura peut-être remarqué que 
cela nous donne $\tau$ moins d'exemples de données,
puisque nous n'avons pas suffisamment d'historique pour $y_1, \ldots, y_\tau$. 
Bien que nous puissions remplir les premières séquences $\tau$ avec des zéros,
pour garder les choses simples, nous les abandonnons pour le moment. 
L'ensemble de données résultant contient $T - \tau$ exemples,
où chaque entrée du modèle
a une longueur de séquence $\tau$.
Nous (**créons un itérateur de données sur les 600 premiers exemples**),
couvrant une période de la fonction sinus.

```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

Le modèle à entraîner est simple : juste une régression linéaire.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## Prédiction

Voyons comment le modèle prédit. 
La première chose à vérifier est 
[**predicting what happens just in the next time step**],
à savoir la *one-step-ahead prediction*.

```{.python .input}
%%tab all
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x', 
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

Les prédictions à un pas de temps ont l'air bien. 
Même vers la fin de la période $t=1000$, les prédictions semblent toujours dignes de confiance.
Cependant, il y a un petit problème :
si nous n'observons les données de la séquence que jusqu'à l'étape 604 (`n_train + tau`), 
nous ne pouvons pas espérer recevoir les données d'entrée 
pour toutes les futures prédictions à une étape.
Au lieu de cela, nous devons utiliser les prédictions antérieures 
comme entrées de notre modèle pour ces prédictions futures, 
une étape à la fois :

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots
$$

Généralement, pour une séquence observée $x_1, \ldots, x_t$, 
sa sortie prédite $\hat{x}_{t+k}$ au pas de temps $t+k$ 
est appelée la prédiction $k$*-step-ahead*. 
Comme nous avons observé jusqu'à $x_{604}$, 
sa prédiction $k-step-ahead$ est $\hat{x}_{604+k}$.
En d'autres termes, nous devrons continuer à utiliser nos propres prédictions 
pour faire des prédictions multi-étapes-ahead.
Voyons comment cela se passe.

```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)    
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]], 
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time', 
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

Comme le montre l'exemple ci-dessus, c'est un échec spectaculaire. 
Les prédictions tombent à une constante 
assez rapidement après quelques étapes de prédiction.
Pourquoi l'algorithme a-t-il si mal fonctionné ?
Cela est finalement dû au fait que les erreurs s'accumulent.
Disons qu'après l'étape 1, nous avons une erreur $\epsilon_1 = \bar\epsilon$.
Maintenant, l'*entrée* de l'étape 2 est perturbée par $\epsilon_1$,
ainsi nous souffrons donc d'une erreur de l'ordre de 
$\epsilon_2 = \bar\epsilon + c \epsilon_1$ pour une certaine constante $c$, et ainsi de suite. 
L'erreur peut diverger assez rapidement des véritables observations. 
Il s'agit d'un phénomène courant. 
Par exemple, les prévisions météorologiques pour les 24 heures à venir ont tendance à être assez précises 
mais au-delà, la précision diminue rapidement. 
Nous discuterons des méthodes permettant d'améliorer cette précision 
tout au long de ce chapitre et au-delà.

Examinons [**de plus près les difficultés des prévisions à $k$ étapes avant la prédiction**]
en calculant des prévisions sur la séquence entière pour $k = 1, 4, 16, 64$.

```{.python .input}
%%tab all
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:], 
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x', 
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

Cela illustre clairement comment la qualité de la prédiction change 
lorsque nous essayons de prédire plus loin dans le futur.
Alors que les prédictions à 4 pas à l'avance semblent toujours bonnes, 
tout ce qui est au-delà est presque inutile.

## Résumé

* Il y a une grande différence de difficulté entre l'interpolation et l'extrapolation. 
  Par conséquent, si vous avez une séquence, respectez toujours 
  l'ordre temporel des données lors de la formation, 
  c'est-à-dire ne vous entraînez jamais sur des données futures.
* Les modèles de séquence nécessitent des outils statistiques spécialisés pour l'estimation. 
  Deux choix populaires sont les modèles autorégressifs 
  et les modèles autorégressifs à variables latentes.
* Pour les modèles causaux (par exemple, le temps qui avance), 
  l'estimation de la direction avant est généralement 
  beaucoup plus facile que la direction arrière.
* Pour une séquence observée jusqu'au pas de temps $t$, 
  sa sortie prédite au pas de temps $t+k$ 
  est la prédiction $k$*-step-ahead*. 
  Lorsque l'on prédit plus loin dans le temps en augmentant $k$, 
  les erreurs s'accumulent et la qualité de la prédiction se dégrade,
  souvent de façon spectaculaire.

## Exercices

1. Améliorez le modèle dans l'expérience de cette section.
   1. Incorporez-vous plus que les 4 dernières observations ? Combien en avez-vous vraiment besoin ?
   1. De combien d'observations passées auriez-vous besoin s'il n'y avait pas de bruit ? Conseil : vous pouvez écrire $\sin$ et $\cos$ comme une équation différentielle.
   1. Pouvez-vous incorporer des observations plus anciennes tout en maintenant constant le nombre total de caractéristiques ? Cela améliore-t-il la précision ? Pourquoi ?
 1. Modifiez l'architecture du réseau neuronal et évaluez les performances. Vous pouvez entraîner le nouveau modèle avec plus d'époques. Qu'observez-vous ?
1. Un investisseur souhaite trouver un bon titre à acheter. 
   Il examine les rendements passés pour décider lequel est susceptible de bien se comporter. 
   Qu'est-ce qui pourrait mal tourner dans cette stratégie ?
1. La causalité s'applique-t-elle également au texte ? Dans quelle mesure ?
1. Donnez un exemple de cas où un modèle autorégressif latent 
pourrait être nécessaire pour capturer la dynamique des données.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

 :begin_tab:`pytorch` 
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

 :begin_tab:`tensorflow` 
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:
