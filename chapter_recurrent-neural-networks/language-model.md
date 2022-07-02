# Modèles de langage
:label:`sec_language-model` 

 
 Dans :numref:`sec_text-sequence` , nous voyons comment mapper des séquences de texte en jetons, ces jetons pouvant être considérés comme une séquence d'observations discrètes, telles que des mots ou des caractères. Supposons que les tokens d'une séquence de texte de longueur $T$ soient à leur tour $x_1, x_2, \ldots, x_T$.
L'objectif des *modèles de langage*
est d'estimer la probabilité conjointe de l'ensemble de la séquence :

$$P(x_1, x_2, \ldots, x_T),$$ 

 où les outils statistiques
dans :numref:`sec_sequence` 
 peuvent être appliqués.

Les modèles de langage sont incroyablement utiles. Par exemple, un modèle de langage idéal serait capable de générer du texte naturel par lui-même, simplement en dessinant un jeton à la fois $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Contrairement au singe qui utilise une machine à écrire, tout texte issu d'un tel modèle passerait pour un langage naturel, par exemple un texte anglais. De plus, il serait suffisant pour générer un dialogue significatif, simplement en conditionnant le texte à des fragments de dialogue antérieurs.
Il est clair que nous sommes encore très loin de concevoir un tel système, puisqu'il devrait *comprendre* le texte plutôt que de simplement générer un contenu grammaticalement sensé.

Néanmoins, les modèles de langage sont d'une grande utilité, même sous leur forme limitée.
Par exemple, les expressions "reconnaître la parole" et "détruire une belle plage" se ressemblent beaucoup.
Cela peut provoquer une ambiguïté dans la reconnaissance de la parole,
qui est facilement résolue par un modèle de langage qui rejette la deuxième traduction comme étant farfelue.
De même, dans un algorithme de résumé de documents
, il est utile de savoir que "le chien mord l'homme" est beaucoup plus fréquent que "l'homme mord le chien", ou que "je veux manger grand-mère" est une déclaration plutôt inquiétante, alors que "je veux manger, grand-mère" est beaucoup plus bénin.


## Apprendre des modèles de langage

La question évidente est de savoir comment nous devrions modéliser un document, ou même une séquence de tokens. 
Supposons que nous tokenisions les données textuelles au niveau des mots.
Commençons par appliquer les règles de probabilité de base :

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$ 

 Par exemple, 
la probabilité d'une séquence de texte contenant quatre mots serait donnée comme suit :

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$ 

 ### Modèles de Markov et $n$-grammes

Parmi ces analyses de modèles de séquences dans :numref:`sec_sequence` ,
appliquons les modèles de Markov à la modélisation du langage.
Une distribution sur les séquences satisfait à la propriété de Markov du premier ordre si $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. Les ordres supérieurs correspondent à des dépendances plus longues. Cela conduit à un certain nombre d'approximations que nous pourrions appliquer pour modéliser une séquence :

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Les formules de probabilité qui impliquent une, deux et trois variables sont généralement appelées modèles *unigramme*, *bigramme* et *trigramme*, respectivement. 
Afin de calculer le modèle de langage, nous devons calculer la probabilité des mots
et la probabilité conditionnelle d'un mot étant donné
les quelques mots précédents.
Notez que
ces probabilités sont des paramètres du modèle de langage
.




### Fréquence des mots

Ici, nous supposons que l'ensemble de données d'apprentissage est un grand corpus de texte, tel que toutes les
entrées de Wikipedia, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg),
et tous les textes publiés sur le
Web.
La probabilité des mots peut être calculée à partir de la fréquence relative des mots
d'un mot donné dans l'ensemble de données d'apprentissage.
Par exemple, l'estimation $\hat{P}(\text{deep})$ peut être calculée comme la probabilité
de toute phrase commençant par le mot "deep". Une approche légèrement moins précise
consisterait à compter toutes les occurrences de
le mot "deep" et à les diviser par le nombre total de mots dans
le corpus.
Cette méthode fonctionne assez bien, en particulier pour les mots fréquents
. Ensuite, nous pourrions essayer d'estimer

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$ 

 où $n(x)$ et $n(x, x')$ sont le nombre d'occurrences de singletons
et de paires de mots consécutifs, respectivement.
Malheureusement, 
estimer la probabilité
d'une paire de mots est un peu plus difficile, car les occurrences
de "deep learning" sont beaucoup moins fréquentes. 
En particulier, pour certaines combinaisons de mots inhabituelles, il peut être difficile de trouver
suffisamment d'occurrences pour obtenir des estimations précises.
Comme le suggèrent les résultats empiriques de :numref:`subsec_natural-lang-stat` ,
, les choses se gâtent pour les combinaisons de trois mots et plus.
Il y aura de nombreuses combinaisons plausibles de trois mots que nous ne verrons probablement pas dans notre ensemble de données.
À moins de trouver une solution pour attribuer à ces combinaisons de mots un nombre non nul, nous ne pourrons pas les utiliser dans un modèle de langage. Si l'ensemble de données est petit ou si les mots sont très rares, il se peut que nous ne trouvions même pas un seul d'entre eux.

### Lissage de Laplace

Une stratégie courante consiste à effectuer une forme de *lissage de Laplace*.
La solution consiste à
ajouter une petite constante à tous les comptes. 
Désignez par $n$ le nombre total de mots dans
l'ensemble d'apprentissage
et $m$ le nombre de mots uniques.
Cette solution est utile pour les singletons, par exemple, via

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Ici, $\epsilon_1,\epsilon_2$, et $\epsilon_3$ sont des hyperparamètres.
Prenons l'exemple de $\epsilon_1$:
lorsque $\epsilon_1 = 0$, aucun lissage n'est appliqué ;
lorsque $\epsilon_1$ s'approche de l'infini positif,
$\hat{P}(x)$ s'approche de la probabilité uniforme $1/m$. 
Ce qui précède est une variante plutôt primitive de ce que
d'autres techniques peuvent accomplir :cite:`Wood.Gasthaus.Archambeau.ea.2011` .


Malheureusement, les modèles de ce type deviennent rapidement difficiles à manier
pour les raisons suivantes. 
Premièrement, 
comme nous l'avons vu dans :numref:`subsec_natural-lang-stat` ,
de nombreux $n$-grammes se produisent très rarement, 
rendant le lissage de Laplace plutôt inadapté à la modélisation du langage.
Deuxièmement, nous devons stocker tous les comptes.
Troisièmement, cela ne tient absolument pas compte du sens des mots. Par exemple,
, "chat" et "félin" devraient apparaître dans des contextes connexes.
Il est assez difficile d'ajuster de tels modèles à des contextes supplémentaires,
alors que les modèles de langage basés sur l'apprentissage profond sont bien adaptés pour
prendre cela en compte.
Enfin, il est presque certain que les longues séquences de mots
sont nouvelles, et qu'un modèle qui se contente de compter la fréquence des séquences de mots déjà vues (
) ne donnera pas de bons résultats.
Par conséquent, nous nous concentrons sur l'utilisation des réseaux neuronaux pour la modélisation du langage
dans le reste du chapitre.


## Perplexité
:label:`subsec_perplexity` 

 Ensuite, voyons comment mesurer la qualité d'un modèle de langue, qui sera utilisée pour évaluer nos modèles dans les sections suivantes.
L'un des moyens consiste à vérifier le caractère surprenant du texte.
Un bon modèle de langage est capable de prédire avec
des tokens de haute précision que ce que nous allons voir ensuite.
Considérons les continuations suivantes de la phrase "Il pleut", telles que proposées par différents modèles de langage :

1. "Il pleut dehors"
1. "Il pleut sur le bananier"
1. "Il pleut piouw;kcj pwepoiut"

En termes de qualité, l'exemple 1 est clairement le meilleur. Les mots sont sensés et logiquement cohérents.
Même s'il ne reflète pas exactement le mot qui suit sémantiquement ("à San Francisco" et "en hiver" auraient été des extensions parfaitement raisonnables), le modèle est capable de saisir le type de mot qui suit.
L'exemple 2 est considérablement plus mauvais, car il produit une extension qui n'a aucun sens. Néanmoins, le modèle a au moins appris comment épeler les mots et un certain degré de corrélation entre les mots. Enfin, l'exemple 3 indique un modèle mal entraîné qui ne s'adapte pas correctement aux données.

Nous pourrions mesurer la qualité du modèle en calculant la vraisemblance de la séquence.
Malheureusement, il s'agit d'un nombre difficile à comprendre et à comparer.
Après tout, les séquences les plus courtes sont beaucoup plus susceptibles de se produire que les plus longues,
. Ainsi, l'évaluation du modèle sur le magnum opus de Tolstoï
*Guerre et Paix* produira inévitablement une probabilité beaucoup plus faible que, par exemple, sur la novella de Saint-Exupéry *Le Petit Prince*. Ce qui manque, c'est l'équivalent d'une moyenne.

La théorie de l'information est utile ici.
Nous avons défini l'entropie, la surprise et l'entropie croisée
lorsque nous avons présenté la régression softmax
(:numref:`subsec_info_theory_basics` ).
Si nous voulons compresser du texte, nous pouvons demander à
de prédire le prochain token en fonction de l'ensemble actuel de tokens.
Un meilleur modèle de langage devrait nous permettre de prédire le prochain token avec plus de précision.
Ainsi, il devrait nous permettre de dépenser moins de bits dans la compression de la séquence.
Nous pouvons donc la mesurer par la perte d'entropie croisée moyennée sur
sur tous les tokens $n$ d'une séquence :

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$ 
 :eqlabel:`eq_avg_ce_for_lm` 

 où $P$ est donné par un modèle de langage et $x_t$ est le token réel observé au pas de temps $t$ de la séquence.
Les performances sur des documents de longueurs différentes sont ainsi comparables. Pour des raisons historiques, les scientifiques du traitement du langage naturel préfèrent utiliser une quantité appelée *perplexité*. En bref, il s'agit de l'exponentielle de :eqref:`eq_avg_ce_for_lm` :

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$ 

 La perplexité peut être comprise comme la moyenne géométrique du nombre de choix réels qui s'offrent à nous lorsque nous décidons du prochain jeton à sélectionner. Examinons un certain nombre de cas :

* Dans le meilleur des cas, le modèle estime toujours parfaitement la probabilité du jeton cible à 1. Dans ce cas, la perplexité du modèle est de 1.
* Dans le pire des cas, le modèle prédit toujours la probabilité du jeton cible à 0. Dans cette situation, la perplexité est une infinité positive.
* Dans le scénario de base, le modèle prédit une distribution uniforme sur tous les tokens disponibles du vocabulaire. Dans ce cas, la perplexité est égale au nombre de tokens uniques du vocabulaire. En fait, si nous devions stocker la séquence sans aucune compression, ce serait le mieux que nous puissions faire pour la coder. Par conséquent, cela fournit une limite supérieure non triviale que tout modèle utile doit dépasser.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Partitionnement des séquences
:label:`subsec_partitioning-seqs` 

 Nous allons concevoir des modèles de langage à l'aide de réseaux neuronaux
et utiliser la perplexité pour évaluer 
la capacité du modèle à 
prédire le prochain token en fonction de l'ensemble actuel de tokens
dans les séquences de texte.
Avant de présenter le modèle,
supposons qu'il
traite un minilot de séquences de longueur prédéfinie
à la fois.
La question est maintenant de savoir comment [**lire des minis lots de séquences d'entrée et de séquences cibles de manière aléatoire**].


Supposons que le jeu de données prenne la forme d'une séquence d'indices de jetons $T$ dans `corpus`.
Nous allons
le partitionner
en sous-séquences, où chaque sous-séquence a $n$ tokens (pas de temps).
Pour itérer sur 
(presque) tous les tokens de l'ensemble de données 
pour chaque époque
et obtenir toutes les sous-séquences de longueur$n$ possibles,
nous pouvons introduire du hasard.
Plus concrètement,
au début de chaque époque,
élimine les premiers tokens $d$,
où $d\in [0,n)$ est uniformément échantillonné au hasard.
Le reste de la séquence
est ensuite partitionné
en sous-séquences $m=\lfloor (T-d)/n \rfloor$.
On désigne par $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$ la sous-séquence de longueur$n$ commençant à partir du jeton $x_t$ au pas de temps $t$. 
Les sous-séquences partitionnées $m$ résultantes
sont 
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}.$
Chaque sous-séquence sera utilisée comme une séquence d'entrée dans le modèle de langage.


Pour la modélisation du langage,
le but est de prédire le prochain token sur la base des tokens que nous avons vus jusqu'à présent, donc les cibles (étiquettes) sont la séquence originale, décalée d'un token.
La séquence cible pour toute séquence d'entrée $\mathbf x_t$
 est $\mathbf x_{t+1}$ avec la longueur $n$.

![Obtaining 5 pairs of input sequences and target sequences from partitioned length-5 subsequences.](../img/lang-model-data.svg) 
 :label:`fig_lang_model_data` 

 :numref:`fig_lang_model_data` montre un exemple d'obtention de 5 paires de séquences d'entrée et de séquences cibles avec $n=5$ et $d=2$.

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = d2l.tensor([corpus[i:i+num_steps+1] 
                        for i in range(0, len(corpus)-num_steps-1)])
    self.X, self.Y = array[:,:-1], array[:,1:]
```

Pour entraîner les modèles de langage,
nous allons échantillonner aléatoirement 
paires de séquences d'entrée et de séquences cibles
en minibatchs.
Le chargeur de données suivant génère aléatoirement un minibatch à partir de l'ensemble de données à chaque fois.
L'argument `batch_size` spécifie le nombre d'exemples de sous-séquences (`self.b`) dans chaque minilot
et `num_steps` est la longueur de la sous-séquence en tokens (`self.n`).

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

Comme nous pouvons le voir dans ce qui suit, 
un minilot de séquences cibles
peut être obtenu 
en décalant les séquences d'entrée
d'un token.

```{.python .input  n=7}
%%tab all
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
```

## Résumé

* Les modèles de langue estiment la probabilité conjointe d'une séquence de texte.
* $n$-grams fournit un modèle pratique pour traiter les longues séquences en tronquant la dépendance.
* Il y a beaucoup de structure mais pas assez de fréquence pour traiter efficacement les combinaisons de mots peu fréquentes via le lissage de Laplace.
* Pour entraîner les modèles de langage, nous pouvons échantillonner aléatoirement des paires de séquences d'entrée et de séquences cibles dans des minibatchs.


## Exercices

1. Supposons qu'il y ait $100,000$ mots dans l'ensemble de données d'entraînement. Combien de fréquences de mots et de fréquences adjacentes multi-mots un quatre-grammes doit-il stocker ?
1. Comment modéliseriez-vous un dialogue ?
1. Quelles autres méthodes pouvez-vous imaginer pour la lecture de longues séquences de données ?
1. Considérez notre méthode pour écarter un nombre uniformément aléatoire des premiers tokens au début de chaque époque.
   1. Cela conduit-il vraiment à une distribution parfaitement uniforme sur les séquences du document ?
   1. Que faudrait-il faire pour rendre les choses encore plus uniformes ? 
1. Si nous voulons qu'un exemple de séquence soit une phrase complète, quel type de problème cela introduit-il dans l'échantillonnage par minibatchs ? Comment pouvons-nous résoudre ce problème ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
