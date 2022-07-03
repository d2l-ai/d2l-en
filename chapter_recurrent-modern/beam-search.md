# Beam Search
:label:`sec_beam-search` 

Dans :numref:`sec_seq2seq`,
nous avons prédit la séquence de sortie token par token
jusqu'à ce que le token spécial de fin de séquence "&lt;eos&gt;"
soit prédit.
Dans cette section,
nous commencerons par formaliser cette stratégie de recherche *avide*
et par explorer les problèmes qu'elle pose,
puis nous comparerons cette stratégie avec d'autres alternatives :
*recherche exhaustive* et *recherche par faisceau*.

Avant une introduction formelle à la recherche gloutonne,
formalisons le problème de recherche
en utilisant
la même notation mathématique que celle utilisée pour :numref:`sec_seq2seq`.
À tout pas de temps $t'$, 
la probabilité de la sortie du décodeur $y_{t'}$ 
est conditionnelle 
à la sous-séquence de sortie
$y_1, \ldots, y_{t'-1}$ avant $t'$ et 
la variable de contexte $\mathbf{c}$ qui
code l'information de la séquence d'entrée.
Pour quantifier le coût de calcul,
désigne par 
$\mathcal{Y}$ (il contient "&lt;eos&gt;")
le vocabulaire de sortie.
La cardinalité $\left|\mathcal{Y}\right|$ de cet ensemble de vocabulaire
est donc la taille du vocabulaire.
Spécifions également le nombre maximum de tokens
d'une séquence de sortie comme $T'$.
Par conséquent,
notre objectif est de rechercher une sortie idéale
parmi toutes les 
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ 
séquences de sortie possibles.
Bien sûr, 
pour toutes ces séquences de sortie,
les portions incluant et suivant "&lt;eos&gt;" seront rejetées
dans la sortie réelle.

## Greedy Search

Tout d'abord, examinons 
une stratégie simple : *greedy search*.
Cette stratégie a été utilisée pour prédire des séquences dans :numref:`sec_seq2seq`.
Dans la recherche par gourmandise,
à n'importe quel pas de temps $t'$ de la séquence de sortie, 
nous recherchons le jeton 
avec la probabilité conditionnelle la plus élevée de $\mathcal{Y}$, c'est-à-dire, 

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$ 

comme sortie. 
Une fois que "&lt;eos&gt;" est sorti ou que la séquence de sortie a atteint sa longueur maximale $T'$, la séquence de sortie est terminée.

Alors, qu'est-ce qui peut mal se passer avec la recherche avide ?
En fait,
la séquence *optimale*
devrait être la séquence de sortie
avec le maximum 
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$ ,
qui est
la probabilité conditionnelle de générer une séquence de sortie basée sur la séquence d'entrée.
Malheureusement, il n'y a aucune garantie
que la séquence optimale sera obtenue
par recherche gloutonne.

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Illustrons cela par un exemple.
Supposons qu'il y ait quatre jetons 
"A", "B", "C", et "&lt;eos&gt;" dans le dictionnaire de sortie.
Dans :numref:`fig_s2s-prob1`,
les quatre nombres sous chaque pas de temps représentent les probabilités conditionnelles de générer "A", "B", "C" et "&lt;eos&gt;" à ce pas de temps, respectivement. 
À chaque pas de temps, la recherche avide 
sélectionne le jeton ayant la probabilité conditionnelle la plus élevée. 
Par conséquent, la séquence de sortie "A", "B", "C" et "&lt;eos&gt;" sera prédite 
dans :numref:`fig_s2s-prob1`. 
La probabilité conditionnelle de cette séquence de sortie est $0.5\times0.4\times0.4\times0.6 = 0.048$.

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`


Ensuite, examinons un autre exemple 
dans :numref:`fig_s2s-prob2`. 
Contrairement à :numref:`fig_s2s-prob1`, 
au pas de temps 2
nous sélectionnons le jeton "C"
dans :numref:`fig_s2s-prob2`, 
qui a la *deuxième* probabilité conditionnelle la plus élevée.
Puisque les sous-séquences de sortie aux étapes 1 et 2, 
sur lesquelles l'étape 3 est basée, 
ont changé de "A" et "B" dans :numref:`fig_s2s-prob1` à "A" et "C" dans :numref:`fig_s2s-prob2`, 
la probabilité conditionnelle de chaque jeton 
à l'étape 3 a également changé dans :numref:`fig_s2s-prob2`. 
Supposons que nous choisissions le jeton "B" à l'étape 3. 
Maintenant, le pas de temps 4 est conditionnel à
la sous-séquence de sortie aux trois premiers pas de temps
"A", "C" et "B", 
qui est différente de "A", "B" et "C" dans :numref:`fig_s2s-prob1`. 
Par conséquent, la probabilité conditionnelle de générer chaque jeton au pas de temps 4 dans :numref:`fig_s2s-prob2` est également différente de celle de :numref:`fig_s2s-prob1`. 
Par conséquent, 
la probabilité conditionnelle de la séquence de sortie "A", "C", "B" et "&lt;eos&gt;" 
dans :numref:`fig_s2s-prob2` 
est $0.5\times0.3 \times0.6\times0.6=0.054$, 
ce qui est supérieur à celle de la recherche par la méthode classique dans :numref:`fig_s2s-prob1`. 
Dans cet exemple, 
la séquence de sortie "A", "B", "C" et "&lt;eos&gt;" obtenue par la recherche par la méthode classique n'est pas une séquence optimale.

## Recherche exhaustive

Si l'objectif est d'obtenir la séquence optimale, nous pouvons envisager d'utiliser la *recherche exhaustive* : 
énumérer de manière exhaustive toutes les séquences de sortie possibles avec leurs probabilités conditionnelles,
puis sortir celle 
dont la probabilité conditionnelle est la plus élevée.

Bien que nous puissions utiliser la recherche exhaustive pour obtenir la séquence optimale, 
son coût de calcul $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ est susceptible d'être excessivement élevé. 
Par exemple, lorsque $|\mathcal{Y}|=10000$ et $T'=10$, nous devrons évaluer $10000^{10} = 10^{40}$ séquences. C'est pratiquement impossible !
D'autre part,
le coût de calcul de la recherche avide est 
$\mathcal{O}(\left|\mathcal{Y}\right|T')$ : 
il est généralement nettement inférieur à
celui de la recherche exhaustive. Par exemple, lorsque $|\mathcal{Y}|=10000$ et $T'=10$, nous n'avons besoin d'évaluer que les séquences $10000\times10=10^5$.


## Recherche de faisceaux

Les décisions relatives aux stratégies de recherche de séquences
se situent sur un spectre,
avec des questions faciles aux deux extrêmes.
Que se passe-t-il si seule la précision compte ?
Évidemment, la recherche exhaustive.
Et si seul le coût de calcul compte ?
La recherche avide est évidente.
Une application du monde réel pose généralement
une question complexe,
quelque part entre ces deux extrêmes.

*La recherche par faisceau* est une version améliorée de la recherche avide. Elle possède un hyperparamètre appelé *beam size*, $k$. 
Au pas de temps 1, 
nous sélectionnons $k$ les tokens ayant les probabilités conditionnelles les plus élevées.
Chacun d'entre eux sera le premier jeton des séquences de sortie candidates 
$k$ , respectivement.
A chaque pas de temps suivant, 
sur la base des séquences de sortie candidates $k$
au pas de temps précédent,
nous continuons à sélectionner $k$ des séquences de sortie candidates 
avec les probabilités conditionnelles les plus élevées 
parmi $k\left|\mathcal{Y}\right|$ choix possibles.

![The process of beam search (beam size: 2, maximum length of an output sequence: 3) . Les séquences de sortie candidates sont $A$, $C$, $AB$, $CE$, $ABD$, et $CED$.](../img/beam-search.svg)
:label:`fig_beam-search` 

 
:numref:`fig_beam-search` démontre le processus de recherche de faisceau 
à l'aide d'un exemple. 
Supposons que le vocabulaire de sortie
ne contienne que cinq éléments : 
$\mathcal{Y} = \{A, B, C, D, E\}$, 
où l'un d'entre eux est "&lt;eos&gt;". 
Supposons que la taille du faisceau soit de 2 et que 
la longueur maximale d'une séquence de sortie soit de 3. 
Au pas de temps 1, 
supposons que les tokens ayant les probabilités conditionnelles les plus élevées $P(y_1 \mid \mathbf{c})$ sont $A$ et $C$. À l'étape 2, pour tous les $y_2 \in \mathcal{Y},$, nous calculons 

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$ 

et choisissons les deux plus grandes valeurs parmi ces dix, soit
$P(A, B \mid \mathbf{c})$ et $P(C, E \mid \mathbf{c})$.
Ensuite, à l'étape 3, pour tous les $y_3 \in \mathcal{Y}$, nous calculons 

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

et choisissons les deux valeurs les plus grandes parmi ces dix valeurs, à savoir 
$P(A, B, D \mid \mathbf{c})$ et $P(C, E, D \mid  \mathbf{c}).$
En conséquence, nous obtenons six séquences de sortie candidates : (i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; et (vi) $C$, $E$, $D$. 


Au final, nous obtenons l'ensemble des séquences de sortie candidates finales sur la base de ces six séquences (par exemple, nous écartons les parties incluant et suivant "&lt;eos&gt;").
Ensuite,
nous choisissons la séquence ayant le score le plus élevé parmi les suivants comme séquence de sortie :

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$ 
:eqlabel:`eq_beam-search-score` 

où $L$ est la longueur de la séquence candidate finale et $\alpha$ est généralement fixé à 0,75. 
Étant donné qu'une séquence plus longue comporte plus de termes logarithmiques dans la somme de :eqref:`eq_beam-search-score`,
le terme $L^\alpha$ au dénominateur pénalise
les longues séquences.

Le coût de calcul de la recherche par faisceau est de $\mathcal{O}(k\left|\mathcal{Y}\right|T')$. 
Ce résultat se situe entre celui de la recherche avide et celui de la recherche exhaustive. En fait, la recherche avide peut être traitée comme un type spécial de recherche par faisceau avec 
une taille de faisceau de 1. 
Avec un choix flexible de la taille du faisceau,
la recherche par faisceau offre un compromis entre
la précision et le coût de calcul.



## Résumé

* Les stratégies de recherche de séquences comprennent la recherche avide, la recherche exhaustive et la recherche par faisceau.
* La recherche par faisceau offre un compromis entre la précision et le coût de calcul grâce au choix flexible de la taille du faisceau.


## Exercices

1. Peut-on considérer la recherche exhaustive comme un type particulier de recherche par faisceau ? Pourquoi ou pourquoi pas ?
1. Appliquez la recherche par faisceau au problème de la traduction automatique dans :numref:`sec_seq2seq`. Comment la taille du faisceau affecte-t-elle les résultats de la traduction et la vitesse de prédiction ?
1. Nous avons utilisé la modélisation du langage pour générer du texte suivant des préfixes fournis par l'utilisateur dans :numref:`sec_rnn-scratch`. Quel type de stratégie de recherche utilise-t-elle ? Pouvez-vous l'améliorer ?

[Discussions](https://discuss.d2l.ai/t/338)
