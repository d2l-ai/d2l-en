# Embedded word (word2vec)
:label:`sec_word2vec` 

 
 Le langage naturel est un système complexe utilisé pour exprimer des significations.
Dans ce système, les mots sont l'unité de base de la signification.
Comme son nom l'indique,
*word vectors* sont des vecteurs utilisés pour représenter les mots,
et peuvent également être considérés comme des vecteurs de caractéristiques ou des représentations de mots.
La technique de mise en correspondance des mots avec des vecteurs réels
est appelée *word embedding*.
Ces dernières années,
word embedding est progressivement devenu
la connaissance de base du traitement du langage naturel.


## Les vecteurs à un coup sont un mauvais choix

Nous avons utilisé des vecteurs à un coup pour représenter les mots (les caractères sont des mots) dans :numref:`sec_rnn-scratch` .
Supposons que le nombre de mots différents dans le dictionnaire (la taille du dictionnaire) soit $N$,
et que chaque mot corresponde à
un nombre entier différent (index) de $0$ à $N-1$.
Pour obtenir la représentation vectorielle à un coup
pour tout mot d'indice $i$,
nous créons un vecteur de longueur$N$ avec tous les 0
et définissons l'élément en position $i$ à 1.
De cette façon, chaque mot est représenté comme un vecteur de longueur $N$, et il
peut être utilisé directement par les réseaux neuronaux.


Bien que les vecteurs de mots " one-hot " soient faciles à construire,
ils ne constituent généralement pas un bon choix.
L'une des principales raisons est que les vecteurs de mots à un coup ne peuvent pas exprimer avec précision la similarité entre différents mots, comme la similarité en cosinus * que nous utilisons souvent.
Pour les vecteurs $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, leur similarité en cosinus est le cosinus de l'angle qui les sépare :


 $$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$ 

 
 Comme la similarité en cosinus entre les vecteurs one-hot de deux mots différents est égale à 0,
les vecteurs one-hot ne peuvent pas coder les similarités entre les mots.


## Self-Supervised word2vec

L'outil [word2vec](https://code.google.com/archive/p/word2vec/) a été proposé pour résoudre le problème ci-dessus.
Il fait correspondre chaque mot à un vecteur de longueur fixe, et ces vecteurs peuvent mieux exprimer la relation de similarité et d'analogie entre différents mots.
L'outil word2vec contient deux modèles, à savoir *skip-gram* :cite:`Mikolov.Sutskever.Chen.ea.2013` et *continuous bag of words* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013` .
Pour les représentations sémantiquement significatives,
leur formation repose sur
des probabilités conditionnelles
qui peuvent être considérées comme la prédiction de
certains mots à l'aide de certains de leurs mots environnants
dans des corpus.
Puisque la supervision provient des données sans étiquettes,
les modèles de saut de programme et de sac de mots continu
sont des modèles auto-supervisés.

Dans ce qui suit, nous allons présenter ces deux modèles et leurs méthodes de formation.


## Le modèle Skip-Gram
:label:`subsec_skip-gram` 

 Le modèle *skip-gram* suppose qu'un mot peut être utilisé pour générer les mots qui l'entourent dans une séquence de texte.
Prenons comme exemple la séquence de texte "le", "homme", "aime", "son", "fils".
Choisissons "aime" comme *mot central* et fixons la taille de la fenêtre de contexte à 2.
Comme indiqué dans :numref:`fig_skip_gram` ,
étant donné le mot central "aime",
le modèle de saut de programme considère
la probabilité conditionnelle de générer les *mots de contexte* : "le", "homme", "son" et "fils",
qui ne sont pas éloignés de plus de 2 mots du mot central :

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$ 

 Supposons que
les mots contextuels soient générés indépendamment
étant donné le mot central (c'est-à-dire, l'indépendance conditionnelle).
Dans ce cas, la probabilité conditionnelle ci-dessus
peut être réécrite comme suit :

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$ 

 ![The skip-gram model considers the conditional probability of generating the surrounding context words given a center word.](../img/skip-gram.svg) 
:label:`fig_skip_gram` 

 Dans le modèle de saut de programme, chaque mot
a deux représentations vectorielles $d$-dimensionnelles
pour calculer les probabilités conditionnelles.
Plus concrètement,
pour tout mot ayant l'indice $i$ dans le dictionnaire,
désigne par $\mathbf{v}_i\in\mathbb{R}^d$
 et $\mathbf{u}_i\in\mathbb{R}^d$
 ses deux vecteurs
lorsqu'il est utilisé comme mot *centre* et comme mot *contexte*, respectivement.
La probabilité conditionnelle de générer n'importe quel mot de contexte
 $w_o$ (avec l'indice $o$ dans le dictionnaire) étant donné le mot central $w_c$ (avec l'indice $c$ dans le dictionnaire) peut être modélisée par
une opération softmax sur les produits scalaires :


 $$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$ 
 :eqlabel:`eq_skip-gram-softmax` 

 où l'ensemble des indices du vocabulaire $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$.
Étant donné une séquence de texte de longueur $T$, où le mot au pas de temps $t$ est désigné par $w^{(t)}$.
Supposons que les mots de contexte
soient générés de manière indépendante
pour un mot central quelconque.
Pour la taille de la fenêtre de contexte $m$,
la fonction de vraisemblance du modèle de saut de programme
est la probabilité de générer tous les mots de contexte
étant donné un mot central quelconque :


 $$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$ 

 où tout pas de temps inférieur à $1$ ou supérieur à $T$ peut être omis.

### Formation

Les paramètres du modèle de saut de programme sont le vecteur du mot central et le vecteur du mot contextuel pour chaque mot du vocabulaire.
Lors de la formation, nous apprenons les paramètres du modèle en maximisant la fonction de vraisemblance (c'est-à-dire l'estimation de vraisemblance maximale). Cela équivaut à minimiser la fonction de perte suivante :

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$ 

 Lorsque l'on utilise la descente de gradient stochastique pour minimiser la perte,
à chaque itération
nous pouvons
échantillonner de manière aléatoire une sous-séquence plus courte pour calculer le gradient (stochastique) de cette sous-séquence afin de mettre à jour les paramètres du modèle.
Pour calculer ce gradient (stochastique),
nous devons obtenir
les gradients de
la probabilité conditionnelle logarithmique par rapport au vecteur mot central et au vecteur mot contextuel.
En général, selon :eqref:`eq_skip-gram-softmax` 
 la probabilité conditionnelle logarithmique
impliquant toute paire du mot central $w_c$ et
le mot de contexte $w_o$ est


 $$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$ 
 :eqlabel:`eq_skip-gram-log` 

 Par différenciation, nous pouvons obtenir son gradient
par rapport au vecteur du mot central $\mathbf{v}_c$ comme

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$ 
 :eqlabel:`eq_skip-gram-grad` 

 
 Notez que le calcul dans :eqref:`eq_skip-gram-grad` nécessite les probabilités conditionnelles de tous les mots du dictionnaire avec $w_c$ comme mot central.
Les gradients des autres vecteurs mots peuvent être obtenus de la même manière.


Après l'apprentissage, pour tout mot ayant l'index $i$ dans le dictionnaire, nous obtenons les deux vecteurs de mots
$\mathbf{v}_i$ (comme mot central) et $\mathbf{u}_i$ (comme mot de contexte).
Dans les applications de traitement du langage naturel, les vecteurs de mots centraux du modèle de saut de programme sont généralement
utilisés comme représentations de mots.


## Le modèle du sac continu de mots (CBOW)


 Le modèle du sac *continu de mots* (CBOW) est similaire au modèle du skip-gramme.
La principale différence
par rapport au modèle skip-gram est que
le modèle de sac continu de mots
suppose qu'un mot central est généré
sur la base des mots contextuels qui l'entourent dans la séquence de texte.
Par exemple,
dans la même séquence de texte "le", "homme", "aime", "son" et "fils", avec "aime" comme mot central et la taille de la fenêtre de contexte étant 2,
le modèle de sac de mots continu
considère
la probabilité conditionnelle de générer le mot central "aime" sur la base des mots de contexte "le", "man", "his" et "son" (comme indiqué sur :numref:`fig_cbow` ), qui est

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$ 

 ![The continuous bag of words model considers the conditional probability of generating the center word given its surrounding context words.](../img/cbow.svg) 
:label:`fig_cbow` 

 
 Étant donné qu'il existe plusieurs mots contextuels
dans le modèle de sac de mots continu,
la moyenne de ces vecteurs de mots contextuels est calculée sur
dans le calcul de la probabilité conditionnelle.
Plus précisément,
pour tout mot ayant l'index $i$ dans le dictionnaire,
désigne par $\mathbf{v}_i\in\mathbb{R}^d$
 et $\mathbf{u}_i\in\mathbb{R}^d$
 ses deux vecteurs
lorsqu'il est utilisé comme mot *contextuel* et comme mot *central*
(les significations sont interverties dans le modèle skip-gram), respectivement.
La probabilité conditionnelle de générer n'importe quel mot central
 $w_c$ (avec l'indice $c$ dans le dictionnaire) étant donné les mots contextuels qui l'entourent $w_{o_1}, \ldots, w_{o_{2m}}$ (avec l'indice $o_1, \ldots, o_{2m}$ dans le dictionnaire) peut être modélisée par



 $$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$ 
 :eqlabel:`fig_cbow-full` 

 
 Par souci de concision, laissons $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ et $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$. Alors, :eqref:`fig_cbow-full` peut être simplifié comme suit :

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$ 

 Étant donné une séquence de texte de longueur $T$, où le mot au pas de temps $t$ est désigné par $w^{(t)}$.
Pour la taille de la fenêtre de contexte $m$,
la fonction de vraisemblance du modèle de sac de mots continu
est la probabilité de générer tous les mots centraux
étant donné leurs mots de contexte :


 $$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$ 

### Formation

La formation des modèles de sac de mots continus
est presque la même que la formation des modèles de saut de programme
.
L'estimation par maximum de vraisemblance du modèle de sac de mots continu
est équivalente à la minimisation de la fonction de perte suivante :



$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Remarquez que

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$ 

 Par différenciation, nous pouvons obtenir son gradient
par rapport à tout vecteur de mots contextuel $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$)
comme


 $$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$ 
 :eqlabel:`eq_cbow-gradient` 

 
 Les gradients des autres vecteurs de mots peuvent être obtenus de la même manière.
Contrairement au modèle de saut de programme,
le modèle de sac de mots continu
typiquement
utilise des vecteurs de mots contextuels comme représentations des mots.




## Résumé

* Les vecteurs de mots sont des vecteurs utilisés pour représenter les mots, et peuvent également être considérés comme des vecteurs de caractéristiques ou des représentations de mots. La technique de mise en correspondance des mots avec des vecteurs réels est appelée intégration de mots.
* L'outil word2vec contient à la fois le modèle "skip-gram" et le modèle "continuous bag of words".
* Le modèle skip-gram suppose qu'un mot peut être utilisé pour générer les mots qui l'entourent dans une séquence de texte, tandis que le modèle continuous bag of words suppose qu'un mot central est généré sur la base des mots qui l'entourent.



## Exercices

1. Quelle est la complexité de calcul de chaque gradient ? Quel pourrait être le problème si la taille du dictionnaire est énorme ?
1. Certaines phrases fixes en anglais sont composées de plusieurs mots, comme "new york". Comment former leurs vecteurs de mots ? Conseil : voir la section 4 du document word2vec :cite:`Mikolov.Sutskever.Chen.ea.2013` .
1. Réfléchissons à la conception de word2vec en prenant le modèle de saut de gramme comme exemple. Quelle est la relation entre le produit scalaire de deux vecteurs de mots dans le modèle skip-gram et la similarité en cosinus ? Pour une paire de mots ayant une sémantique similaire, pourquoi la similarité en cosinus de leurs vecteurs de mots (formés par le modèle skip-gram) peut-elle être élevée ?

[Discussions](https://discuss.d2l.ai/t/381)
