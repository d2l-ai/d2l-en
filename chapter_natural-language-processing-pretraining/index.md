# Traitement du langage naturel : Préformation
:label:`chap_nlp_pretrain` 

 
 Les êtres humains ont besoin de communiquer.
Ce besoin fondamental de la condition humaine est à l'origine de la production quotidienne d'une grande quantité de textes écrits.
Compte tenu de la richesse du texte dans les médias sociaux, les applications de chat, les courriels, les critiques de produits, les articles d'actualité, les documents de recherche et les livres, il devient vital de permettre aux ordinateurs de les comprendre pour offrir une assistance ou prendre des décisions basées sur le langage humain.

*Le traitement du langage naturel* étudie les interactions entre les ordinateurs et les humains à l'aide de langages naturels.
En pratique, il est très courant d'utiliser des techniques de traitement du langage naturel pour traiter et analyser des données textuelles (langage naturel humain), comme les modèles de langage dans :numref:`sec_language-model` et les modèles de traduction automatique dans :numref:`sec_machine_translation` .

Pour comprendre un texte, nous pouvons commencer par apprendre
ses représentations.
En s'appuyant sur les séquences de texte existantes
provenant de grands corpus,
*l'apprentissage auto-supervisé*
a été largement utilisé
pour pré-former les représentations de texte,
par exemple en prédisant une partie cachée du texte
en utilisant une autre partie du texte environnant.
De cette façon, les modèles
apprennent par supervision
à partir de données textuelles *massives*
sans efforts *coûteux* d'étiquetage !


Comme nous le verrons dans ce chapitre,
lorsque l'on traite chaque mot ou sous-mot comme un jeton individuel,
la représentation de chaque jeton peut être pré-entraînée
à l'aide des modèles word2vec, GloVe ou subword embedding
sur de grands corpus.
Après le pré-entraînement, la représentation de chaque token peut être un vecteur,
. Cependant, elle reste la même quel que soit le contexte.
Par exemple, la représentation vectorielle de "bank" est la même
dans les deux cas :
"go to the bank to deposit some money"
et
"go to the bank to sit down".
Ainsi, de nombreux modèles de pré-entraînement plus récents adaptent la représentation du même jeton
à différents contextes.
Parmi eux se trouve BERT, un modèle auto-supervisé beaucoup plus profond basé sur l'encodeur transformateur.
Dans ce chapitre, nous nous concentrerons sur la manière de prétraîner de telles représentations pour le texte,
comme mis en évidence dans :numref:`fig_nlp-map-pretrain` .

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on the upstream text representation pretraining.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`


Pour avoir une vue d'ensemble,
:numref:`fig_nlp-map-pretrain` montre que
les représentations de texte pré-entraînées peuvent être alimentées par
une variété d'architectures d'apprentissage profond pour différentes applications de traitement du langage naturel en aval.
Nous les aborderons dans :numref:`chap_nlp_app` .

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```

