# Traitement du langage naturel : Applications
:label:`chap_nlp_app` 

 Nous avons vu comment représenter les tokens dans des séquences de texte et entraîner leurs représentations dans :numref:`chap_nlp_pretrain` .
Ces représentations textuelles pré-entraînées peuvent être utilisées dans divers modèles pour différentes tâches de traitement du langage naturel en aval.

En fait,
les chapitres précédents ont déjà abordé certaines applications de traitement du langage naturel
*sans pré-entraînement*,
juste pour expliquer les architectures d'apprentissage profond.
Par exemple, dans :numref:`chap_rnn` ,
, nous nous sommes appuyés sur les RNN pour concevoir des modèles de langage permettant de générer des textes de type roman.
Dans :numref:`chap_modern_rnn` et :numref:`chap_attention` ,
, nous avons également conçu des modèles basés sur les RNN et des mécanismes d'attention pour la traduction automatique.

Cependant, ce livre n'a pas pour objectif de couvrir toutes ces applications de manière exhaustive.
Au lieu de cela,
, nous nous concentrons sur *la manière d'appliquer l'apprentissage (profond) de la représentation des langues pour résoudre les problèmes de traitement du langage naturel*.
Compte tenu des représentations de texte pré-entraînées,
ce chapitre explorera deux 
tâches de traitement du langage naturel populaires et représentatives
en aval :
l'analyse des sentiments et l'inférence du langage naturel,
qui analysent respectivement un texte unique et les relations des paires de textes.

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

Comme indiqué dans :numref:`fig_nlp-map-app` ,
ce chapitre se concentre sur la description des idées de base de la conception de modèles de traitement du langage naturel en utilisant différents types d'architectures d'apprentissage profond, comme les MLP, les CNN, les RNN et l'attention.
Bien qu'il soit possible de combiner n'importe quelle représentation de texte pré-entraînée avec n'importe quelle architecture pour n'importe quelle application dans :numref:`fig_nlp-map-app` ,
nous sélectionnons quelques combinaisons représentatives.
Plus précisément, nous allons explorer les architectures populaires basées sur les RNN et les CNN pour l'analyse des sentiments.
Pour l'inférence en langage naturel, nous choisissons l'attention et les MLP pour démontrer comment analyser les paires de textes.
Enfin, nous présentons la manière d'affiner un modèle BERT pré-entraîné
pour un large éventail d'applications de traitement du langage naturel,
, notamment au niveau de la séquence (classification d'un seul texte et classification de paires de textes)
et au niveau du token (étiquetage de textes et réponse à des questions).
Comme cas empirique concret,
nous allons affiner BERT pour l'inférence en langage naturel.

Comme nous l'avons présenté dans :numref:`sec_bert` ,
BERT nécessite des modifications minimales de l'architecture
pour une large gamme d'applications de traitement du langage naturel.
Cependant, cet avantage a pour contrepartie le réglage fin
d'un grand nombre de paramètres BERT pour les applications en aval.
Lorsque l'espace ou le temps est limité,
les modèles élaborés basés sur les MLP, CNN, RNN et l'attention
sont plus réalisables.
Dans ce qui suit, nous commençons par l'application d'analyse de sentiments
et illustrons la conception de modèles basés sur les RNN et les CNN, respectivement.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```

