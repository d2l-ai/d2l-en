# Attention et transformateurs
:label:`chap_attention` 

Le nerf optique du système visuel d'un primate
reçoit des entrées sensorielles massives,
dépassant de loin ce que le cerveau peut traiter entièrement.
Heureusement,
tous les stimuli ne sont pas créés égaux.
La focalisation et la concentration de la conscience
ont permis aux primates de diriger leur attention
vers des objets d'intérêt,
tels que les proies et les prédateurs,
dans l'environnement visuel complexe.
La capacité de ne prêter attention qu'à
une petite fraction des informations
a une signification évolutive,
permettant aux êtres humains
de vivre et de réussir.

Les scientifiques étudient l'attention
dans le domaine des neurosciences cognitives
depuis le 19e siècle.
Dans ce chapitre,
nous commencerons par passer en revue un cadre populaire
expliquant comment l'attention est déployée dans une scène visuelle.
Inspirés par les indices d'attention de ce cadre,
nous allons concevoir des modèles
qui exploitent ces indices d'attention.
Notamment, la régression à noyau de Nadaraya-Watson
en 1964 est une démonstration simple de l'apprentissage automatique avec des *mécanismes d'attention*.
Ensuite, nous présenterons les fonctions d'attention
qui ont été largement utilisées dans
la conception de modèles d'attention en apprentissage profond.
Plus précisément,
nous montrerons comment utiliser ces fonctions
pour concevoir l'attention *Bahdanau*,
un modèle d'attention révolutionnaire dans l'apprentissage profond
qui peut s'aligner de manière bidirectionnelle et est différentiable.


Dotée de
les modèles plus récents d'*attention multi-têtes*
et d'*auto-attention*,
l'architecture du *transformateur* est uniquement
basée sur les mécanismes d'attention.
Nous décrirons ensuite sa conception originale d'encodeur-décodeur pour la traduction automatique.
Nous montrerons ensuite comment son encodeur peut 
représenter des images, ce qui a conduit au développement de transformateurs de vision.
Lors de l'entraînement de très grands modèles sur de très grands ensembles de données (par exemple, 300 millions d'images), les transformateurs de vision
surpassent les ResNets de manière significative dans la classification d'images, ce qui démontre l'évolutivité supérieure des transformateurs.
Ainsi, les transformateurs ont été largement utilisés dans le *pré-entraînement* à grande échelle, qui peut être adapté pour effectuer différentes tâches avec la mise à jour du modèle (par exemple, *réglage fin*) ou non (par exemple, *quelques clichés*).
Enfin, nous examinerons comment prétraîner les transformateurs en tant qu'encodeur seulement (par exemple, BERT), encodeur-décodeur (par exemple, T5) et décodeur seulement (par exemple, la série GPT).
Le succès convaincant du pré-entraînement à grande échelle avec des transformateurs dans des domaines aussi divers que
le langage,
la vision, la parole,
et l'apprentissage par renforcement
suggère que de meilleures performances bénéficient de modèles plus grands, de plus de données d'entraînement et de plus de calcul d'entraînement.

```toc
:maxdepth: 2

attention-cues
nadaraya-watson
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```

