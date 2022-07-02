# Réseaux neuronaux récurrents modernes
:label:`chap_modern_rnn` 

 Nous avons présenté les bases des RNN,
qui peuvent mieux traiter les données de séquence.
À titre de démonstration,
nous avons implémenté des modèles de langage
basés sur les RNN sur des données textuelles.
Cependant, 
ces techniques peuvent ne pas être suffisantes
pour les praticiens lorsqu'ils sont confrontés à
un large éventail de problèmes d'apprentissage de séquences de nos jours.

Par exemple,
un problème notable dans la pratique
est l'instabilité numérique des RNN.
Bien que nous ayons appliqué des astuces de mise en œuvre
telles que l'écrêtage du gradient,
ce problème peut être atténué davantage
avec des conceptions plus sophistiquées des modèles de séquence.
Plus précisément, les RNN à déclenchement
sont beaucoup plus courants dans la pratique.
Nous commencerons par présenter deux de ces réseaux largement utilisés,
à savoir les *gated recurrent units* (GRUs) et les *long short-term memory* (LSTM).
En outre, nous développerons l'architecture RNN
avec une seule couche cachée unidirectionnelle
qui a été discutée jusqu'à présent.
Nous décrirons des architectures profondes avec
plusieurs couches cachées,
et discuterons de la conception bidirectionnelle
avec des calculs récurrents en avant et en arrière.
De telles expansions sont fréquemment adoptées
dans les réseaux récurrents modernes.
Lorsque nous expliquons ces variantes de RNN,
, nous continuons à considérer
le même problème de modélisation du langage que celui présenté dans :numref:`chap_rnn` .

En fait, la modélisation du langage
ne révèle qu'une petite fraction de ce dont 
est capable en matière d'apprentissage de séquences.
Dans une variété de problèmes d'apprentissage de séquences,
tels que la reconnaissance automatique de la parole, la conversion de texte en parole et la traduction automatique,
les entrées et les sorties sont des séquences de longueur arbitraire.
Pour expliquer comment adapter ce type de données,
nous prendrons la traduction automatique comme exemple,
et présenterons l'architecture codeur-décodeur basée sur
RNNs et la recherche de faisceau pour la génération de séquences.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

