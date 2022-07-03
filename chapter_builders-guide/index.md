# Builders' Guide
:label:`chap_computation` 

 Outre les ensembles de données géants et le matériel puissant,
d'excellents outils logiciels ont joué un rôle indispensable
dans la progression rapide de l'apprentissage profond.
En commençant par la bibliothèque Theano, publiée en 2007,
des outils open-source flexibles ont permis aux chercheurs
de prototyper rapidement des modèles, en évitant le travail répétitif
en recyclant des composants standard
tout en conservant la possibilité d'apporter des modifications de bas niveau.
Au fil du temps, les bibliothèques d'apprentissage profond ont évolué
pour offrir des abstractions de plus en plus grossières.
Tout comme les concepteurs de semi-conducteurs sont passés de la spécification des transistors
aux circuits logiques à l'écriture de code,
les chercheurs en réseaux neuronaux sont passés de la réflexion sur
le comportement de neurones artificiels individuels
à la conception de réseaux en termes de couches entières,
et conçoivent désormais souvent des architectures avec des *blocs* beaucoup plus grossiers en tête.


Jusqu'à présent, nous avons présenté quelques concepts de base de l'apprentissage automatique,
pour passer à des modèles d'apprentissage profond entièrement fonctionnels.
Dans le dernier chapitre,
nous avons implémenté chaque composant d'un MLP à partir de zéro
et nous avons même montré comment exploiter les API de haut niveau
pour déployer les mêmes modèles sans effort.
Pour aller aussi loin et aussi vite, nous avons *fait appel aux bibliothèques,
mais nous avons ignoré les détails plus avancés sur *leur fonctionnement*.
Dans ce chapitre, nous allons lever le voile,
et approfondir les composants clés du calcul de l'apprentissage profond,
à savoir la construction de modèles, l'accès aux paramètres et l'initialisation,
la conception de couches et de blocs personnalisés, la lecture et l'écriture de modèles sur le disque,
et l'exploitation des GPU pour obtenir des accélérations spectaculaires.
Ces informations vous feront passer du statut d'utilisateur *final* à celui d'utilisateur *puissant*,
en vous donnant les outils nécessaires pour profiter des avantages
d'une bibliothèque d'apprentissage profond mature tout en conservant la flexibilité
d'implémenter des modèles plus complexes, y compris ceux que vous inventez vous-même !
Bien que ce chapitre ne présente pas de nouveaux modèles ou jeux de données,
les chapitres de modélisation avancée qui suivent s'appuient largement sur ces techniques.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```

