# Réseaux neuronaux convolutifs modernes
:label:`chap_modern_cnn` 

Maintenant que nous avons compris les bases du câblage des CNN, nous allons
vous faire découvrir les architectures CNN modernes. Cette visite est, par
nécessité, incomplète, grâce à la pléthore de nouvelles conceptions passionnantes
qui sont ajoutées. Leur importance découle du fait que non seulement
ils peuvent être utilisés directement pour des tâches de vision, mais ils servent également de générateurs de caractéristiques de base
pour des tâches plus avancées telles que le suivi
:cite:`Zhang.Sun.Jiang.ea.2021`, la segmentation :cite:`Long.Shelhamer.Darrell.2015`, la détection d'objets
 :cite:`Redmon.Farhadi.2018`, ou la transformation de style
:cite:`Gatys.Ecker.Bethge.2016`.  Dans ce chapitre, la plupart des sections
correspondent à une architecture CNN importante qui a été à un moment donné
(ou actuellement) le modèle de base sur lequel de nombreux projets de recherche et
systèmes déployés ont été construits.  Chacun de ces réseaux a été brièvement une architecture dominante
et beaucoup d'entre eux ont été gagnants ou finalistes du concours
[ImageNet competition](https://www.image-net.org/challenges/LSVRC/) 
qui sert de baromètre des progrès de l'apprentissage supervisé en
vision par ordinateur depuis 2010.

Ces modèles comprennent l'AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`,
le premier réseau à grande échelle déployé pour battre les méthodes conventionnelles de vision par ordinateur
dans un défi de vision à grande échelle ; le réseau VGG
:cite:`Simonyan.Zisserman.2014`, qui utilise un certain nombre de
blocs d'éléments répétitifs ; le réseau en réseau (NiN) qui
convolue des réseaux neuronaux entiers par patch sur des entrées
:cite:`Lin.Chen.Yan.2013` ; le GoogLeNet qui utilise des réseaux avec des convolutions multibranches
 :cite:`Szegedy.Liu.Jia.ea.2015` ; le réseau résiduel
(ResNet) :cite:`He.Zhang.Ren.ea.2016`, qui reste l'une des
architectures sur étagère les plus populaires dans le domaine de la vision par ordinateur ;
les blocs ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017` 
pour des connexions plus éparses ;
et 
le DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017` pour une généralisation de l'architecture résiduelle
.
En outre,
en supposant des blocs standard et fixes,
nous simplifions progressivement les espaces de conception avec de meilleurs modèles,

conduisant au RegNet
:cite:`Radosavovic.Kosaraju.Girshick.ea.2020`. 


Bien que l'idée des réseaux neuronaux *profonds* soit assez simple (empiler
un tas de couches), les performances peuvent varier énormément selon les architectures
et les choix d'hyperparamètres.  Les réseaux neuronaux
décrits dans ce chapitre sont le fruit de l'intuition, de quelques connaissances mathématiques
et de nombreux essais et erreurs.  Nous présentons ces modèles
par ordre chronologique, en partie pour donner un sens à l'histoire
afin que vous puissiez vous forger vos propres intuitions sur l'orientation du domaine
et peut-être développer vos propres architectures.  Par exemple, la normalisation des lots
et les connexions résiduelles décrites dans ce chapitre
ont offert deux idées populaires pour l'entrainement et la conception de modèles profonds,
toutes deux ayant depuis été appliquées à des architectures autres que la vision par ordinateur
également.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```

