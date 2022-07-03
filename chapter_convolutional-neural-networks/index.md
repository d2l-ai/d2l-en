# Réseaux neuronaux convolutifs
:label:`chap_cnn` 

 Les données d'image sont représentées sous la forme d'une grille bidimensionnelle de pixels, qu'elle soit
monochromatique ou en couleur. En conséquence, chaque pixel correspond respectivement à une
ou plusieurs valeurs numériques. Jusqu'à présent, nous avons ignoré cette riche structure
et les avons traitées comme des vecteurs de nombres en *aplatissant* les images
, sans tenir compte de la relation spatiale entre les pixels. Cette approche
profondément insatisfaisante était nécessaire pour faire passer les vecteurs unidimensionnels
résultants par un MLP entièrement connecté.

Comme ces réseaux sont invariants par rapport à l'ordre des caractéristiques, nous
pourrions obtenir des résultats similaires, que nous préservions un ordre
correspondant à la structure spatiale des pixels ou que nous permutions
les colonnes de notre matrice de conception avant d'ajuster les paramètres du MLP.
Il serait préférable de tirer parti de notre connaissance préalable du fait que les pixels voisins
sont généralement liés les uns aux autres, afin de construire des modèles efficaces pour
l'apprentissage à partir de données d'image.

Ce chapitre présente les réseaux de neurones convolutifs * (CNN)
:cite:`LeCun.Jackel.Bottou.ea.1995` , une puissante famille de réseaux de neurones qui
sont conçus précisément dans ce but.
Les architectures basées sur les CNN sont
maintenant omniprésentes dans le domaine de la vision par ordinateur.
Par exemple, sur la collection Imagnet
:cite:`Deng.Dong.Socher.ea.2009` , seule l'utilisation de réseaux neuronaux convolutifs
, en bref les Convnets, a permis d'obtenir des améliorations significatives des performances
 :cite:`Krizhevsky.Sutskever.Hinton.2012` .

Les CNN modernes, comme on les appelle familièrement, doivent leur conception à
des inspirations de la biologie, de la théorie des groupes et à une bonne dose de
bricolage expérimental.  Outre leur efficacité en matière d'échantillonnage
pour obtenir des modèles précis, les CNN ont tendance à être efficaces sur le plan du calcul,
à la fois parce qu'ils nécessitent moins de paramètres que les architectures entièrement connectées
et parce que les convolutions sont faciles à paralléliser sur
les cœurs des GPU :cite:`Chetlur.Woolley.Vandermersch.ea.2014` .  Par conséquent, les praticiens
appliquent souvent les CNN chaque fois que cela est possible, et de plus en plus, ils sont apparus comme des concurrents crédibles
même pour des tâches avec une structure de séquence unidimensionnelle
, comme l'audio :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014` , le texte
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014` , et l'analyse de séries temporelles
:cite:`LeCun.Bengio.ea.1995` , où les réseaux neuronaux récurrents sont
conventionnellement utilisés.  Certaines adaptations astucieuses des CNN ont également
permis de les utiliser sur des données structurées en graphes :cite:`Kipf.Welling.2016` et
dans les systèmes de recommandation.

Tout d'abord, nous allons nous plonger plus profondément dans la motivation des réseaux neuronaux convolutifs
. Nous passerons ensuite en revue les opérations de base
qui constituent la colonne vertébrale de tous les réseaux convolutifs.
Il s'agit notamment des couches convolutionnelles elles-mêmes,
des détails de détail tels que le padding et le stride,
des couches de mise en commun utilisées pour agréger les informations
dans des régions spatiales adjacentes,
de l'utilisation de plusieurs canaux à chaque couche,
et d'une discussion approfondie de la structure des architectures modernes.
Nous conclurons ce chapitre par un exemple fonctionnel complet de LeNet,
le premier réseau convolutif déployé avec succès,
bien avant l'essor de l'apprentissage profond moderne.
Dans le chapitre suivant, nous nous plongerons dans les implémentations complètes
de quelques architectures CNN populaires et comparativement récentes
dont les conceptions représentent la plupart des techniques
couramment utilisées par les praticiens modernes.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

