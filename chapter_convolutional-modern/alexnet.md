```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Réseaux neuronaux convolutifs profonds (AlexNet)
:label:`sec_alexnet` 

 
 Bien que les CNN soient bien connus
dans les communautés de la vision par ordinateur et de l'apprentissage automatique
après l'introduction de LeNet :cite:`LeCun.Jackel.Bottou.ea.1995` ,
ils n'ont pas immédiatement dominé le domaine.
Bien que LeNet ait obtenu de bons résultats sur les premiers petits ensembles de données,
la performance et la faisabilité de l'entraînement des CNN
sur des ensembles de données plus grands et plus réalistes n'avaient pas encore été établies.
En fait, pendant la majeure partie du temps écoulé entre le début des années 1990
et les résultats décisifs de 2012 :cite:`Krizhevsky.Sutskever.Hinton.2012` ,
les réseaux neuronaux ont souvent été dépassés par d'autres méthodes d'apprentissage automatique,
telles que les méthodes à noyau :cite:`Scholkopf.Smola.2002` , les méthodes d'ensemble :cite:`Freund.Schapire.ea.1996` ,
et l'estimation structurée :cite:`Taskar.Guestrin.Koller.2004` .

Pour la vision par ordinateur, cette comparaison n'est peut-être pas juste.
En effet, bien que les entrées des réseaux convolutifs
soient constituées de valeurs de pixels brutes ou légèrement traitées (par exemple, par centrage), les praticiens n'introduisent jamais de pixels bruts dans les modèles traditionnels.
Au lieu de cela, les pipelines de vision par ordinateur typiques
consistaient en des pipelines d'extraction de caractéristiques conçus manuellement, tels que SIFT :cite:`Lowe.2004` , SURF :cite:`Bay.Tuytelaars.Van-Gool.2006` , et les sacs de mots visuels :cite:`Sivic.Zisserman.2003` .
Plutôt que d'*apprendre les caractéristiques*, les caractéristiques étaient *fabriquées*.
La plupart des progrès ont été réalisés grâce à des idées plus ingénieuses pour les caractéristiques et à une connaissance approfondie de la géométrie :cite:`Hartley.Zisserman.2000` . L'algorithme d'apprentissage était souvent considéré comme une réflexion après coup.

Bien que certains accélérateurs de réseaux neuronaux aient été disponibles dans les années 1990,
ils n'étaient pas encore suffisamment puissants pour réaliser
des CNN profonds multicanaux et multicouches
avec un grand nombre de paramètres. Par exemple, le GeForce 256 de NVIDIA de 1999
était capable de traiter au maximum 480 millions d'opérations par seconde, sans aucun cadre de programmation significatif
pour les opérations au-delà des jeux. Les accélérateurs d'aujourd'hui sont capables d'effectuer plus de 300 TFLOPs par dispositif (Ampere A100 de NVIDIA),
où *FLOPs*
sont des opérations en virgule flottante en nombre de multiplications-additions.
En outre, les ensembles de données étaient encore relativement petits : L'OCR sur 60 000 images à basse résolution était considérée comme une tâche très difficile.
En plus de ces obstacles, il manquait encore des astuces clés pour l'entraînement des réseaux neuronaux
, notamment des heuristiques d'initialisation des paramètres :cite:`Glorot.Bengio.2010` ,
des variantes intelligentes de la descente de gradient stochastique :cite:`Kingma.Ba.2014` ,
des fonctions d'activation non écrasantes :cite:`Nair.Hinton.2010` ,
et des techniques de régularisation efficaces :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` .

Ainsi, plutôt que de former des systèmes *de bout en bout* (du pixel à la classification),
les pipelines classiques ressemblaient davantage à ceci :

1. Obtenir un ensemble de données intéressant. Au début, ces jeux de données nécessitaient des capteurs coûteux. Par exemple, le site [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake) de 1994 affichait une résolution énorme de 0,3 mégapixel (VGA), capable de stocker jusqu'à 8 images, le tout pour le prix de \_ 224 \_fois 224 \_mois1$,000.
1. Prétraiter l'ensemble des données avec des caractéristiques fabriquées à la main, en se basant sur des connaissances en optique, en géométrie, sur d'autres outils analytiques, et parfois sur les découvertes fortuites d'étudiants diplômés chanceux.
1. Faites passer les données par un ensemble standard d'extracteurs de caractéristiques tels que le SIFT (scale-invariant feature transform) :cite:`Lowe.2004` , le SURF (speeded up robust features) :cite:`Bay.Tuytelaars.Van-Gool.2006` , ou un certain nombre d'autres pipelines réglés à la main.
1. Introduisez les représentations résultantes dans votre classificateur préféré, probablement un modèle linéaire ou une méthode à noyau, pour entraîner un classificateur.

Si vous avez parlé à des chercheurs en apprentissage automatique,
ils pensaient que l'apprentissage automatique était à la fois important et beau.
Des théories élégantes prouvaient les propriétés de divers classificateurs :cite:`Boucheron.Bousquet.Lugosi.2005` et l'optimisation convexe
 :cite:`Boyd.Vandenberghe.2004` était devenue le pilier pour les obtenir.
Le domaine de l'apprentissage automatique était florissant, rigoureux et éminemment utile. Cependant,
si vous parliez à un chercheur en vision par ordinateur,
vous entendriez une histoire très différente.
La vérité crasse de la reconnaissance d'images, vous diraient-ils,
est que les progrès sont dus aux caractéristiques, à la géométrie :cite:`Hartley.Zisserman.2000` , et à l'ingénierie,
plutôt qu'aux nouveaux algorithmes d'apprentissage.
Les chercheurs en vision par ordinateur croyaient à juste titre
qu'un ensemble de données légèrement plus grand ou plus propre
ou un pipeline d'extraction de caractéristiques légèrement amélioré
comptaient beaucoup plus pour la précision finale que n'importe quel algorithme d'apprentissage.

## Apprentissage des représentations

Une autre façon de présenter la situation est que
la partie la plus importante du pipeline était la représentation.
Et jusqu'en 2012, la représentation était calculée de manière essentiellement mécanique.
En fait, l'ingénierie d'un nouvel ensemble de fonctions de caractéristiques, l'amélioration des résultats et la rédaction de la méthode constituaient un genre d'article important.
SIFT :cite:`Lowe.2004` ,
SURF :cite:`Bay.Tuytelaars.Van-Gool.2006` ,
HOG (histogrammes de gradient orienté) :cite:`Dalal.Triggs.2005` ,
[bags of visual words] (https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)
et autres extracteurs de caractéristiques similaires ont dominé.

Un autre groupe de chercheurs,
dont Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari, et Juergen Schmidhuber,
avait d'autres projets.
Ils pensaient que les caractéristiques elles-mêmes devaient être apprises.
De plus, ils pensaient que pour être raisonnablement complexes,
les caractéristiques devaient être composées de manière hiérarchique
avec plusieurs couches apprises conjointement, chacune avec des paramètres apprenables.
Dans le cas d'une image, les couches les plus basses pourraient venir
pour détecter les bords, les couleurs et les textures, par analogie avec la façon dont le système visuel des animaux
traite ses entrées.

Le premier CNN moderne :cite:`Krizhevsky.Sutskever.Hinton.2012` , nommé
*AlexNet* d'après l'un de ses inventeurs, Alex Krizhevsky, est en grande partie une amélioration évolutive
par rapport à LeNet. Il a obtenu d'excellentes performances lors du concours ImageNet 2012.

![Filtres d'image appris par la première couche d'AlexNet (from :cite:`Krizhevsky.Sutskever.Hinton.2012`).](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Il est intéressant de noter que dans les couches les plus basses du réseau,
le modèle a appris des extracteurs de caractéristiques qui ressemblent à certains filtres traditionnels.
:numref:`fig_filters` 
montre des descripteurs d'image de niveau inférieur.
Les couches supérieures du réseau pourraient s'appuyer sur ces représentations
pour représenter des structures plus grandes, comme des yeux, des nez, des brins d'herbe, etc.
Des couches encore plus élevées peuvent représenter des objets entiers
comme des personnes, des avions, des chiens ou des frisbees.
En fin de compte, l'état caché final apprend une représentation compacte
de l'image qui résume son contenu
de sorte que les données appartenant à différentes catégories puissent être facilement séparées.

AlexNet (2012) et son précurseur LeNet (1995) partagent de nombreux éléments architecturaux. On peut donc se demander pourquoi il a fallu autant de temps.
L'une des principales différences est qu'au cours des deux dernières décennies, les données et les calculs ont considérablement augmenté. AlexNet était donc beaucoup plus grand :
il a été entraîné sur beaucoup plus de données et sur des GPU beaucoup plus rapides que les CPU disponibles en 1995.

#### Ingrédient manquant : Données

Les modèles profonds comportant de nombreuses couches nécessitent de grandes quantités de données
afin d'entrer dans le régime
où ils surpassent de manière significative les méthodes traditionnelles
basées sur des optimisations convexes (par exemple, les méthodes linéaires et à noyau).
Cependant, étant donné la capacité de stockage limitée des ordinateurs,
le coût relatif des capteurs (d'imagerie),
et les budgets de recherche comparativement plus restreints dans les années 1990,
la plupart des recherches se sont appuyées sur de minuscules ensembles de données.
De nombreux articles ont traité de la collection de jeux de données de l'UCI,
dont beaucoup ne contenaient que des centaines ou (quelques) milliers d'images
capturées en basse résolution et souvent avec un arrière-plan artificiellement propre.

En 2009, le jeu de données ImageNet a été publié sur :cite:`Deng.Dong.Socher.ea.2009` ,
et a mis les chercheurs au défi d'apprendre des modèles à partir d'un million d'exemples,
1000 chacun provenant de 1000 catégories d'objets distinctes. Les catégories elles-mêmes
étaient basées sur les nœuds nominaux les plus populaires de WordNet :cite:`Miller.1995`.
L'équipe d'ImageNet a utilisé Google Image Search pour préfiltrer de grands ensembles de candidats
pour chaque catégorie et a employé
le pipeline de crowdsourcing Amazon Mechanical Turk
pour confirmer pour chaque image si elle appartenait à la catégorie associée.
Cette échelle était sans précédent, dépassant les autres par plus d'un ordre de grandeur
(par exemple, CIFAR-100 compte 60 000 images). $Un$autre aspect était que les images étaient à
une résolution assez élevée de $224 \times 224$ pixels, contrairement à l'ensemble de données TinyImages
de 80 millions de pixels :cite:`Torralba.Fergus.Freeman.2008`, composé de$ vignettes de $32 \times 32$ pixels.
Cela a permis l'entrainement de caractéristiques de plus haut niveau.
Le concours associé, baptisé ImageNet Large Scale Visual Recognition
Challenge ([ILSVRC](https://www.image-net.org/challenges/LSVRC/))
a fait avancer la recherche sur la vision par ordinateur et l'apprentissage automatique,
en mettant les chercheurs au défi d'identifier les modèles les plus performants
à une échelle plus grande que celle envisagée auparavant par les universitaires.

#### L'ingrédient manquant : Matériel

Les modèles d'apprentissage profond sont des consommateurs voraces de cycles de calcul.
L'apprentissage peut prendre des centaines d'époques, et chaque itération
nécessite de faire passer les données par de nombreuses couches d'opérations d'algèbre linéaire coûteuses en calcul
.
C'est l'une des principales raisons pour lesquelles, dans les années 1990 et au début des années 2000, on préférait
les algorithmes simples basés sur les objectifs convexes
optimisés de manière plus efficace.

*Les unités de traitement graphique* (GPU) ont changé la donne
en rendant l'apprentissage profond possible.
Ces puces ont été développées depuis longtemps pour accélérer le traitement graphique
au profit des jeux vidéo.
Elles ont notamment été optimisées pour les $$produits matrice-vecteur $ à$ haut débit 4 $\times 4$
, nécessaires à de nombreuses tâches d'infographie.
Heureusement, les mathématiques sont étonnamment similaires
à celles requises pour calculer les couches convolutionnelles.
À cette époque, NVIDIA et ATI avaient commencé à optimiser les GPU
pour les opérations de calcul général :cite:`Fernando.2004`,
allant jusqu'à les commercialiser sous le nom de *GPU polyvalents* (GPGPU).

Pour donner une idée de ce qu'est un GPU, considérons les cœurs d'un microprocesseur moderne
(CPU).
Chacun de ces cœurs est assez puissant, tourne à une fréquence d'horloge élevée
et possède de grandes caches (jusqu'à plusieurs mégaoctets de L3).
Chaque cœur est bien adapté à l'exécution d'une large gamme d'instructions,
avec des prédicteurs de branchement, un pipeline profond, des unités d'exécution spécialisées,
l'exécution spéculative,
et bien d'autres fonctionnalités
qui lui permettent d'exécuter une grande variété de programmes avec un flux de contrôle sophistiqué.
Cette force apparente est cependant aussi son talon d'Achille :
les cœurs à usage général sont très coûteux à construire. Ils excellent dans le code polyvalent
avec un flux de contrôle important.
Cela nécessite une grande surface de puce, non seulement pour l'UAL (unité arithmétique et logique) de
où les calculs sont effectués, mais aussi pour
tous les éléments susmentionnés, plus
les interfaces mémoire, la logique de mise en cache entre les cœurs,
les interconnexions à grande vitesse, etc. Les processeurs sont
comparativement mauvais dans une seule tâche par rapport au matériel dédié.
Les ordinateurs portables modernes ont 4 à 8 cœurs,
et même les serveurs haut de gamme dépassent rarement 64 cœurs par socket,
simplement parce que ce n'est pas rentable.

En comparaison, les GPU peuvent être composés de milliers de petits éléments de traitement (les dernières puces Ampere de NIVIDA ont jusqu'à 6912 cœurs CUDA), souvent regroupés en groupes plus importants (NVIDIA les appelle warps).
Les détails diffèrent quelque peu entre NVIDIA, AMD, ARM et les autres fournisseurs de puces. Alors que chaque cœur est relativement faible,
fonctionnant à une fréquence d'horloge d'environ 1 GHz,
c'est le nombre total de ces cœurs qui rend les GPU plusieurs fois plus rapides que les CPU.
Par exemple, la récente génération Ampere de NVIDIA offre plus de 300 TFLOPs par puce pour les multiplications matricielles spécialisées de précision 16 bits (BFLOAT16), et jusqu'à 20 TFLOPs pour les opérations à virgule flottante plus générales (FP32).
Dans le même temps, les performances en virgule flottante des CPU dépassent rarement 1 TFLOP (le Graviton 2 d'AWS, par exemple, atteint 2 TFLOP en pointe pour les opérations de précision 16 bits).
La raison pour laquelle cela est possible est en fait assez simple :
. Tout d'abord, la consommation d'énergie a tendance à croître *quadratiquement* avec la fréquence d'horloge.
Par conséquent, pour le budget énergétique d'un cœur de CPU qui fonctionne 4 fois plus vite (un chiffre typique),
vous pouvez utiliser 16 cœurs de GPU à \frac{1}{4} $$ la vitesse,
ce qui donne $16 \times \frac{1}{4} = 4$ fois la performance.
En outre, les cœurs GPU sont beaucoup plus simples
(en fait, pendant longtemps, ils n'étaient même pas *capables*
d'exécuter du code à usage général),
ce qui les rend plus efficaces sur le plan énergétique.
Enfin, de nombreuses opérations d'apprentissage profond nécessitent une bande passante mémoire élevée.
Là encore, les GPU brillent par leurs bus au moins 10 fois plus larges que ceux de nombreux CPU.

Retour en 2012. Une percée majeure a eu lieu à l'adresse
lorsque Alex Krizhevsky et Ilya Sutskever
ont mis en œuvre un CNN profond
qui pouvait fonctionner sur des GPU.
Ils ont réalisé que les goulots d'étranglement des CNN,
convolutions et multiplications matricielles,
sont toutes des opérations qui peuvent être parallélisées dans le matériel.
En utilisant deux NVIDIA GTX 580 avec 3 Go de mémoire, l'une ou l'autre étant capable de 1,5 TFLOP,
ils ont implémenté des convolutions rapides.
Le code [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)
était suffisamment bon pour que pendant plusieurs années
il soit le standard de l'industrie et alimente
les deux premières années du boom de l'apprentissage profond.

## AlexNet

AlexNet, qui employait un CNN à 8 couches,
a remporté le ImageNet Large Scale Visual Recognition Challenge 2012
par une large marge :cite:`Russakovsky.Deng.Huang.ea.2013` .
Ce réseau a montré, pour la première fois,
que les caractéristiques obtenues par apprentissage peuvent transcender les caractéristiques conçues manuellement, brisant ainsi le paradigme précédent en vision par ordinateur.

Les architectures d'AlexNet et de LeNet sont étonnamment similaires,
comme l'illustre :numref:`fig_alexnet` .
Notez que nous fournissons une version légèrement simplifiée d'AlexNet
en supprimant certaines bizarreries de conception qui étaient nécessaires en 2012
pour faire tenir le modèle sur deux petits GPU.

![De LeNet (gauche) à AlexNet (droite)](../img/alexnet.svg)
:label:`fig_alexnet` 

 Il existe également des différences significatives entre AlexNet et LeNet.
Tout d'abord, AlexNet est beaucoup plus profond que LeNet5, qui est relativement petit.
AlexNet se compose de huit couches : cinq couches convolutionnelles,
deux couches cachées entièrement connectées et une couche de sortie entièrement connectée.
Deuxièmement, AlexNet utilise la ReLU au lieu de la sigmoïde
comme fonction d'activation. Entrons dans les détails ci-dessous.

### Architecture

Dans la première couche d'AlexNet, la forme de la fenêtre de convolution est de $11 \times11$.
Comme les images d'ImageNet sont huit fois plus hautes et plus larges
que les images du MNIST,
les objets dans les données d'ImageNet ont tendance à occuper plus de pixels avec plus de détails visuels.
Par conséquent, une fenêtre de convolution plus grande est nécessaire pour capturer l'objet.
La forme de la fenêtre de convolution dans la deuxième couche
est réduite à 5 $\times5$, suivie de 3 $\times3$.
En outre, après les première, deuxième et cinquième couches de convolution,
le réseau ajoute des couches de max-pooling
avec une forme de fenêtre de $3 \times3$ et un stride de 2.
De plus, AlexNet possède dix fois plus de canaux de convolution que LeNet.

Après la dernière couche convolutionnelle, il y a deux couches entièrement connectées
avec 4096 sorties.
Ces deux énormes couches entièrement connectées produisent des paramètres de modèle de près de 1 GB.
En raison de la mémoire limitée des premiers GPU,
l'AlexNet original utilisait une conception à double flux de données,
afin que chacun de ses deux GPU puisse être responsable
du stockage et du calcul de sa seule moitié du modèle.
Heureusement, la mémoire des GPU est relativement abondante aujourd'hui,
. Il est donc rare de devoir répartir les modèles entre les GPU aujourd'hui
(notre version du modèle AlexNet s'écarte
de l'article original sur cet aspect).

#### Fonctions d'activation

En outre, AlexNet a remplacé la fonction d'activation sigmoïde par une fonction d'activation ReLU plus simple. D'une part, le calcul de la fonction d'activation ReLU est plus simple. Par exemple, elle ne comporte pas l'opération d'exponentiation présente dans la fonction d'activation sigmoïde.
 D'autre part, la fonction d'activation ReLU facilite l'apprentissage du modèle lors de l'utilisation de différentes méthodes d'initialisation des paramètres. En effet, lorsque la sortie de la fonction d'activation sigmoïde est très proche de 0 ou 1, le gradient de ces régions est presque nul, de sorte que la rétropropagation ne peut pas continuer à mettre à jour certains des paramètres du modèle. En revanche, le gradient de la fonction d'activation ReLU dans l'intervalle positif est toujours égal à 1 (:numref:$$`subsec_activation-functions` ). Par conséquent, si les paramètres du modèle ne sont pas correctement initialisés, la fonction sigmoïde peut obtenir un gradient de presque 0 dans l'intervalle positif, de sorte que le modèle ne peut pas être entraîné efficacement.

### Contrôle de la capacité et prétraitement

AlexNet contrôle la complexité du modèle de la couche entièrement connectée
par dropout (:numref:`sec_dropout` ),
alors que LeNet n'utilise que la décroissance des poids.
Pour augmenter encore plus les données, la boucle d'apprentissage d'AlexNet
a ajouté un grand nombre d'augmentations d'images,
telles que des retournements, des coupures et des changements de couleur.
Cela rend le modèle plus robuste et la taille plus importante de l'échantillon réduit efficacement le surajustement.
Nous aborderons l'augmentation des données plus en détail dans :numref:`sec_image_augmentation` . Voir également :cite:`Buslaev.Iglovikov.Khvedchenya.ea.2020` pour un examen approfondi de ces étapes de prétraitement.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab all
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

Nous [**construisons un exemple de données à canal unique**] avec une hauteur et une largeur de 224 (**pour observer la forme de la sortie de chaque couche**). Il correspond à l'architecture d'AlexNet dans :numref:`fig_alexnet` .

```{.python .input}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

## Entrainement

Bien qu'AlexNet ait été formé sur ImageNet dans :cite:`Krizhevsky.Sutskever.Hinton.2012`,
nous utilisons ici Fashion-MNIST
car l'entrainement d'un modèle ImageNet jusqu'à convergence peut prendre des heures ou des jours
même sur un GPU moderne.
L'un des problèmes liés à l'application directe d'AlexNet sur [**Fashion-MNIST**]
est que ses (**images ont une résolution**) inférieure ($28 \times 28$ pixels)
(**que les images ImageNet.**)
Pour que les choses fonctionnent, (**nous les sur-échantillonnons à $224 \times 224$**).
Ce n'est généralement pas une pratique intelligente,
mais nous le faisons ici pour être fidèles à l'architecture d'AlexNet.
Nous effectuons ce redimensionnement avec l'argument `resize` dans le constructeur `d2l.FashionMNIST`.

Maintenant, nous pouvons [**commencer l'entraînement d'AlexNet.**]
Par rapport à LeNet dans :numref:`sec_lenet` ,
le principal changement ici est l'utilisation d'un taux d'apprentissage plus faible
et un entraînement beaucoup plus lent en raison du réseau plus profond et plus large,
de la résolution d'image plus élevée, et des convolutions plus coûteuses.

```{.python .input}
%%tab pytorch, mxnet
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## Discussion

La structure d'AlexNet présente une ressemblance frappante avec LeNet, avec un certain nombre d'améliorations critiques, tant pour la précision (dropout) que pour la facilité d'apprentissage (ReLU). Ce qui est tout aussi frappant, c'est la quantité de progrès réalisés en termes d'outils d'apprentissage profond. Ce qui représentait plusieurs mois de travail en 2012 peut maintenant être accompli en une douzaine de lignes de code en utilisant n'importe quel framework moderne.

En examinant l'architecture, on constate qu'AlexNet a un talon d'Achille en matière d'efficacité : les deux dernières couches cachées nécessitent des matrices de taille $6400 \times 4096$ et $4096 \times 4096$, respectivement. Cela correspond à 164 Mo de mémoire et à 81 MFLOPs de calcul, ce qui représente une dépense non négligeable, surtout sur les petits appareils, comme les téléphones portables. C'est l'une des raisons pour lesquelles AlexNet a été dépassé par des architectures beaucoup plus efficaces que nous aborderons dans les sections suivantes. Néanmoins, il s'agit d'une étape clé pour passer des réseaux peu profonds aux réseaux profonds qui sont utilisés de nos jours. Bien qu'il semble qu'il n'y ait que quelques lignes de plus dans l'implémentation d'AlexNet que dans celle de LeNet, il a fallu de nombreuses années à la communauté universitaire pour adopter ce changement conceptuel et tirer parti de ses excellents résultats expérimentaux. Cela était également dû à l'absence d'outils de calcul efficaces. À l'époque, ni DistBelief :cite:`Dean.Corrado.Monga.ea.2012` ni Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014` n'existaient, et Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` manquait encore de nombreuses caractéristiques distinctives. Ce n'est que la disponibilité de TensorFlow :cite:`Abadi.Barham.Chen.ea.2016` qui a radicalement changé cette situation.

## Exercices

1. Suite à la discussion ci-dessus, analysez les performances de calcul d'AlexNet.
   1. Calculez l'empreinte mémoire des convolutions et des couches entièrement connectées, respectivement. Laquelle domine ?
   1. Calculez le coût de calcul pour les convolutions et les couches entièrement connectées.
   1. Comment la bande passante de la mémoire affecte-t-elle le calcul ?
1. En tant que concepteur de puces, vous devez trouver un compromis entre le calcul et la largeur de bande de la mémoire. Par exemple, une puce plus rapide nécessite plus de surface et plus d'énergie, et une plus grande largeur de bande de mémoire nécessite plus de broches et de logique de contrôle, donc également plus de surface. Comment l'optimiser ?
1. Essayez d'augmenter le nombre d'époques lors de l'entrainement d'AlexNet. Par rapport à LeNet, en quoi les résultats diffèrent-ils ? Pourquoi ?
1. AlexNet est peut-être trop complexe pour le jeu de données Fashion-MNIST, notamment en raison de la faible résolution des images initiales.
   1. Essayez de simplifier le modèle pour rendre l'entraînement plus rapide, tout en vous assurant que la précision ne baisse pas de manière significative.
   1. Concevoir un meilleur modèle qui fonctionne directement sur $28 \fois 28$images.
1. Modifiez la taille du lot et observez les changements dans le débit (images/s), la précision et la mémoire du GPU.
1. Appliquez dropout et ReLU à LeNet-5. Est-ce que cela s'améliore ? Pouvez-vous améliorer encore les choses en effectuant un prétraitement pour tirer parti des invariances inhérentes aux images ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
