# Préface

Il y a quelques années à peine, il n'y avait légions de scientifiques spécialisés dans l'apprentissage profond
développant des produits et services intelligents dans les grandes entreprises et les startups.
Lorsque nous sommes entrés dans ce domaine, l'apprentissage automatique 
ne faisait pas les gros titres des journaux quotidiens.
Nos parents n'avaient aucune idée de ce qu'était l'apprentissage automatique,
et encore moins des raisons pour lesquelles nous pourrions le préférer à une carrière en médecine ou en droit.
L'apprentissage automatique était une discipline académique au ciel bleu
dont l'importance industrielle était limitée
à un ensemble restreint d'applications du monde réel,
dont la reconnaissance vocale et la vision par ordinateur.
En outre, nombre de ces applications
nécessitaient une telle connaissance du domaine
qu'elles étaient souvent considérées comme des domaines entièrement distincts 
dont l'apprentissage automatique n'était qu'une petite composante.
À cette époque, les réseaux neuronaux - les prédécesseurs 
des méthodes d'apprentissage profond
sur lesquelles nous nous concentrons dans ce livre - étaient 
généralement considérés comme dépassés.


Au cours des dernières années, l'apprentissage profond a pris le monde par surprise,
entraînant des progrès rapides dans des domaines aussi divers que 
la vision par ordinateur, le traitement du langage naturel, 
la reconnaissance vocale automatique, l'apprentissage par renforcement, 
et l'informatique biomédicale.
De plus, le succès de l'apprentissage profond
sur tant de tâches d'intérêt pratique
a même catalysé des développements 
dans l'apprentissage automatique théorique et les statistiques.
Grâce à ces avancées, 
nous pouvons désormais construire des voitures qui se conduisent toutes seules
avec plus d'autonomie que jamais 
(et moins d'autonomie que certaines entreprises voudraient vous faire croire),
des systèmes de réponse intelligents qui rédigent automatiquement les courriels les plus banals,
qui aident les gens à se sortir de boîtes de réception trop volumineuses,
et des agents logiciels qui dominent les meilleurs humains du monde
à des jeux de société comme le go, un exploit que l'on croyait autrefois impossible à réaliser avant des décennies.
Ces outils exercent déjà un impact de plus en plus important sur l'industrie et la société,
changeant la façon dont les films sont réalisés, dont les maladies sont diagnostiquées,
et jouant un rôle croissant dans les sciences fondamentales - de l'astrophysique à la biologie.



## À propos de ce livre

Ce livre représente notre tentative de rendre l'apprentissage profond accessible,
en vous enseignant les *concepts*, le *contexte* et le *code*.

### Un support combinant Code, Mathématiques et HTML

Pour qu'une technologie informatique atteigne son plein impact,
elle doit être bien comprise, bien documentée et soutenue par
des outils matures et bien maintenus.
Les idées clés doivent être clairement distillées,
minimisant ainsi le temps nécessaire à 
la mise à niveau des nouveaux praticiens.
Les bibliothèques matures doivent automatiser les tâches courantes,
et le code exemplaire doit permettre aux praticiens
de modifier, d'appliquer et d'étendre facilement les applications courantes pour répondre à leurs besoins.
Prenons l'exemple des applications web dynamiques.
Bien qu'un grand nombre d'entreprises, comme Amazon,
aient développé avec succès des applications Web basées sur des bases de données dans les années 1990,
le potentiel de cette technologie pour aider les entrepreneurs créatifs
a été réalisé dans une bien plus large mesure au cours des dix dernières années,
en partie grâce au développement de cadres puissants et bien documentés.


Tester le potentiel de l'apprentissage profond présente des défis uniques
car toute application unique fait appel à diverses disciplines.
L'application de l'apprentissage profond exige de comprendre simultanément
(i) les motivations qui poussent à formuler un problème d'une manière particulière; 
(ii) la forme mathématique d'un modèle donné; 
(iii) les algorithmes d'optimisation pour adapter les modèles aux données; 
(iv) les principes statistiques qui nous disent 
quand nous devons nous attendre à ce que nos modèles 
se généralisent à des données inédites
et les méthodes pratiques pour certifier 
qu'ils se sont effectivement généralisés;
et (v) les techniques d'ingénierie
nécessaires pour former des modèles de manière efficace,
naviguer dans les pièges du calcul numérique
et tirer le meilleur parti du matériel disponible.
Enseigner à la fois les compétences de pensée critique 
nécessaires pour formuler des problèmes,
les mathématiques pour les résoudre,
et les outils logiciels pour mettre en œuvre ces solutions 
tout en un seul endroit présente des défis considérables.
L'objectif de ce livre est de présenter une ressource unifiée
pour mettre à niveau les praticiens en herbe.

Lorsque nous avons commencé ce projet de livre,
il n'existait aucune ressource qui, à la fois,
(i) était à jour, (ii) couvrait toute l'étendue
de l'apprentissage automatique moderne avec une profondeur technique substantielle,
et (iii) intercalait une exposition 
de la qualité que l'on attend 
d'un manuel attrayant 
avec le code exécutable propre
que l'on s'attend à trouver dans les tutoriels pratiques.
Nous avons trouvé de nombreux exemples de code pour
comment utiliser un cadre d'apprentissage profond donné
(par exemple, comment effectuer des calculs numériques de base avec des matrices dans TensorFlow)
ou pour mettre en œuvre des techniques particulières
(par exemple, des extraits de code pour LeNet, AlexNet, ResNets, etc.)
dispersés dans divers articles de blog et dépôts GitHub.
Cependant, ces exemples se concentrent généralement sur
*comment* mettre en œuvre une approche donnée,
mais laissent de côté la discussion sur 
*pourquoi* certaines décisions algorithmiques sont prises.
Bien que certaines ressources interactives 
soient apparues sporadiquement
pour traiter un sujet particulier, 
par exemple, les articles de blog attrayants
publiés sur le site [Distill](http://distill.pub), ou les blogs personnels,
ils ne couvraient que certains sujets de l'apprentissage profond,
et manquaient souvent de code associé.
D'autre part, bien que plusieurs manuels d'apprentissage profond 
aient vu le jour - par exemple, :cite:`Goodfellow.Bengio.Courville.2016`, 
qui offre une étude complète 
sur les bases de l'apprentissage profond - ces ressources 
n'associent pas les descriptions
aux réalisations des concepts dans le code,
laissant parfois les lecteurs désemparés 
quant à la manière de les mettre en œuvre.
De plus, trop de ressources 
sont cachées derrière les murs payants
des fournisseurs de cours commerciaux.

Nous avons entrepris de créer une ressource qui pourrait
(i) être librement accessible à tous ;
(ii) offrir une profondeur technique suffisante 
pour fournir un point de départ sur le chemin
pour devenir réellement un scientifique de l'apprentissage automatique appliqué ;
(iii) inclure du code exécutable, montrant aux lecteurs
*comment* résoudre des problèmes dans la pratique;
(iv) permettre des mises à jour rapides, à la fois par nous
et par la communauté dans son ensemble ;
et (v) être complété par un site [forum](http://discuss.d2l.ai)
pour une discussion interactive des détails techniques et pour répondre aux questions.

Ces objectifs étaient souvent en conflit.
Les équations, les théorèmes et les citations 
sont mieux gérés et mis en page en LaTeX.
Le code est mieux décrit en Python.
Et les pages Web sont natives en HTML et JavaScript.
En outre, nous voulons que le contenu soit
accessible à la fois sous forme de code exécutable, sous forme de livre physique,
sous forme de PDF téléchargeable, et sur Internet sous forme de site Web.
À l'heure actuelle, il n'existe aucun outil ni aucun flux de travail
parfaitement adapté à ces exigences, 
nous avons donc dû assembler les nôtres.
Nous décrivons notre approche en détail 
dans :numref:`sec_how_to_contribute`.
Nous avons choisi GitHub pour partager la source 
et faciliter les contributions de la communauté,
les carnets Jupyter pour mélanger le code, les équations et le texte,
Sphinx comme moteur de rendu 
pour générer des sorties multiples,
et Discourse pour le forum.
Bien que notre système ne soit pas encore parfait,
ces choix offrent un bon compromis 
entre les préoccupations concurrentes.
Nous pensons que ce livre pourrait être 
le premier publié
en utilisant un tel flux de travail intégré.


### Apprendre par la pratique

De nombreux manuels présentent les concepts les uns après les autres, 
couvrant chacun d'entre eux de manière exhaustive.
Par exemple, l'excellent manuel de Chris Bishop :cite:`Bishop.2006`,
enseigne chaque sujet de manière si approfondie
que le chapitre
sur la régression linéaire nécessite 
une quantité de travail non négligeable.
Si les experts adorent ce livre 
précisément pour son exhaustivité,
pour les vrais débutants, cette propriété limite 
son utilité en tant que texte d'introduction.

Dans ce livre, nous enseignerons la plupart des concepts *juste à temps*.
En d'autres termes, vous apprendrez les concepts au moment même
où ils sont nécessaires pour atteindre un objectif pratique.
Bien que nous prenions un peu de temps dès le début pour enseigner
les préliminaires fondamentaux, comme l'algèbre linéaire et les probabilités,
nous voulons que vous goûtiez à la satisfaction de former votre premier modèle
avant de vous préoccuper de distributions de probabilités plus ésotériques.

Hormis quelques cahiers préliminaires qui fournissent un cours accéléré
sur le contexte mathématique de base,
chaque chapitre suivant introduit à la fois un nombre raisonnable de nouveaux concepts
et fournit des exemples de travail autonomes uniques---utilisant des ensembles de données réels.
Cela présente un défi organisationnel.
Certains modèles pourraient logiquement être regroupés dans un seul cahier.
Et certaines idées pourraient être mieux enseignées 
en exécutant plusieurs modèles successivement.
D'un autre côté, il y a un grand avantage à adhérer à
à une politique de *un exemple de travail, un notebook* :
Cela vous permet de faciliter au maximum
le lancement de vos propres projets de recherche en exploitant notre code.
Il suffit de copier un notebook et de commencer à le modifier.

Nous entrelacerons le code exécutable avec du matériel de base si nécessaire.
En général, nous préférons souvent rendre les outils
disponibles avant de les expliquer complètement (et nous poursuivrons en 
expliquant le contexte plus tard à l'adresse).
Par exemple, nous pouvons utiliser la *descente de gradient stochastique*
avant d'expliquer en détail son utilité ou son fonctionnement.
Cela permet de donner aux praticiens les munitions nécessaires
pour résoudre rapidement les problèmes,
au prix de demander au lecteur
de nous faire confiance pour certaines décisions curatoriales.

Ce livre enseigne les concepts d'apprentissage profond à partir de zéro.
Parfois, nous souhaitons entrer dans les détails des modèles
qui seraient généralement cachés à l'utilisateur
par les abstractions avancées des cadres d'apprentissage profond.
Cela se produit surtout dans les tutoriels de base,
où nous voulons que vous compreniez tout
ce qui se passe dans une couche ou un optimiseur donné.
Dans ce cas, nous présentons souvent deux versions de l'exemple :
une où nous implémentons tout à partir de zéro,
en nous appuyant uniquement sur les fonctionnalités de type NumPy
et la différenciation automatique,
et un autre exemple, plus pratique,
où nous écrivons un code succinct en utilisant 
les API de haut niveau des frameworks d'apprentissage profond.
Une fois que nous vous avons appris le fonctionnement d'un composant,
nous pouvons simplement utiliser les API de haut niveau dans les tutoriels suivants.


### Contenu et structure

Le livre peut être divisé grossièrement en trois parties,
portant sur les préliminaires, les techniques d'apprentissage profond,
et les sujets avancés portant sur les systèmes réels 
et les applications (:numref:`fig_book_org` ).

![Book structure](../img/book-org.svg)
:label:`fig_book_org`


* La première partie couvre les bases et les préliminaires.
:numref:`chap_introduction` offre 
une introduction à l'apprentissage profond.
Puis, dans :numref:`chap_preliminaries`,
nous vous présentons rapidement 
les conditions préalables requises
pour l'apprentissage profond pratique, 
telles que le stockage et la manipulation des données,
et l'application de diverses opérations numériques 
basées sur les concepts de base de l'algèbre linéaire, 
du calcul et des probabilités.
:numref:`chap_linear` et :numref:`chap_perceptrons` 
couvrent les concepts et techniques les plus fondamentaux de l'apprentissage profond, 
notamment la régression et la classification;
les modèles linéaires et les perceptrons multicouches; 
et l'overfitting et la régularisation. 

* Les six chapitres suivants sont consacrés aux techniques modernes d'apprentissage profond.   
:numref:`chap_computation` décrit
les principaux composants informatiques des systèmes 
d'apprentissage profond et pose les bases 
pour nos implémentations ultérieures 
de modèles plus complexes. 
Ensuite, :numref:`chap_cnn` et :numref:`chap_modern_cnn`,
présentent les réseaux de neurones convolutifs (CNN),
des outils puissants qui constituent la colonne vertébrale
de la plupart des systèmes modernes de
vision par ordinateur.
De même, :numref:`chap_rnn` et :numref:`chap_modern_rnn`
présentent les réseaux neuronaux récurrents (RNN),
modèles qui exploitent la structure séquentielle (par exemple, temporelle)
des données et sont couramment utilisés 
pour le traitement du langage naturel 
et la prédiction des séries temporelles.
Dans :numref:`chap_attention`,
nous présentons une classe relativement nouvelle de modèles
basés sur des mécanismes dits d'attention
qui ont remplacé les RNN comme architecture dominante
pour la plupart des tâches de traitement du langage naturel (NLP).
Ces sections vous permettront de vous familiariser
avec les outils les plus puissants et les plus généraux
qui sont largement utilisés par les praticiens de l'apprentissage profond. 

* La troisième partie traite de l'évolutivité, de l'efficacité et des applications. 
Tout d'abord, dans :numref:`chap_optimization`,
nous abordons plusieurs algorithmes d'optimisation courants
utilisés pour former des modèles d'apprentissage profond.
Le chapitre suivant, :numref:`chap_performance`,
examine plusieurs facteurs clés
qui influence les performances de calcul 
de votre code d'apprentissage profond. 
Dans :numref:`chap_cv`,
nous illustrons les principales applications 
de l'apprentissage profond en vision par ordinateur. 
Dans :numref:`chap_nlp_pretrain` et :numref:`chap_nlp_app`,
nous montrons comment pré-entraîner des modèles de représentation du langage
et les appliquer à des tâches de traitement du langage naturel.


### Code
:label:`sec_code`
La plupart des sections de ce livre comportent du code exécutable. 
Nous pensons que certaines intuitions se développent mieux
par essais et erreurs,
en modifiant légèrement le code et en observant les résultats.
Idéalement, une théorie mathématique élégante pourrait nous indiquer
précisément comment modifier notre code pour obtenir le résultat souhaité. 
Cependant, aujourd'hui, les praticiens de l'apprentissage profond
doivent souvent s'aventurer là où aucune théorie convaincante
ne peut les guider fermement.
Malgré tous nos efforts, les explications formelles de l'efficacité 
des différentes techniques font toujours défaut,
à la fois parce que les mathématiques permettant de caractériser 
ces modèles peuvent être très difficiles
et aussi parce que les recherches sérieuses sur ces 
sujets n'ont été lancées que récemment.
Nous espérons qu'au fur et à mesure que
la théorie de l'apprentissage profond progresse,
les futures éditions de ce livre
pourront fournir des informations qui éclipseront
celles actuellement disponibles. 

To avoid unnecessary repetition, we encapsulate 
Pour éviter les répétitions inutiles, nous encapsulons
certaines de nos fonctions et classes les plus fréquemment 
importées et référencées dans la librarie `d2l`.
Pour indiquer un bloc de code, tel qu'une fonction,
une classe, ou une collection d'instructions d'importation,
qui sera ensuite accessible via la librarie `d2l`,
nous le marquerons avec `#@save`.
Nous offrons un aperçu détaillé
de ces fonctions et classes dans :numref:`sec_d2l`.
La librarie `d2l` est léger et ne nécessite que
les dépendances suivantes :

 
 

 
 













 
 
 


 



 



 
 


```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
La plupart du code de ce livre est basé sur Apache MXNet,
un framework open-source pour l'apprentissage profond
qui est le choix préféré 
d'AWS (Amazon Web Services),
ainsi que de nombreux collèges et entreprises.
L'ensemble du code de ce livre a passé les tests 
sous la dernière version de MXNet.
Cependant, en raison du développement rapide de l'apprentissage profond, 
certains codes *de l'édition imprimée* 
peuvent ne pas fonctionner correctement dans les futures versions de MXNet.
Nous prévoyons de maintenir la version en ligne à jour.
Si vous rencontrez des problèmes,
veuillez consulter :ref:`chap_installation` 
pour mettre à jour votre code et votre environnement d'exécution.

Voici comment nous importons les modules de MXNet.
:end_tab:

:begin_tab:`pytorch`
La plupart du code de ce livre est basé sur PyTorch,
un framework open-source extrêmement populaire
qui a été adopté avec enthousiasme 
par la communauté de recherche en apprentissage profond.
L'ensemble du code de ce livre a passé les tests 
sous la dernière version stable de PyTorch.
Cependant, en raison du développement rapide de l'apprentissage profond,
certains codes *de l'édition imprimée* 
peuvent ne pas fonctionner correctement dans les futures versions de PyTorch.
Nous prévoyons de maintenir la version en ligne à jour.
Si vous rencontrez des problèmes,
veuillez consulter :ref:`chap_installation` 
pour mettre à jour votre code et votre environnement d'exécution.

Voici comment nous importons les modules de PyTorch.
:end_tab:

:begin_tab:`tensorflow`
La plupart du code de ce livre est basé sur TensorFlow,
un framework open-source pour l'apprentissage profond
qui est largement adopté dans l'industrie
et populaire parmi les chercheurs.
L'ensemble du code de ce livre a passé les tests 
sous la dernière version stable de TensorFlow.
Cependant, en raison du développement rapide de l'apprentissage profond, 
certains codes *de l'édition imprimée* 
peuvent ne pas fonctionner correctement dans les futures versions de TensorFlow.
Nous prévoyons de maintenir la version en ligne à jour.
Si vous rencontrez des problèmes,
veuillez consulter :ref:`chap_installation` 
pour mettre à jour votre code et votre environnement d'exécution.

Voici comment nous importons les modules de TensorFlow.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### Public cible

Ce livre s'adresse aux étudiants (de premier et deuxième cycles),
aux ingénieurs et aux chercheurs qui souhaitent acquérir une solide maîtrise
des techniques pratiques de l'apprentissage profond.
Comme nous expliquons chaque concept à partir de zéro,
aucune connaissance préalable en apprentissage profond ou en apprentissage automatique n'est requise.
L'explication complète des méthodes d'apprentissage profond
nécessite un peu de mathématiques et de programmation,
mais nous supposons que vous avez déjà des bases,
y compris des notions d'algèbre linéaire, 
de calcul, de probabilité et de programmation Python.
Au cas où vous auriez oublié les bases,
l'annexe fournit une remise à niveau 
sur la plupart des mathématiques 
que vous trouverez dans ce livre.
La plupart du temps, nous donnerons la priorité à 
l'intuition et aux idées
sur la rigueur mathématique.
Si vous souhaitez approfondir ces bases 
au-delà des conditions préalables à la compréhension de notre livre,
nous vous recommandons volontiers d'autres ressources formidables :
Linear Analysis de Bela Bollobas :cite:`Bollobas.1999` 
couvre l'algèbre linéaire et l'analyse fonctionnelle de manière très approfondie.
All of Statistics :cite:`Wasserman.2013` 
constitue une merveilleuse introduction aux statistiques.
Les ouvrages de Joe Blitzstein [books](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) 
et [courses](https://projects.iq.harvard.edu/stat110/home) 
sur les probabilités et l'inférence sont des joyaux pédagogiques.
Et si vous n'avez jamais utilisé Python auparavant,
vous voudrez peut-être parcourir ce [Python tutorial](http://learnpython.org/).


### Forum

Associé à ce livre, nous avons lancé un forum de discussion,
situé à l'adresse [discuss.d2l.ai](https://discuss.d2l.ai/).
Si vous avez des questions sur une section du livre,
vous trouverez un lien vers la page de discussion associée
à la fin de chaque cahier.


## Remerciements

Nous sommes redevables aux centaines de contributeurs pour les versions anglaise et chinoise de
.
Ils nous ont aidés à améliorer le contenu et nous ont fait part de leurs précieux commentaires.
Nous remercions tout particulièrement tous les contributeurs de cette version anglaise
qui ont contribué à l'améliorer pour tous.
Leurs identifiants ou noms GitHub sont (sans ordre particulier) :
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg.

Nous remercions Amazon Web Services, en particulier Swami Sivasubramanian, Peter DeSantis, Adam Selipsky,
et Andrew Jassy pour leur généreux soutien dans la rédaction de ce livre. 
Sans le temps disponible, les ressources, les discussions avec les collègues, 
et les encouragements continus, ce livre n'aurait pas vu le jour.


## Résumé

* L'apprentissage profond a révolutionné la reconnaissance des formes, introduisant une technologie qui alimente désormais un large éventail de technologies, notamment la vision par ordinateur, le traitement du langage naturel et la reconnaissance automatique de la parole.
* Pour appliquer avec succès l'apprentissage profond, vous devez comprendre comment formuler un problème, les mathématiques de la modélisation, les algorithmes pour adapter vos modèles aux données, et les techniques d'ingénierie pour mettre tout cela en œuvre.
* Ce livre présente une ressource complète, y compris la prose, les chiffres, les mathématiques et le code, le tout en un seul endroit.
* Pour répondre aux questions relatives à ce livre, visitez notre forum à l'adresse https://discuss.d2l.ai/.
* Tous les cahiers sont disponibles en téléchargement sur GitHub.


## Exercices

1. Enregistrez un compte sur le forum de discussion de ce livre [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Installez Python sur votre ordinateur.
1. Suivez les liens en bas de la section pour accéder au forum, où vous pourrez demander de l'aide, discuter du livre et trouver des réponses à vos questions en faisant appel aux auteurs et à une communauté plus large.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
