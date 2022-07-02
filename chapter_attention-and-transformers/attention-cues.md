```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Les indices d'attention
:label:`sec_attention-cues` 

 Merci de l'attention que vous portez à ce livre
.
L'attention est une ressource rare :
en ce moment
vous lisez ce livre
et ignorez le reste.
Ainsi, comme pour l'argent,
votre attention est payée avec un coût d'opportunité.
Pour s'assurer que l'investissement de votre attention
en ce moment en vaut la peine,
nous avons été très motivés pour accorder notre attention avec soin
afin de produire un beau livre.
L'attention
est la clé de voûte de l'arc de la vie et
détient la clé de l'exceptionnalité de toute œuvre.


Puisque l'économie étudie l'allocation de ressources rares,
nous sommes
dans l'ère de l'économie de l'attention,
où l'attention humaine est traitée comme un bien limité, précieux et rare
qui peut être échangé.
De nombreux modèles commerciaux ont été
développés pour en tirer profit.
Sur les services de streaming musical ou vidéo,
nous prêtons attention à leurs publicités
ou nous payons pour les cacher.
Pour se développer dans le monde des jeux en ligne,
, soit nous prêtons attention à
participer à des batailles, qui attirent de nouveaux joueurs,
soit nous payons de l'argent pour devenir instantanément puissant.
Rien n'est gratuit.

En somme,
l'information dans notre environnement n'est pas rare,
l'attention l'est.
Lorsque nous inspectons une scène visuelle,
notre nerf optique reçoit des informations
de l'ordre de $10^8$ bits par seconde,
dépassant de loin ce que notre cerveau peut traiter entièrement.
Heureusement,
nos ancêtres avaient appris par l'expérience (également appelée données)
que *toutes les entrées sensorielles ne sont pas créées égales*.
Tout au long de l'histoire de l'humanité,
la capacité de diriger l'attention
vers une fraction seulement des informations intéressantes
a permis à notre cerveau
d'allouer les ressources de manière plus intelligente
pour survivre, grandir et socialiser,
comme la détection des prédateurs, des proies et des partenaires.



### Les signaux d'attention en biologie

Pour expliquer comment notre attention se déploie dans le monde visuel,
un cadre à deux composantes a émergé
et s'est répandu.
Cette idée remonte à William James dans les années 1890,
, qui est considéré comme le "père de la psychologie américaine" :cite:`James.2007` .
Dans ce cadre,
les sujets dirigent sélectivement le point de mire de l'attention
en utilisant à la fois le *repère non-volontaire* et le *repère volitif*.

L'indice non-volontaire est basé sur
la saillance et la conspicuité des objets dans l'environnement.
Imaginez que cinq objets se trouvent devant vous :
un journal, un document de recherche, une tasse de café, un cahier et un livre, comme dans :numref:`fig_eye-coffee` .
Alors que tous les produits en papier sont imprimés en noir et blanc,
la tasse de café est rouge.
En d'autres termes,
ce café est intrinsèquement saillant et ostensible dans
cet environnement visuel,
attirant automatiquement et involontairement l'attention.
Vous amenez donc la fovéa (le centre de la macula, où l'acuité visuelle est la plus élevée) sur le café, comme le montre l'illustration :numref:`fig_eye-coffee` .

![Using the nonvolitional cue based on saliency (red cup, non-paper) , l'attention est involontairement dirigée vers le café.](../img/eye-coffee.svg)
:width:`400px` 
 :label:`fig_eye-coffee` 

 Après avoir bu du café,
vous êtes en manque de caféine et
vous avez envie de lire un livre.
Vous tournez donc la tête, vous recentrez vos yeux,
et vous regardez le livre comme indiqué dans :numref:`fig_eye-book` .
À la différence de
, le cas de :numref:`fig_eye-coffee` 
 où le café vous incite à
faire une sélection basée sur la saillance,
dans ce cas dépendant de la tâche, vous sélectionnez le livre sous
contrôle cognitif et volontaire.
En utilisant l'indice volitif basé sur des critères de sélection variables,
cette forme d'attention est plus délibérée.
Elle est également plus puissante grâce à l'effort volontaire du sujet.

![Using the volitional cue (want to read a book) qui est dépendant de la tâche, l'attention est dirigée vers le livre sous contrôle volitif.](../img/eye-book.svg)
:width:`400px` 
 :label:`fig_eye-book` 

 
 ## Requêtes, clés et valeurs

Inspirés par les indices d'attention non volitifs et volitifs qui expliquent le déploiement attentionnel,
dans ce qui suit, nous allons
décrire un cadre pour
concevoir des mécanismes d'attention
en incorporant ces deux indices d'attention.

Pour commencer, considérons le cas le plus simple où seuls
les signaux non volitifs sont disponibles.
Pour biaiser la sélection sur les entrées sensorielles,
nous pouvons simplement utiliser
une couche entièrement connectée paramétrée
ou même une mise en commun maximale ou moyenne non paramétrée
.

Par conséquent,
ce qui différencie les mécanismes d'attention
des couches entièrement connectées
ou des couches de mise en commun
, c'est l'inclusion des indices volitifs.
Dans le contexte des mécanismes d'attention,
nous nous référons aux indices volitifs comme des *requêtes*.
Pour toute requête,
les mécanismes d'attention
biaisent la sélection sur les entrées sensorielles (par exemple, les représentations de caractéristiques intermédiaires)
via la *mise en commun de l'attention*.
Ces entrées sensorielles sont appelées *valeurs* dans le contexte des mécanismes d'attention.
Plus généralement,
chaque valeur est associée à une *clé*,
qui peut être considérée comme l'indice non-volatile de cette entrée sensorielle.
Comme le montrent les exemples :numref:`fig_qkv` et
, nous pouvons concevoir le regroupement de l'attention
de manière à ce que la requête donnée (indice volitif) puisse interagir avec des clés (indices non volitifs),
, qui guident la sélection de biais sur les valeurs (entrées sensorielles).

![Attention mechanisms bias selection over values (sensory inputs) via la mise en commun de l'attention, qui intègre les requêtes (indices volitifs) et les clés (indices non volitifs)](../img/qkv.svg)
:label:`fig_qkv` 

 Notez qu'il existe de nombreuses alternatives pour la conception des mécanismes d'attention.
Par exemple,
nous pouvons concevoir un modèle d'attention non-différenciable
qui peut être entraîné à l'aide de méthodes d'apprentissage par renforcement :cite:`Mnih.Heess.Graves.ea.2014` .
Étant donné la prédominance du cadre de travail de :numref:`fig_qkv` ,
les modèles de ce cadre
seront au centre de notre attention dans ce chapitre.


## Visualisation de l'attention

La mise en commun de l'attention
peut être traitée comme une moyenne pondérée des entrées,
où les poids sont uniformes.
En pratique, la mise en commun de l'attention
agrège les valeurs en utilisant une moyenne pondérée, où les poids sont calculés entre la requête donnée et différentes clés.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

Pour visualiser les poids d'attention,
nous définissons la fonction `show_heatmaps`.
Son entrée `matrices` a la forme (nombre de lignes à afficher, nombre de colonnes à afficher, nombre de requêtes, nombre de clés).

```{.python .input}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    num_rows, num_cols = len(matrices), len(matrices[0])
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

Pour la démonstration,
nous considérons un cas simple où
le poids d'attention est de un seulement lorsque la requête et la clé sont les mêmes ; sinon il est de zéro.

```{.python .input}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

Dans les sections suivantes,
nous invoquerons souvent cette fonction pour visualiser les poids d'attention.

## Résumé

* L'attention humaine est une ressource limitée, précieuse et rare.
* Les sujets dirigent sélectivement leur attention en utilisant à la fois les indices non volitifs et volitifs. Le premier est basé sur la saillance et le second est dépendant de la tâche.
* Les mécanismes d'attention sont différents des couches entièrement connectées ou des couches de mise en commun en raison de l'inclusion des indices volontaires.
* Les mécanismes d'attention biaisent la sélection sur les valeurs (entrées sensorielles) via la mise en commun de l'attention, qui incorpore des requêtes (indices volitifs) et des clés (indices non volitifs). Les clés et les valeurs sont appariées.
* Nous pouvons visualiser les poids de l'attention entre les requêtes et les clés.

## Exercices

1. Quel peut être l'indice volitif lors du décodage d'une séquence mot à mot en traduction automatique ? Quels sont les indices non volitifs et les entrées sensorielles ?
1. Générez aléatoirement une matrice $10 \times 10$ et utilisez l'opération softmax pour vous assurer que chaque ligne est une distribution de probabilité valide. Visualisez les poids d'attention de sortie.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
