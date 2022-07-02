```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Données de régression synthétiques
:label:`sec_synthetic-regression-data` 

 
 L'apprentissage automatique consiste à extraire des informations des données.
Vous pourriez donc vous demander ce que nous pourrions apprendre des données synthétiques
Bien que nous puissions ne pas nous soucier intrinsèquement des modèles 
que nous avons nous-mêmes intégrés dans un modèle de génération de données artificielles,
de tels ensembles de données sont néanmoins utiles à des fins didactiques,
nous aidant à évaluer les propriétés de nos algorithmes d'apprentissage 
et à confirmer que nos implémentations fonctionnent comme prévu.
Par exemple, si nous créons des données pour lesquelles les paramètres corrects sont connus *a priori*,
nous pouvons alors vérifier que notre modèle peut effectivement les récupérer.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Génération de l'ensemble de données

Pour cet exemple, nous travaillerons en basse dimension
pour des raisons de concision.
L'extrait de code suivant génère 1000 exemples
avec des caractéristiques bidimensionnelles tirées 
d'une distribution normale standard.
La matrice de conception résultante $\mathbf{X}$
 appartient à $\mathbb{R}^{1000 \times 2}$. 
Nous générons chaque étiquette en appliquant 
une fonction linéaire *vérité terrain*, 
corrompue par un bruit additif $\epsilon$, 
tiré indépendamment et identiquement pour chaque exemple :

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**)

Par commodité, nous supposons que $\epsilon$ est tiré 
d'une distribution normale avec la moyenne $\mu= 0$ 
 et l'écart type $\sigma = 0.01$.
Notez que pour une conception orientée objet
, nous ajoutons le code à la méthode `__init__` d'une sous-classe de `d2l.DataModule` (présentée dans :numref:`oo-design-data` ). 
Une bonne pratique consiste à permettre la définition de tout hyperparamètre supplémentaire. 
Nous y parvenons avec `save_hyperparameters()`. 
L'adresse `batch_size` sera déterminée ultérieurement.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise            
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Ci-dessous, nous fixons les paramètres réels à $\mathbf{w} = [2, -3.4]^\top$ et $b = 4.2$.
Plus tard, nous pourrons vérifier nos paramètres estimés par rapport à ces valeurs de *vérité terrain*.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Chaque ligne de `features` est constituée d'un vecteur dans $\mathbb{R}^2$ et chaque ligne de `labels` est un scalaire.**] Examinons la première entrée.

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Lecture de l'ensemble de données

L'apprentissage de modèles d'apprentissage automatique nécessite souvent de multiples passages sur un ensemble de données, 
en saisissant un mini-batch d'exemples à la fois. 
Ces données sont ensuite utilisées pour mettre à jour le modèle. 
Pour illustrer ce fonctionnement, nous 
[**implémentons la fonction `get_dataloader`,**] 
en l'enregistrant comme méthode dans la classe `SyntheticRegressionData` via `add_to_class` (introduite dans :numref:`oo-design-utilities` ).
Elle (**prend une taille de lot, une matrice de caractéristiques,
et un vecteur d'étiquettes, et génère des minilots de taille `batch_size`.**)
Ainsi, chaque minilot est constitué d'un tuple de caractéristiques et d'étiquettes. 
Notez que nous devons tenir compte du fait que nous sommes en mode formation ou validation : 
dans le premier cas, nous voudrons lire les données dans un ordre aléatoire, 
alors que dans le second cas, la possibilité de lire les données dans un ordre prédéfini 
peut être importante à des fins de débogage.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet') or tab.selected('pytorch'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

Pour construire une certaine intuition, inspectons le premier minibatch de données
. Chaque mini lot de caractéristiques nous fournit à la fois sa taille et la dimensionnalité des caractéristiques d'entrée.
De même, notre minilot d'étiquettes aura une forme correspondante donnée par `batch_size`.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Bien qu'apparemment inoffensive, l'invocation de 
de `iter(data.train_dataloader())` 
 illustre la puissance de la conception orientée objet de Python. 
Notez que nous avons ajouté une méthode à la classe `SyntheticRegressionData`
 *après* avoir créé l'objet `data`. 
Néanmoins, l'objet bénéficie de 
l'ajout *ex post facto* de la fonctionnalité à la classe.

Tout au long de l'itération, nous obtenons des mini-séquences distinctes
jusqu'à ce que l'ensemble des données soit épuisé (essayez ceci).
Bien que l'itération mise en œuvre ci-dessus soit bonne à des fins didactiques,
elle est inefficace d'une manière qui pourrait nous mettre en difficulté sur des problèmes réels.
Par exemple, elle exige que nous chargions toutes les données en mémoire
et que nous effectuions beaucoup d'accès aléatoires à la mémoire.
Les itérateurs intégrés mis en œuvre dans un cadre d'apprentissage profond
sont considérablement plus efficaces et peuvent traiter
des sources telles que des données stockées dans des fichiers, 
des données reçues via un flux, 
et des données générées ou traitées à la volée. 
Essayons maintenant d'implémenter la même fonction en utilisant des itérateurs intégrés.

## Implémentation concise du chargeur de données

Plutôt que d'écrire notre propre itérateur,
nous pouvons [**appeler l'API existante dans un cadre pour charger des données.**]
Comme précédemment, nous avons besoin d'un ensemble de données avec des caractéristiques `X` et des étiquettes `y`. 
Au-delà de cela, nous définissons `batch_size` dans le chargeur de données intégré 
et le laissons se charger de mélanger les exemples efficacement.

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)

@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

Le nouveau chargeur de données se comporte exactement comme le précédent, à ceci près qu'il est plus efficace et dispose de quelques fonctionnalités supplémentaires.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Par exemple, le chargeur de données fourni par l'API du framework 
prend en charge la méthode intégrée `__len__`, 
afin que nous puissions interroger sa longueur, 
c'est-à-dire le nombre de lots.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## Résumé

Les chargeurs de données sont un moyen pratique d'abstraire 
le processus de chargement et de manipulation des données. 
Ainsi, le même *algorithme d'apprentissage automatique* 
est capable de traiter de nombreux types et sources de données différents 
sans qu'il soit nécessaire de le modifier. 
L'un des avantages des chargeurs de données 
est qu'ils peuvent être composés. 
Par exemple, nous pourrions charger des images 
et avoir ensuite un filtre de post-traitement 
qui les recadre ou les modifie autrement. 
En tant que tels, les chargeurs de données peuvent être utilisés 
pour décrire un pipeline de traitement de données complet. 

Quant au modèle lui-même, le modèle linéaire bidimensionnel 
est à peu près le modèle le plus simple que nous puissions rencontrer. 
Il nous permet de tester la précision des modèles de régression 
sans nous soucier d'avoir des quantités insuffisantes de données 
ou un système d'équations sous-déterminé. 
Nous allons en faire bon usage dans la section suivante. 


## Exercices

1. Que se passe-t-il si le nombre d'exemples ne peut pas être divisé par la taille du lot. Comment changer ce comportement en spécifiant un argument différent en utilisant l'API du framework ?
1. Que se passe-t-il si nous voulons générer un énorme ensemble de données, où la taille du vecteur de paramètres `w` et le nombre d'exemples `num_examples` sont tous deux importants ? 
    1. Que se passe-t-il si nous ne pouvons pas contenir toutes les données en mémoire ?
   1. Comment mélanger les données si elles sont stockées sur le disque ? Votre tâche consiste à concevoir un algorithme *efficace* qui ne nécessite pas trop de lectures ou d'écritures aléatoires. Conseil : [pseudorandom permutation generators](https://en.wikipedia.org/wiki/Pseudorandom_permutation) vous permet de concevoir un remaniement sans avoir besoin de stocker explicitement la table de permutation :cite:`Naor.Reingold.1999` . 
1. Implémentez un générateur de données qui produit de nouvelles données à la volée, chaque fois que l'itérateur est appelé. 
1. Comment concevriez-vous un générateur de données aléatoires qui génère *les mêmes* données à chaque fois qu'il est appelé ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6664)
:end_tab:
