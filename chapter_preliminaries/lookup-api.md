```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Documentation
:begin_tab:`mxnet` 
 Bien que nous ne puissions pas présenter chaque fonction et classe MXNet 
(et les informations pourraient devenir rapidement obsolètes), 
le [API documentation](https://mxnet.apache.org/versions/1.8.0/api) 
 et des [tutorials](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) et exemples supplémentaires 
fournissent une telle documentation. 
Cette section fournit des conseils sur la manière d'explorer l'API MXNet.
:end_tab:

:begin_tab:`pytorch`
Bien qu'il ne soit pas possible de présenter toutes les fonctions et classes de PyTorch 
(les informations risquent d'être rapidement dépassées), 
, [API documentation](https://pytorch.org/docs/stable/index.html), [tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html) et des exemples supplémentaires 
fournissent une telle documentation.
Cette section fournit quelques conseils sur la façon d'explorer l'API de PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Bien qu'il ne soit pas possible de présenter chaque fonction et classe TensorFlow 
(les informations risquent d'être rapidement dépassées), 
, [API documentation](https://www.tensorflow.org/api_docs), [tutorials](https://www.tensorflow.org/tutorials) et les exemples 
fournissent une telle documentation. 
Cette section fournit quelques conseils sur la façon d'explorer l'API TensorFlow.
:end_tab:


## Fonctions et classes dans un module

Afin de savoir quelles fonctions et classes peuvent être appelées dans un module,
nous invoquons la fonction `dir`. Par exemple, nous pouvons
(**interroger toutes les propriétés du module de génération de nombres aléatoires**) :

```{.python .input  n=1}
%%tab mxnet
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

En général, nous pouvons ignorer les fonctions qui commencent et se terminent par `__` (objets spéciaux en Python) 
ou les fonctions qui commencent par un seul `_`(généralement des fonctions internes). 
D'après les noms de fonctions ou d'attributs restants, 
nous pouvons supposer que ce module offre 
diverses méthodes pour générer des nombres aléatoires, 
y compris l'échantillonnage à partir de la distribution uniforme (`uniform`), 
la distribution normale (`normal`), et la distribution multinomiale (`multinomial`).

## Fonctions et classes spécifiques

Pour obtenir des instructions plus spécifiques sur la manière d'utiliser une fonction ou une classe donnée,
nous pouvons invoquer la fonction `help`. À titre d'exemple, nous allons
[**explorer les instructions d'utilisation de la fonction `ones` des tenseurs**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

D'après la documentation, nous pouvons voir que la fonction `ones` 
 crée un nouveau tenseur avec la forme spécifiée 
et attribue la valeur 1 à tous les éléments. 
Dans la mesure du possible, vous devriez (**exécuter un test rapide**) 
pour confirmer votre interprétation :

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

Dans le carnet Jupyter, nous pouvons utiliser `?` pour afficher le document dans une autre fenêtre
. Par exemple, `list?` créera un contenu
qui est presque identique à `help(list)`,
l'affichant dans une nouvelle fenêtre du navigateur.
En outre, si nous utilisons deux points d'interrogation, comme `list??`,
le code Python implémentant la fonction sera également affiché.

La documentation officielle fournit de nombreuses descriptions et exemples qui dépassent le cadre de cet ouvrage. 
Nous mettons l'accent sur la couverture des cas d'utilisation importants 
qui vous permettront de vous lancer rapidement dans des problèmes pratiques, 
plutôt que sur l'exhaustivité de la couverture. 
Nous vous encourageons également à étudier le code source des bibliothèques 
pour voir des exemples d'implémentations de haute qualité pour le code de production. 
En faisant cela, vous deviendrez un meilleur ingénieur 
en plus de devenir un meilleur scientifique.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
