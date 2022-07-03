# Compilateurs et interprètes
:label:`sec_hybridize` 

 Jusqu'à présent, ce livre s'est concentré sur la programmation impérative, qui utilise des instructions telles que `print`, `+` et `if` pour modifier l'état d'un programme. Considérons l'exemple suivant d'un programme impératif simple.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python est un langage *interprété*. Lorsqu'il évalue la fonction `fancy_func` ci-dessus, il exécute les opérations constituant le corps de la fonction *dans l'ordre*. En d'autres termes, elle évalue `e = add(a, b)` et stocke les résultats dans la variable `e`, modifiant ainsi l'état du programme. Les deux instructions suivantes `f = add(c, d)` et `g = add(e, f)` seront exécutées de manière similaire, en effectuant des additions et en stockant les résultats sous forme de variables. :numref:`fig_compute_graph` illustre le flux de données.

![Data flow in an imperative program.](../img/computegraph.svg)
:label:`fig_compute_graph`

Bien que la programmation impérative soit pratique, elle peut être inefficace. D'une part, même si la fonction `add` est appelée à plusieurs reprises tout au long de `fancy_func`, Python exécutera les trois appels de fonction individuellement. Si ceux-ci sont exécutés, par exemple, sur un GPU (ou même sur plusieurs GPU), la surcharge due à l'interpréteur Python peut devenir écrasante. De plus, il devra sauvegarder les valeurs des variables de `e` et `f` jusqu'à ce que toutes les instructions de `fancy_func` aient été exécutées. En effet, nous ne savons pas si les variables `e` et `f` seront utilisées par d'autres parties du programme après l'exécution des instructions `e = add(a, b)` et `f = add(c, d)`.

## Programmation symbolique

Considérez l'alternative, la programmation *symbolique*, où le calcul n'est généralement effectué qu'une fois le processus entièrement défini. Cette stratégie est utilisée par plusieurs cadres d'apprentissage profond, notamment Theano et TensorFlow (ce dernier a acquis des extensions impératives). Elle implique généralement les étapes suivantes :

1. Définir les opérations à exécuter.
1. Compiler les opérations dans un programme exécutable.
1. Fournir les entrées requises et appeler le programme compilé pour l'exécution.

Cela permet une optimisation importante. Tout d'abord, nous pouvons sauter l'interpréteur Python dans de nombreux cas, éliminant ainsi un goulot d'étranglement des performances qui peut devenir important sur plusieurs GPU rapides associés à un seul thread Python sur un CPU. 
Deuxièmement, un compilateur peut optimiser et réécrire le code ci-dessus en `print((1 + 2) + (3 + 4))` ou même `print(10)`. Cela est possible car un compilateur peut voir le code complet avant de le transformer en instructions machine. Par exemple, il peut libérer de la mémoire (ou ne jamais l'allouer) lorsqu'une variable n'est plus nécessaire. Ou encore, il peut transformer entièrement le code en un morceau équivalent.
Pour vous faire une meilleure idée, considérez la simulation suivante de la programmation impérative (c'est Python après tout) ci-dessous.

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

Les différences entre la programmation impérative (interprétée) et la programmation symbolique sont les suivantes :

* La programmation impérative est plus facile. Lorsque la programmation impérative est utilisée en Python, la majorité du code est simple et facile à écrire. Il est également plus facile de déboguer le code de programmation impérative. En effet, il est plus facile d'obtenir et d'imprimer toutes les valeurs de variables intermédiaires pertinentes, ou d'utiliser les outils de débogage intégrés à Python.
* La programmation symbolique est plus efficace et plus facile à porter. La programmation symbolique permet d'optimiser plus facilement le code lors de la compilation, tout en ayant la possibilité de porter le programme dans un format indépendant de Python. Cela permet au programme d'être exécuté dans un environnement non Python, évitant ainsi tout problème potentiel de performance lié à l'interpréteur Python.


## Programmation hybride

Historiquement, la plupart des cadres d'apprentissage profond choisissent entre une approche impérative ou symbolique. Par exemple, Theano, TensorFlow (inspiré par le premier), Keras et CNTK formulent les modèles de manière symbolique. À l'inverse, Chainer et PyTorch adoptent une approche impérative. Un mode impératif a été ajouté à TensorFlow 2.0 et Keras dans des révisions ultérieures.

:begin_tab:`mxnet`
Lors de la conception de Gluon, les développeurs se sont demandé s'il était possible de combiner les avantages des deux paradigmes de programmation. Cela a conduit à un modèle hybride qui permet aux utilisateurs de développer et de déboguer avec une programmation impérative pure, tout en ayant la possibilité de convertir la plupart des programmes en programmes symboliques à exécuter lorsque les performances de calcul et le déploiement au niveau du produit sont requis.

En pratique, cela signifie que nous construisons des modèles en utilisant la classe `HybridBlock` ou `HybridSequential`. Par défaut, l'une ou l'autre de ces classes est exécutée de la même manière que la classe `Block` ou `Sequential` dans la programmation impérative. 
La classe `HybridSequential` est une sous-classe de `HybridBlock` (tout comme `Sequential` sous-classe `Block`). Lorsque la fonction `hybridize` est appelée, Gluon compile le modèle sous la forme utilisée en programmation symbolique. Cela permet d'optimiser les composants à forte intensité de calcul sans sacrifier la manière dont le modèle est implémenté. Nous allons illustrer ces avantages ci-dessous, en nous concentrant sur les modèles et les blocs séquentiels.
:end_tab:

:begin_tab:`pytorch`
Comme mentionné ci-dessus, PyTorch est basé sur la programmation impérative et utilise des graphes de calcul dynamiques. Dans le but de tirer parti de la portabilité et de l'efficacité de la programmation symbolique, les développeurs se sont demandé s'il était possible de combiner les avantages des deux modèles de programmation. Cela a conduit à un torchscript qui permet aux utilisateurs de développer et de déboguer en utilisant la programmation impérative pure, tout en ayant la possibilité de convertir la plupart des programmes en programmes symboliques à exécuter lorsque les performances de calcul et le déploiement au niveau du produit sont nécessaires.
:end_tab:

:begin_tab:`tensorflow`
Le paradigme de la programmation impérative est désormais la valeur par défaut dans Tensorflow 2, un changement bienvenu pour ceux qui découvrent le langage. Cependant, les mêmes techniques de programmation symbolique et les graphes de calcul qui en découlent existent toujours dans TensorFlow, et on peut y accéder grâce au décorateur facile à utiliser `tf.function`. Cela a apporté le paradigme de la programmation impérative à TensorFlow, a permis aux utilisateurs de définir des fonctions plus intuitives, puis de les envelopper et de les compiler dans des graphes de calcul automatiquement en utilisant une fonctionnalité que l'équipe de TensorFlow appelle [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
:end_tab:

## Hybridation de la classe `Sequential`

 La façon la plus simple de se faire une idée du fonctionnement de l'hybridation est de considérer des réseaux profonds à plusieurs couches. Conventionnellement, l'interpréteur Python devra exécuter le code de toutes les couches pour générer une instruction qui pourra ensuite être transmise à un CPU ou à un GPU. Pour un seul dispositif de calcul (rapide), cela ne pose pas de problème majeur. En revanche, si nous utilisons un serveur avancé à 8 GPU, comme une instance AWS P3dn.24xlarge, Python aura du mal à occuper tous les GPU. L'interpréteur Python monofilaire devient alors le goulot d'étranglement. Voyons comment nous pouvons résoudre ce problème pour des parties importantes du code en remplaçant `Sequential` par `HybridSequential`. Nous commençons par définir un MLP simple.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
En appelant la fonction `hybridize`, nous sommes en mesure de compiler et d'optimiser le calcul dans le MLP. Le résultat du calcul du modèle reste inchangé.
:end_tab:

:begin_tab:`pytorch`
En convertissant le modèle à l'aide de la fonction `torch.jit.script`, nous sommes en mesure de compiler et d'optimiser le calcul dans le MLP. Le résultat du calcul du modèle reste inchangé.
:end_tab:

:begin_tab:`tensorflow`
Auparavant, toutes les fonctions construites dans TensorFlow étaient construites comme un graphe de calcul, et donc compilées en JIT par défaut. Cependant, avec la sortie de TensorFlow 2.X et EagerTensor, ce n'est plus le comportement par défaut. 
Nous pouvons réactiver cette fonctionnalité avec tf.function. tf.function est plus communément utilisé comme décorateur de fonction, cependant il est possible de l'appeler directement comme une fonction python normale, comme montré ci-dessous. Le résultat du calcul du modèle reste inchangé.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Cela semble presque trop beau pour être vrai : il suffit de désigner un bloc pour être `HybridSequential`, d'écrire le même code que précédemment et d'invoquer `hybridize`. Une fois que cela se produit, le réseau est optimisé (nous évaluerons les performances ci-dessous). Malheureusement, cela ne fonctionne pas comme par magie pour chaque couche. Cela dit, une couche ne sera pas optimisée si elle hérite de la classe `Block` au lieu de la classe `HybridBlock`.
:end_tab:

:begin_tab:`pytorch`
Cela semble presque trop beau pour être vrai : écrivez le même code que précédemment et convertissez simplement le modèle en utilisant `torch.jit.script`. Une fois cela fait, le réseau est optimisé (nous évaluerons les performances ci-dessous).
:end_tab:

:begin_tab:`tensorflow`
Cela semble presque trop beau pour être vrai : écrivez le même code que précédemment et convertissez simplement le modèle en utilisant `tf.function`. Le réseau est alors construit comme un graphe de calcul dans la représentation intermédiaire MLIR de TensorFlow et est fortement optimisé au niveau du compilateur pour une exécution rapide (nous évaluerons les performances ci-dessous).
L'ajout explicite de l'indicateur `jit_compile = True` à l'appel `tf.function()` active la fonctionnalité XLA (Accelerated Linear Algebra) dans TensorFlow. XLA peut optimiser davantage le code compilé JIT dans certains cas. L'exécution en mode graphique est activée sans cette définition explicite, mais XLA peut rendre certaines grandes opérations d'algèbre linéaire (dans la veine de celles que nous voyons dans les applications d'apprentissage profond) beaucoup plus rapides, en particulier dans un environnement GPU.
:end_tab:

### Accélération par hybridation

Pour démontrer l'amélioration des performances obtenue par la compilation, nous comparons le temps nécessaire pour évaluer `net(x)` avant et après hybridation. Définissons d'abord une classe pour mesurer ce temps. Elle nous sera utile tout au long du chapitre lorsque nous chercherons à mesurer (et à améliorer) les performances.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
Nous pouvons maintenant invoquer le réseau deux fois, une fois avec et une fois sans hybridation.
:end_tab:

:begin_tab:`pytorch`
Maintenant, nous pouvons invoquer le réseau deux fois, une fois avec et une fois sans torchscript.
:end_tab:

:begin_tab:`tensorflow`
Maintenant, nous pouvons invoquer le réseau trois fois, une fois exécuté avec empressement, une fois avec une exécution en mode graphique, et une fois encore en utilisant XLA compilé en JIT.
:end_tab:

```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
Comme on peut l'observer dans les résultats ci-dessus, après qu'une instance de `HybridSequential` ait appelé la fonction `hybridize`, les performances de calcul sont améliorées grâce à l'utilisation de la programmation symbolique.
:end_tab:

:begin_tab:`pytorch`
Comme on peut l'observer dans les résultats ci-dessus, après qu'une instance `nn.Sequential` a été scriptée en utilisant la fonction `torch.jit.script`, les performances de calcul sont améliorées par l'utilisation de la programmation symbolique.
:end_tab:

:begin_tab:`tensorflow`
Comme on peut l'observer dans les résultats ci-dessus, après qu'une instance de `tf.keras.Sequential` ait été scriptée à l'aide de la fonction `tf.function`, les performances de calcul sont améliorées par l'utilisation de la programmation symbolique via l'exécution en mode graphique dans tensorflow. 
:end_tab:

### Sérialisation

:begin_tab:`mxnet` 
 L'un des avantages de la compilation des modèles est que nous pouvons sérialiser (enregistrer) le modèle et ses paramètres sur le disque. Cela nous permet de stocker un modèle d'une manière qui est indépendante du langage frontal de choix. Cela nous permet de déployer des modèles formés sur d'autres appareils et d'utiliser facilement d'autres langages de programmation frontaux. En même temps, le code est souvent plus rapide que ce qui peut être réalisé en programmation impérative. Voyons la fonction `export` en action.
:end_tab:

:begin_tab:`pytorch`
L'un des avantages de la compilation des modèles est que nous pouvons sérialiser (enregistrer) le modèle et ses paramètres sur le disque. Cela nous permet de stocker un modèle d'une manière qui est indépendante du langage frontal de choix. Cela nous permet de déployer des modèles formés sur d'autres appareils et d'utiliser facilement d'autres langages de programmation frontaux. En même temps, le code est souvent plus rapide que ce qui peut être réalisé en programmation impérative. Voyons la fonction `save` en action.
:end_tab:

:begin_tab:`tensorflow`
L'un des avantages de la compilation des modèles est que nous pouvons sérialiser (enregistrer) le modèle et ses paramètres sur le disque. Cela nous permet de stocker un modèle d'une manière qui est indépendante du langage frontal de choix. Cela nous permet de déployer des modèles formés sur d'autres appareils et d'utiliser facilement d'autres langages de programmation frontaux ou d'exécuter un modèle formé sur un serveur. En même temps, le code est souvent plus rapide que ce qui peut être réalisé en programmation impérative. 
L'API de bas niveau qui nous permet d'enregistrer dans tensorflow est `tf.saved_model`. 
Voyons l'instance `saved_model` en action.
:end_tab:

```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
Le modèle est décomposé en un (gros fichier binaire) de paramètres et une description JSON du programme nécessaire pour exécuter le calcul du modèle. Les fichiers peuvent être lus par d'autres langages frontaux supportés par Python ou MXNet, tels que C++, R, Scala et Perl. Examinons les premières lignes de la description du modèle.
:end_tab:

```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
Plus tôt, nous avons démontré que, après avoir appelé la fonction `hybridize`, le modèle est capable d'atteindre des performances de calcul et une portabilité supérieures. Notez cependant que l'hybridation peut affecter la flexibilité du modèle, en particulier en termes de flux de contrôle. 

En outre, contrairement à l'instance `Block`, qui doit utiliser la fonction `forward`, pour une instance `HybridBlock`, nous devons utiliser la fonction `hybrid_forward`.
:end_tab:

```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
Le code ci-dessus implémente un réseau simple avec 4 unités cachées et 2 sorties. La fonction `hybrid_forward` prend un argument supplémentaire `F`. Ceci est nécessaire car, selon que le code a été hybridé ou non, il utilisera une bibliothèque légèrement différente (`ndarray` ou `symbol`) pour le traitement. Les deux classes exécutent des fonctions très similaires et MXNet détermine automatiquement l'argument. Pour comprendre ce qui se passe, nous imprimons les arguments dans le cadre de l'invocation de la fonction.
:end_tab:

```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
Répéter le calcul avant conduira au même résultat (nous omettons les détails). Voyons maintenant ce qui se passe si nous invoquons la fonction `hybridize`.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Au lieu d'utiliser `ndarray`, nous utilisons maintenant le module `symbol` pour `F`. De plus, même si l'entrée est de type `ndarray`, les données qui circulent dans le réseau sont maintenant converties en type `symbol` dans le cadre du processus de compilation. La répétition de l'appel de fonction conduit à un résultat surprenant:
:end_tab :

```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
Ce résultat est très différent de ce que nous avons vu précédemment. Toutes les instructions print, telles que définies dans , sont omises. En effet, après l'hybridation, l'exécution de n'implique plus l'interpréteur Python. Cela signifie que tout code Python parasite est omis (comme les instructions d'impression) en faveur d'une exécution beaucoup plus rationnelle et de meilleures performances. Au lieu de cela, MXNet appelle directement le backend C++. Notez également que certaines fonctions ne sont pas prises en charge par le module (par exemple, ) et que les opérations in-place telles que et doivent être réécrites sous . Néanmoins, la compilation de modèles en vaut la peine lorsque la vitesse est importante. L'avantage peut aller de petits points de pourcentage à plus du double de la vitesse, en fonction de la complexité du modèle, de la vitesse du CPU, et de la vitesse et du nombre de GPU.## Résumé  * La programmation impérative facilite la conception de nouveaux modèles puisqu'il est possible d'écrire du code avec un flux de contrôle et la possibilité d'utiliser une grande partie de l'écosystème logiciel Python. * La programmation symbolique nécessite de spécifier le programme et de le compiler avant de l'exécuter. L'avantage est l'amélioration des performances. * MXNet est capable de combiner les avantages des deux approches selon les besoins. * Les modèles construits par les classes et sont capables de convertir des programmes impératifs en programmes symboliques en appelant la fonction .## Exercices     1. Ajoutez à la première ligne de la fonction de la classe de cette section. Exécutez le code et observez les erreurs que vous rencontrez. Pourquoi se produisent-elles ? 1. Que se passe-t-il si nous ajoutons le flux de contrôle, c'est-à-dire les instructions Python et dans la fonction ? 1. Passez en revue les modèles qui vous ont intéressé dans les chapitres précédents. Pouvez-vous améliorer leurs performances de calcul en les réimplantant ? 1. Passez en revue les modèles qui vous ont intéressé dans les chapitres précédents. Pouvez-vous améliorer leurs performances de calcul en les réimplantant ? `hybrid_forward` `net(x)` `symbol` `asnumpy` `a += b` `a[:] = a + b` `a = a + b`
:end_tab:







:begin_tab:`mxnet` 

 `HybridSequential` `HybridBlock` `hybridize`
:end_tab:





:begin_tab:`mxnet` 
 `x.asnumpy()` `hybrid_forward` `HybridNet`
 `if` `for` `hybrid_forward`

:end_tab:

:begin_tab:`pytorch,tensorflow` 

:end_tab:




:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2492)
:end_tab:
