# Parallélisation automatique
:label:`sec_auto_para`


Les cadres d'apprentissage profond (par exemple, MXNet et PyTorch) construisent automatiquement des graphes de calcul en arrière-plan. En utilisant un graphe de calcul
, le système est conscient de toutes les dépendances,
et peut exécuter sélectivement plusieurs tâches non interdépendantes en parallèle pour
améliorer la vitesse. Par exemple, :numref:`fig_asyncgraph` dans :numref:`sec_async` initialise deux variables indépendamment. Par conséquent, le système peut choisir de les exécuter en parallèle.


Généralement, un seul opérateur utilisera toutes les ressources de calcul de tous les CPU ou d'un seul GPU. Par exemple, l'opérateur `dot` utilisera tous les cœurs (et threads) de tous les CPU, même s'il y a plusieurs processeurs CPU sur une seule machine. Il en va de même pour un seul GPU. Par conséquent, la parallélisation n'est pas aussi utile pour les ordinateurs à périphérique unique. Avec plusieurs appareils, les choses sont plus importantes. Alors que la parallélisation est généralement plus pertinente entre plusieurs GPU, l'ajout du CPU local augmentera légèrement les performances. Par exemple, consultez :cite:`Hadjis.Zhang.Mitliagkas.ea.2016` qui se concentre sur l'entraînement de modèles de vision par ordinateur en combinant un GPU et un CPU. Grâce à la commodité d'un cadre de parallélisation automatique, nous pouvons atteindre le même objectif en quelques lignes de code Python. Plus largement, notre discussion sur le calcul parallèle automatique se concentre sur le calcul parallèle utilisant à la fois les CPU et les GPU, ainsi que sur la parallélisation du calcul et de la communication.

Notez que nous avons besoin d'au moins deux GPU pour exécuter les expériences de cette section.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## Calcul parallèle sur GPUs

Commençons par définir une charge de travail de référence à tester : la fonction `run` ci-dessous effectue 10 multiplications matricielles sur le périphérique de notre choix en utilisant des données allouées dans deux variables :`x_gpu1` et `x_gpu2`.

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
Nous appliquons maintenant la fonction aux données. Pour s'assurer que la mise en cache ne joue pas un rôle dans les résultats, nous chauffons les dispositifs en effectuant une seule passe sur l'un d'entre eux avant de procéder à la mesure.
:end_tab:

:begin_tab:`pytorch`
Nous appliquons maintenant la fonction aux données. Pour s'assurer que la mise en cache ne joue pas de rôle dans les résultats, nous réchauffons les périphériques en effectuant un seul passage sur l'un d'entre eux avant de procéder à la mesure. `torch.cuda.synchronize()` attend que tous les noyaux de tous les flux sur un périphérique CUDA soient terminés. Elle prend un argument `device`, le périphérique pour lequel nous devons nous synchroniser. Elle utilise le périphérique actuel, donné par `current_device()`, si l'argument du périphérique est `None` (par défaut).
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Si nous supprimons l'instruction `waitall` entre les deux tâches, le système est libre de paralléliser automatiquement le calcul sur les deux périphériques.
:end_tab:

:begin_tab:`pytorch`
Si nous supprimons l'instruction `synchronize` entre les deux tâches, le système est libre de paralléliser automatiquement le calcul sur les deux dispositifs.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

Dans le cas ci-dessus, le temps d'exécution total est inférieur à la somme de ses parties, car le cadre d'apprentissage profond planifie automatiquement le calcul sur les deux GPU sans que l'utilisateur ait besoin d'un code sophistiqué.



## Calcul et Communication en parallèle

Dans de nombreux cas, nous devons déplacer des données entre différents dispositifs, par exemple entre le CPU et le GPU, ou entre différents GPU. 
Par exemple,
cela se produit lorsque nous voulons effectuer une optimisation distribuée où nous devons agréger les gradients sur plusieurs cartes accélératrices. Simulons cela en calculant sur le GPU, puis en recopiant les résultats sur le CPU.

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Cette méthode est quelque peu inefficace. Notez que nous pourrions déjà commencer à copier des parties de `y` vers le CPU alors que le reste de la liste est toujours en cours de calcul. Cette situation se produit, par exemple, lorsque nous calculons le gradient sur un minibatch. Les gradients de certains des paramètres seront disponibles plus tôt que ceux des autres. Il est donc avantageux pour nous de commencer à utiliser la largeur de bande du bus PCI-Express pendant que le GPU est encore en fonctionnement. La suppression de `waitall` entre les deux parties nous permet de simuler ce scénario.
:end_tab:

:begin_tab:`pytorch`
C'est quelque peu inefficace. Notez que nous pourrions déjà commencer à copier des parties de `y` vers le CPU alors que le reste de la liste est toujours en cours de calcul. Cette situation se produit, par exemple, lorsque nous calculons le gradient (backprop) sur un minibatch. Les gradients de certains des paramètres seront disponibles plus tôt que ceux des autres. Il est donc avantageux de commencer à utiliser la bande passante du bus PCI-Express pendant que le GPU est encore en fonctionnement. Dans PyTorch, plusieurs fonctions telles que `to()` et `copy_()` admettent un argument explicite `non_blocking`, qui permet à l'appelant de contourner la synchronisation lorsqu'elle est inutile. Le réglage de `non_blocking=True` nous permet de simuler ce scénario.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Le temps total requis pour les deux opérations est (comme prévu) inférieur à la somme de leurs parties.
Notez que cette tâche est différente du calcul parallèle car elle utilise une ressource différente : le bus entre le CPU et les GPU. En fait, nous pourrions calculer sur les deux dispositifs et communiquer, le tout en même temps. Comme indiqué ci-dessus, il existe une dépendance entre le calcul et la communication :`y[i]` doit être calculé avant de pouvoir être copié sur le CPU. Heureusement, le système peut copier `y[i-1]` tout en calculant `y[i]` pour réduire le temps d'exécution total.

Nous conclurons par une illustration du graphe de calcul et de ses dépendances pour un simple MLP à deux couches lors de l'entraînement sur un CPU et deux GPU, comme décrit dans :numref:`fig_twogpu` . Il serait assez pénible de programmer manuellement le programme parallèle qui en résulte. C'est là qu'il est avantageux d'avoir un backend de calcul basé sur le graphe pour l'optimisation.

![Le graphe de calcul et ses dépendances d'un MLP à deux couches sur un CPU et deux GPU.](../img/twogpu.svg)
:label:`fig_twogpu`


## Résumé

* Les systèmes modernes possèdent une variété de dispositifs, tels que plusieurs GPU et CPU. Ils peuvent être utilisés en parallèle, de manière asynchrone. 
* Les systèmes modernes disposent également d'une variété de ressources pour la communication, comme PCI Express, le stockage (généralement des disques durs solides ou via des réseauxet)la bande passante du réseau. Ils peuvent être utilisés en parallèle pour une efficacité maximale. 
* Le backend peut améliorer les performances grâce à des calculs et des communications parallèles automatiques. 

## Exercices

1. Huit opérations ont été effectuées dans la fonction `run` définie dans cette section. Il n'y a pas de dépendances entre elles. Concevez une expérience pour voir si le cadre d'apprentissage profond les exécutera automatiquement en parallèle.
1. Lorsque la charge de travail d'un opérateur individuel est suffisamment faible, la parallélisation peut être utile même sur un seul CPU ou GPU. Concevez une expérience pour vérifier cela. 
1. Concevez une expérience qui utilise le calcul parallèle sur les CPU, les GPU et la communication entre les deux appareils.
1. Utilisez un débogueur tel que [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) de NVIDIA pour vérifier que votre code est efficace. 
1. Concevez des tâches de calcul qui incluent des dépendances de données plus complexes, et réalisez des expériences pour voir si vous pouvez obtenir les bons résultats tout en améliorant les performances.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:
