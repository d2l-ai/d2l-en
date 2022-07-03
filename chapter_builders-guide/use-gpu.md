```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# GPUs
:label:`sec_use_gpu`

Dans :numref:`tab_intro_decade`, nous avons abordé la croissance rapide de la
du calcul au cours des deux dernières décennies.
En un mot, les performances des GPU ont augmenté
par un facteur de 1000 chaque décennie depuis 2000.
Cela offre de grandes opportunités, mais cela suggère également
un besoin important de fournir de telles performances.


Dans cette section, nous commençons à discuter de la façon d'exploiter ces performances de calcul pour vos recherches.
cette performance de calcul pour votre recherche.
Tout d'abord en utilisant des GPU uniques et plus tard,
comment utiliser plusieurs GPU et plusieurs serveurs (avec plusieurs GPU).

Plus précisément, nous allons voir comment
d'utiliser un seul GPU NVIDIA pour les calculs.
Tout d'abord, vérifiez que vous avez au moins un GPU NVIDIA installé.
Ensuite, téléchargez le [pilote NVIDIA et CUDA](https://developer.nvidia.com/cuda-downloads)
et suivez les instructions pour définir le chemin d'accès approprié.
Une fois ces préparatifs terminés,
la commande `nvidia-smi' peut être utilisée
pour (**voir les informations de la carte graphique**).

```{.python .input}
%%tab all
!nvidia-smi
```

:begin_tab:`mxnet`
Vous avez peut-être remarqué qu'un tenseur MXNet
est presque identique à un `ndarray` de NumPy.
Mais il y a quelques différences cruciales.
L'une des principales caractéristiques qui distingue MXNet
de NumPy est son support de divers dispositifs matériels.

Dans MXNet, chaque tableau a un contexte.
Jusqu'à présent, par défaut, toutes les variables
et les calculs associés
ont été assignés au CPU.
Typiquement, d'autres contextes pourraient être divers GPUs.
Les choses peuvent devenir encore plus difficiles lorsque
nous déployons des tâches sur plusieurs serveurs.
En assignant intelligemment les tableaux aux contextes,
nous pouvons minimiser le temps passé
transfert de données entre les périphériques.
Par exemple, lors de l'entrainement de réseaux neuronaux sur un serveur avec un GPU,
nous préférons généralement que les paramètres du modèle se trouvent sur le GPU.

Ensuite, nous devons confirmer que
la version GPU de MXNet est installée.
Si une version CPU de MXNet est déjà installée,
nous devons d'abord la désinstaller.
Par exemple, utilisez la commande `pip uninstall mxnet`,
puis installez la version correspondante de MXNet
en fonction de votre version CUDA.
En supposant que vous avez CUDA 10.0 installé,
vous pouvez installer la version de MXNet
qui prend en charge CUDA 10.0 via la commande `pip install mxnet-cu100`.
:end_tab:

:begin_tab:`pytorch`
Dans PyTorch, chaque tableau a un dispositif, nous l'appelons souvent un contexte.
Jusqu'à présent, par défaut, toutes les variables
et les calculs associés
ont été assignés au CPU.
Typiquement, d'autres contextes pourraient être divers GPUs.
Les choses peuvent devenir encore plus compliquées lorsque
nous déployons des tâches sur plusieurs serveurs.
En assignant intelligemment les tableaux aux contextes,
nous pouvons minimiser le temps passé
transfert de données entre les périphériques.
Par exemple, lors de l'entrainement de réseaux neuronaux sur un serveur avec un GPU,
nous préférons généralement que les paramètres du modèle soient stockés sur le GPU.
:end_tab:

Pour exécuter les programmes de cette section,
vous avez besoin d'au moins deux GPU.
Notez que cela peut être extravagant pour la plupart des ordinateurs de bureau
mais c'est facilement disponible dans le cloud, par ex,
en utilisant les instances multi-GPU d'AWS EC2.
Presque toutes les autres sections ne nécessitent *pas* de GPU multiples.
Il s'agit simplement d'illustrer
comment les données circulent entre les différents périphériques.

## [**Computing Devices**]

Nous pouvons spécifier des périphériques, tels que les CPU et les GPU,
pour le stockage et le calcul.
Par défaut, les tenseurs sont créés dans la mémoire principale
et utilisent ensuite le CPU pour les calculer.

:begin_tab:`mxnet`
Dans MXNet, le CPU et le GPU peuvent être indiqués par `cpu()` et `gpu()`.
Il est à noter que `cpu()`
(ou tout autre nombre entier entre parenthèses)
signifie tous les processeurs physiques et la mémoire.
Cela signifie que les calculs de MXNet
essaieront d'utiliser tous les cœurs du CPU.
Cependant, `gpu()` ne représente qu'une seule carte
et la mémoire correspondante.
S'il y a plusieurs GPUs, nous utilisons `gpu(i)`
pour représenter le $i^\mathrm{th}$ GPU ($i$ commence à 0).
Aussi, `gpu(0)` et `gpu()` sont équivalents.
:end_tab:

:begin_tab:`pytorch`
Dans PyTorch, le CPU et le GPU peuvent être indiqués par `torch.device('cpu')` et `torch.device('cuda')`.
Il est à noter que le dispositif `cpu` désigne tous les processeurs physiques et la mémoire.
désigne tous les processeurs physiques et la mémoire.
Cela signifie que les calculs de PyTorch vont essayer
essaieront d'utiliser tous les cœurs du CPU.
Cependant, un périphérique `gpu` représente seulement une carte
et la mémoire correspondante.
S'il y a plusieurs GPUs, nous utilisons `torch.device(f'cuda:{i}')`
pour représenter le $i^\mathrm{th}$ GPU ($i$ commence à 0).
Aussi, `gpu:0` et `gpu` sont équivalents.
:end_tab:

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
def cpu():  #@save
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('pytorch'):
        return torch.device('cpu')
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')

def gpu(i=0):  #@save
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('pytorch'):
        return torch.device(f'cuda:{i}')
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')

cpu(), gpu(), gpu(1)
```

Nous pouvons (**interroger le nombre de GPUs disponibles.**)

```{.python .input}
%%tab all
def num_gpus():  #@save
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('pytorch'):
        return torch.cuda.device_count()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))

num_gpus()
```

Maintenant, nous [**définissons deux fonctions pratiques qui nous permettent d'exécuter du code même si les GPU demandés n'existent pas.**].

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensors and GPUs

Par défaut, les tenseurs sont créés sur le CPU.
Nous pouvons [**interroger le périphérique où se trouve le tenseur.**]

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

Il est important de noter que chaque fois que nous voulons
opérer sur plusieurs termes,
ils doivent être sur le même dispositif.
Par exemple, si nous additionnons deux tenseurs,
nous devons nous assurer que les deux arguments
se trouvent sur le même dispositif, sinon l'infrastructure
ne saurait pas où stocker le résultat
ou même comment décider où effectuer le calcul.

### Stockage sur le GPU

Il existe plusieurs façons de [**stocker un tenseur sur le GPU**].
Par exemple, nous pouvons spécifier un périphérique de stockage lors de la création d'un tenseur.
Ensuite, nous créons la variable tensorielle `X` sur le premier `gpu`.
Le tenseur créé sur un GPU ne consomme que la mémoire de ce GPU.
Nous pouvons utiliser la commande `nvidia-smi` pour visualiser l'utilisation de la mémoire du GPU.
En général, nous devons nous assurer que nous ne créons pas de données qui dépassent la limite de mémoire du GPU.

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

En supposant que vous avez au moins deux GPU, le code suivant va (**créer un tenseur aléatoire sur le deuxième GPU.**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### Copying

[**Si nous voulons calculer "X + Y", nous devons décider où effectuer cette opération. **]
Par exemple, comme indiqué dans :numref:`fig_copyto`,
nous pouvons transférer `X` vers le second GPU
et y effectuer l'opération.
*Ne vous contentez pas* d'additionner `X` et `Y`,
car cela entraînerait une exception.
Le moteur d'exécution ne saurait pas quoi faire :
il ne trouverait pas de données sur le même périphérique et il échouerait.
Puisque `Y` se trouve sur le second GPU,
nous devons y déplacer `X` avant de pouvoir ajouter les deux.

![Copier des données pour effectuer une opération sur la même device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

Maintenant que [**les données sont sur le même GPU
( `Z` et `Y` le sont tous les deux),
nous pouvons les additionner.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
Imaginez que votre variable `Z` se trouve déjà sur votre deuxième GPU.
Que se passe-t-il si nous appelons encore `Z.copyto(gpu(1))`?
Il fera une copie et allouera une nouvelle mémoire,
même si cette variable vit déjà sur le périphérique souhaité.
Il arrive que, selon l'environnement dans lequel notre code s'exécute,
deux variables puissent déjà se trouver sur le même périphérique.
Nous voulons donc faire une copie uniquement si les variables
se trouvent actuellement sur des périphériques différents.
Dans ce cas, nous pouvons appeler `as_in_ctx`.
Si la variable se trouve déjà dans le périphérique spécifié
, il n'y a rien à faire.
À moins que vous ne souhaitiez spécifiquement faire une copie,
`as_in_ctx` est la méthode à privilégier.
:end_tab:

:begin_tab:`pytorch`
Imaginez que votre variable `Z` se trouve déjà sur votre deuxième GPU.
Que se passe-t-il si nous appelons toujours `Z.cuda(1)`?
Il retournera `Z` au lieu de faire une copie et d'allouer une nouvelle mémoire.
:end_tab:

:begin_tab:`tensorflow`
Imaginez que votre variable `Z` se trouve déjà sur votre deuxième GPU.
Que se passe-t-il si nous appelons toujours `Z2 = Z` dans le même périmètre de périphérique ?
Il renverra `Z` au lieu d'en faire une copie et d'allouer une nouvelle mémoire.
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

#### Side Notes

Les gens utilisent les GPU pour faire de l'apprentissage automatique
parce qu'ils s'attendent à ce qu'ils soient rapides.
Mais le transfert de variables entre périphériques est lent.
Nous voulons donc que vous soyez 100% certain
que vous voulez faire quelque chose de lent avant de vous laisser le faire.
Si le cadre d'apprentissage profond effectue automatiquement la copie
sans planter, vous ne vous rendrez peut-être pas compte
que vous avez écrit du code lent.

De plus, le transfert de données entre les appareils (CPU, GPU et autres machines)
est quelque chose de beaucoup plus lent que le calcul.
Cela rend également la parallélisation beaucoup plus difficile,
puisque nous devons attendre que les données soient envoyées (ou plutôt reçues)
avant de pouvoir procéder à d'autres opérations.
C'est pourquoi les opérations de copie doivent être prises avec beaucoup de précautions.
En règle générale, plusieurs petites opérations
sont bien pires qu'une grosse opération.
De plus, plusieurs opérations à la fois
sont bien meilleures que de nombreuses opérations uniques intercalées dans le code
sauf si vous savez ce que vous faites.
En effet, ces opérations peuvent se bloquer si un périphérique
doit attendre l'autre avant de pouvoir faire autre chose.
C'est un peu comme commander son café dans une file d'attente
plutôt que de le précommander par téléphone
et de découvrir qu'il est prêt quand vous l'êtes.

Enfin, lorsque nous imprimons des tenseurs ou convertissons des tenseurs au format NumPy,
si les données ne se trouvent pas dans la mémoire principale,
le framework les copiera d'abord dans la mémoire principale,
ce qui entraîne une surcharge de transmission supplémentaire.
Pire encore, il est maintenant soumis au redoutable verrou global de l'interpréteur
qui fait que tout attend la fin de Python.


## [**Réseaux neuronaux et GPUs**]

De même, un modèle de réseau neuronal peut spécifier des périphériques.
Le code suivant place les paramètres du modèle sur le GPU.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

Nous verrons beaucoup plus d'exemples de
comment exécuter des modèles sur des GPU dans les chapitres suivants,
simplement parce qu'ils deviendront un peu plus intensifs en termes de calcul.

Lorsque l'entrée est un tenseur sur le GPU, le modèle calculera le résultat sur le même GPU.

```{.python .input}
%%tab all
net(X)
```

Confirmons (**que les paramètres du modèle sont stockés sur le même GPU.**)

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

Que l'entraîneur supporte le GPU.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

En bref, tant que toutes les données et tous les paramètres sont sur le même dispositif, nous pouvons apprendre des modèles efficacement. Dans les chapitres suivants, nous verrons plusieurs exemples de ce type.

## Résumé

* Nous pouvons spécifier des dispositifs pour le stockage et le calcul, comme le CPU ou le GPU.
 Par défaut, les données sont créées dans la mémoire principale
 et utilisent ensuite le CPU pour les calculs.
* Le cadre d'apprentissage profond exige que toutes les données d'entrée pour le calcul
 soient sur le même périphérique,
 que ce soit le CPU ou le même GPU.
* Vous pouvez perdre des performances importantes en déplaçant les données sans précaution.
 Une erreur typique est la suivante : calculer la perte
 pour chaque minibatch sur le GPU et la rapporter
 à l'utilisateur sur la ligne de commande (ou l'enregistrer dans un NumPy `ndarray`)
 déclenchera un verrouillage global de l'interpréteur qui bloquera tous les GPU.
 Il est bien mieux d'allouer de la mémoire
 pour la journalisation à l'intérieur du GPU et de ne déplacer que les journaux les plus importants.

## Exercices

1. Essayez une tâche de calcul plus importante, comme la multiplication de grandes matrices,
 et voyez la différence de vitesse entre le CPU et le GPU.
  Qu'en est-il d'une tâche comportant une petite quantité de calculs ?
1. Comment devons-nous lire et écrire les paramètres du modèle sur le GPU ?
1. Mesurez le temps nécessaire pour calculer 1000 multiplications matricielles
 de matrices $100 \times 100$
 et enregistrer la norme de Frobenius de la matrice de sortie, un résultat à la fois
 ou garder un journal sur le GPU et transférer uniquement le résultat final.
1. Mesurez le temps nécessaire pour effectuer deux multiplications matricielles
 sur deux GPU en même temps par rapport à la séquence
 sur un GPU. Indice : vous devriez constater une mise à l'échelle presque linéaire.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:


