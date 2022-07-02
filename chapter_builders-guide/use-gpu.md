```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# GPUs
:label:`sec_use_gpu`

In :numref:`tab_intro_decade`, we discussed the rapid growth
of computation over the past two decades.
In a nutshell, GPU performance has increased
by a factor of 1000 every decade since 2000.
This offers great opportunities but it also suggests
a significant need to provide such performance.


In this section, we begin to discuss how to harness
this computational performance for your research.
First by using single GPUs and at a later point,
how to use multiple GPUs and multiple servers (with multiple GPUs).

Specifically, we will discuss how
to use a single NVIDIA GPU for calculations.
First, make sure you have at least one NVIDIA GPU installed.
Then, download the [NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads)
and follow the prompts to set the appropriate path.
Once these preparations are complete,
the `nvidia-smi` command can be used
to (**view the graphics card information**).

```{.python .input}
%%tab all
!nvidia-smi
```

:begin_tab:`mxnet`
You might have noticed that a MXNet tensor
looks almost identical to a NumPy `ndarray`.
But there are a few crucial differences.
One of the key features that distinguishes MXNet
from NumPy is its support for diverse hardware devices.

In MXNet, every array has a context.
So far, by default, all variables
and associated computation
have been assigned to the CPU.
Typically, other contexts might be various GPUs.
Things can get even hairier when
we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently,
we can minimize the time spent
transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.

Next, we need to confirm that
the GPU version of MXNet is installed.
If a CPU version of MXNet is already installed,
we need to uninstall it first.
For example, use the `pip uninstall mxnet` command,
then install the corresponding MXNet version
according to your CUDA version.
Assuming you have CUDA 10.0 installed,
you can install the MXNet version
that supports CUDA 10.0 via `pip install mxnet-cu100`.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, every array has a device, we often refer it as a context.
So far, by default, all variables
and associated computation
have been assigned to the CPU.
Typically, other contexts might be various GPUs.
Things can get even hairier when
we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently,
we can minimize the time spent
transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.
:end_tab:

To run the programs in this section,
you need at least two GPUs.
Note that this might be extravagant for most desktop computers
but it is easily available in the cloud, e.g.,
by using the AWS EC2 multi-GPU instances.
Almost all other sections do *not* require multiple GPUs.
Instead, this is simply to illustrate
how data flow between different devices.

## [**Computing Devices**]

We can specify devices, such as CPUs and GPUs,
for storage and calculation.
By default, tensors are created in the main memory
and then use the CPU to calculate it.

:begin_tab:`mxnet`
In MXNet, the CPU and GPU can be indicated by `cpu()` and `gpu()`.
It should be noted that `cpu()`
(or any integer in the parentheses)
means all physical CPUs and memory.
This means that MXNet's calculations
will try to use all CPU cores.
However, `gpu()` only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `gpu(i)`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu(0)` and `gpu()` are equivalent.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the CPU and GPU can be indicated by `torch.device('cpu')` and `torch.device('cuda')`.
It should be noted that the `cpu` device
means all physical CPUs and memory.
This means that PyTorch's calculations
will try to use all CPU cores.
However, a `gpu` device only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `torch.device(f'cuda:{i}')`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu:0` and `gpu` are equivalent.
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

We can (**query the number of available GPUs.**)

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

Now we [**define two convenient functions that allow us
to run code even if the requested GPUs do not exist.**]

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

By default, tensors are created on the CPU.
We can [**query the device where the tensor is located.**]

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

It is important to note that whenever we want
to operate on multiple terms,
they need to be on the same device.
For instance, if we sum two tensors,
we need to make sure that both arguments
live on the same device---otherwise the framework
would not know where to store the result
or even how to decide where to perform the computation.

### Storage on the GPU

There are several ways to [**store a tensor on the GPU.**]
For example, we can specify a storage device when creating a tensor.
Next, we create the tensor variable `X` on the first `gpu`.
The tensor created on a GPU only consumes the memory of this GPU.
We can use the `nvidia-smi` command to view GPU memory usage.
In general, we need to make sure that we do not create data that exceeds the GPU memory limit.

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

Assuming that you have at least two GPUs, the following code will (**create a random tensor on the second GPU.**)

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

[**If we want to compute `X + Y`,
we need to decide where to perform this operation.**]
For instance, as shown in :numref:`fig_copyto`,
we can transfer `X` to the second GPU
and perform the operation there.
*Do not* simply add `X` and `Y`,
since this will result in an exception.
The runtime engine would not know what to do:
it cannot find data on the same device and it fails.
Since `Y` lives on the second GPU,
we need to move `X` there before we can add the two.

![Copy data to perform an operation on the same device.](../img/copyto.svg)
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


