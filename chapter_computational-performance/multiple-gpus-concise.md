# Mise en œuvre concise pour plusieurs GPU
:label:`sec_multi_gpu_concise` 

Mettre en œuvre le parallélisme à partir de zéro pour chaque nouveau modèle n'est pas une partie de plaisir. De plus, il y a un avantage significatif à optimiser les outils de synchronisation pour une haute performance. Dans ce qui suit, nous allons montrer comment le faire en utilisant les API de haut niveau des cadres d'apprentissage profond.
Les mathématiques et les algorithmes sont les mêmes que dans :numref:`sec_multi_gpu`.
Sans surprise, vous aurez besoin d'au moins deux GPU pour exécuter le code de cette section.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**A Toy Network**]

Utilisons un réseau un peu plus significatif que le LeNet de :numref:`sec_multi_gpu` qui est encore suffisamment facile et rapide à entraîner.
Nous choisissons une variante de ResNet-18 :cite:`He.Zhang.Ren.ea.2016`. Comme les images d'entrée sont minuscules, nous le modifions légèrement. En particulier, la différence avec :numref:`sec_resnet` est que nous utilisons un noyau de convolution, un stride et un padding plus petits au début.
De plus, nous supprimons la couche de max-pooling.

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Initialisation du réseau

:begin_tab:`mxnet` 
La fonction `initialize` nous permet d'initialiser les paramètres sur un dispositif de notre choix.
Pour un rappel sur les méthodes d'initialisation, voir :numref:`sec_numerical_stability`. Ce qui est particulièrement pratique, c'est qu'elle nous permet également d'initialiser le réseau sur *plusieurs* périphériques simultanément. Voyons comment cela fonctionne en pratique.
:end_tab:

:begin_tab:`pytorch`
Nous allons initialiser le réseau à l'intérieur de la boucle d'apprentissage.
Pour un rappel sur les méthodes d'initialisation, voir :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
À l'aide de la fonction `split_and_load` introduite dans :numref:`sec_multi_gpu`, nous pouvons diviser un minilot de données et en copier des portions dans la liste de périphériques fournie par la variable `devices`. L'instance de réseau *automatiquement* utilise le GPU approprié pour calculer la valeur de la propagation vers l'avant. Ici, nous générons 4 observations et les répartissons sur les GPU.
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Une fois que les données passent par le réseau, les paramètres correspondants sont initialisés *sur le dispositif par lequel les données sont passées*.
Cela signifie que l'initialisation se fait sur une base par dispositif. Puisque nous avons choisi le GPU 0 et le GPU 1 pour l'initialisation, le réseau n'est initialisé que là, et pas sur le CPU. En fait, les paramètres n'existent même pas sur le CPU. Nous pouvons le vérifier en imprimant les paramètres et en observant toute erreur qui pourrait survenir.
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Ensuite, remplaçons le code pour [**évaluer la précision**] par un code qui fonctionne (**en parallèle sur plusieurs périphériques**). Cette fonction remplace la fonction `evaluate_accuracy_gpu` de :numref:`sec_lenet`. La principale différence est que nous divisons un minibatch avant d'invoquer le réseau. Tout le reste est essentiellement identique.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Training**]

Comme précédemment, le code de entrainement doit exécuter plusieurs fonctions de base pour un parallélisme efficace :

* Les paramètres du réseau doivent être initialisés sur tous les appareils.
* Lors de l'itération sur le jeu de données, les minibatchs doivent être répartis sur tous les appareils.
* Nous calculons la perte et son gradient en parallèle sur tous les dispositifs.
* Les gradients sont agrégés et les paramètres sont mis à jour en conséquence.

À la fin, nous calculons la précision (encore une fois en parallèle) pour rendre compte de la performance finale du réseau. La routine de entrainement est assez similaire aux implémentations des chapitres précédents, sauf que nous devons diviser et agréger les données.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Voyons comment cela fonctionne en pratique. Pour nous échauffer, nous [**entraînons le réseau sur un seul GPU.**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Ensuite, nous [**utilisons 2 GPU pour l'entraînement**]. Comparé à LeNet
évalué dans :numref:`sec_multi_gpu`,
le modèle de ResNet-18 est considérablement plus complexe. C'est là que la parallélisation montre son avantage. Le temps de calcul est significativement plus important que le temps de synchronisation des paramètres. Cela améliore l'extensibilité puisque l'overhead de la parallélisation est moins important.

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Résumé

:begin_tab:`mxnet` 
* Gluon fournit des primitives pour l'initialisation du modèle sur plusieurs dispositifs en fournissant une liste de contexte.
:end_tab:
* Les données sont automatiquement évaluées sur les dispositifs où elles peuvent être trouvées.
* Prenez soin d'initialiser les réseaux sur chaque dispositif avant d'essayer d'accéder aux paramètres sur ce dispositif. Sinon, vous rencontrerez une erreur.
* Les algorithmes d'optimisation s'agrègent automatiquement sur plusieurs GPU.



## Exercices

:begin_tab:`mxnet` 
 1. Cette section utilise ResNet-18. Essayez différentes époques, tailles de lots et taux d'apprentissage. Utilisez plus de GPU pour le calcul. Que se passe-t-il si vous essayez avec 16 GPU (par exemple, sur une instance AWS p2.16xlarge) ?
1. Parfois, différents dispositifs fournissent une puissance de calcul différente. Nous pourrions utiliser les GPU et le CPU en même temps. Comment devrions-nous diviser le travail ? Le jeu en vaut-il la chandelle ? Pourquoi ? Pourquoi pas ?
1. Que se passe-t-il si nous abandonnons `npx.waitall()`? Comment modifieriez-vous l'entrainement de manière à obtenir un chevauchement de deux étapes au maximum pour le parallélisme ?
:end_tab:

:begin_tab:`pytorch`
1. Cette section utilise ResNet-18. Essayez différentes époques, tailles de lots et taux d'apprentissage. Utilisez plus de GPU pour le calcul. Que se passe-t-il si vous essayez avec 16 GPU (par exemple, sur une instance AWS p2.16xlarge) ?
1. Parfois, différents dispositifs fournissent une puissance de calcul différente. Nous pourrions utiliser les GPU et le CPU en même temps. Comment devrions-nous diviser le travail ? Le jeu en vaut-il la chandelle ? Pourquoi ? Pourquoi pas ?
:end_tab:



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
