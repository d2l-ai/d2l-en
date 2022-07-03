# Taux d'apprentissage Planification
:label:`sec_scheduler` 

Jusqu'à présent, nous nous sommes principalement concentrés sur les *algorithmes d'optimisation* de la manière de mettre à jour les vecteurs de poids plutôt que sur le *taux* auquel ils sont mis à jour. Néanmoins, l'ajustement du taux d'apprentissage est souvent tout aussi important que l'algorithme lui-même. Plusieurs aspects sont à prendre en compte :

* De toute évidence, l'ampleur du taux d'apprentissage est importante. S'il est trop grand, l'optimisation diverge, s'il est trop petit, l'apprentissage prend trop de temps ou nous nous retrouvons avec un résultat sous-optimal. Nous avons vu précédemment que le nombre de conditions du problème est important (voir par exemple :numref:`sec_momentum` pour plus de détails). Intuitivement, il s'agit du rapport entre la quantité de changement dans la direction la moins sensible et la direction la plus sensible.
* Deuxièmement, le taux de décroissance est tout aussi important. Si le taux d'apprentissage reste élevé, nous pouvons simplement finir par rebondir autour du minimum et donc ne pas atteindre l'optimalité. :numref:`sec_minibatch_sgd` en a discuté de manière assez détaillée et nous avons analysé les garanties de performance dans :numref:`sec_sgd`. En bref, nous voulons que le taux décroisse, mais probablement plus lentement que $\mathcal{O}(t^{-\frac{1}{2}})$ qui serait un bon choix pour les problèmes convexes.
* Un autre aspect tout aussi important est l'*initialisation*. Il s'agit à la fois de la façon dont les paramètres sont définis initialement (voir :numref:`sec_numerical_stability` pour plus de détails) et de la façon dont ils évoluent initialement. C'est ce qu'on appelle le *warmup*, c'est-à-dire la rapidité avec laquelle on commence à se rapprocher de la solution. De grandes étapes au début peuvent ne pas être bénéfiques, en particulier parce que l'ensemble initial de paramètres est aléatoire. Les directions de mise à jour initiales peuvent également être tout à fait insignifiantes.
* Enfin, il existe un certain nombre de variantes d'optimisation qui effectuent un ajustement cyclique du taux d'apprentissage. Cela dépasse le cadre du présent chapitre. Nous recommandons au lecteur d'examiner les détails dans :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`, par exemple, comment obtenir de meilleures solutions en faisant la moyenne sur un *chemin* entier de paramètres.

Étant donné que la gestion des taux d'apprentissage nécessite beaucoup de détails, la plupart des cadres d'apprentissage profond disposent d'outils permettant de traiter cette question automatiquement. Dans le présent chapitre, nous passerons en revue les effets que les différents horaires ont sur la précision et nous montrerons également comment cela peut être géré efficacement via un *planificateur de taux d'apprentissage*.

## Problème fictif

Nous commençons par un problème fictif qui est assez bon marché pour être calculé facilement, mais suffisamment non trivial pour illustrer certains des aspects clés. Pour cela, nous choisissons une version légèrement modernisée de LeNet (`relu` au lieu de `sigmoid` activation, MaxPooling au lieu de AveragePooling), telle qu'appliquée à Fashion-MNIST. De plus, nous hybridons le réseau pour en améliorer les performances. Comme la plupart du code est standard, nous nous contentons de présenter les bases sans discussion détaillée. Voir :numref:`chap_cnn` pour un rafraîchissement si nécessaire.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0, 
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Voyons ce qui se passe si nous invoquons cet algorithme avec les paramètres par défaut, tels qu'un taux d'apprentissage de $0.3$ et un entraînement pendant $30$ itérations. Notez comment la précision de l'apprentissage continue d'augmenter alors que la progression en termes de précision du test stagne au-delà d'un point. L'écart entre les deux courbes indique un surajustement.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Planificateurs

Une façon d'ajuster le taux d'apprentissage est de le définir explicitement à chaque étape. La méthode `set_learning_rate` permet d'y parvenir facilement. Nous pourrions l'ajuster à la baisse après chaque époque (ou même après chaque minilot), par exemple, de manière dynamique en réponse à la façon dont l'optimisation progresse.

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

Plus généralement, nous voulons définir un planificateur. Lorsqu'il est invoqué avec le nombre de mises à jour, il renvoie la valeur appropriée du taux d'apprentissage. Définissons-en un simple qui fixe le taux d'apprentissage à $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Traçons son comportement sur une gamme de valeurs.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Voyons maintenant comment cela se passe pour l'entrainement sur Fashion-MNIST. Nous fournissons simplement le planificateur comme un argument supplémentaire à l'algorithme d'apprentissage.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Cela fonctionne un peu mieux que précédemment. Deux choses ressortent : la courbe est plus lisse que précédemment. Deuxièmement, il y a moins d'overfitting. Malheureusement, la question de savoir pourquoi certaines stratégies conduisent à moins d'overfitting n'est pas bien résolue en *théorie*. Il existe un argument selon lequel une taille de pas plus petite conduira à des paramètres plus proches de zéro et donc plus simples. Cependant, cela n'explique pas entièrement le phénomène puisque nous ne nous arrêtons pas vraiment tôt mais réduisons simplement le taux d'apprentissage en douceur.

## Politiques

Bien que nous ne puissions pas couvrir toute la variété des ordonnanceurs de taux d'apprentissage, nous essayons de donner un bref aperçu des politiques populaires ci-dessous. Les choix les plus courants sont la décroissance polynomiale et les planifications constantes par morceaux. En outre, les ordonnancements à taux d'apprentissage en cosinus se sont avérés efficaces de manière empirique pour certains problèmes. Enfin, pour certains problèmes, il est utile de faire chauffer l'optimiseur avant d'utiliser des taux d'apprentissage élevés.

### Planificateur factoriel

Une alternative à une décroissance polynomiale serait une décroissance multiplicative, c'est-à-dire $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ pour $\alpha \in (0, 1)$. Pour empêcher la décroissance du taux d'apprentissage au-delà d'une limite inférieure raisonnable, l'équation de mise à jour est souvent modifiée en $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Ceci peut également être accompli par un planificateur intégré dans MXNet via l'objet `lr_scheduler.FactorScheduler`. Il prend quelques paramètres supplémentaires, tels que la période d'échauffement, le mode d'échauffement (linéaire ou constant), le nombre maximum de mises à jour souhaitées, etc. A l'avenir, nous utiliserons les ordonnanceurs intégrés comme il convient et nous n'expliquerons ici que leur fonctionnalité. Comme illustré, il est assez simple de construire votre propre ordonnanceur si nécessaire.

### Planificateur multifactoriel

Une stratégie courante pour l'entraînement des réseaux profonds consiste à maintenir le taux d'apprentissage constant et à le diminuer d'une quantité donnée de temps en temps. C'est-à-dire, étant donné un ensemble de moments où il faut diminuer le taux, comme $s = \{5, 10, 20\}$ diminuer $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ chaque fois que $t \in s$. En supposant que les valeurs sont divisées par deux à chaque étape, nous pouvons mettre en œuvre cette méthode comme suit.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler) 
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
  
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

L'intuition derrière ce programme de taux d'apprentissage constant par morceaux est que l'on laisse l'optimisation se poursuivre jusqu'à ce qu'un point stationnaire ait été atteint en termes de distribution des vecteurs de poids. Ensuite (et seulement ensuite), nous diminuons le taux de manière à obtenir un proxy de meilleure qualité vers un bon minimum local. L'exemple ci-dessous montre comment cela peut produire des solutions légèrement meilleures.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Planificateur en cosinus

Une heuristique plutôt déroutante a été proposée par :cite:`Loshchilov.Hutter.2016`. Elle s'appuie sur l'observation que nous ne voulons peut-être pas diminuer trop radicalement le taux d'apprentissage au début et, de plus, que nous voulons peut-être "affiner" la solution à la fin en utilisant un taux d'apprentissage très faible. Il en résulte un programme de type cosinus avec la forme fonctionnelle suivante pour des taux d'apprentissage dans la plage $t \in [0, T]$.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


Ici, $\eta_0$ est le taux d'apprentissage initial, $\eta_T$ est le taux cible au moment $T$. De plus, pour $t > T$ nous épinglons simplement la valeur à $\eta_T$ sans l'augmenter à nouveau. Dans l'exemple suivant, nous fixons le pas de mise à jour maximal à $T = 20$.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
  
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Dans le contexte de la vision par ordinateur, ce programme *peut* conduire à de meilleurs résultats. Notez cependant que de telles améliorations ne sont pas garanties (comme on peut le voir ci-dessous).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

#### Warmup

Dans certains cas, l'initialisation des paramètres n'est pas suffisante pour garantir une bonne solution. Ceci est particulièrement un problème pour certaines conceptions de réseau avancées qui peuvent conduire à des problèmes d'optimisation instables. Nous pourrions résoudre ce problème en choisissant un taux d'apprentissage suffisamment faible pour éviter la divergence au début. Malheureusement, cela signifie que la progression est lente. Inversement, un taux d'apprentissage élevé conduit initialement à la divergence.

Une solution assez simple à ce dilemme consiste à utiliser une période d'échauffement au cours de laquelle le taux d'apprentissage *augmente* jusqu'à son maximum initial et à refroidir le taux jusqu'à la fin du processus d'optimisation. Pour simplifier, on utilise généralement une augmentation linéaire à cette fin. Cela conduit à un programme de la forme indiquée ci-dessous.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Notez que le réseau converge mieux initialement (observez en particulier les performances pendant les 5 premières époques).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

L'échauffement peut être appliqué à n'importe quel ordonnanceur (pas seulement le cosinus). Pour une discussion plus détaillée des ordonnanceurs de taux d'apprentissage et de nombreuses autres expériences, voir également :cite:`Gotmare.Keskar.Xiong.ea.2018`. Ils constatent notamment qu'une phase de réchauffement limite l'ampleur de la divergence des paramètres dans les réseaux très profonds. Cela est intuitivement logique puisque nous nous attendons à une divergence significative due à une initialisation aléatoire dans les parties du réseau qui prennent le plus de temps pour progresser au début.

## Résumé

* La diminution du taux d'apprentissage pendant l'entrainement peut conduire à une amélioration de la précision et (ce qui laisse le plus perplexe) à une réduction du surajustement du modèle.
* Une diminution par morceaux du taux d'apprentissage chaque fois que la progression atteint un plateau est efficace dans la pratique. Essentiellement, cela garantit que nous convergeons efficacement vers une solution appropriée et que nous réduisons ensuite la variance inhérente des paramètres en réduisant le taux d'apprentissage.
* Les ordonnanceurs Cosinus sont populaires pour certains problèmes de vision par ordinateur. Voir par exemple, [GluonCV](http://gluon-cv.mxnet.io) pour les détails d'un tel ordonnanceur.
* Une période de réchauffement avant l'optimisation peut empêcher la divergence.
* L'optimisation a plusieurs objectifs dans l'apprentissage profond. Outre la minimisation de l'objectif d'apprentissage, différents choix d'algorithmes d'optimisation et d'ordonnancement du taux d'apprentissage peuvent conduire à des quantités assez différentes de généralisation et de suradaptation sur l'ensemble de test (pour la même quantité d'erreur d'apprentissage).

## Exercices

1. Expérimentez le comportement d'optimisation pour un taux d'apprentissage fixe donné. Quel est le meilleur modèle que vous pouvez obtenir de cette manière ?
1. Comment la convergence change-t-elle si vous modifiez l'exposant de la diminution du taux d'apprentissage ? Utilisez `PolyScheduler` pour vous faciliter la tâche dans les expériences.
1. Appliquez l'ordonnanceur cosinus à de gros problèmes de vision par ordinateur, par exemple, l'entrainement d'ImageNet. Comment affecte-t-il les performances par rapport aux autres ordonnanceurs ?
1. Combien de temps doit durer l'échauffement ?
1. Pouvez-vous relier l'optimisation et l'échantillonnage ? Commencez par utiliser les résultats de :cite:`Welling.Teh.2011` sur la dynamique de Langevin à gradient stochastique.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:
