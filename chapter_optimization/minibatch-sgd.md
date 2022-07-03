# Minibatch Stochastic Gradient Descent
:label:`sec_minibatch_sgd` 

Jusqu'à présent, nous avons rencontré deux extrêmes dans l'approche de l'apprentissage par gradient : :numref:`sec_gd` utilise l'ensemble des données pour calculer les gradients et mettre à jour les paramètres, une passe à la fois. Inversement, :numref:`sec_sgd` traite un seul exemple d'entraînement à la fois pour progresser.
Chacune de ces méthodes présente ses propres inconvénients.
La descente de gradient n'est pas particulièrement *efficace en termes de données* lorsque les données sont très similaires.
La descente de gradient stochastique n'est pas particulièrement *efficace sur le plan du calcul* puisque les CPU et les GPU ne peuvent pas exploiter toute la puissance de la vectorisation.
Cela suggère qu'il pourrait y avoir quelque chose entre les deux,
et en fait, c'est ce que nous avons utilisé jusqu'à présent dans les exemples que nous avons discutés.

## Vectorisation et caches

L'efficacité de calcul est au cœur de la décision d'utiliser des minibatchs. On le comprend mieux lorsqu'on envisage la parallélisation vers plusieurs GPU et plusieurs serveurs. Dans ce cas, nous devons envoyer au moins une image à chaque GPU. Avec 8 GPU par serveur et 16 serveurs, nous arrivons déjà à une taille de minibatch non inférieure à 128.

Les choses sont un peu plus subtiles lorsqu'il s'agit de GPU uniques ou même de CPU. Ces dispositifs ont plusieurs types de mémoire, souvent plusieurs types d'unités de calcul et différentes contraintes de bande passante entre eux.
Par exemple, un CPU possède un petit nombre de registres, puis les caches L1, L2 et, dans certains cas, L3 (qui sont partagés entre les différents cœurs du processeur).
Ces caches ont une taille et une latence croissantes (et en même temps une bande passante décroissante).
Il suffit de dire que le processeur est capable d'effectuer beaucoup plus d'opérations que ce que l'interface de la mémoire principale est capable de fournir.

Tout d'abord, un CPU de 2 GHz avec 16 cœurs et une vectorisation AVX-512 peut traiter jusqu'à $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ octets par seconde. Les capacités des GPU dépassent facilement ce chiffre par un facteur de 100. D'autre part, un processeur de serveur de milieu de gamme peut ne pas avoir beaucoup plus de 100 Go/s de bande passante, c'est-à-dire moins d'un dixième de ce qui serait nécessaire pour alimenter le processeur. Pour aggraver les choses, tous les accès à la mémoire ne sont pas égaux : les interfaces mémoire ont généralement une largeur de 64 bits ou plus (par exemple, sur les GPU, jusqu'à 384 bits), ce qui signifie que la lecture d'un seul octet entraîne le coût d'un accès beaucoup plus large.

Deuxièmement, il y a une surcharge importante pour le premier accès alors que l'accès séquentiel est relativement bon marché (c'est ce qu'on appelle souvent une lecture en rafale). Il y a beaucoup d'autres choses à garder à l'esprit, comme la mise en cache lorsque nous avons plusieurs sockets, chiplets et autres structures.
Consultez le site [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy) 
pour une discussion plus approfondie.

La façon d'alléger ces contraintes est d'utiliser une hiérarchie de caches CPU qui sont réellement assez rapides pour alimenter le processeur en données. C'est la *force motrice du batching dans l'apprentissage profond. Pour simplifier les choses, considérons la multiplication matrice-matrice, disons $\mathbf{A} = \mathbf{B}\mathbf{C}$. Nous disposons d'un certain nombre d'options pour calculer $\mathbf{A}$. Par exemple, nous pouvons essayer ce qui suit :

1. Nous pouvons calculer $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$, c'est-à-dire que nous pouvons le calculer par éléments au moyen de produits scalaires.
1. Nous pourrions calculer $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$, c'est-à-dire que nous pourrions le calculer une colonne à la fois. De même, nous pourrions calculer $\mathbf{A}$ une ligne de $\mathbf{A}_{i,:}$ à la fois.
1. Nous pourrions simplement calculer $\mathbf{A} = \mathbf{B} \mathbf{C}$.
1. Nous pourrions diviser $\mathbf{B}$ et $\mathbf{C}$ en matrices de blocs plus petites et calculer $\mathbf{A}$ un bloc à la fois.

Si nous choisissons la première option, nous devrons copier un vecteur ligne et un vecteur colonne dans le CPU chaque fois que nous voudrons calculer un élément de $\mathbf{A}_{ij}$. Pire encore, étant donné que les éléments de la matrice sont alignés séquentiellement, nous devons accéder à de nombreux emplacements disjoints pour l'un des deux vecteurs lorsque nous les lisons en mémoire. La deuxième option est beaucoup plus favorable. Elle nous permet de conserver le vecteur colonne $\mathbf{C}_{:,j}$ dans le cache du CPU tout en continuant à parcourir $\mathbf{B}$. Cela permet de diviser par deux la bande passante requise en mémoire avec un accès plus rapide correspondant. Bien entendu, l'option 3 est la plus souhaitable. Malheureusement, la plupart des matrices ne peuvent pas être entièrement placées dans le cache (c'est ce dont nous parlons après tout). Cependant, l'option 4 offre une alternative utile en pratique : nous pouvons déplacer des blocs de la matrice dans le cache et les multiplier localement. Les bibliothèques optimisées s'en chargent pour nous. Voyons dans quelle mesure ces opérations sont efficaces en pratique.

Au-delà de l'efficacité de calcul, les frais généraux introduits par Python et par le cadre d'apprentissage profond lui-même sont considérables. Rappelons qu'à chaque fois que nous exécutons une commande, l'interpréteur Python envoie une commande au moteur MXNet qui doit l'insérer dans le graphe de calcul et la traiter pendant l'ordonnancement. Une telle surcharge peut être tout à fait préjudiciable. En résumé, il est fortement conseillé d'utiliser la vectorisation (et les matrices) chaque fois que cela est possible.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Puisque nous évaluerons fréquemment le temps d'exécution dans le reste du livre, définissons un temporisateur.

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    
timer = Timer()
```

L'affectation par éléments itère simplement sur toutes les lignes et colonnes de $\mathbf{B}$ et $\mathbf{C}$ respectivement pour affecter la valeur à $\mathbf{A}$.

```{.python .input}
#@tab mxnet
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Une stratégie plus rapide consiste à effectuer une affectation par colonne.

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

Enfin, la manière la plus efficace est d'effectuer l'ensemble de l'opération en un seul bloc. Voyons la vitesse respective de ces opérations.

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatchs

:label:`sec_minibatches` 

Dans le passé, nous avons considéré comme acquis le fait de lire des *minibatchs* de données plutôt que des observations uniques pour mettre à jour les paramètres. Nous en donnons maintenant une brève justification. Le traitement d'observations uniques nous oblige à effectuer de nombreuses multiplications matrice-vecteur (ou même vecteur-vecteur), ce qui est assez coûteux et entraîne une surcharge importante pour le cadre d'apprentissage profond sous-jacent. Cela s'applique à la fois à l'évaluation d'un réseau lorsqu'il est appliqué aux données (souvent appelé inférence) et au calcul des gradients pour mettre à jour les paramètres. C'est-à-dire que cela s'applique chaque fois que nous exécutons $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ où

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$ 

Nous pouvons augmenter l'efficacité *computationnelle* de cette opération en l'appliquant à un minibatch d'observations à la fois. En d'autres termes, nous remplaçons le gradient $\mathbf{g}_t$ sur une seule observation par un gradient sur un petit lot

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$ 

Voyons ce que cela donne aux propriétés statistiques de $\mathbf{g}_t$: puisque $\mathbf{x}_t$ et tous les éléments du minilot $\mathcal{B}_t$ sont tirés uniformément au hasard de l'ensemble d'apprentissage, l'espérance du gradient reste inchangée. La variance, par contre, est réduite de manière significative. Puisque le gradient du minilot est composé de $b := |\mathcal{B}_t|$ gradients indépendants dont on fait la moyenne, son écart type est réduit par un facteur de $b^{-\frac{1}{2}}$. En soi, c'est une bonne chose, car cela signifie que les mises à jour sont alignées de manière plus fiable sur le gradient complet.

Naïvement, cela pourrait indiquer que le choix d'un grand minibatch $\mathcal{B}_t$ serait universellement souhaitable. Hélas, après un certain point, la réduction supplémentaire de l'écart-type est minime par rapport à l'augmentation linéaire du coût de calcul. En pratique, nous choisissons un minibatch qui est suffisamment grand pour offrir une bonne efficacité de calcul tout en tenant dans la mémoire d'un GPU. Pour illustrer les économies réalisées, voyons un peu de code. Nous y effectuons la même multiplication matrice-matrice, mais cette fois-ci divisée en "minibatchs" de 64 colonnes à la fois.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

Comme nous pouvons le voir, le calcul sur le minilot est essentiellement aussi efficace que sur la matrice complète. Une mise en garde s'impose. Dans :numref:`sec_batch_norm`, nous avons utilisé un type de régularisation qui dépendait fortement de la quantité de variance dans un minibatch. Lorsque nous augmentons cette dernière, la variance diminue et avec elle le bénéfice de l'injection de bruit due à la normalisation des lots. Voir, par exemple, :cite:`Ioffe.2017` pour plus de détails sur la façon de remettre à l'échelle et de calculer les termes appropriés.

## Lecture du jeu de données

Voyons comment les minibatchs sont générés efficacement à partir des données. Dans ce qui suit, nous utilisons un jeu de données développé par la NASA pour tester l'aile [noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) afin de comparer ces algorithmes d'optimisation. Par commodité, nous n'utilisons que les premiers exemples $1,500$. Les données sont blanchies pour le prétraitement, c'est-à-dire que nous supprimons la moyenne et rééchelonnons la variance à $1$ par coordonnée.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Implémentation à partir de zéro

Rappelez-vous l'implémentation de la descente de gradient stochastique en minibatch de :numref:`sec_linear_scratch`. Dans ce qui suit, nous fournissons une implémentation légèrement plus générale. Par commodité, elle a la même signature d'appel que les autres algorithmes d'optimisation présentés plus loin dans ce chapitre. Plus précisément, nous ajoutons l'état
à l'entrée `states` et plaçons l'hyperparamètre dans le dictionnaire `hyperparams`. En outre,
nous calculons la moyenne de la perte de chaque exemple de minilots dans la fonction d'apprentissage,
de sorte que le gradient dans l'algorithme d'optimisation n'a pas besoin d'être
divisé par la taille du lot.

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Ensuite, nous implémentons une fonction d'apprentissage générique pour faciliter l'utilisation des autres algorithmes d'optimisation présentés plus loin dans ce chapitre. Elle initialise un modèle de régression linéaire et peut être utilisée pour entraîner le modèle avec la descente de gradient stochastique en minibatchs et d'autres algorithmes présentés ultérieurement.

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Voyons comment l'optimisation se déroule pour la descente de gradient par lots. Pour ce faire, il suffit de fixer la taille du minibatch à 1500 (c'est-à-dire au nombre total d'exemples). Par conséquent, les paramètres du modèle ne sont mis à jour qu'une fois par époque. Il y a peu de progrès. En fait, après 6 étapes, le progrès s'arrête.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Lorsque la taille du lot est égale à 1, nous utilisons la descente de gradient stochastique pour l'optimisation. Pour simplifier l'implémentation, nous avons choisi un taux d'apprentissage constant (bien que petit). Dans la descente de gradient stochastique, les paramètres du modèle sont mis à jour chaque fois qu'un exemple est traité. Dans notre cas, cela équivaut à 1500 mises à jour par époque. Comme nous pouvons le constater, la baisse de la valeur de la fonction objectif ralentit après une époque. Bien que les deux procédures aient traité 1500 exemples en une époque, la descente de gradient stochastique consomme plus de temps que la descente de gradient dans notre expérience. Cela est dû au fait que la descente de gradient stochastique met à jour les paramètres plus fréquemment et qu'il est moins efficace de traiter les observations une par une.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Enfin, lorsque la taille du lot est égale à 100, nous utilisons la descente de gradient stochastique en minibatch pour l'optimisation. Le temps requis par époque est plus court que le temps requis pour la descente de gradient stochastique et le temps pour la descente de gradient par lots.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

En réduisant la taille du lot à 10, le temps pour chaque époque augmente car la charge de travail pour chaque lot est moins efficace à exécuter.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Nous pouvons maintenant comparer le temps par rapport à la perte pour les quatre expériences précédentes. Comme on peut le voir, bien que la descente de gradient stochastique converge plus rapidement que GD en termes de nombre d'exemples traités, elle utilise plus de temps pour atteindre la même perte que GD car le calcul du gradient exemple par exemple n'est pas aussi efficace. La descente de gradient stochastique par minilots est capable d'équilibrer la vitesse de convergence et l'efficacité du calcul. Une taille de minibatch de 10 est plus efficace que la descente de gradient stochastique ; une taille de minibatch de 100 surpasse même GD en termes de temps d'exécution.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Mise en œuvre concise

Dans Gluon, nous pouvons utiliser la classe `Trainer` pour appeler les algorithmes d'optimisation. Ceci est utilisé pour implémenter une fonction d'entraînement générique. Nous l'utiliserons tout au long du présent chapitre.

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `Erreur quadratique moyenne` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

L'utilisation de Gluon pour répéter la dernière expérience montre un comportement identique.

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Résumé

* La vectorisation rend le code plus efficace en raison de la réduction de l'overhead provenant du cadre d'apprentissage profond et en raison d'une meilleure localité de la mémoire et de la mise en cache sur les CPU et les GPU.
* Il existe un compromis entre l'efficacité statistique découlant de la descente de gradient stochastique et l'efficacité informatique découlant du traitement de grands lots de données à la fois.
* La descente de gradient stochastique en mini-lots offre le meilleur des deux mondes : efficacité informatique et statistique.
* Dans la descente de gradient stochastique en mini-lots, nous traitons des lots de données obtenus par une permutation aléatoire des données d'apprentissage (c'est-à-dire que chaque observation n'est traitée qu'une fois par époque, mais dans un ordre aléatoire).
* Il est conseillé de décroître les taux d'apprentissage pendant la formation.
* En général, la descente de gradient stochastique en minibatch est plus rapide que la descente de gradient stochastique et la descente de gradient pour la convergence vers un risque plus faible, lorsqu'elle est mesurée en termes de temps d'horloge.

## Exercices

1. Modifiez la taille des lots et le taux d'apprentissage et observez le taux de décroissance de la valeur de la fonction objectif et le temps consommé à chaque époque.
1. Lisez la documentation MXNet et utilisez la fonction de la classe `Trainer` `set_learning_rate` pour réduire le taux d'apprentissage de la descente de gradient stochastique en minibatch à 1/10 de sa valeur précédente après chaque époque.
1. Comparez la descente de gradient stochastique en minibatchs avec une variante qui *échantillonne avec remplacement* de l'ensemble d'apprentissage. Que se passe-t-il ?
1. Un génie maléfique réplique votre ensemble de données sans vous le dire (c'est-à-dire que chaque observation se produit deux fois et que votre ensemble de données atteint le double de sa taille initiale, mais personne ne vous l'a dit). Comment le comportement de la descente de gradient stochastique, de la descente de gradient stochastique en minibatch et de la descente de gradient change-t-il ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
