# Image Augmentation
:label:`sec_image_augmentation` 

Dans :numref:`sec_alexnet`, 
nous avons mentionné que de grands ensembles de données 
sont une condition préalable
pour le succès de
réseaux neuronaux profonds
dans diverses applications.
*Augmentation d'image* 
génère des exemples d'entraînement similaires mais distincts
après une série de modifications aléatoires des images d'entraînement, augmentant ainsi la taille de l'ensemble d'entraînement.
Par ailleurs, l'augmentation d'image
peut être motivée
par le fait que 
les modifications aléatoires des exemples d'entraînement 
permettent aux modèles de moins dépendre de
certains attributs, améliorant ainsi leur capacité de généralisation.
Par exemple, nous pouvons recadrer une image de différentes manières pour faire apparaître l'objet d'intérêt dans différentes positions, réduisant ainsi la dépendance d'un modèle à la position de l'objet. 
Nous pouvons également ajuster des facteurs tels que la luminosité et la couleur pour réduire la sensibilité d'un modèle à la couleur.
Il est probablement vrai
que l'augmentation de l'image était indispensable
pour le succès d'AlexNet à cette époque.
Dans cette section, nous allons aborder cette technique largement utilisée en vision par ordinateur.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## Méthodes courantes d'augmentation d'image

Dans notre étude des méthodes courantes d'augmentation d'image, nous utiliserons l'image suivante $400\times 500$ comme exemple.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

La plupart des méthodes d'augmentation d'image ont un certain degré d'aléatoire. Pour faciliter l'observation de l'effet de l'augmentation de l'image, nous définissons ensuite une fonction auxiliaire `apply`. Cette fonction exécute plusieurs fois la méthode d'augmentation d'image `aug` sur l'image d'entrée `img` et affiche tous les résultats.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Retournement et recadrage

:begin_tab:`mxnet` 
[**Retourner l'image à gauche et à droite**] ne change généralement pas la catégorie de l'objet. 
C'est l'une des premières méthodes d'augmentation de l'image et l'une des plus utilisées.
Ensuite, nous utilisons le module `transforms` pour créer l'instance `RandomFlipLeftRight`, qui retourne
une image à gauche et à droite avec une chance sur deux.
:end_tab:

:begin_tab:`pytorch`
[**Retourner l'image à gauche et à droite**] ne change généralement pas la catégorie de l'objet. 
Il s'agit de l'une des premières méthodes d'augmentation d'image et l'une des plus utilisées.
Ensuite, nous utilisons le module `transforms` pour créer l'instance `RandomHorizontalFlip`, qui retourne
une image à gauche et à droite avec une chance sur deux.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**Renverser de haut en bas**] n'est pas aussi courant que de renverser de gauche à droite. Mais au moins pour cet exemple d'image, le retournement de haut en bas n'entrave pas la reconnaissance.
Ensuite, nous créons une instance `RandomFlipTopBottom` pour retourner
une image de haut en bas avec une chance sur deux.
:end_tab:

:begin_tab:`pytorch`
[**Retourner de haut en bas**] n'est pas aussi courant que de retourner de gauche à droite. Mais au moins pour cet exemple d'image, le retournement de haut en bas n'entrave pas la reconnaissance.
Ensuite, nous créons une instance `RandomVerticalFlip` pour retourner
une image de haut en bas avec une chance sur deux.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

Dans l'image d'exemple que nous avons utilisée, le chat se trouve au milieu de l'image, mais ce n'est peut-être pas le cas en général. 
Dans :numref:`sec_pooling`, nous avons expliqué que la couche de mise en commun peut réduire la sensibilité d'une couche convolutive à la position de la cible.
En outre, nous pouvons également recadrer l'image de manière aléatoire pour que les objets apparaissent à différentes positions dans l'image et à différentes échelles, ce qui peut également réduire la sensibilité d'un modèle à la position cible.

Dans le code ci-dessous, nous [**recadrons de manière aléatoire**] une zone avec une surface de $10\% \sim 100\%$ de la zone d'origine à chaque fois, et le rapport entre la largeur et la hauteur de cette zone est choisi de manière aléatoire parmi $0.5 \sim 2$. Ensuite, la largeur et la hauteur de la région sont toutes deux mises à l'échelle à 200 pixels. 
Sauf indication contraire, le nombre aléatoire entre $a$ et $b$ dans cette section fait référence à une valeur continue obtenue par échantillonnage aléatoire et uniforme à partir de l'intervalle $[a, b]$.

```{.python .input}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

#### Changement de couleurs

Une autre méthode d'augmentation est le changement de couleurs. Nous pouvons modifier quatre aspects de la couleur de l'image : la luminosité, le contraste, la saturation et la teinte. Dans l'exemple ci-dessous, nous [**changeons aléatoirement la luminosité**] de l'image à une valeur comprise entre 50% ($1-0.5$) et 150% ($1+0.5$) de l'image originale.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

De même, nous pouvons [**modifier de manière aléatoire la teinte**] de l'image.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Nous pouvons également créer une instance `RandomColorJitter` et définir comment [**modifier de manière aléatoire les valeurs `brightness`, `contrast`, `saturation` et `hue` de l'image en même temps**].

```{.python .input}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Combinaison de plusieurs méthodes d'augmentation d'image

Dans la pratique, nous allons [**combiner plusieurs méthodes d'augmentation d'image**]. 
Par exemple,
nous pouvons combiner les différentes méthodes d'augmentation d'image définies ci-dessus et les appliquer à chaque image via une instance `Compose`.

```{.python .input}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**Entrainement avec l'augmentation d'image**]

Formons un modèle avec l'augmentation d'image.
Nous utilisons ici le jeu de données CIFAR-10 au lieu du jeu de données Fashion-MNIST que nous avons utilisé précédemment. 
En effet, la position et la taille des objets dans le jeu de données Fashion-MNIST ont été normalisées, alors que la couleur et la taille des objets dans le jeu de données CIFAR-10 présentent des différences plus importantes. 
Les 32 premières images d'entraînement du jeu de données CIFAR-10 sont présentées ci-dessous.

```{.python .input}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Afin d'obtenir des résultats définitifs lors de la prédiction, nous n'appliquons généralement l'augmentation d'image qu'aux exemples d'entraînement et n'utilisons pas l'augmentation d'image avec des opérations aléatoires pendant la prédiction. 
[**Nous utilisons ici uniquement la méthode aléatoire la plus simple de retournement gauche-droite**]. En outre, nous utilisons une instance `ToTensor` pour convertir un minibatch d'images dans le format requis par le cadre d'apprentissage profond, c'est-à-dire 
des nombres à virgule flottante de 32 bits entre 0 et 1 avec la forme de (taille du lot, nombre de canaux, hauteur, largeur).

```{.python .input}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
Ensuite, nous définissons une fonction auxiliaire pour faciliter la lecture de l'image et
l'application de l'augmentation de l'image. 
La fonction `transform_first` fournie par les ensembles de données
de Gluon applique l'augmentation d'image au premier élément de chaque exemple d'entraînement
(image et étiquette), c'est-à-dire l'image. 
Pour
une introduction détaillée à `DataLoader`, veuillez vous référer à :numref:`sec_fashion_mnist`.
:end_tab: 

 :begin_tab:`pytorch` 
Ensuite, nous [**définissons une fonction auxiliaire pour faciliter la lecture de l'image et l'application de l'augmentation d'image**].
L'argument `transform` fourni par le jeu de données
de PyTorch applique l'augmentation pour transformer les images.
Pour
une introduction détaillée à `DataLoader`, veuillez vous référer à :numref:`sec_fashion_mnist`.
:end_tab:


```{.python .input}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Entraînement multi-GPU

Nous entraînons le modèle ResNet-18 de
:numref:`sec_resnet` sur le jeu de données
CIFAR-10.
Rappelez-vous l'introduction à l'entraînement multi-GPU de
dans :numref:`sec_multi_gpu_concise`.
Dans ce qui suit,
[**nous définissons une fonction pour entraîner et évaluer le modèle en utilisant plusieurs GPU**].

```{.python .input}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `Vrai` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Maintenant, nous pouvons [**définir la fonction `train_with_data_aug` pour entraîner le modèle avec l'augmentation de l'image**].
Cette fonction récupère tous les GPU disponibles, 
utilise Adam comme algorithme d'optimisation,
applique l'augmentation d'image à l'ensemble de données d'entraînement,
et enfin appelle la fonction `train_ch13` qui vient d'être définie pour entraîner et évaluer le modèle.

```{.python .input}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

Entraînons [**le modèle**] en utilisant l'augmentation d'image basée sur le renversement gauche-droite aléatoire.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Résumé

* L'augmentation d'image génère des images aléatoires basées sur des données d'entraînement existantes pour améliorer la capacité de généralisation des modèles.
* Afin d'obtenir des résultats définitifs lors de la prédiction, nous n'appliquons généralement l'augmentation d'image qu'aux exemples d'entraînement et n'utilisons pas l'augmentation d'image avec des opérations aléatoires pendant la prédiction.
* Les cadres d'apprentissage profond fournissent de nombreuses méthodes d'augmentation d'image différentes, qui peuvent être appliquées simultanément.


## Exercices

1. Entraînez le modèle sans utiliser l'augmentation d'image :`train_with_data_aug(test_augs, test_augs)`. Comparez la précision de l'entraînement et du test en utilisant et en n'utilisant pas l'augmentation d'image. Cette expérience comparative peut-elle soutenir l'argument selon lequel l'augmentation d'image peut atténuer l'overfitting ? Pourquoi ?
1. Combinez plusieurs méthodes d'augmentation d'image différentes dans l'entrainement du modèle sur l'ensemble de données CIFAR-10. Cela améliore-t-il la précision du test ? 
1. Reportez-vous à la documentation en ligne du cadre d'apprentissage profond. Quelles sont les autres méthodes d'augmentation de l'image qu'il propose ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:
