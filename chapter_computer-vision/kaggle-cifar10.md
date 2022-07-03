# Classification d'images (CIFAR-10) sur Kaggle
:label:`sec_kaggle_cifar10` 

Jusqu'à présent, nous avons utilisé les API de haut niveau des cadres d'apprentissage profond pour obtenir directement des ensembles de données d'images au format tenseur.
Cependant, les jeux de données d'images personnalisés
se présentent souvent sous la forme de fichiers d'images.
Dans cette section, nous allons partir de
fichiers d'images brutes,
et les organiser, les lire, puis les transformer
en format tenseur étape par étape.

Nous avons expérimenté avec le jeu de données CIFAR-10 dans :numref:`sec_image_augmentation`,
qui est un jeu de données important en vision par ordinateur.
Dans cette section,
nous allons appliquer les connaissances acquises
dans les sections précédentes
pour pratiquer la compétition Kaggle de
classification d'images CIFAR-10.
(**L'adresse web de la compétition est https://www.kaggle.com/c/cifar-10**)

:numref:`fig_kaggle_cifar10` montre les informations sur la page web de la compétition.
Afin de soumettre les résultats,
vous devez enregistrer un compte Kaggle.

![CIFAR-10 image classification competition webpage information. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

## Obtention et organisation du jeu de données

Le jeu de données de la compétition est divisé en
un jeu d'entraînement et un jeu de test,
qui contiennent respectivement 50 000 et 300 000 images.
Dans l'ensemble de test,
10000 images seront utilisées pour l'évaluation,
tandis que les 290000 images restantes
ne seront pas évaluées :
elles sont incluses uniquement
pour qu'il soit difficile
de tricher avec
*manuellement* les résultats étiquetés de l'ensemble de test.
Les images de ce jeu de données
sont toutes des fichiers d'images en couleur (canaux RVB) au format png,
dont la hauteur et la largeur sont toutes deux de 32 pixels.
Les images couvrent un total de 10 catégories, à savoir avions, voitures, oiseaux, chats, cerfs, chiens, grenouilles, chevaux, bateaux et camions.
Le coin supérieur gauche de :numref:`fig_kaggle_cifar10` montre quelques images d'avions, de voitures et d'oiseaux de l'ensemble de données.


#### Téléchargement du jeu de données

Après s'être connecté à Kaggle, nous pouvons cliquer sur l'onglet "Données" de la page Web du concours de classification d'images CIFAR-10 illustré sur :numref:`fig_kaggle_cifar10` et télécharger le jeu de données en cliquant sur le bouton "Télécharger tout".
Après avoir décompressé le fichier téléchargé dans `../data`, et décompressé `train.7z` et `test.7z` à l'intérieur, vous trouverez l'ensemble des données dans les chemins suivants :

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

où les répertoires `train` et `test` contiennent respectivement les images d'entraînement et de test, `trainLabels.csv` fournit les étiquettes pour les images d'entraînement, et `sample_submission.csv` est un exemple de fichier de soumission.

Pour faciliter la prise en main, [**nous fournissons un échantillon à petite échelle de l'ensemble de données qui
contient les 1000 premières images d'entraînement et 5 images de test aléatoires.**]
Pour utiliser l'ensemble de données complet du concours Kaggle, vous devez définir la variable `demo` sur `False`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# If you use the full dataset downloaded for the Kaggle competition, set
# `demo` to False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**Organiser le jeu de données**]

Nous devons organiser les jeux de données pour faciliter l'apprentissage et le test des modèles.
Commençons par lire les étiquettes du fichier csv.
La fonction suivante renvoie un dictionnaire qui fait correspondre
la partie sans extension du nom de fichier à son étiquette.

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# training examples:', len(labels))
print('# classes:', len(set(labels.values())))
```

Ensuite, nous définissons la fonction `reorg_train_valid` pour [**séparer l'ensemble de validation de l'ensemble de entrainement original.**]
L'argument `valid_ratio` de cette fonction est le rapport entre le nombre d'exemples de l'ensemble de validation et le nombre d'exemples de l'ensemble de entrainement original.
Plus concrètement,
laisse $n$ être le nombre d'images de la classe ayant le moins d'exemples, et $r$ être le ratio.
L'ensemble de validation répartira les images
$\max(\lfloor nr\rfloor,1)$ pour chaque classe.
Prenons l'exemple de `valid_ratio=0.1`. Étant donné que l'ensemble d'entraînement original comporte 50 000 images,
45 000 images seront utilisées pour l'entraînement dans le chemin `train_valid_test/train`,
tandis que les 5 000 autres images seront réparties dans
comme ensemble de validation dans le chemin `train_valid_test/valid`. Après avoir organisé l'ensemble de données, les images de la même classe seront placées dans le même dossier.

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

La fonction `reorg_test` ci-dessous [**organise l'ensemble de test pour le chargement des données pendant la prédiction.**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

Enfin, nous utilisons une fonction pour [**invoquer**]
les fonctions `read_csv_labels`, `reorg_train_valid`, et `reorg_test` (**définies ci-dessus.**)

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

Ici, nous avons uniquement fixé la taille du lot à 32 pour l'échantillon à petite échelle de l'ensemble de données.
Lors de l'entraînement et du test de
l'ensemble complet de données de la compétition Kaggle,
`batch_size` doit être fixé à un nombre entier plus grand, tel que 128.
Nous avons séparé 10% des exemples d'entraînement comme ensemble de validation pour le réglage des hyperparamètres.

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**Augmentation d'image**]

Nous utilisons l'augmentation d'image pour traiter l'ajustement excessif.
Par exemple, les images peuvent être retournées horizontalement de manière aléatoire pendant la formation.
Nous pouvons également effectuer une normalisation pour les trois canaux RVB des images couleur. Vous trouverez ci-dessous une liste de certaines de ces opérations que vous pouvez modifier.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    gluon.data.vision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    torchvision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

Pendant les tests,
nous n'effectuons la normalisation que sur les images
afin de
supprimer le caractère aléatoire des résultats de l'évaluation.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## Lecture du jeu de données

Ensuite, nous [**lisons le jeu de données organisé composé de fichiers d'images brutes**]. Chaque exemple comprend une image et une étiquette.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

Pendant la formation,
nous devons [**spécifier toutes les opérations d'augmentation d'image définies ci-dessus**].
Lorsque l'ensemble de validation
est utilisé pour l'évaluation du modèle lors de l'ajustement des hyperparamètres,
aucun aléa provenant de l'augmentation de l'image ne doit être introduit.
Avant la prédiction finale,
nous entraînons le modèle sur l'ensemble d'entraînement et l'ensemble de validation combinés afin d'utiliser pleinement toutes les données étiquetées.

```{.python .input}
#@tab mxnet
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

### Définition du [**Modèle**]

:begin_tab:`mxnet` 
Ici, nous construisons les blocs résiduels en fonction de la classe `HybridBlock`, ce qui est
légèrement différent de l'implémentation décrite dans
:numref:`sec_resnet`.
Cela permet d'améliorer l'efficacité des calculs.
:end_tab:

```{.python .input}
#@tab mxnet
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
Ensuite, nous définissons le modèle ResNet-18.
:end_tab:

```{.python .input}
#@tab mxnet
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
Nous utilisons l'initialisation de Xavier décrite dans :numref:`subsec_xavier` avant de commencer la formation.
:end_tab:

:begin_tab:`pytorch`
Nous définissons le modèle ResNet-18 décrit dans
:numref:`sec_resnet`.
:end_tab:

```{.python .input}
#@tab mxnet
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## Définition de la [**Fonction d'entraînement**]

Nous allons sélectionner les modèles et régler les hyperparamètres en fonction de la performance du modèle sur l'ensemble de validation.
Dans ce qui suit, nous définissons la fonction d'apprentissage du modèle `train`.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Formation et validation du modèle**]

Maintenant, nous pouvons former et valider le modèle.
Tous les hyperparamètres suivants peuvent être ajustés.
Par exemple, nous pouvons augmenter le nombre d'époques.
Lorsque `lr_period` et `lr_decay` sont définis sur 4 et 0,9, respectivement, le taux d'apprentissage de l'algorithme d'optimisation sera multiplié par 0,9 toutes les 4 époques. Pour faciliter la démonstration,
nous n'entraînons ici que 20 époques.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
net(next(iter(train_iter))[0])
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**Classifier l'ensemble de test**] et soumettre les résultats sur Kaggle

Après avoir obtenu un modèle prometteur avec des hyperparamètres,
nous utilisons toutes les données étiquetées (y compris l'ensemble de validation) pour réentraîner le modèle et classifier l'ensemble de test.

```{.python .input}
#@tab mxnet
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
net(next(iter(train_valid_iter))[0])
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

Le code ci-dessus
va générer un fichier `submission.csv`,
dont le format
répond aux exigences du concours Kaggle.
La méthode
pour soumettre les résultats à Kaggle
est similaire à celle de :numref:`sec_kaggle_house`.

## Résumé

* Nous pouvons lire des ensembles de données contenant des fichiers d'images brutes après les avoir organisés dans le format requis.

:begin_tab:`mxnet`
* Nous pouvons utiliser les réseaux de neurones convolutifs, l'augmentation d'image et la programmation hybride dans une compétition de classification d'images.
:end_tab:

:begin_tab:`pytorch`
* Nous pouvons utiliser les réseaux de neurones convolutifs et l'augmentation d'image dans un concours de classification d'images.
:end_tab:

## Exercices

1. Utilisez le jeu de données CIFAR-10 complet pour cette compétition Kaggle. Définissez les hyperparamètres comme suit :`batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50` et `lr_decay = 0.1`.  Voyez quelle précision et quel classement vous pouvez obtenir dans cette compétition. Pouvez-vous encore les améliorer ?
1. Quelle précision pouvez-vous obtenir sans utiliser l'augmentation d'image ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1479)
:end_tab:
