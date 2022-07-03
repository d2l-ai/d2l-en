# Identification de races de chiens (ImageNet Dogs) sur Kaggle

Dans cette section, nous allons pratiquer
le problème d'identification de races de chiens sur
Kaggle. (**L'adresse web de cette compétition est https://www.kaggle.com/c/dog-breed-identification**)

Dans cette compétition,
120 races de chiens différentes seront reconnues.
En fait,
le jeu de données pour ce concours est
un sous-ensemble du jeu de données ImageNet.
Contrairement aux images du jeu de données CIFAR-10 dans :numref:`sec_kaggle_cifar10`,
les images du jeu de données ImageNet sont à la fois plus hautes et plus larges dans différentes dimensions.
:numref:`fig_kaggle_dog` montre les informations sur la page web du concours. Vous avez besoin d'un compte Kaggle
pour soumettre vos résultats.


![The dog breed identification competition website. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## Obtention et organisation du jeu de données

Le jeu de données de la compétition est divisé en un jeu d'entraînement et un jeu de test, qui contiennent respectivement 10222 et 10357 images JPEG
de trois canaux RVB (couleur).
Dans l'ensemble de données d'entraînement,
il y a 120 races de chiens
telles que les Labradors, les Caniches, les Teckels, les Samoyèdes, les Huskies, les Chihuahuas et les Yorkshire Terriers.


#### Téléchargement de l'ensemble de données

Après vous être connecté à Kaggle,
vous pouvez cliquer sur l'onglet "Données" sur la page Web de la compétition
illustrée dans :numref:`fig_kaggle_dog` et télécharger l'ensemble de données en cliquant sur le bouton "Télécharger tout".
Après avoir décompressé le fichier téléchargé dans `../data`, vous trouverez l'ensemble des données dans les chemins suivants :

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* .../data/dog-breed-identification/test

Vous avez peut-être remarqué que la structure ci-dessus est
similaire à celle du concours CIFAR-10 dans :numref:`sec_kaggle_cifar10`, où les dossiers `train/` et `test/` contiennent respectivement les images de chiens d'entraînement et de test, et `labels.csv` contient
les étiquettes des images d'entraînement.
De même, pour faciliter la prise en main, [**nous fournissons un petit échantillon de l'ensemble de données**] mentionné ci-dessus :`train_valid_test_tiny.zip`.
Si vous comptez utiliser le jeu de données complet pour le concours Kaggle, vous devez changer la variable `demo` ci-dessous en `False`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `Faux`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**Organiser l'ensemble de données**]

Nous pouvons organiser l'ensemble de données de la même manière que nous l'avons fait dans :numref:`sec_kaggle_cifar10`, à savoir séparer
un ensemble de validation de l'ensemble d'entraînement original, et déplacer les images dans des sous-dossiers regroupés par étiquettes.

La fonction `reorg_dog_data` ci-dessous lit
les étiquettes des données d'entraînement, sépare l'ensemble de validation et organise l'ensemble d'entraînement.

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**Image Augmentation**]

Rappelons que cet ensemble de données sur les races de chiens
est un sous-ensemble de l'ensemble de données ImageNet,
dont les images
sont plus grandes que celles de l'ensemble de données CIFAR-10
dans :numref:`sec_kaggle_cifar10`.
La liste suivante
énumère quelques opérations d'augmentation d'image
qui pourraient être utiles pour des images relativement plus grandes.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

Pendant la prédiction,
nous utilisons uniquement des opérations de prétraitement d'images
sans caractère aléatoire.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**Lire le jeu de données**]

Comme dans :numref:`sec_kaggle_cifar10`,
nous pouvons lire le jeu de données organisé
composé de fichiers d'images brutes.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

Ci-dessous, nous créons des instances d'itérateurs de données
de la même manière
que dans :numref:`sec_kaggle_cifar10`.

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

## [**Réglage fin d'un modèle pré-entraîné**]

Encore une fois,
le jeu de données pour cette compétition est un sous-ensemble du jeu de données ImageNet.
Par conséquent, nous pouvons utiliser l'approche décrite à l'adresse
:numref:`sec_fine_tuning` 
pour sélectionner un modèle pré-entraîné sur le jeu de données ImageNet complet
et l'utiliser pour extraire des caractéristiques d'image à introduire dans un réseau de sortie personnalisé à petite échelle
.
Les API de haut niveau des cadres d'apprentissage profond
fournissent un large éventail de modèles
pré-entraînés sur le jeu de données ImageNet.
Ici, nous choisissons
un modèle ResNet-34 pré-entraîné,
où nous réutilisons simplement
l'entrée de la couche de sortie de ce modèle
(c'est-à-dire les caractéristiques extraites
).
Nous pouvons ensuite remplacer la couche de sortie d'origine par un petit réseau de sortie personnalisé
qui peut être entraîné,
tel que l'empilement de deux couches entièrement connectées
.
Contrairement à l'expérience menée sur
:numref:`sec_fine_tuning`,
l'expérience suivante
ne réentraîne pas le modèle pré-entraîné utilisé pour l'extraction des caractéristiques.
Cela réduit le temps d'entraînement et la mémoire
pour le stockage des gradients.

Rappelez-vous que nous avons normalisé
les images en utilisant
les moyennes et les écarts types des trois canaux RVB pour l'ensemble complet de données ImageNet.
En fait,
cela est également cohérent avec l'opération de normalisation
effectuée par le modèle pré-entraîné sur ImageNet.

```{.python .input}
#@tab mxnet
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

Avant de [**calculer la perte**],
nous obtenons d'abord l'entrée de la couche de sortie du modèle pré-entraîné, c'est-à-dire la caractéristique extraite.
Ensuite, nous utilisons cette caractéristique comme entrée pour notre petit réseau de sortie personnalisé afin de calculer la perte.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## Définir [**la fonction d'entraînement**]

Nous allons sélectionner le modèle et régler les hyperparamètres en fonction des performances du modèle sur l'ensemble de validation. La fonction d'entraînement du modèle `train` ne fait que
itérer les paramètres du petit réseau de sortie personnalisé.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Entrainement et validation du modèle**]

Nous pouvons maintenant former et valider le modèle.
Les hyperparamètres suivants sont tous réglables.
Par exemple, le nombre d'époques peut être augmenté. Comme `lr_period` et `lr_decay` sont respectivement définis sur 2 et 0,9, le taux d'apprentissage de l'algorithme d'optimisation sera multiplié par 0,9 toutes les 2 époques.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**Classifier l'ensemble de test**] et soumettre les résultats sur Kaggle


Similaire à l'étape finale dans :numref:`sec_kaggle_cifar10`,
à la fin toutes les données étiquetées (y compris l'ensemble de validation) sont utilisées pour l'entraînement du modèle et la classification de l'ensemble de test.
Nous utiliserons le réseau de sortie personnalisé formé
pour la classification.

```{.python .input}
#@tab mxnet
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

Le code ci-dessus
va générer un fichier `submission.csv`
 à soumettre
à Kaggle de la même manière que celle décrite dans :numref:`sec_kaggle_house`.


## Résumé


* Les images de l'ensemble de données ImageNet sont plus grandes (avec des dimensions variables) que les images CIFAR-10. Nous pouvons modifier les opérations d'augmentation d'image pour des tâches sur un jeu de données différent.
* Pour classer un sous-ensemble du jeu de données ImageNet, nous pouvons utiliser des modèles pré-entraînés sur l'ensemble du jeu de données ImageNet pour extraire des caractéristiques et n'entraîner qu'un réseau de sortie personnalisé à petite échelle. Cela permettra de réduire le temps de calcul et le coût de la mémoire.


## Exercices

1. En utilisant le jeu de données complet de la compétition Kaggle, quels résultats pouvez-vous obtenir en augmentant `batch_size` (taille du lot) et `num_epochs` (nombre d'époques) tout en fixant d'autres hyperparamètres comme `lr = 0.01`, `lr_period = 10` et `lr_decay = 0.1`?
1. Obtenez-vous de meilleurs résultats si vous utilisez un modèle pré-entraîné plus profond ? Comment ajuster les hyperparamètres ? Pouvez-vous encore améliorer les résultats ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1481)
:end_tab:
