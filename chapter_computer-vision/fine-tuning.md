# Ajustement fin
:label:`sec_fine_tuning` 

 Dans les chapitres précédents, nous avons expliqué comment entraîner des modèles sur le jeu de données d'entraînement Fashion-MNIST contenant seulement 60000 images. Nous avons également décrit ImageNet, le jeu de données d'images à grande échelle le plus utilisé dans le monde universitaire, qui contient plus de 10 millions d'images et 1000 objets. Cependant, la taille du jeu de données que nous rencontrons habituellement se situe entre celles de ces deux jeux de données.


Supposons que nous voulions reconnaître différents types de chaises à partir d'images, puis recommander des liens d'achat aux utilisateurs. 
Une méthode possible est d'abord d'identifier
100 chaises courantes,
de prendre 1000 images de différents angles pour chaque chaise, 
et ensuite d'entraîner un modèle de classification sur le jeu de données d'images collectées.
Bien que ce jeu de données de chaises soit plus important que le jeu de données Fashion-MNIST,
le nombre d'exemples est toujours inférieur à un dixième de 
celui d'ImageNet.
Cela peut conduire à un ajustement excessif des modèles complexes 
qui sont adaptés à ImageNet sur ce jeu de données de chaises.
En outre, en raison de la quantité limitée d'exemples d'entraînement,
la précision du modèle entraîné
peut ne pas répondre aux exigences pratiques.


Afin de résoudre les problèmes ci-dessus,
une solution évidente est de collecter plus de données.
Cependant, la collecte et l'étiquetage des données peuvent prendre beaucoup de temps et d'argent.
Par exemple, pour collecter le jeu de données ImageNet, les chercheurs ont dépensé des millions de dollars provenant de fonds de recherche.
Bien que le coût actuel de la collecte de données ait été considérablement réduit, ce coût ne peut toujours pas être ignoré.


Une autre solution consiste à appliquer l'apprentissage par transfert * pour transférer les connaissances apprises de l'ensemble de données source * à l'ensemble de données cible *.
Par exemple, bien que la plupart des images de l'ensemble de données ImageNet n'aient rien à voir avec les chaises, le modèle formé sur cet ensemble de données peut extraire des caractéristiques d'image plus générales, qui peuvent aider à identifier les bords, les textures, les formes et la composition des objets.
Ces caractéristiques similaires peuvent
également être efficaces pour reconnaître les chaises.


## Étapes


 Dans cette section, nous allons introduire une technique courante dans l'apprentissage par transfert : le *réglage fin*. Comme le montre le site :numref:`fig_finetune` , l'ajustement fin comprend les quatre étapes suivantes :


 1. Préentraîner un modèle de réseau neuronal, c'est-à-dire le *modèle source*, sur un jeu de données source (par exemple, le jeu de données ImageNet).
1. Créez un nouveau modèle de réseau neuronal, c'est-à-dire le *modèle cible*. Cette opération copie tous les modèles et leurs paramètres sur le modèle source, à l'exception de la couche de sortie. Nous supposons que ces paramètres de modèle contiennent les connaissances acquises à partir de l'ensemble de données source et que ces connaissances seront également applicables à l'ensemble de données cible. Nous supposons également que la couche de sortie du modèle source est étroitement liée aux étiquettes de l'ensemble de données source ; elle n'est donc pas utilisée dans le modèle cible.
1. Ajoutez une couche de sortie au modèle cible, dont le nombre de sorties est le nombre de catégories dans l'ensemble de données cible. Puis initialiser aléatoirement les paramètres du modèle de cette couche.
1. Entraînez le modèle cible sur le jeu de données cible, tel qu'un jeu de données de chaises. La couche de sortie sera formée à partir de zéro, tandis que les paramètres de toutes les autres couches sont affinés sur la base des paramètres du modèle source.

![Fine tuning.](../img/finetune.svg)
:label:`fig_finetune`

Lorsque les jeux de données cibles sont beaucoup plus petits que les jeux de données sources, l'ajustement fin permet d'améliorer la capacité de généralisation des modèles.


## Reconnaissance de hot-dogs

Démontrons l'ajustement fin par un cas concret :
reconnaissance de hot-dogs. 
Nous allons affiner un modèle ResNet sur un petit jeu de données,
qui a été pré-entraîné sur le jeu de données ImageNet.
Ce petit jeu de données est composé de
milliers d'images avec et sans hot-dogs.
Nous utiliserons le modèle affiné pour reconnaître 
hot dogs à partir d'images.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### Lecture du jeu de données

[**Le jeu de données de hot-dogs que nous utilisons provient d'images en ligne**].
Ce jeu de données est composé de
1400 images de classe positive contenant des hot dogs,
et autant d'images de classe négative contenant d'autres aliments.
1000 images des deux classes sont utilisées pour l'entrainement et le reste pour les tests.


Après avoir décompressé le jeu de données téléchargé,
nous obtenons deux dossiers `hotdog/train` et `hotdog/test`. Les deux dossiers ont des sous-dossiers `hotdog` et `not-hotdog`, chacun d'entre eux contenant des images de
la classe correspondante.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

Nous créons deux instances pour lire tous les fichiers d'images dans les ensembles de données de entrainement et de test, respectivement.

```{.python .input}
#@tab mxnet
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

Les 8 premiers exemples positifs et les 8 dernières images négatives sont présentés ci-dessous. Comme vous pouvez le constater, [**les images varient en taille et en rapport d'aspect**].

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

Pendant l'apprentissage, nous découpons d'abord une zone aléatoire de taille et de rapport hauteur/largeur aléatoires dans l'image
, puis nous mettons à l'échelle cette zone
pour obtenir une image d'entrée $224 \times 224$. 
Lors du test, nous mettons à l'échelle la hauteur et la largeur d'une image à 256 pixels, puis nous découpons une zone centrale $224 \times 224$ en entrée.
En outre, 
pour les trois canaux de couleur RVB (rouge, vert et bleu)
nous *normalisons* leurs valeurs canal par canal.
Concrètement,
la valeur moyenne d'un canal est soustraite de chaque valeur de ce canal, puis le résultat est divisé par l'écart-type de ce canal.

[~)~)Augmentations des données~)~)]

```{.python .input}
#@tab mxnet
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**Définir et initialiser le modèle**]

Nous utilisons ResNet-18, qui a été pré-entraîné sur le jeu de données ImageNet, comme modèle source. Ici, nous spécifions `pretrained=True` pour télécharger automatiquement les paramètres du modèle pré-entraîné. 
Si ce modèle est utilisé pour la première fois, une connexion Internet
est nécessaire pour le téléchargement.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
L'instance du modèle source pré-entraîné contient deux variables membres :`features` et `output`. La première contient toutes les couches du modèle sauf la couche de sortie, et la seconde est la couche de sortie du modèle. 
L'objectif principal de cette division est de faciliter le réglage fin des paramètres du modèle de toutes les couches sauf la couche de sortie. La variable membre `output` du modèle source est présentée ci-dessous.
:end_tab:

:begin_tab:`pytorch`
L'instance de modèle source pré-entraînée contient un certain nombre de couches de caractéristiques et une couche de sortie `fc`.
L'objectif principal de cette division est de faciliter le réglage fin des paramètres du modèle de toutes les couches sauf la couche de sortie. La variable membre `fc` du modèle source est indiquée ci-dessous.
:end_tab:

```{.python .input}
#@tab mxnet
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

En tant que couche entièrement connectée, elle transforme les sorties du pooling moyen global final de ResNet en 1000 sorties de classe de l'ensemble de données ImageNet.
Nous construisons ensuite un nouveau réseau neuronal en tant que modèle cible. Il est défini de la même manière que le modèle source pré-entraîné, sauf que
son nombre de sorties dans la couche finale
est fixé à
le nombre de classes dans le jeu de données cible (plutôt que 1000).

Dans le code ci-dessous, les paramètres du modèle avant la couche de sortie de l'instance du modèle cible `finetune_net` sont initialisés aux paramètres du modèle des couches correspondantes du modèle source.
Puisque ces paramètres de modèle ont été obtenus via un pré-entraînement sur ImageNet, 
ils sont efficaces.
Par conséquent, nous pouvons seulement utiliser 
un petit taux d'apprentissage pour *affiner* ces paramètres pré-entraînés.
En revanche, les paramètres du modèle dans la couche de sortie sont initialisés de manière aléatoire et nécessitent généralement un taux d'apprentissage plus élevé pour être appris à partir de zéro.
Si le taux d'apprentissage de base est $\eta$, un taux d'apprentissage de $10\eta$ sera utilisé pour itérer les paramètres du modèle dans la couche de sortie.

```{.python .input}
#@tab mxnet
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**Réglage fin du modèle**]

Tout d'abord, nous définissons une fonction d'apprentissage `train_fine_tuning` qui utilise le réglage fin pour pouvoir être appelée plusieurs fois.

```{.python .input}
#@tab mxnet
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

Nous [**fixons le taux d'apprentissage de base à une petite valeur**]
afin de *régler finement* les paramètres du modèle obtenus via le pré-entraînement. Sur la base des paramètres précédents, nous allons entraîner les paramètres de la couche de sortie du modèle cible à partir de zéro en utilisant un taux d'apprentissage dix fois supérieur.

```{.python .input}
#@tab mxnet
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**Pour comparaison,**] nous définissons un modèle identique, mais (**initialisons tous ses paramètres de modèle à des valeurs aléatoires**). Puisque le modèle entier doit être entraîné à partir de zéro, nous pouvons utiliser un taux d'apprentissage plus élevé.

```{.python .input}
#@tab mxnet
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

Comme nous pouvons le constater, le modèle finement ajusté tend à être plus performant pour la même époque
car les valeurs initiales de ses paramètres sont plus efficaces.


## Résumé

* L'apprentissage par transfert transfère les connaissances acquises de l'ensemble de données source à l'ensemble de données cible. Le réglage fin est une technique courante pour l'apprentissage par transfert.
* Le modèle cible copie tous les modèles et leurs paramètres du modèle source, à l'exception de la couche de sortie, et ajuste ces paramètres en fonction de l'ensemble de données cible. En revanche, la couche de sortie du modèle cible doit être formée à partir de zéro.
* En général, l'ajustement fin des paramètres utilise un taux d'apprentissage plus faible, tandis que l'entrainement de la couche de sortie à partir de zéro peut utiliser un taux d'apprentissage plus élevé.


## Exercices

1. Continuez à augmenter le taux d'apprentissage de `finetune_net`. Comment la précision du modèle change-t-elle ?
2. Ajustez encore les hyperparamètres de `finetune_net` et `scratch_net` dans l'expérience comparative. Leur précision est-elle toujours différente ?
3. Réglez les paramètres avant la couche de sortie de `finetune_net` sur ceux du modèle source et ne les mettez *pas* à jour pendant la formation. Comment la précision du modèle change-t-elle ? Vous pouvez utiliser le code suivant.

```{.python .input}
#@tab mxnet
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. En fait, il existe une classe "hotdog" dans l'ensemble de données `ImageNet`. Son paramètre de poids correspondant dans la couche de sortie peut être obtenu via le code suivant. Comment pouvons-nous tirer parti de ce paramètre de poids ?

```{.python .input}
#@tab mxnet
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1439)
:end_tab:
