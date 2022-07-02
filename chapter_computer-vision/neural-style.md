# Transfert de style neuronal

Si vous êtes un passionné de photographie,
vous connaissez peut-être ce filtre.
Il peut modifier le style de couleur des photos
de sorte que les photos de paysages deviennent plus nettes
ou que les photos de portraits soient blanchies.
Cependant,
un filtre ne modifie généralement
qu'un seul aspect de la photo.
Pour appliquer un style idéal
à une photo,
vous devrez probablement
essayer de nombreuses combinaisons de filtres différentes.
Ce processus est
aussi complexe que le réglage des hyperparamètres d'un modèle.



Dans cette section, nous allons
exploiter les représentations par couches d'un CNN
pour appliquer automatiquement le style d'une image
à une autre image, c'est-à-dire le *transfert de style* :cite:`Gatys.Ecker.Bethge.2016` .
Cette tâche nécessite deux images d'entrée :
l'une est l'image *de contenu* et
l'autre est l'image *de style*.
Nous utiliserons les réseaux neuronaux
pour modifier l'image de contenu
afin de la rendre proche de l'image de style en termes de style.
Par exemple,
l'image de contenu dans :numref:`fig_style_transfer` est une photo de paysage prise par nous
dans le parc national du Mont Rainier dans la banlieue de Seattle, tandis que l'image de style est une peinture à l'huile
sur le thème des chênes d'automne.
Dans l'image synthétisée de sortie,
les coups de pinceau à l'huile de l'image de style
sont appliqués, ce qui donne des couleurs plus vives,
tout en préservant la forme principale des objets
dans l'image de contenu.

![Given content and style images, style transfer outputs a synthesized image.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## Méthode

:numref:`fig_style_transfer_model` illustre
la méthode de transfert de style basée sur CNN à l'aide d'un exemple simplifié.
Tout d'abord, nous initialisons l'image synthétisée,
par exemple, dans l'image de contenu.
Cette image synthétisée est la seule variable qui doit être mise à jour pendant le processus de transfert de style,
c'est-à-dire les paramètres du modèle à mettre à jour pendant l'apprentissage.
Nous choisissons ensuite un CNN pré-entraîné
pour extraire les caractéristiques de l'image et figer les paramètres de son modèle
pendant l'apprentissage.
Ce CNN profond utilise plusieurs couches
pour extraire
des caractéristiques hiérarchiques des images.
Nous pouvons choisir la sortie de certaines de ces couches comme caractéristiques de contenu ou de style.
Prenons l'exemple de :numref:`fig_style_transfer_model` .
Le réseau neuronal pré-entraîné comporte ici 3 couches convolutionnelles,
où la deuxième couche produit les caractéristiques de contenu,
et les première et troisième couches produisent les caractéristiques de style.

![CNN-based style transfer process. Solid lines show the direction of forward propagation and dotted lines show backward propagation. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`

Ensuite, nous calculons la fonction de perte du transfert de style par propagation avant (direction des flèches pleines), et nous mettons à jour les paramètres du modèle (l'image synthétisée en sortie) par rétropropagation (direction des flèches pointillées).
La fonction de perte couramment utilisée dans le transfert de style se compose de trois parties :
(i) *perte de contenu* rend l'image synthétisée et l'image de contenu proches en termes de caractéristiques de contenu ;
(ii) *perte de style* rend l'image synthétisée et l'image de style proches en termes de caractéristiques de style ;
et (iii) *perte de variation totale* aide à réduire le bruit dans l'image synthétisée.
Enfin, lorsque l'apprentissage du modèle est terminé, nous sortons les paramètres du modèle de transfert de style pour générer
l'image synthétisée finale.



Dans ce qui suit,
nous expliquerons les détails techniques du transfert de style via une expérience concrète.


## [**Lecture des images de contenu et de style**]

Tout d'abord, nous lisons les images de contenu et de style.
D'après leurs axes de coordonnées imprimés,
nous pouvons dire que ces images ont des tailles différentes.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
#@tab mxnet
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**Prétraitement et post-traitement**]

Ci-dessous, nous définissons deux fonctions pour le prétraitement et le post-traitement des images.
La fonction `preprocess` normalise
chacun des trois canaux RVB de l'image d'entrée et transforme les résultats au format d'entrée CNN.
La fonction `postprocess` rétablit les valeurs des pixels de l'image de sortie à leur valeur originale avant la normalisation.
Étant donné que la fonction d'impression d'image exige que chaque pixel ait une valeur à virgule flottante comprise entre 0 et 1,
nous remplaçons toute valeur inférieure à 0 ou supérieure à 1 par 0 ou 1, respectivement.

```{.python .input}
#@tab mxnet
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [**Extraction de caractéristiques**]

Nous utilisons le modèle VGG-19 pré-entraîné sur le jeu de données ImageNet pour extraire les caractéristiques des images :cite:`Gatys.Ecker.Bethge.2016` .

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

Afin d'extraire les caractéristiques de contenu et de style de l'image, nous pouvons sélectionner la sortie de certaines couches du réseau VGG.
D'une manière générale, plus la couche d'entrée est proche, plus il est facile d'extraire les détails de l'image, et inversement, plus il est facile d'extraire l'information globale de l'image. Afin d'éviter de retenir excessivement
les détails de l'image de contenu dans l'image synthétisée,
nous choisissons une couche VGG plus proche de la sortie comme *couche de contenu* pour sortir les caractéristiques de contenu de l'image.
Nous sélectionnons également la sortie de différentes couches VGG pour extraire les caractéristiques de style locales et globales.
Ces couches sont également appelées *couches de style*.
Comme mentionné dans :numref:`sec_vgg` ,
le réseau VGG utilise 5 blocs convolutifs.
Dans l'expérience, nous choisissons la dernière couche convolutive du quatrième bloc convolutif comme couche de contenu, et la première couche convolutive de chaque bloc convolutif comme couche de style.
Les indices de ces couches peuvent être obtenus en imprimant l'instance `pretrained_net`.

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

Lors de l'extraction de caractéristiques à l'aide de couches VGG,
nous n'avons besoin d'utiliser que toutes celles
de la couche d'entrée à la couche de contenu ou à la couche de style qui est la plus proche de la couche de sortie.
Construisons une nouvelle instance de réseau `net`, qui ne retient que toutes les couches VGG à utiliser pour l'extraction de caractéristiques (
).

```{.python .input}
#@tab mxnet
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

Étant donné l'entrée `X`, si nous invoquons simplement
la propagation vers l'avant `net(X)`, nous ne pouvons obtenir que la sortie de la dernière couche.
Comme nous avons également besoin des sorties des couches intermédiaires,
nous devons effectuer un calcul couche par couche et conserver
les sorties des couches de contenu et de style.

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

Deux fonctions sont définies ci-dessous :
la fonction `get_contents` extrait les caractéristiques de contenu de l'image de contenu,
et la fonction `get_styles` extrait les caractéristiques de style de l'image de style.
Comme il n'est pas nécessaire de mettre à jour les paramètres du modèle du VGG pré-entraîné pendant l'apprentissage,
nous pouvons extraire les caractéristiques de contenu et de style
avant même le début de l'apprentissage.
Comme l'image synthétisée
est un ensemble de paramètres de modèle à mettre à jour
pour le transfert de style,
nous ne pouvons extraire le contenu et les caractéristiques de style de l'image synthétisée qu'en appelant la fonction `extract_features` pendant la formation.

```{.python .input}
#@tab mxnet
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**Définir la fonction de perte**]

Nous allons maintenant décrire la fonction de perte pour le transfert de style. La fonction de perte se compose de
la perte de contenu, la perte de style et la perte de variation totale.

### Perte de contenu

Semblable à la fonction de perte dans la régression linéaire,
la perte de contenu mesure la différence
dans les caractéristiques du contenu
entre l'image synthétisée et l'image de contenu via
la fonction de perte au carré.
Les deux entrées de la fonction de perte au carré
sont les deux sorties
de la couche de contenu calculée par la fonction `extract_features`.

```{.python .input}
#@tab mxnet
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

#### Perte de style

La perte de style, similaire à la perte de contenu,
utilise également la fonction de perte au carré pour mesurer la différence de style entre l'image synthétisée et l'image de style.
Pour exprimer la sortie de style de toute couche de style,
nous utilisons d'abord la fonction `extract_features` pour
calculer la sortie de la couche de style.
Supposons que la sortie ait
1 exemple, $c$ canaux,
hauteur $h$, et largeur $w$,
nous pouvons transformer cette sortie en
matrice $\mathbf{X}$ avec $c$ lignes et $hw$ colonnes.
Cette matrice peut être considérée comme
la concaténation de
$c$ vecteurs $\mathbf{x}_1, \ldots, \mathbf{x}_c$,
dont chacun a une longueur de $hw$.
Ici, le vecteur $\mathbf{x}_i$ représente la caractéristique de style du canal $i$.

Dans la *matrice de Gram* de ces vecteurs $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$, l'élément $x_{ij}$ dans la ligne $i$ et la colonne $j$ est le produit scalaire des vecteurs $\mathbf{x}_i$ et $\mathbf{x}_j$.
Il représente la corrélation des caractéristiques de style des canaux $i$ et $j$.
Nous utilisons cette matrice de Gram pour représenter la sortie de style de toute couche de style.
Notez que lorsque la valeur de $hw$ est plus grande que celle de
, cela entraîne probablement des valeurs plus grandes dans la matrice de Gram.
Notez également que la hauteur et la largeur de la matrice de Gram sont toutes deux le nombre de canaux $c$.
Pour que la perte de style ne soit pas affectée
par ces valeurs,
la fonction `gram` ci-dessous divise
la matrice de Gram par le nombre de ses éléments, c'est-à-dire $chw$.

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

Évidemment,
les deux entrées de la matrice de Gram de la fonction de perte au carré pour la perte de style sont basées sur
les sorties de la couche de style pour
l'image synthétisée et l'image de style.
On suppose ici que la matrice de Gram `gram_Y` basée sur l'image de style a été précalculée.

```{.python .input}
#@tab mxnet
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### Perte totale de variation

Parfois, l'image synthétisée apprise
présente beaucoup de bruit à haute fréquence,
c'est-à-dire des pixels particulièrement clairs ou sombres.
Une méthode courante de réduction du bruit est le débruitage par variation totale
**.
Désignez par $x_{i, j}$ la valeur du pixel à la coordonnée $(i, j)$.
La réduction de la perte de variation totale

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$ 

 rapproche les valeurs des pixels voisins sur l'image synthétisée.

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### Fonction de perte

[**La fonction de perte du transfert de style est la somme pondérée de la perte de contenu, de la perte de style et de la perte de variation totale**].
En ajustant ces hyperparamètres de pondération,
nous pouvons équilibrer entre
la conservation du contenu,
le transfert de style,
et la réduction du bruit sur l'image synthétisée.

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**Initialisation de l'image synthétisée**]

Dans le transfert de style,
l'image synthétisée est la seule variable qui doit être mise à jour pendant la formation.
Ainsi, nous pouvons définir un modèle simple, `SynthesizedImage`, et traiter l'image synthétisée comme les paramètres du modèle.
Dans ce modèle, la propagation directe renvoie simplement les paramètres du modèle.

```{.python .input}
#@tab mxnet
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

Ensuite, nous définissons la fonction `get_inits`.
Cette fonction crée une instance de modèle d'image synthétisée et l'initialise à l'image `X`.
Les matrices de Gram pour l'image de style aux différentes couches de style, `styles_Y_gram`, sont calculées avant l'apprentissage.

```{.python .input}
#@tab mxnet
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**Training**]


 Lors de l'apprentissage du modèle de transfert de style,
nous extrayons continuellement
les caractéristiques de contenu et les caractéristiques de style de l'image synthétisée, et nous calculons la fonction de perte.
La boucle d'apprentissage est définie ci-dessous.

```{.python .input}
#@tab mxnet
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

Maintenant, nous [**commençons à former le modèle**].
Nous redimensionnons la hauteur et la largeur des images de contenu et de style à 300 par 450 pixels.
Nous utilisons l'image de contenu pour initialiser l'image synthétisée.

```{.python .input}
#@tab mxnet
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

Nous pouvons constater que l'image synthétisée
conserve le paysage et les objets de l'image de contenu,
et transfère en même temps la couleur de l'image de style
.
Par exemple,
l'image synthétisée a des blocs de couleur comme
ceux de l'image de style.
Certains de ces blocs ont même la texture subtile des coups de pinceau.




## Résumé

* La fonction de perte couramment utilisée dans le transfert de style se compose de trois parties : (i) la perte de contenu rend l'image synthétisée et l'image de contenu proches dans les caractéristiques de contenu ; (ii) la perte de style rend l'image synthétisée et l'image de style proches dans les caractéristiques de style ; et (iii) la perte de variation totale aide à réduire le bruit dans l'image synthétisée.
* Nous pouvons utiliser un CNN pré-entraîné pour extraire les caractéristiques de l'image et minimiser la fonction de perte pour mettre à jour continuellement l'image synthétisée comme paramètres du modèle pendant l'entraînement.
* Nous utilisons des matrices de Gram pour représenter les sorties de style des couches de style.


## Exercices

1. Comment la sortie change-t-elle lorsque vous sélectionnez différentes couches de contenu et de style ?
1. Ajustez les hyperparamètres de poids dans la fonction de perte. La sortie retient-elle plus de contenu ou a-t-elle moins de bruit ?
1. Utilisez des images de contenu et de style différents. Pouvez-vous créer des images synthétisées plus intéressantes ?
1. Peut-on appliquer le transfert de style au texte ? Conseil : vous pouvez vous référer à l'étude de Hu et al. :cite:`Hu.Lee.Aggarwal.ea.2020` .

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:
