```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch'])
```

# Transformateur de vision
:label:`sec_vision-transformer` 

 L'architecture de transformateur a été initialement proposée pour l'apprentissage de séquence à séquence, comme pour la traduction automatique. 
D'une grande efficacité, les transformateurs
sont ensuite devenus le modèle de choix dans diverses tâches de traitement du langage naturel :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring` . 
Cependant, 
dans le domaine de la vision par ordinateur
l'architecture dominante
a été basée sur
CNNs (:numref:`chap_modern_cnn` ).
*Peut-on adapter les transformateurs
pour modéliser les données d'image* ?
Cette question a suscité un immense intérêt
dans la communauté de la vision par ordinateur.
:cite:`ramachandran2019stand` a proposé de remplacer la convolution par l'auto-attention. 
Cependant, son utilisation de modèles spécialisés dans l'attention rend difficile la mise à l'échelle des modèles sur les accélérateurs matériels.
:cite:`cordonnier2020relationship` a prouvé théoriquement que l'auto-attention peut apprendre à se comporter de manière similaire à la convolution. Empiriquement, les patchs de $2 \times 2$ ont été extraits d'images en entrée, mais la petite taille des patchs rend le modèle uniquement applicable aux données d'image à faible résolution.

Sans contraintes spécifiques sur la taille des patchs,
*vision transformers* (ViTs)
extraient les patchs des images
et les alimentent dans un encodeur transformateur
pour obtenir une représentation globale,
qui sera finalement transformée pour la classification :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` .
Les transformateurs présentent notamment une meilleure évolutivité que les CNN :
lors de l'apprentissage de modèles plus grands sur des ensembles de données plus importants,
les transformateurs de vision surpassent les ResNets par une marge significative. À l'instar de la conception de l'architecture des réseaux dans le traitement du langage naturel, les transformateurs
ont également changé la donne dans le domaine de la vision par ordinateur.


## Modèle

:numref:`fig_vit` représente
l'architecture modèle des transformateurs de vision.
Cette architecture se compose d'une tige
qui patchifie les images, 
d'un corps basé sur le codeur de transformateur multicouche,
et d'une tête qui transforme la représentation globale
en étiquette de sortie.

![The vision transformer architecture. In this example, an image is split into 9 patches. A special “&lt;cls&gt;” token and the 9 flattened image patches are transformed via patch embedding and $n$ transformer encoder blocks into 10 representations, respectively. The “&lt;cls&gt;” representation is further transformed into the output label.](../img/vit.svg)
:label:`fig_vit`

Considérons une image d'entrée avec des canaux de hauteur $h$, de largeur $w$,
et $c$.
En spécifiant la hauteur et la largeur du patch comme $p$,
l'image est divisée en une séquence de patchs $m = hw/p^2$,
où chaque patch est aplati en un vecteur de longueur $cp^2$.
De cette façon, les patchs d'image peuvent être traités de manière similaire aux tokens dans les séquences de texte par les codeurs transformateurs.
Un jeton spécial "&lt;cls&gt;" (classe) et
les patchs d'image aplatis de $m$ sont projetés linéairement
dans une séquence de vecteurs $m+1$,
additionnés avec des incorporations positionnelles apprenables.
Le codeur transformateur multicouche
transforme les vecteurs d'entrée $m+1$
 en une même quantité de représentations vectorielles de sortie de même longueur.
Il fonctionne exactement de la même manière que le codeur transformateur original dans :numref:`fig_transformer` ,
à la seule différence de la position de normalisation.
Puisque le jeton "&lt;cls&gt;" s'occupe de tous les patchs de l'image par auto-attention (voir :numref:`fig_cnn-rnn-self-attention` ),
sa représentation de la sortie du codeur transformateur
sera transformée en étiquette de sortie.

```{.python .input  n=1}
from d2l import torch as d2l
import torch
from torch import nn
```

## Intégration de patchs

Pour mettre en œuvre un transformateur de vision, commençons par l'intégration de patchs dans :numref:`fig_vit` . La division d'une image en patchs et la projection linéaire de ces patchs aplatis peuvent être simplifiées en une seule opération de convolution, où la taille du noyau et la taille du pas sont toutes deux fixées à la taille du patch.

```{.python .input  n=2}
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

Dans l'exemple suivant, en prenant en entrée des images dont la hauteur et la largeur sont de `img_size`, l'incorporation de patchs produit des patchs `(img_size//patch_size)**2` qui sont projetés linéairement sur des vecteurs de longueur `num_hiddens`.

```{.python .input  n=9}
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.randn(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Encodeur transformateur de vision
:label:`subsec_vit-encoder` 

 Le MLP de l'encodeur transformateur de vision
est légèrement différent du FFN positionnel de l'encodeur transformateur original (voir :numref:`subsec_positionwise-ffn` ). Tout d'abord, la fonction d'activation utilise ici l'unité linéaire d'erreur gaussienne (GELU), qui peut être considérée comme une version plus lisse de la ReLU :cite:`hendrycks2016gaussian` .
Ensuite, le dropout est appliqué à la sortie de chaque couche entièrement connectée dans le MLP pour la régularisation.

```{.python .input}
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

L'implémentation du bloc codeur du transformateur de vision
suit simplement la conception de la prénormalisation dans :numref:`fig_vit` ,
où la normalisation est appliquée juste *avant* l'attention multi-têtes ou le MLP.
Contrairement à la post-normalisation ("add &amp; norm" dans :numref:`fig_transformer` ), où la normalisation est placée juste *après* les connexions résiduelles, la pré-normalisation de
conduit à une entrainement plus efficace ou efficiente pour les transformateurs :cite:`baevski2018adaptive,wang2019learning,xiong2020layer` .

```{.python .input}
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, 
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = self.ln1(X)
        return X + self.mlp(self.ln2(
            X + self.attention(X, X, X, valid_lens)))
```

Comme dans :numref:`subsec_transformer-encoder` ,
tout bloc encodeur de transformateur de vision ne change pas sa forme d'entrée.

```{.python .input}
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

## Putting All Things Together

Le passage en avant des transformateurs de vision ci-dessous est simple.
Tout d'abord, les images d'entrée sont introduites dans une instance de `PatchEmbedding`,
dont la sortie est concaténée avec l'encapsulage du token "&lt;cls&gt;". Elles sont additionnées avec des incorporations positionnelles apprenables avant d'être abandonnées.
Ensuite, la sortie est introduite dans l'encodeur transformateur qui empile les instances `num_blks` de la classe `ViTBlock`.
Enfin, la représentation du token "&lt;cls&gt;" est projetée par la tête de réseau.

```{.python .input}
class ViT(d2l.Classifier):
    """Vision transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

## Formation

l'entrainement d'un transformateur de vision sur le jeu de données Fashion-MNIST se fait de la même manière que l'entrainement des CNN sur :numref:`chap_modern_cnn` .

```{.python .input}
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Résumé et discussion

Vous pouvez remarquer que pour les petits ensembles de données comme Fashion-MNIST, notre transformateur de vision implémenté ne surpasse pas le ResNet dans :numref:`sec_resnet` .
Des observations similaires peuvent être faites même sur le jeu de données ImageNet (1,2 million d'images).
Cela est dû au fait que les transformateurs *manquent* de ces principes utiles dans la convolution, tels que l'invariance de la traduction et la localité (:numref:`sec_why-conv` ).
Cependant, le tableau change lorsqu'on entraîne des modèles plus grands sur des ensembles de données plus importants (par exemple, 300 millions d'images),
où les transformateurs de vision surpassent les ResNets par une marge importante dans la classification d'images, démontrant
la supériorité intrinsèque des transformateurs en matière d'évolutivité :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` .
L'introduction des transformateurs de vision
a changé le paysage de la conception des réseaux pour la modélisation des données d'image.
Ils se sont rapidement révélés efficaces sur le jeu de données ImageNet avec des stratégies d'entraînement efficaces en termes de données de DeiT :cite:`touvron2021training` .
Vers un réseau dorsal polyvalent en vision par ordinateur,
Les transformateurs de Swin ont résolu le problème de la complexité informatique quadratique par rapport à la taille de l'image (:numref:`subsec_cnn-rnn-self-attention` )
et ont ajouté des priors de type convolution,
étendant l'applicabilité des transformateurs à une gamme de tâches de vision par ordinateur au-delà de la classification d'images avec des résultats de pointe :cite:`liu2021swin` .


## Exercices

1. Comment la valeur de `img_size` affecte-t-elle le temps de entrainement ?
1. Au lieu de projeter la représentation du token "&lt;cls&gt;" sur la sortie, comment projeter les représentations moyennes des patchs ? Implémentez ce changement et voyez comment il affecte la précision.
1. Pouvez-vous modifier les hyperparamètres pour améliorer la précision du transformateur de vision ?


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8943)
:end_tab:
