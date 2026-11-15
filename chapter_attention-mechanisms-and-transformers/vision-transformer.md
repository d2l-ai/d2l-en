```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# Transformeriai regai
:label:`sec_vision-transformer`

Transformerių architektūra iš pradžių buvo pasiūlyta
sekos į seką mokymuisi,
daugiausia dėmesio skiriant mašininiam vertimui.
Vėliau transformeriai tapo pasirinktu modeliu
įvairiose natūraliosios kalbos apdorojimo užduotyse :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`.
Tačiau kompiuterinės regos srityje
dominuojanti architektūra liko
CNN (:numref:`chap_modern_cnn`).
Natūralu, kad tyrėjai pradėjo svarstyti,
ar galima pasiekti geresnių rezultatų
pritaikius transformerių modelius vaizdų duomenims.
Šis klausimas sukėlė didžiulį susidomėjimą
kompiuterinės regos bendruomenėje.
Neseniai :citet:`ramachandran2019stand` pasiūlė
schemą, kaip konvoliuciją pakeisti saviraiške dėmesio sutelkimo transformacija.
Tačiau dėl specialių šablonų naudojimo dėmesio sutelkime
sunku didinti modelių mastelį aparatinės įrangos greitintuvuose.
Tada :citet:`cordonnier2020relationship` teoriškai įrodė,
kad saviraiškė dėmesio sutelkimo transformacija gali išmokti elgtis panašiai kaip konvoliucija.
Empiriškai kaip įvestys buvo imami $2 \times 2$ vaizdo fragmentai (angl. *patches*),
tačiau dėl mažo fragmento dydžio modelis
tinka tik žemos skiriamosios gebos vaizdų duomenims.

Be specialių fragmento dydžio apribojimų,
*regimieji transformeriai* (angl. *vision Transformers*, ViT)
išskiria fragmentus iš vaizdų
ir paduoda juos į transformerio kodavimo bloką,
kad gautų globalų vaizdą,
kuris galiausiai transformuojamas klasifikavimui :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Pažymėtina, kad transformeriai rodo geresnį mastelio didinimą nei CNN:
apmokant didesnius modelius didesniuose duomenų rinkiniuose,
regimieji transformeriai pranoksta ResNet žymia persvara.
Panašiai kaip tinklų architektūros projektavimo kraštovaizdyje natūraliosios kalbos apdorojime,
transformeriai taip pat tapo lūžio tašku kompiuterinėje regoje.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Modelis

:numref:`fig_vit` pavaizduota
regimųjų transformerių modelio architektūra.
Šią architektūrą sudaro įvadas (angl. *stem*),
kuris suskaido vaizdus į fragmentus,
kūnas, pagrįstas daugiasluoksniu transformerio kodavimo bloku,
ir galva, kuri paverčia globalų vaizdą
išvesties žymeniu.

![Regimojo transformerio architektūra. Šiame pavyzdyje vaizdas padalytas į devynis fragmentus. Specialus „&lt;cls&gt;“ teksto vienetas ir devyni išlyginti vaizdo fragmentai per fragmentų įterpimą ir $\mathit{n}$ transformerio kodavimo bloko blokų paverčiami atitinkamai dešimčia vaizdų. „&lt;cls&gt;“ vaizdas toliau paverčiamas išvesties žymeniu.](../img/vit.svg)
:label:`fig_vit`

Apsvarstykime įvesties vaizdą, kurio aukštis $h$, plotis $w$
ir $c$ kanalų.
Nurodžius, kad fragmento aukštis ir plotis abu yra $p$,
vaizdas suskaidomas į $m = hw/p^2$ fragmentų seką,
kur kiekvienas fragmentas išlyginamas į $cp^2$ ilgio vektorių.
Taip transformerio kodavimo blokai gali traktuoti vaizdo fragmentus panašiai kaip teksto sekų teksto vienetus.
Specialus „&lt;cls&gt;“ (klasės) teksto vienetas ir
$m$ išlygintų vaizdo fragmentų tiesiškai projektuojami
į $m+1$ vektorių seką,
sudedamą su mokomais poziciniais įterpiniais.
Daugiasluoksnis transformerio kodavimo blokas
transformuoja $m+1$ įvesties vektorių
į tiek pat to paties ilgio išvesties vektorių vaizdų.
Jis veikia lygiai taip pat kaip originalus transformerio kodavimo blokas :numref:`fig_transformer`,
skiriasi tik normalizacijos vieta.
Kadangi „&lt;cls&gt;“ teksto vienetas sutelkia dėmesį į visus vaizdo fragmentus
per saviraišką dėmesio sutelkimo transformaciją (žr. :numref:`fig_cnn-rnn-self-attention`),
jo vaizdas iš transformerio kodavimo bloko išvesties
bus toliau paverčiamas išvesties žymeniu.

## Fragmentų įterpimas

Kad įgyvendintume regimąjį transformerį, pradėkime
nuo fragmentų įterpimo :numref:`fig_vit`.
Vaizdo suskaidymą į fragmentus
ir šių išlygintų fragmentų tiesinį projektavimą
galima supaprastinti iki vienos konvoliucijos operacijos,
kur ir branduolio dydis, ir žingsnio dydis nustatomi lygūs fragmento dydžiui.

```{.python .input}
%%tab pytorch
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

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

Toliau pateiktame pavyzdyje, kaip įvestis imant vaizdus, kurių aukštis ir plotis yra `img_size`,
fragmentų įterpimas išveda `(img_size//patch_size)**2` fragmentų,
kurie tiesiškai projektuojami į `num_hiddens` ilgio vektorius.

```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Regimojo transformerio kodavimo blokas
:label:`subsec_vit-encoder`

Regimojo transformerio kodavimo bloko MLP šiek tiek skiriasi
nuo originalaus transformerio kodavimo bloko pozicinio tiesioginio sklidimo tinklo
(žr. :numref:`subsec_positionwise-ffn`).
Pirma, čia aktyvacijos funkcija naudoja Gauso paklaidos tiesinį vienetą (GELU),
kurį galima laikyti glotnesne ReLU versija :cite:`Hendrycks.Gimpel.2016`.
Antra, reguliarizacijai dropout taikomas kiekvieno pilnai sujungto MLP sluoksnio išvesčiai.

```{.python .input}
%%tab pytorch
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

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

Regimojo transformerio kodavimo bloko bloko įgyvendinimas
tiesiog seka išankstinės normalizacijos dizainą :numref:`fig_vit`,
kur normalizacija taikoma iškart *prieš* lygiagrečią dėmesio sutelkimo transformaciją arba MLP.
Skirtingai nei po-normalizacija („sudėk ir normalizuok“ :numref:`fig_transformer`),
kai normalizacija dedama iškart *po* liekamųjų jungčių,
išankstinė normalizacija lemia veiksmingesnį ar efektyvesnį transformerių apmokymą :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.

```{.python .input}
%%tab pytorch
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
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

Kaip ir :numref:`subsec_transformer-encoder`,
joks regimojo transformerio kodavimo bloko blokas nekeičia savo įvesties formos.

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## Viskas kartu

Toliau pateiktas regimųjų transformerių tiesioginis sklidimas yra paprastas.
Pirmiausia įvesties vaizdai paduodami į `PatchEmbedding` egzempliorių,
kurio išvestis sujungiama su „&lt;cls&gt;“ teksto vieneto įterpiniu.
Jie sudedami su mokomais poziciniais įterpiniais prieš dropout.
Tada išvestis paduodama į transformerio kodavimo bloką, kuris sukrauna `num_blks` klasės `ViTBlock` egzempliorių.
Galiausiai „&lt;cls&gt;“ teksto vieneto vaizdą projektuoja tinklo galva.

```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """Vision Transformer."""
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

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """Vision Transformer."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## Mokymas

Regimojo transformerio apmokymas Fashion-MNIST duomenų rinkinyje yra tarsi CNN apmokymas :numref:`chap_modern_cnn`.

```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Santrauka ir diskusija

Galbūt pastebėjote, kad mažuose duomenų rinkiniuose, tokiuose kaip Fashion-MNIST,
mūsų įgyvendintas regimasis transformeris
nepralenkia ResNet iš :numref:`sec_resnet`.
Panašių pastebėjimų galima padaryti net ImageNet duomenų rinkinyje (1,2 milijono vaizdų).
Taip yra todėl, kad transformeriams *trūksta* tų naudingų konvoliucijos principų,
tokių kaip postūmio invariantiškumas ir lokalumas (:numref:`sec_why-conv`).
Tačiau vaizdas pasikeičia apmokant didesnius modelius didesniuose duomenų rinkiniuose (pvz., 300 milijonų vaizdų),
kur regimieji transformeriai vaizdų klasifikavime pranoksta ResNet didele persvara, parodydami
būdingą transformerių pranašumą mastelio didinimo prasme :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Regimųjų transformerių pristatymas
pakeitė tinklų projektavimo vaizdų duomenims modeliuoti kraštovaizdį.
Netrukus buvo parodyta, kad jie veiksmingi ImageNet duomenų rinkinyje
su duomenis taupančiomis DeiT apmokymo strategijomis :cite:`touvron2021training`.
Tačiau saviraiškės dėmesio sutelkimo transformacijos kvadratinis sudėtingumas
(:numref:`sec_self-attention-and-positional-encoding`)
daro transformerių architektūrą
mažiau tinkamą didesnės skiriamosios gebos vaizdams.
Siekdami bendrosios paskirties pagrindinio tinklo kompiuterinėje regoje,
Swin transformeriai sprendė kvadratinį skaičiavimo sudėtingumą
vaizdo dydžio atžvilgiu (:numref:`subsec_cnn-rnn-self-attention`)
ir grąžino į konvoliuciją panašias prielaidas,
išplėsdami transformerių pritaikomumą įvairioms kompiuterinės regos užduotims,
ne tik vaizdų klasifikavimui, su pažangiausiais rezultatais :cite:`liu2021swin`.

## Pratimai

1. Kaip `img_size` reikšmė veikia apmokymo laiką?
1. Užuot projektavus „&lt;cls&gt;“ teksto vieneto vaizdą į išvestį, kaip projektuotumėte vidurkintus fragmentų vaizdus? Įgyvendinkite šį pakeitimą ir pažiūrėkite, kaip jis veikia tikslumą.
1. Ar galite pakeisti hiperparametrus, kad pagerintumėte regimojo transformerio tikslumą?

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/8943)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18032)
:end_tab:
