```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Daugiasluoksnių perceptronų įgyvendinimas
:label:`sec_mlp-implementation`

Daugiasluoksnius perceptronus (MLP) įgyvendinti nėra daug sudėtingiau nei paprastus tiesinius modelius. Pagrindinis konceptualus
skirtumas yra tas, kad dabar sujungiame kelis sluoksnius.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Įgyvendinimas nuo nulio

Vėl pradėkime įgyvendindami tokį tinklą nuo nulio.

### Modelio parametrų inicializavimas

Prisiminkite, kad Fashion-MNIST turi 10 klasių,
o kiekvienas vaizdas sudarytas iš $28 \times 28 = 784$
pilkumo atspalvių pikselių reikšmių tinklelio.
Kaip ir anksčiau, kol kas nepaisysime erdvinės struktūros
tarp pikselių,
todėl tai galime laikyti klasifikavimo duomenų rinkiniu
su 784 įvesties požymiais ir 10 klasių.
Pradžioje [**įgyvendinsime MLP
su vienu paslėptuoju sluoksniu ir 256 paslėptaisiais vienetais.**]
Ir sluoksnių skaičius, ir jų plotis yra reguliuojami
(jie laikomi hiperparametrais).
Paprastai sluoksnių pločius renkamės dalius iš didesnių 2 laipsnių.
Tai skaičiavimo požiūriu efektyvu dėl to, kaip
aparatinėje įrangoje paskirstoma ir adresuojama atmintis.

Vėlgi, savo parametrus atvaizduosime keliais tenzoriais.
Atkreipkite dėmesį, kad *kiekvienam sluoksniui* turime sekti
vieną svorių matricą ir vieną poslinkių vektorių.
Kaip visada, skiriame atmintį
nuostolių gradientams šių parametrų atžvilgiu.

:begin_tab:`mxnet`
Žemiau pateiktame kode pirmiausia apibrėžiame ir inicializuojame parametrus,
o tada įjungiame gradientų sekimą.
:end_tab:

:begin_tab:`pytorch`
Žemiau pateiktame kode naudojame `nn.Parameter`,
kad automatiškai užregistruotume
klasės atributą kaip parametrą, kurį seks `autograd` (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
Žemiau pateiktame kode naudojame `tf.Variable`
modelio parametrui apibrėžti.
:end_tab:

:begin_tab:`jax`
Žemiau pateiktame kode naudojame `flax.linen.Module.param`
modelio parametrui apibrėžti.
:end_tab:

```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### Modelis

Kad įsitikintume, jog suprantame, kaip viskas veikia,
ReLU aktyvaciją [**įgyvendinsime patys**],
užuot tiesiogiai kvietę įtaisytąją funkciją `relu`.

```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

Kadangi nepaisome erdvinės struktūros,
kiekvieną dvimatį vaizdą pertvarkome (`reshape`) į
plokščią vektorių, kurio ilgis `num_inputs`.
Galiausiai (**įgyvendiname savo modelį**)
vos keliomis kodo eilutėmis. Kadangi naudojame karkaso įtaisytąjį autograd, to pakanka.

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Apmokymas

Laimei, [**MLP apmokymo ciklas
yra lygiai toks pat kaip softmax regresijos.**] Apibrėžiame modelį, duomenis ir apmokytoją, o tada kviečiame modelio ir duomenų metodą `fit`.

```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Glaustas įgyvendinimas

Kaip galite tikėtis, pasitelkę aukšto lygio API, MLP galime įgyvendinti dar glausčiau.

### Modelis

Palyginti su glaustu softmax regresijos įgyvendinimu
(:numref:`sec_softmax_concise`),
vienintelis skirtumas yra tas, kad pridedame
*du* pilnai sujungtus sluoksnius ten, kur anksčiau pridėjome tik *vieną*.
Pirmasis yra [**paslėptasis sluoksnis**],
antrasis – išvesties sluoksnis.

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

Anksčiau modeliams apibrėžėme `forward` metodus, kad transformuotume įvestį naudodami modelio parametrus.
Šios operacijos iš esmės yra konvejeris:
paimate įvestį ir
pritaikote transformaciją (pvz.,
matricos daugyba iš svorių, po kurios pridedamas poslinkis),
tada pakartotinai naudojate dabartinės transformacijos išvestį kaip
kitos transformacijos įvestį.
Tačiau galbūt pastebėjote, kad
čia `forward` metodas neapibrėžtas.
Iš tikrųjų `MLP` paveldi `forward` metodą iš klasės `Module` (:numref:`subsec_oo-design-models`), kad
tiesiog iškviestų `self.net(X)` (`X` yra įvestis),
kuris dabar apibrėžtas kaip transformacijų seka
per klasę `Sequential`.
Klasė `Sequential` abstrahuoja tiesioginį procesą,
leisdama mums susitelkti į transformacijas.
Plačiau aptarsime, kaip veikia klasė `Sequential`, :numref:`subsec_model-construction-sequential`.


### Apmokymas

[**Apmokymo ciklas**] yra lygiai toks pat,
kaip įgyvendinant softmax regresiją.
Šis moduliškumas leidžia atskirti
klausimus, susijusius su modelio architektūra,
nuo ortogonalių svarstymų.

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Santrauka

Dabar, kai turime daugiau praktikos projektuojant giliuosius tinklus, žingsnis nuo vieno iki kelių giliųjų tinklų sluoksnių nebekelia tokio didelio iššūkio. Visų pirma, galime pakartotinai naudoti apmokymo algoritmą ir duomenų įkroviklį. Vis dėlto atkreipkite dėmesį, kad MLP įgyvendinimas nuo nulio yra netvarkingas: modelio parametrų pavadinimai ir jų sekimas apsunkina modelių plėtimą. Pavyzdžiui, įsivaizduokite, kad norite įterpti dar vieną sluoksnį tarp 42 ir 43 sluoksnių. Tai dabar galėtų būti 42b sluoksnis, nebent norėtume atlikti nuoseklų pervadinimą. Be to, jei tinklą įgyvendiname nuo nulio, karkasui gerokai sunkiau atlikti prasmingas našumo optimizacijas.

Vis dėlto dabar pasiekėte 9-ojo dešimtmečio pabaigos technikos lygį, kai pilnai sujungti gilieji tinklai buvo pasirenkamas neuroninių tinklų modeliavimo metodas. Kitas mūsų konceptualus žingsnis bus vaizdų nagrinėjimas. Prieš tai turime apžvelgti keletą statistikos pagrindų ir išsamiau aptarti, kaip efektyviai skaičiuoti modelius.


## Pratimai

1. Pakeiskite paslėptųjų vienetų skaičių `num_hiddens` ir nubraižykite, kaip jų skaičius veikia modelio tikslumą. Kokia geriausia šio hiperparametro reikšmė?
1. Pabandykite pridėti paslėptąjį sluoksnį ir pažiūrėkite, kaip tai veikia rezultatus.
1. Kodėl blogas sumanymas įterpti paslėptąjį sluoksnį su vienu neuronu? Kas gali nepavykti?
1. Kaip mokymosi sparto keitimas pakeičia jūsų rezultatus? Kai visi kiti parametrai fiksuoti, kuris mokymosi sparta duoda geriausius rezultatus? Kaip tai susiję su epochų skaičiumi?
1. Optimizuokime visus hiperparametrus kartu, t. y. mokymosi spartą, epochų skaičių, paslėptųjų sluoksnių skaičių ir paslėptųjų vienetų skaičių viename sluoksnyje.
    1. Koks geriausias rezultatas, kurį galite gauti optimizuodami visus juos?
    1. Kodėl gerokai sunkiau tvarkytis su keliais hiperparametrais?
    1. Aprašykite efektyvią strategiją keliems parametrams optimizuoti kartu.
1. Palyginkite karkaso ir įgyvendinimo nuo nulio greitį sudėtingai problemai. Kaip jis kinta didėjant tinklo sudėtingumui?
1. Išmatuokite tenzorių ir matricų daugybos greitį gerai sulygiuotoms ir nesulygiuotoms matricoms. Pavyzdžiui, išbandykite matricas, kurių dimensijos 1024, 1025, 1026, 1028 ir 1032.
    1. Kaip tai skiriasi tarp GPU ir CPU?
    1. Nustatykite savo CPU ir GPU atminties magistralės plotį.
1. Išbandykite skirtingas aktyvacijos funkcijas. Kuri veikia geriausiai?
1. Ar yra skirtumas tarp tinklo svorių inicializavimų? Ar tai svarbu?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17985)
:end_tab:
