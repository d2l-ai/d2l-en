```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Softmax regresijos įgyvendinimas nuo nulio
:label:`sec_softmax_scratch`

Kadangi softmax regresija tokia fundamentali,
manome, kad turėtumėte mokėti
ją įgyvendinti patys.
Čia apsiribojame modelio
softmax specifinių aspektų apibrėžimu
ir pakartotinai naudojame kitus komponentus
iš savo tiesinės regresijos skyrelio,
įskaitant apmokymo ciklą.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
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
from functools import partial
```

## Softmax funkcija

Pradėkime nuo svarbiausios dalies:
atvaizdžio iš skaliarų į tikimybes.
Kad atsigaivintumėte, prisiminkite sumavimo operatoriaus veikimą
pagal konkrečias tenzoriaus dimensijas,
kaip aptarta :numref:`subsec_lin-alg-reduction`
ir :numref:`subsec_lin-alg-non-reduction`.
[**Turint matricą `X`, galime sumuoti per visus elementus (pagal numatytuosius nustatymus) arba tik
per tos pačios ašies elementus.**]
Kintamasis `axis` leidžia apskaičiuoti eilučių ir stulpelių sumas:

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Softmax apskaičiavimas reikalauja trijų žingsnių:
(i) kiekvieno nario kėlimo eksponente;
(ii) sumos per kiekvieną eilutę, kad apskaičiuotume kiekvieno pavyzdžio normalizavimo konstantą;
(iii) kiekvienos eilutės dalybos iš jos normalizavimo konstantos,
užtikrinant, kad rezultato suma lygi 1:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

Vardiklio (logaritmas) vadinamas (logaritmine) *statistine suma*
(angl. *partition function*).
Ji buvo pristatyta [statistinėje fizikoje](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)),
sumuojant per visas galimas termodinaminio ansamblio būsenas.
Įgyvendinimas paprastas:

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Bet kuriai įvesčiai `X` [**kiekvieną elementą paverčiame
neneigiamu skaičiumi.
Kiekvienos eilutės suma lygi 1,**]
kaip ir reikalaujama iš tikimybės. Dėmesio: aukščiau pateiktas kodas *nėra* atsparus labai dideliems ar labai mažiems argumentams. Nors jo pakanka, kad parodytų, kas vyksta, *neturėtumėte* naudoti šio kodo tiesiogiai jokiam rimtam tikslui. Giliojo mokymosi karkasuose tokios apsaugos jau yra įtaisytos, ir toliau naudosime įtaisytąjį softmax.

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## Modelis

Dabar turime viską, ko reikia
[**softmax regresijos modeliui**] įgyvendinti.
Kaip ir mūsų tiesinės regresijos pavyzdyje,
kiekvienas egzempliorius bus atvaizduojamas
fiksuoto ilgio vektoriumi.
Kadangi čia neapdoroti duomenys sudaryti
iš $28 \times 28$ pikselių vaizdų,
[**kiekvieną vaizdą išlyginame,
laikydami jį 784 ilgio vektoriumi.**]
Vėlesniuose skyriuose pristatysime
konvoliucinius neuroninius tinklus,
kurie erdvinę struktūrą išnaudoja
patenkinamesniu būdu.


Softmax regresijoje
mūsų tinklo išvesčių skaičius
turėtų būti lygus klasių skaičiui.
(**Kadangi mūsų duomenų rinkinyje yra 10 klasių,
mūsų tinklo išvesties dimensija yra 10.**)
Todėl mūsų svoriai sudaro $784 \times 10$ matricą
plius $1 \times 10$ eilutės vektorių poslinkiams.
Kaip ir su tiesine regresija,
svorius `W` inicializuojame
Gauso triukšmu.
Poslinkiai inicializuojami nuliais.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

Toliau pateiktas kodas apibrėžia, kaip tinklas
atvaizduoja kiekvieną įvestį į išvestį.
Atkreipkite dėmesį, kad kiekvieną minirinkinio $28 \times 28$ pikselių vaizdą
išlyginame į vektorių naudodami `reshape`
prieš perduodami duomenis per mūsų modelį.

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## Kryžminės entropijos nuostoliai

Toliau turime įgyvendinti kryžminės entropijos nuostolių funkciją
(pristatytą :numref:`subsec_softmax-regression-loss-func`).
Tai gali būti dažniausia nuostolių funkcija
visame giliajame mokyme.
Šiuo metu giliojo mokymosi taikymų,
lengvai apibūdinamų kaip klasifikavimo uždaviniai,
yra gerokai daugiau nei tų, kuriuos geriau traktuoti kaip regresijos uždavinius.

Prisiminkite, kad kryžminė entropija ima prognozuotos tikimybės,
priskirtos tikrajam žymeniui, neigiamą logaritminį tikėtinumą.
Dėl efektyvumo vengiame Python ciklų „for" ir vietoj to naudojame indeksavimą.
Visų pirma fiktyvusis kodavimas $\mathbf{y}$
leidžia mums pasirinkti atitinkamus $\hat{\mathbf{y}}$ narius.

Kad pamatytume tai veikiant, [**sukuriame pavyzdinius duomenis `y_hat`
su 2 pavyzdžių prognozuotomis tikimybėmis per 3 klases ir jų atitinkamais žymenimis `y`.**]
Teisingi žymenys yra atitinkamai $0$ ir $2$ (t. y. pirmoji ir trečioji klasė).
[**Naudodami `y` kaip tikimybių `y_hat` indeksus,**]
galime efektyviai išrinkti narius.

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
Dabar galime (**įgyvendinti kryžminės entropijos nuostolių funkciją**) suvidurkindami pasirinktų tikimybių logaritmus.
:end_tab:

:begin_tab:`jax`
Dabar galime (**įgyvendinti kryžminės entropijos nuostolių funkciją**) suvidurkindami pasirinktų tikimybių logaritmus.

Atkreipkite dėmesį, kad norint pasinaudoti `jax.jit` JAX įgyvendinimams pagreitinti
ir užtikrinti, kad `loss` būtų gryna funkcija, funkcija `cross_entropy` iš naujo apibrėžiama
`loss` viduje, kad būtų išvengta bet kokių globalių kintamųjų ar funkcijų naudojimo,
dėl kurių funkcija `loss` galėtų tapti negryna.
Besidominčius skaitytojus nukreipiame į [JAX dokumentaciją](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) apie `jax.jit` ir grynąsias funkcijas.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return cross_entropy(y_hat, y), {}
```

## Apmokymas

Pakartotinai naudojame metodą `fit`, apibrėžtą :numref:`sec_linear_scratch`, kad [**apmokytume modelį 10 epochų.**]
Atkreipkite dėmesį, kad epochų skaičius (`max_epochs`),
minirinkinio dydis (`batch_size`)
ir mokymosi sparta (`lr`)
yra reguliuojami hiperparametrai.
Tai reiškia, kad nors šių reikšmių
neišmokstame per pagrindinį apmokymo ciklą,
jos vis tiek veikia mūsų modelio
našumą, tiek apmokymo,
tiek apibendrinimo požiūriu.
Praktikoje šias reikšmes norėsite rinktis
pagal *validavimo* duomenų dalį
ir galiausiai įvertinti savo galutinį modelį
*testavimo* dalyje.
Kaip aptarta :numref:`subsec_generalization-model-selection`,
Fashion-MNIST testavimo duomenis laikysime
validavimo aibe, taigi
šioje dalyje pateiksime validavimo nuostolius ir validavimo tikslumą.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Prognozavimas

Kai apmokymas baigtas,
mūsų modelis yra pasirengęs [**klasifikuoti kai kuriuos vaizdus.**]

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

Labiau domina vaizdai, kuriuos žymime *neteisingai*. Vizualizuojame juos
palygindami jų tikruosius žymenis
(pirmoji teksto išvesties eilutė)
su modelio prognozėmis
(antroji teksto išvesties eilutė).

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Santrauka

Dabar jau pradedame įgyti patirties
sprendžiant tiesinės regresijos
ir klasifikavimo uždavinius.
Su ja pasiekėme tai, kas, galima sakyti, yra
1960–1970-ųjų statistinio modeliavimo pažangos viršūnė.
Kitame skyrelyje parodysime, kaip pasinaudoti
giliojo mokymosi karkasais šiam modeliui įgyvendinti
daug efektyviau.

## Pratimai

1. Šiame skyrelyje softmax funkciją įgyvendinome tiesiogiai pagal softmax operacijos matematinį apibrėžimą. Kaip aptarta :numref:`sec_softmax`, tai gali sukelti skaitinį nestabilumą.
    1. Patikrinkite, ar `softmax` vis dar veikia teisingai, jei įvestis turi reikšmę $100$.
    1. Patikrinkite, ar `softmax` vis dar veikia teisingai, jei didžiausia iš visų įvesčių yra mažesnė nei $-100$.
    1. Įgyvendinkite pataisą, nagrinėdami reikšmę, palyginti su didžiausiu argumento įrašu.
1. Įgyvendinkite funkciją `cross_entropy`, kuri atitinka kryžminės entropijos nuostolių funkcijos apibrėžimą $\sum_i y_i \log \hat{y}_i$.
    1. Išbandykite ją šio skyrelio kodo pavyzdyje.
    1. Kodėl, jūsų manymu, ji veikia lėčiau?
    1. Ar turėtumėte ją naudoti? Kada tai būtų prasminga?
    1. Į ką reikia būti atsargiems? Užuomina: apsvarstykite logaritmo apibrėžimo sritį.
1. Ar visada gera mintis grąžinti labiausiai tikėtiną žymenį? Pavyzdžiui, ar taip darytumėte medicininei diagnozei? Kaip bandytumėte tai spręsti?
1. Tarkime, kad norime naudoti softmax regresiją kitam žodžiui prognozuoti pagal kai kuriuos požymius. Kokios problemos gali kilti dėl didelio žodyno?
1. Eksperimentuokite su šio skyrelio kodo hiperparametrais. Visų pirma:
    1. Nubraižykite, kaip kinta validavimo nuostoliai keičiant mokymosi spartą.
    1. Ar validavimo ir apmokymo nuostoliai keičiasi keičiant minirinkinio dydį? Kiek didelį ar mažą jį reikia pasirinkti, kad pamatytumėte poveikį?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17982)
:end_tab:
