```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Parametrų valdymas

Kai pasirenkame architektūrą
ir nustatome hiperparametrus,
pereiname prie apmokymo ciklo,
kuriame mūsų tikslas yra rasti parametrų reikšmes,
minimizuojančias mūsų nuostolių funkciją.
Po apmokymo šių parametrų mums prireiks
ateities prognozėms daryti.
Be to, kartais norėsime
išgauti parametrus,
galbūt norėdami pakartotinai juos panaudoti kitame kontekste,
išsaugoti savo modelį diske, kad
jį būtų galima vykdyti kitoje programinėje įrangoje,
arba ištirti jų, tikėdamiesi
įgyti mokslinį supratimą.

Daugeliu atvejų galėsime
nepaisyti smulkių detalių,
kaip parametrai deklaruojami
ir manipuliuojami, pasikliaudami giliojo mokymosi karkasais,
kad jie atliktų sunkiąją dalį.
Tačiau kai tolstame nuo
sukrautų architektūrų su standartiniais sluoksniais,
kartais reikės gilintis į
parametrų deklaravimo ir manipuliavimo niuansus.
Šiame skyrelyje aptariame:

* Parametrų pasiekimą derinimui, diagnostikai ir vizualizacijoms.
* Parametrų bendrinimą tarp skirtingų modelio komponentų.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

(**Pradėkime susitelkdami į MLP su vienu paslėptuoju sluoksniu.**)

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**Parametrų pasiekimas**]
:label:`subsec_param-access`

Pradėkime nuo to, kaip pasiekti parametrus
iš jums jau pažįstamų modelių.

:begin_tab:`mxnet, pytorch, tensorflow`
Kai modelis apibrėžiamas per klasę `Sequential`,
pirmiausia galime pasiekti bet kurį sluoksnį, indeksuodami
modelį taip, tarsi jis būtų sąrašas.
Kiekvieno sluoksnio parametrai patogiai
yra jo atribute.
:end_tab:

:begin_tab:`jax`
Flax ir JAX atskiria modelį ir parametrus, kaip
galbūt pastebėjote anksčiau apibrėžtuose modeliuose.
Kai modelis apibrėžiamas per klasę `Sequential`,
pirmiausia turime inicializuoti tinklą, kad sugeneruotume
parametrų žodyną. Bet kurio sluoksnio parametrus galime pasiekti
per šio žodyno raktus.
:end_tab:

Antrojo pilnai sujungto sluoksnio parametrus galime patikrinti taip.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

Matome, kad šis pilnai sujungtas sluoksnis
turi du parametrus,
atitinkamai atitinkančius to sluoksnio
svorius ir poslinkius.


### [**Tikslinių parametrų pasiekimas**]

Atkreipkite dėmesį, kad kiekvienas parametras atvaizduojamas
kaip parametro klasės egzempliorius.
Kad su parametrais padarytume ką nors naudinga,
pirmiausia turime pasiekti pagrindines skaitines reikšmes.
Tai galima padaryti keliais būdais.
Kai kurie paprastesni, o kiti bendresni.
Šis kodas išgauna poslinkį
iš antrojo neuroninio tinklo sluoksnio, kuris grąžina parametro klasės egzempliorių, ir
toliau pasiekia to parametro reikšmę.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
Parametrai yra sudėtingi objektai,
kuriuose yra reikšmės, gradientai
ir papildoma informacija.
Todėl reikšmės turime prašyti aiškiai.

Be reikšmės, kiekvienas parametras taip pat leidžia pasiekti gradientą. Kadangi šiam tinklui dar nekvietėme atgalinio sklidimo, jis yra pradinės būsenos.
:end_tab:

:begin_tab:`jax`
Skirtingai nei kiti karkasai, JAX nesaugo neuroninio tinklo parametrų
gradientų – vietoj to parametrai ir tinklas yra atskirti.
Tai leidžia naudotojui išreikšti savo skaičiavimą kaip
Python funkciją ir tam pačiam tikslui naudoti transformaciją `grad`.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**Visi parametrai iš karto**]

Kai reikia atlikti operacijas su visais parametrais,
pasiekti juos po vieną gali tapti varginanti.
Situacija gali tapti ypač nepatogi,
kai dirbame su sudėtingesniais, pvz., įdėtaisiais, moduliais,
nes turėtume rekursyviai
apeiti visą medį, kad išgautume
kiekvieno submodulio parametrus. Žemiau pademonstruojame visų sluoksnių parametrų pasiekimą.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**Susieti parametrai**]

Dažnai norime bendrinti parametrus tarp kelių sluoksnių.
Pažiūrėkime, kaip tai padaryti elegantiškai.
Toliau paskiriame pilnai sujungtą sluoksnį
ir tada naudojame jo parametrus būtent
kito sluoksnio parametrams nustatyti.
Čia turime paleisti tiesioginį sklidimą
`net(X)` prieš pasiekdami parametrus.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

Šis pavyzdys rodo, kad antrojo ir trečiojo sluoksnių parametrai
yra susieti.
Jie ne tik lygūs, bet ir
atvaizduojami tuo pačiu tenzoriumi.
Taigi, jei pakeisime vieną iš parametrų,
pasikeis ir kitas.

:begin_tab:`mxnet, pytorch, tensorflow`
Galbūt stebitės,
kas nutinka gradientams, kai parametrai yra susieti.
Kadangi modelio parametrai turi gradientus,
antrojo ir trečiojo paslėptųjų sluoksnių gradientai
atgalinio sklidimo metu sudedami.
:end_tab:


## Santrauka

Turime kelis būdus pasiekti ir susieti modelio parametrus.


## Pratimai

1. Naudokite modelį `NestMLP`, apibrėžtą :numref:`sec_model_construction`, ir pasiekite įvairių sluoksnių parametrus.
1. Sukonstruokite MLP, kuriame yra bendrinamų parametrų sluoksnis, ir apmokykite jį. Apmokymo proceso metu stebėkite kiekvieno sluoksnio modelio parametrus ir gradientus.
1. Kodėl parametrų bendrinimas yra gera idėja?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17990)
:end_tab:
