```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Failų įvestis ir išvestis

Iki šiol aptarėme, kaip apdoroti duomenis ir kaip
kurti, apmokyti ir testuoti giliojo mokymosi modelius.
Tačiau tam tikru momentu, tikėkimės, būsime pakankamai patenkinti
išmoktais modeliais, kad norėsime
išsaugoti rezultatus vėlesniam naudojimui įvairiuose kontekstuose
(galbūt net prognozėms diegimo metu daryti).
Be to, vykdant ilgą apmokymo procesą,
geriausia praktika yra periodiškai išsaugoti tarpinius rezultatus (kontrolinius taškus),
kad nepamestume kelių dienų skaičiavimų,
jei užkliūtume už savo serverio maitinimo laido.
Taigi atėjo laikas išmokti įkelti ir saugoti
ir atskirus svorių vektorius, ir visus modelius.
Šiame skyrelyje nagrinėjami abu klausimai.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**Tenzorių įkėlimas ir išsaugojimas**)

Atskiriems tenzoriams galime tiesiogiai
kviesti funkcijas `load` ir `save`
juos atitinkamai nuskaityti ir įrašyti.
Abi funkcijos reikalauja, kad nurodytume pavadinimą,
o `save` kaip įvesties reikalauja kintamojo, kurį reikia išsaugoti.

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

Dabar galime nuskaityti duomenis iš išsaugoto failo atgal į atmintį.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

Galime [**išsaugoti tenzorių sąrašą ir nuskaityti juos atgal į atmintį.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Galime net [**įrašyti ir nuskaityti žodyną, atvaizduojantį
eilutes į tenzorius.**]
Tai patogu, kai norime
nuskaityti ar įrašyti visus modelio svorius.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Modelio parametrų įkėlimas ir išsaugojimas**]

Atskirų svorių vektorių (ar kitų tenzorių) išsaugojimas yra naudingas,
bet tampa labai varginantis, jei norime išsaugoti
(o vėliau įkelti) visą modelį.
Juk galime turėti šimtus
parametrų grupių, pasklidusių po visą modelį.
Dėl šios priežasties giliojo mokymosi karkasas suteikia įtaisytųjų funkcijų
visiems tinklams įkelti ir išsaugoti.
Svarbi detalė, į kurią reikia atkreipti dėmesį, yra ta, kad tai
išsaugo modelio *parametrus*, o ne visą modelį.
Pavyzdžiui, jei turime 3 sluoksnių MLP,
architektūrą turime nurodyti atskirai.
Priežastis ta, kad patys modeliai gali turėti savavališko kodo,
todėl jų negalima taip natūraliai serializuoti.
Taigi, kad atkurtume modelį, turime
sugeneruoti architektūrą kode
ir tada įkelti parametrus iš disko.
(**Pradėkime nuo mūsų pažįstamo MLP.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

Toliau [**išsaugome modelio parametrus faile**] pavadinimu „mlp.params“.

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

Kad atkurtume modelį, sukuriame
originalaus MLP modelio kloną.
Užuot atsitiktinai inicializavę modelio parametrus,
[**tiesiogiai nuskaitome faile išsaugotus parametrus**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

Kadangi abu egzemplioriai turi tuos pačius modelio parametrus,
tos pačios įvesties `X` skaičiavimo rezultatas turėtų būti toks pat.
Patikrinkime tai.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Santrauka

Funkcijas `save` ir `load` galima naudoti tenzorių objektų failų įvesčiai ir išvesčiai atlikti.
Visas tinklo parametrų aibes galime išsaugoti ir įkelti per parametrų žodyną.
Architektūros išsaugojimas turi būti atliekamas kode, o ne parametruose.

## Pratimai

1. Net jei nereikia diegti apmokytų modelių kitame įrenginyje, kokia praktinė nauda saugoti modelio parametrus?
1. Tarkime, kad norime pakartotinai panaudoti tik tinklo dalis, kad įtrauktume jas į tinklą su kitokia architektūra. Kaip naudotumėte, tarkime, pirmuosius du ankstesnio tinklo sluoksnius naujame tinkle?
1. Kaip išsaugotumėte tinklo architektūrą ir parametrus? Kokius apribojimus architektūrai nustatytumėte?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17994)
:end_tab:
