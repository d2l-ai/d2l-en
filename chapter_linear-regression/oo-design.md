```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Objektinis projektavimas įgyvendinimui
:label:`sec_oo-design`

Pristatydami tiesinę regresiją,
apžvelgėme įvairius komponentus,
įskaitant
duomenis, modelį, nuostolių funkciją
ir optimizavimo algoritmą.
Iš tiesų
tiesinė regresija yra
vienas paprasčiausių mašininio mokymosi modelių.
Tačiau jai apmokyti naudojama daug tų pačių komponentų, kurių reikia kitiems šios knygos modeliams.
Todėl
prieš pasinerdami į įgyvendinimo detales
verta
suprojektuoti kai kurias API,
kurias naudosime visoje knygoje.
Giliojo mokymosi komponentus traktuodami
kaip objektus,
galime pradėti
apibrėždami šių objektų klases
ir jų sąveikas.
Toks objektinis projektavimas
įgyvendinimui
labai
supaprastins pateikimą, ir galbūt net norėsite jį naudoti savo projektuose.


Įkvėpti atvirojo kodo bibliotekų, tokių kaip [PyTorch Lightning](https://www.pytorchlightning.ai/),
aukštu lygiu
norime turėti tris klases:
(i) `Module` talpina modelius, nuostolius ir optimizavimo metodus;
(ii) `DataModule` teikia duomenų krautuvus apmokymui ir validavimui;
(iii) abi klasės sujungiamos naudojant klasę `Trainer`, kuri leidžia
apmokyti modelius įvairiose aparatinės įrangos platformose.
Didžioji šios knygos kodo dalis pritaiko `Module` ir `DataModule`. Prie klasės `Trainer` sugrįšime tik aptardami GPU, CPU, lygiagretų apmokymą ir optimizavimo algoritmus.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## Pagalbinės priemonės
:label:`oo-design-utilities`

Mums reikia keleto pagalbinių priemonių objektiniam programavimui „Jupyter" sąsiuviniuose supaprastinti. Viena iš problemų yra ta, kad klasių apibrėžimai paprastai yra gana ilgi kodo blokai. Sąsiuvinio skaitomumas reikalauja trumpų kodo fragmentų, pertraukiamų paaiškinimais, o tai nesuderinama su Python bibliotekoms įprastu programavimo stiliumi. Pirmoji
pagalbinė funkcija leidžia užregistruoti funkcijas kaip klasės metodus *po to*, kai klasė jau sukurta. Tiesą sakant, tai galime padaryti *net ir po to*, kai jau sukūrėme klasės egzempliorius! Tai leidžia klasės įgyvendinimą padalyti į kelis kodo blokus.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Pažiūrėkime trumpai, kaip ją naudoti. Planuojame įgyvendinti klasę `A` su metodu `do`. Užuot rašę ir `A`, ir `do` kodą tame pačiame kodo bloke, galime pirmiausia deklaruoti klasę `A` ir sukurti egzempliorių `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Toliau apibrėžiame metodą `do` kaip įprasta, bet ne klasės `A` apimtyje. Vietoj to šį metodą dekoruojame `add_to_class`, kurio argumentas yra klasė `A`. Taip darant metodas gali pasiekti `A` narių kintamuosius, lygiai kaip tikėtumėmės, jei jis būtų įtrauktas į `A` apibrėžimą. Pažiūrėkime, kas nutinka iškvietus jį egzemplioriui `a`.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

Antroji yra pagalbinė klasė, išsaugojanti visus klasės metodo `__init__` argumentus kaip klasės atributus. Tai leidžia netiesiogiai išplėsti konstruktoriaus iškvietimo parašus be papildomo kodo.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Jos įgyvendinimą atidedame iki :numref:`sec_utils`. Norėdami ją naudoti, apibrėžiame savo klasę, paveldinčią iš `HyperParameters` ir metode `__init__` iškviečiančią `save_hyperparameters`.

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Paskutinė pagalbinė priemonė leidžia interaktyviai braižyti eksperimento eigą jam vykstant. Pagerbdami kur kas galingesnį (ir sudėtingesnį) [TensorBoard](https://www.tensorflow.org/tensorboard), ją pavadiname `ProgressBoard`. Įgyvendinimas atidedamas iki :numref:`sec_utils`. Kol kas tiesiog pažiūrėkime, kaip ji veikia.

Metodas `draw` nubraižo tašką `(x, y)` paveiksle, o `label` nurodomas legendoje. Pasirenkamas parametras `every_n` išlygina liniją, paveiksle rodydamas tik $1/n$ taškų. Jų reikšmės suvidurkinamos iš $n$ kaimyninių taškų pradiniame paveiksle.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

Toliau pateiktame pavyzdyje braižome `sin` ir `cos` su skirtingu glotnumu. Jei paleisite šį kodo bloką, pamatysite, kaip linijos auga animuotai.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Modeliai
:label:`subsec_oo-design-models`

Klasė `Module` yra visų modelių, kuriuos įgyvendinsime, bazinė klasė. Mažiausiai mums reikia trijų metodų. Pirmasis, `__init__`, saugo mokomus parametrus, metodas `training_step` priima duomenų rinkinį ir grąžina nuostolių reikšmę, o galiausiai `configure_optimizers` grąžina optimizavimo metodą arba jų sąrašą, naudojamą mokomiems parametrams atnaujinti. Pasirinktinai galime apibrėžti `validation_step`, kad pateiktume vertinimo matus.
Kartais išvesties skaičiavimo kodą dedame į atskirą metodą `forward`, kad jį būtų galima naudoti pakartotinai.

:begin_tab:`jax`
Python 3.7 pristačius [duomenų klases](https://docs.python.org/3/library/dataclasses.html),
klasės, dekoruotos `@dataclass`, automatiškai prideda magiškus
metodus, tokius kaip `__init__` ir `__repr__`. Narių kintamieji apibrėžiami
naudojant tipų anotacijas. Visi Flax moduliai yra Python 3.7 duomenų klasės.
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # No need for save_hyperparam when using Python dataclass
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Use default_factory to make sure new plots are generated on each run
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
        # and built-in __call__ magic methods for forward pass. Adding here
        # for consistency
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """To be defined later in :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
Galbūt pastebėjote, kad `Module` yra `nn.Block` poklasis, t. y. Gluon neuroninių tinklų bazinės klasės.
Ji suteikia patogių funkcijų neuroniniams tinklams tvarkyti. Pavyzdžiui, jei apibrėžiame metodą `forward`, pvz., `forward(self, X)`, tai egzemplioriui `a` šį metodą galime iškviesti per `a(X)`. Tai veikia, nes įtaisytajame metode `__call__` iškviečiamas metodas `forward`. Daugiau detalių ir pavyzdžių apie `nn.Block` rasite :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
Galbūt pastebėjote, kad `Module` yra `nn.Module` poklasis, t. y. PyTorch neuroninių tinklų bazinės klasės.
Ji suteikia patogių funkcijų neuroniniams tinklams tvarkyti. Pavyzdžiui, jei apibrėžiame metodą `forward`, pvz., `forward(self, X)`, tai egzemplioriui `a` šį metodą galime iškviesti per `a(X)`. Tai veikia, nes įtaisytajame metode `__call__` iškviečiamas metodas `forward`. Daugiau detalių ir pavyzdžių apie `nn.Module` rasite :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
Galbūt pastebėjote, kad `Module` yra `tf.keras.Model` poklasis, t. y. TensorFlow neuroninių tinklų bazinės klasės.
Ji suteikia patogių funkcijų neuroniniams tinklams tvarkyti. Pavyzdžiui, ji įtaisytajame metode `__call__` iškviečia metodą `call`. Čia nukreipiame `call` į metodą `forward`, išsaugodami jo argumentus kaip klasės atributą. Tai darome, kad mūsų kodas būtų panašesnis į kitų karkasų įgyvendinimus.
:end_tab:

:begin_tab:`jax`
Galbūt pastebėjote, kad `Module` yra `linen.Module` poklasis, t. y. Flax neuroninių tinklų bazinės klasės.
Ji suteikia patogių funkcijų neuroniniams tinklams tvarkyti. Pavyzdžiui, ji tvarko modelio parametrus, pateikia dekoratorių `nn.compact` kodui supaprastinti, be kita ko, iškviečia metodą `__call__`.
Čia taip pat nukreipiame `__call__` į metodą `forward`. Tai darome, kad mūsų kodas būtų panašesnis į kitų karkasų įgyvendinimus.
:end_tab:

##  Duomenys
:label:`oo-design-data`

Klasė `DataModule` yra bazinė duomenų klasė. Gana dažnai metodas `__init__` naudojamas duomenims paruošti. Tai apima atsisiuntimą ir išankstinį apdorojimą, jei reikia. `train_dataloader` grąžina apmokymo duomenų rinkinio duomenų krautuvą. Duomenų krautuvas yra (Python) generatorius, kuris kiekvieną kartą panaudotas pateikia duomenų minirinkinį. Šis minirinkinys tada perduodamas `Module` metodui `training_step` nuostoliams apskaičiuoti. Yra pasirinktinis `val_dataloader`, grąžinantis validavimo duomenų rinkinio krautuvą. Jis elgiasi taip pat, tik pateikia duomenų minirinkinius `Module` metodui `validation_step`.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Apmokymas
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
Klasė `Trainer` apmoko `Module` klasės mokomus parametrus naudodama `DataModule` nurodytus duomenis. Pagrindinis metodas yra `fit`, priimantis du argumentus: `model`, `Module` egzempliorių, ir `data`, `DataModule` egzempliorių. Tada jis `max_epochs` kartų iteruoja per visą duomenų rinkinį modeliui apmokyti. Kaip ir anksčiau, šio metodo įgyvendinimą atidedame vėlesniems skyriams.
:end_tab:

:begin_tab:`jax`
Klasė `Trainer` apmoko mokomus parametrus `params` naudodama `DataModule` nurodytus duomenis. Pagrindinis metodas yra `fit`, priimantis tris argumentus: `model`, `Module` egzempliorių, `data`, `DataModule` egzempliorių, ir `key`, JAX `PRNGKeyArray`. Argumentą `key` čia padarome pasirinktiniu, kad supaprastintume sąsają, tačiau JAX ir Flax rekomenduojama visada perduoti šakninį raktą ir juo inicializuoti modelio parametrus. Tada jis `max_epochs` kartų iteruoja per visą duomenų rinkinį modeliui apmokyti. Kaip ir anksčiau, šio metodo įgyvendinimą atidedame vėlesniems skyriams.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Santrauka

Kad pabrėžtume objektinį projektavimą
mūsų būsimam giliojo mokymosi įgyvendinimui,
aukščiau pateiktos klasės tiesiog parodo, kaip jų objektai
saugo duomenis ir sąveikauja tarpusavyje.
Šių klasių įgyvendinimus toliau turtinsime,
pavyzdžiui, per `@add_to_class`,
likusioje knygos dalyje.
Be to,
šios pilnai įgyvendintos klasės
išsaugotos [D2L bibliotekoje](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
*lengvame įrankių rinkinyje*, palengvinančiame struktūrizuotą giliojo mokymosi modeliavimą.
Visų pirma jis leidžia pakartotinai naudoti daugelį komponentų skirtinguose projektuose beveik nieko nekeičiant. Pavyzdžiui, galime pakeisti tik optimizatorių, tik modelį, tik duomenų rinkinį ir pan.;
šis moduliškumo laipsnis atsiperka visoje knygoje glaustumu ir paprastumu (todėl jį ir pridėjome), ir tą patį jis gali padaryti jūsų projektams.


## Pratimai

1. Suraskite pilnus aukščiau pateiktų klasių įgyvendinimus, išsaugotus [D2L bibliotekoje](https://github.com/d2l-ai/d2l-en/tree/master/d2l). Primygtinai rekomenduojame išsamiai peržiūrėti įgyvendinimą, kai šiek tiek geriau susipažinsite su giliojo mokymosi modeliavimu.
1. Pašalinkite sakinį `save_hyperparameters` klasėje `B`. Ar vis dar galite atspausdinti `self.a` ir `self.b`? Pasirinktinai: jei esate pasinėrę į pilną klasės `HyperParameters` įgyvendinimą, ar galite paaiškinti, kodėl?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17974)
:end_tab:
