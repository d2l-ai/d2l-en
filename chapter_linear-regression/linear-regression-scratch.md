```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Tiesinės regresijos įgyvendinimas nuo nulio
:label:`sec_linear_scratch`

Dabar esame pasirengę išnagrinėti
pilnai veikiantį
tiesinės regresijos įgyvendinimą.
Šiame skyrelyje
(**įgyvendinsime visą metodą nuo nulio,
įskaitant (i) modelį; (ii) nuostolių funkciją;
(iii) minirinkinio stochastinio gradientinio nusileidimo optimizatorių;
ir (iv) apmokymo funkciją,
kuri sujungia visas šias dalis.**)
Galiausiai paleisime savo sintetinių duomenų generatorių
iš :numref:`sec_synthetic-regression-data`
ir pritaikysime savo modelį
gautam duomenų rinkiniui.
Nors šiuolaikiniai giliojo mokymosi karkasai
gali automatizuoti beveik visą šį darbą,
įgyvendinimas nuo nulio yra vienintelis būdas
užtikrinti, kad tikrai žinote, ką darote.
Be to, kai ateis laikas pritaikyti modelius,
apibrėžiant savo sluoksnius ar nuostolių funkcijas,
supratimas, kaip viskas veikia po gaubtu, pravers.
Šiame skyrelyje remsimės tik
tenzoriais ir automatiniu diferencijavimu.
Vėliau pristatysime glaustesnį įgyvendinimą,
pasinaudodami giliojo mokymosi karkasų pranašumais,
išlaikydami toliau pateiktą struktūrą.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Modelio apibrėžimas

[**Prieš pradėdami optimizuoti savo modelio parametrus**] minirinkinio SGD metodu,
(**pirmiausia turime turėti kokius nors parametrus.**)
Toliau svorius inicializuojame imdami
atsitiktinius skaičius iš normaliojo skirstinio su vidurkiu 0
ir standartiniu nuokrypiu 0,01.
Stebuklingasis skaičius 0,01 praktikoje dažnai veikia gerai,
bet galite nurodyti kitą reikšmę
per argumentą `sigma`.
Be to, poslinkį nustatome lygų 0.
Atkreipkite dėmesį, kad objektiniam projektavimui
kodą pridedame prie poklasio `d2l.Module` (pristatyto :numref:`subsec_oo-design-models`) metodo `__init__`.

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

Toliau turime [**apibrėžti savo modelį,
susiedami jo įvestį ir parametrus su išvestimi.**]
Naudodami tą patį žymėjimą kaip :eqref:`eq_linreg-y-vec`
savo tiesiniam modeliui, tiesiog imame matricos ir vektoriaus sandaugą
įvesties požymių $\mathbf{X}$
ir modelio svorių $\mathbf{w}$
ir prie kiekvieno pavyzdžio pridedame poslinkį $b$.
Sandauga $\mathbf{Xw}$ yra vektorius, o $b$ – skaliaras.
Dėl transliavimo mechanizmo
(žr. :numref:`subsec_broadcasting`)
sudėjus vektorių ir skaliarą,
skaliaras pridedamas prie kiekvienos vektoriaus komponentės.
Gautas metodas `forward`
užregistruojamas klasėje `LinearRegressionScratch`
per `add_to_class` (pristatytą :numref:`oo-design-utilities`).

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Nuostolių funkcijos apibrėžimas

Kadangi [**modelio atnaujinimui reikia imti
nuostolių funkcijos gradientą,**]
turėtume (**pirmiausia apibrėžti nuostolių funkciją.**)
Čia naudojame kvadratinių nuostolių funkciją
iš :eqref:`eq_mse`.
Įgyvendinime reikia paversti tikrąją reikšmę `y`
į prognozuotos reikšmės `y_hat` formą.
Šio metodo grąžinamas rezultatas
taip pat turės tą pačią formą kaip `y_hat`.
Taip pat grąžiname vidutinę nuostolių reikšmę
tarp visų minirinkinio pavyzdžių.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Optimizavimo algoritmo apibrėžimas

Kaip aptarta :numref:`sec_linear_regression`,
tiesinė regresija turi uždaros formos sprendinį.
Tačiau mūsų tikslas čia yra iliustruoti,
kaip apmokyti bendresnius neuroninius tinklus,
o tam reikia išmokyti jus
naudoti minirinkinio SGD.
Todėl pasinaudosime proga
pristatyti jūsų pirmąjį veikiantį SGD pavyzdį.
Kiekviename žingsnyje, naudodami minirinkinį,
atsitiktinai paimtą iš mūsų duomenų rinkinio,
įvertiname nuostolių gradientą
parametrų atžvilgiu.
Toliau atnaujiname parametrus
kryptimi, kuri gali sumažinti nuostolius.

Toliau pateiktas kodas taiko atnaujinimą,
turint parametrų rinkinį ir mokymosi spartą `lr`.
Kadangi mūsų nuostoliai apskaičiuojami kaip minirinkinio vidurkis,
mums nereikia koreguoti mokymosi spartos pagal minirinkinio dydį.
Vėlesniuose skyriuose ištirsime,
kaip reikėtų koreguoti mokymosi spartą
labai dideliems minirinkiniams, kokie atsiranda
paskirstytame didelio masto mokyme.
Kol kas šios priklausomybės galime nepaisyti.

:begin_tab:`mxnet`
Apibrėžiame savo klasę `SGD`,
`d2l.HyperParameters` poklasį (pristatytą :numref:`oo-design-utilities`),
kad turėtų panašią API
kaip įtaisytasis SGD optimizatorius.
Parametrus atnaujiname metode `step`.
Jis priima argumentą `batch_size`, kurio galima nepaisyti.
:end_tab:

:begin_tab:`pytorch`
Apibrėžiame savo klasę `SGD`,
`d2l.HyperParameters` poklasį (pristatytą :numref:`oo-design-utilities`),
kad turėtų panašią API
kaip įtaisytasis SGD optimizatorius.
Parametrus atnaujiname metode `step`.
Metodas `zero_grad` visus gradientus nustato lygius 0,
ir jį reikia paleisti prieš atgalinio sklidimo žingsnį.
:end_tab:

:begin_tab:`tensorflow`
Apibrėžiame savo klasę `SGD`,
`d2l.HyperParameters` poklasį (pristatytą :numref:`oo-design-utilities`),
kad turėtų panašią API
kaip įtaisytasis SGD optimizatorius.
Parametrus atnaujiname metode `apply_gradients`.
Jis priima parametrų ir gradientų porų sąrašą.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

Toliau apibrėžiame metodą `configure_optimizers`, kuris grąžina klasės `SGD` egzempliorių.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Apmokymas

Kai jau turime visas dalis
(parametrus, nuostolių funkciją, modelį ir optimizatorių),
esame pasirengę [**įgyvendinti pagrindinį apmokymo ciklą.**]
Labai svarbu, kad visiškai suprastumėte šį kodą,
nes panašius apmokymo ciklus naudosite
kiekvienam kitam šioje knygoje
nagrinėjamam giliojo mokymosi modeliui.
Kiekvienoje *epochoje* iteruojame per
visą apmokymo duomenų rinkinį,
vieną kartą pereidami per kiekvieną pavyzdį
(darant prielaidą, kad pavyzdžių skaičius
dalijasi iš minirinkinio dydžio).
Kiekvienoje *iteracijoje* paimame apmokymo pavyzdžių minirinkinį
ir apskaičiuojame jo nuostolius modelio metodu `training_step`.
Tada apskaičiuojame gradientus kiekvieno parametro atžvilgiu.
Galiausiai iškviesime optimizavimo algoritmą,
kad atnaujintų modelio parametrus.
Apibendrinant, vykdysime tokį ciklą:

* Inicializuoti parametrus $(\mathbf{w}, b)$
* Kartoti, kol baigsime
    * Apskaičiuoti gradientą $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Atnaujinti parametrus $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
Prisiminkite, kad sintetinių regresijos duomenų rinkinys,
kurį sugeneravome :numref:``sec_synthetic-regression-data``,
nepateikia validavimo duomenų rinkinio.
Tačiau daugeliu atvejų
norėsime validavimo duomenų rinkinio
modelio kokybei įvertinti.
Čia validavimo duomenų krautuvą perleidžiame
vieną kartą kiekvienoje epochoje, kad įvertintume modelio veikimą.
Laikydamiesi savo objektinio projektavimo,
metodai `prepare_batch` ir `fit_epoch`
užregistruojami klasėje `d2l.Trainer`
(pristatytoje :numref:`oo-design-training`).

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

Esame beveik pasirengę apmokyti modelį,
bet pirmiausia mums reikia apmokymo duomenų.
Čia naudojame klasę `SyntheticRegressionData`
ir perduodame jai kai kuriuos tikruosius parametrus.
Tada apmokome savo modelį
su mokymosi sparta `lr=0.03`
ir nustatome `max_epochs=3`.
Atkreipkite dėmesį, kad apskritai tiek epochų skaičius,
tiek mokymosi sparta yra hiperparametrai.
Apskritai hiperparametrų nustatymas yra sudėtingas,
ir paprastai norėsime naudoti trijų dalių skaidymą:
vieną aibę apmokymui,
antrą – hiperparametrų parinkimui,
o trečią – galutiniam įvertinimui.
Kol kas šias detales praleidžiame, bet peržiūrėsime jas
vėliau.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Kadangi duomenų rinkinį sukūrėme patys,
tiksliai žinome, kokie yra tikrieji parametrai.
Taigi galime [**įvertinti savo apmokymo sėkmę,
palygindami tikruosius parametrus
su tais, kuriuos išmokome**] per apmokymo ciklą.
Iš tiesų jie pasirodo labai artimi vieni kitiems.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

Neturėtume laikyti savaime suprantamu dalyku gebėjimo tiksliai atkurti
tikruosius parametrus.
Apskritai giliesiems modeliams unikalių parametrų
sprendinių nėra,
o net tiesiniams modeliams
tiksliai atkurti parametrus
įmanoma tik tada, kai joks požymis
nėra tiesiškai priklausomas nuo kitų.
Tačiau mašininiame mokyme
mums dažnai mažiau rūpi
atkurti tikruosius pagrindinius parametrus,
o labiau – parametrai,
kurie veda prie labai tikslios prognozės :cite:`Vapnik.1992`.
Laimei, net sudėtinguose optimizavimo uždaviniuose
stochastinis gradientinis nusileidimas dažnai gali rasti nepaprastai gerų sprendinių,
iš dalies dėl to, kad giliesiems tinklams
egzistuoja daug parametrų konfigūracijų,
kurios veda prie labai tikslios prognozės.


## Santrauka

Šiame skyrelyje žengėme reikšmingą žingsnį
giliojo mokymosi sistemų projektavimo link,
įgyvendindami pilnai veikiantį
neuroninio tinklo modelį ir apmokymo ciklą.
Šio proceso metu sukūrėme duomenų krautuvą,
modelį, nuostolių funkciją, optimizavimo procedūrą
ir vizualizavimo bei stebėsenos įrankį.
Tai padarėme sukomponuodami Python objektą,
kuriame yra visi modeliui apmokyti reikalingi komponentai.
Nors tai dar nėra profesionalaus lygio įgyvendinimas,
jis puikiai veikia, ir toks kodas
jau galėtų padėti greitai spręsti nedidelius uždavinius.
Ateinančiuose skyreliuose pamatysime, kaip tai padaryti
ir *glaustai* (išvengiant šabloninio kodo),
ir *efektyviau* (pilnai išnaudojant mūsų GPU galimybes).



## Pratimai

1. Kas nutiktų, jei svorius inicializuotume nuliais. Ar algoritmas vis tiek veiktų? O jei
   parametrus inicializuotume su dispersija $1000$, o ne $0.01$?
1. Tarkime, kad esate [Georgas Simonas Omas](https://en.wikipedia.org/wiki/Georg_Ohm) ir bandote sukurti
   varžos modelį, siejantį įtampą ir srovę. Ar galite naudoti automatinį
   diferencijavimą savo modelio parametrams išmokti?
1. Ar galite naudoti [Planko dėsnį](https://en.wikipedia.org/wiki/Planck%27s_law) objekto temperatūrai nustatyti
   pagal spektrinį energijos tankį? Kaip nuoroda: iš absoliučiai juodo kūno sklindančios spinduliuotės spektrinis tankis $B$ yra
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Čia
   $\lambda$ yra bangos ilgis, $T$ – temperatūra, $c$ – šviesos greitis, $h$ – Planko konstanta, o $k$ –
   Bolcmano konstanta. Matuojate energiją skirtingiems bangos ilgiams $\lambda$ ir dabar turite pritaikyti spektrinio
   tankio kreivę Planko dėsniui.
1. Su kokiomis problemomis galite susidurti, jei norėtumėte apskaičiuoti antrąsias nuostolių išvestines? Kaip
   jas išspręstumėte?
1. Kodėl funkcijoje `loss` reikia metodo `reshape`?
1. Eksperimentuokite su skirtingomis mokymosi spartomis, kad išsiaiškintumėte, kaip greitai krenta nuostolių funkcijos reikšmė. Ar galite sumažinti
   paklaidą padidindami apmokymo epochų skaičių?
1. Jei pavyzdžių skaičiaus negalima padalyti iš minirinkinio dydžio, kas nutinka `data_iter` epochos pabaigoje?
1. Pabandykite įgyvendinti kitą nuostolių funkciją, pvz., absoliutinės reikšmės nuostolius `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Patikrinkite, kas nutinka įprastiems duomenims.
    1. Patikrinkite, ar elgesys skiriasi, jei aktyviai sutrikdote kai kuriuos $\mathbf{y}$ įrašus, pvz., $y_5 = 10000$.
    1. Ar galite sugalvoti pigų sprendimą, sujungiantį geriausius kvadratinių ir absoliutinės reikšmės nuostolių aspektus?
       Užuomina: kaip išvengti tikrai didelių gradientų reikšmių?
1. Kodėl reikia permaišyti duomenų rinkinį? Ar galite sugalvoti atvejį, kai piktybiškai sukonstruotas duomenų rinkinys kitaip sugriautų optimizavimo algoritmą?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17976)
:end_tab:
