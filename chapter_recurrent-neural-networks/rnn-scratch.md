# Rekurentinio neuroninio tinklo įgyvendinimas nuo nulio
:label:`sec_rnn-scratch`

Dabar esame pasirengę įgyvendinti RNN nuo nulio.
Visų pirma apmokysime šį RNN veikti
kaip simbolių lygio kalbos modelį
(žr. :numref:`sec_rnn`)
ir apmokysime jį tekstyne, sudarytame iš
viso H. G. Wellso *Laiko mašinos* teksto,
laikydamiesi duomenų apdorojimo žingsnių,
aprašytų :numref:`sec_text-sequence`.
Pradedame nuo duomenų rinkinio įkėlimo.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNN modelis

Pradedame apibrėždami klasę
RNN modeliui įgyvendinti
(:numref:`subsec_rnn_w_hidden_states`).
Atkreipkite dėmesį, kad paslėptųjų vienetų skaičius `num_hiddens`
yra derinamas hiperparametras.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**Toliau pateiktas metodas `forward` apibrėžia, kaip apskaičiuoti
išvestį ir paslėptąją būseną bet kuriuo laiko žingsniu,
turint dabartinę įvestį ir modelio būseną
ankstesniu laiko žingsniu.**]
Atkreipkite dėmesį, kad RNN modelis eina ciklu per
išorinę `inputs` dimensiją,
atnaujindamas paslėptąją būseną
po vieną laiko žingsnį.
Čia modelis naudoja $\tanh$ aktyvacijos funkciją (:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

Įvesties sekų minirinkinį į RNN modelį galime paduoti taip.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

Patikrinkime, ar RNN modelis
duoda teisingos formos rezultatus,
kad užtikrintume, jog paslėptosios būsenos dimensija
išlieka nepakitusi.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## RNN pagrįstas kalbos modelis

Toliau pateikta klasė `RNNLMScratch` apibrėžia
RNN pagrįstą kalbos modelį,
kur savo RNN perduodame per
metodo `__init__` argumentą `rnn`.
Apmokant kalbos modelius,
įvestys ir išvestys yra
iš to paties žodyno.
Todėl jos turi tą pačią dimensiją,
lygią žodyno dydžiui.
Atkreipkite dėmesį, kad modeliui vertinti naudojame painumą.
Kaip aptarta :numref:`subsec_perplexity`, tai užtikrina,
kad skirtingo ilgio sekos būtų palyginamos.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**Vienkartinis kodavimas**]

Prisiminkite, kad kiekvienas teksto vienetas pavaizduojamas
skaitiniu indeksu, nurodančiu
atitinkamo žodžio / simbolio / žodžio dalies
poziciją žodyne.
Galite susigundyti sukurti neuroninį tinklą
su vienu įvesties mazgu (kiekvienu laiko žingsniu),
į kurį indeksą būtų galima paduoti kaip skaliarinę reikšmę.
Tai veikia, kai dirbame su skaitinėmis įvestimis,
tokiomis kaip kaina ar temperatūra, kur bet kurios dvi
pakankamai artimos reikšmės
turėtų būti traktuojamos panašiai.
Tačiau tai nelabai prasminga.
Mūsų žodyno $45$-as ir $46$-as žodžiai
yra „their“ ir „said“,
kurių reikšmės nė iš tolo nėra panašios.

Dirbant su tokiais kategoriniais duomenimis,
dažniausia strategija yra atvaizduoti
kiekvieną elementą *vienkartiniu kodavimu*
(prisiminkite iš :numref:`subsec_classification-problem`).
Vienkartinis kodavimas yra vektorius, kurio ilgį
nusako žodyno dydis $N$,
kur visi įrašai nustatyti į $0$,
išskyrus įrašą, atitinkantį
mūsų teksto vienetą, kuris nustatomas į $1$.
Pavyzdžiui, jei žodyne būtų penki elementai,
tai vienkartiniai vektoriai, atitinkantys
indeksus 0 ir 2, būtų tokie.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**Minirinkiniai, kuriuos imame kiekvienoje iteracijoje,
turės formą (rinkinio dydis, laiko žingsnių skaičius).
Kai kiekvieną įvestį pavaizduojame vienkartiniu vektoriumi,
kiekvieną minirinkinį galime laikyti trimačiu tenzoriumi,
kurio ilgį išilgai trečiosios ašies
nusako žodyno dydis (`len(vocab)`).**)
Dažnai transponuojame įvestį, kad gautume išvestį,
kurios forma (laiko žingsnių skaičius, rinkinio dydis, žodyno dydis).
Tai leis patogiau eiti ciklu per išorinę dimensiją,
atnaujinant minirinkinio paslėptąsias būsenas,
po vieną laiko žingsnį
(pvz., aukščiau pateiktame metode `forward`).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### RNN išvesčių transformavimas

Kalbos modelis naudoja pilnai sujungtą išvesties sluoksnį,
kad RNN išvestis transformuotų į teksto vienetų prognozes kiekvienu laiko žingsniu.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Pažiūrėkime, [**ar tiesioginis skaičiavimas
duoda teisingos formos išvestis.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradientų apkarpymas**]


Nors jau pratę galvoti apie neuroninius tinklus
kaip „gilius“ ta prasme, kad daug sluoksnių
skiria įvestį ir išvestį
net vieno laiko žingsnio ribose,
sekos ilgis įveda
naują gylio sąvoką.
Be perėjimo per tinklą
kryptimi nuo įvesties iki išvesties,
įvestys pirmuoju laiko žingsniu
turi praeiti per $T$ sluoksnių grandinę
išilgai laiko žingsnių, kad
paveiktų modelio išvestį
paskutiniu laiko žingsniu.
Žvelgiant atgaline kryptimi,
kiekvienoje iteracijoje
atgaline kryptimi sklidiname gradientus per laiką,
gaudami $\mathcal{O}(T)$ ilgio
matricų sandaugų grandinę.
Kaip minėta :numref:`sec_numerical_stability`,
tai gali lemti skaitinį nestabilumą,
sukeliantį gradientų sprogimą arba išnykimą,
priklausomai nuo svorių matricų savybių.

Susidorojimas su nykstančiais ir sprogstančiais gradientais
yra fundamentali problema projektuojant RNN
ir paskatino kai kuriuos didžiausius
šiuolaikinių neuroninių tinklų architektūrų pasiekimus.
Kitame skyriuje kalbėsime apie
specializuotas architektūras, sukurtas
tikintis sušvelninti nykstančio gradiento problemą.
Tačiau net šiuolaikiniai RNN dažnai kenčia
nuo sprogstančių gradientų.
Vienas nelabai elegantiškas, bet visuotinai paplitęs sprendimas
yra tiesiog apkarpyti gradientus,
priversdami gautus „apkarpytus“ gradientus
įgyti mažesnes reikšmes.


Apskritai, optimizuodami tam tikrą tikslą
gradientiniu nusileidimu, iteratyviai atnaujiname
dominantį parametrą, tarkime, vektorių $\mathbf{x}$,
stumdami jį neigiamo gradiento $\mathbf{g}$ kryptimi
(stochastiniame gradientiniame nusileidime
šį gradientą skaičiuojame
atsitiktinai atrinktame minirinkinyje).
Pavyzdžiui, su mokymosi greičiu $\eta > 0$
kiekvienas atnaujinimas yra pavidalo
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$.
Toliau darykime prielaidą, kad tikslo funkcija $f$
yra pakankamai glodi.
Formaliai sakome, kad tikslas
yra *Lipšico tolydus* su konstanta $L$,
t. y. bet kuriems $\mathbf{x}$ ir $\mathbf{y}$ turime

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Kaip matote, kai atnaujiname parametrų vektorių atimdami $\eta \mathbf{g}$,
tikslo reikšmės pokytis
priklauso nuo mokymosi greičio,
gradiento normos ir $L$ taip:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

Kitaip tariant, tikslas negali
pasikeisti daugiau nei $L \eta \|\mathbf{g}\|$.
Mažos šios viršutinės ribos reikšmės turėjimas
gali būti vertinamas kaip gerai arba blogai.
Neigiama pusė ta, kad ribojame greitį,
kuriuo galime mažinti tikslo reikšmę.
Teigiama pusė ta, kad tai riboja, kiek
daug galime suklysti bet kuriame viename gradiento žingsnyje.


Kai sakome, kad gradientai sprogsta,
turime omenyje, kad $\|\mathbf{g}\|$
tampa pernelyg didelis.
Šiuo blogiausiu atveju galime padaryti tiek daug
žalos viename gradiento žingsnyje, kad
galime panaikinti visą pažangą, padarytą
per tūkstančius apmokymo iteracijų.
Kai gradientai gali būti tokie dideli,
neuroninio tinklo apmokymas dažnai diverguoja,
nepavykdamas sumažinti tikslo reikšmės.
Kitais atvejais apmokymas galiausiai konverguoja,
bet yra nestabilus dėl masyvių nuostolių šuolių.


Vienas būdas apriboti $L \eta \|\mathbf{g}\|$ dydį
yra sumažinti mokymosi greitį $\eta$ iki mažyčių reikšmių.
Tai turi pranašumą, kad nešaliname atnaujinimų.
Bet ką daryti, jei dideli gradientai pasitaiko tik *retai*?
Šis drastiškas žingsnis sulėtina mūsų pažangą visuose žingsniuose,
tik tam, kad susidorotume su retais gradientų sprogimo atvejais.
Populiari alternatyva yra taikyti *gradientų apkarpymo* euristiką,
projektuojančią gradientus $\mathbf{g}$ į tam tikro
duoto spindulio $\theta$ rutulį taip:

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

Tai užtikrina, kad gradiento norma niekada neviršys $\theta$
ir kad atnaujintas gradientas visiškai sutaps
su originalia $\mathbf{g}$ kryptimi.
Tai taip pat turi pageidaujamą šalutinį poveikį –
riboja įtaką, kurią bet kuris duotas minirinkinys
(ir jame bet kuris duotas pavyzdys)
gali daryti parametrų vektoriui.
Tai suteikia modeliui tam tikrą atsparumo laipsnį.
Kad būtų aišku, tai yra gudrybė (angl. *hack*).
Gradientų apkarpymas reiškia, kad ne visada
sekame tikrąjį gradientą, ir sunku
analitiškai samprotauti apie galimus šalutinius poveikius.
Vis dėlto tai labai naudinga gudrybė,
plačiai taikoma RNN įgyvendinimuose
daugumoje giliojo mokymosi karkasų.


Toliau apibrėžiame metodą gradientams apkarpyti,
kurį iškviečia klasės `d2l.Trainer` metodas `fit_epoch`
(žr. :numref:`sec_linear_scratch`).
Atkreipkite dėmesį, kad skaičiuodami gradiento normą,
sujungiame visus modelio parametrus,
traktuodami juos kaip vieną milžinišką parametrų vektorių.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Mokymas

Naudodami *Laiko mašinos* duomenų rinkinį (`data`),
apmokome simbolių lygio kalbos modelį (`model`),
pagrįstą nuo nulio įgyvendintu RNN (`rnn`).
Atkreipkite dėmesį, kad pirmiausia apskaičiuojame gradientus,
tada juos apkarpome, ir galiausiai
atnaujiname modelio parametrus
naudodami apkarpytus gradientus.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Dekodavimas

Kai kalbos modelis išmoktas,
galime jį naudoti ne tik kitam teksto vienetui prognozuoti,
bet ir toliau prognozuoti kiekvieną paskesnį,
traktuodami anksčiau prognozuotą teksto vienetą taip, tarsi
jis būtų kitas įvestyje.
Kartais tiesiog norėsime generuoti tekstą,
tarsi pradėtume nuo
dokumento pradžios.
Tačiau dažnai naudinga sąlygoti
kalbos modelį pagal vartotojo pateiktą priešdėlį.
Pavyzdžiui, jei kurtume
automatinio užbaigimo funkciją paieškos sistemai
arba padėtume vartotojams rašyti el. laiškus,
norėtume paduoti tai, ką jie
iki šiol parašė (priešdėlį),
ir tada sugeneruoti tikėtiną tęsinį.


[**Toliau pateiktas metodas `predict`
generuoja tęsinį po vieną simbolį,
priėmęs vartotojo pateiktą `prefix`**].
Einant ciklu per simbolius `prefix`,
toliau perduodame paslėptąją būseną
į kitą laiko žingsnį,
bet negeneruojame jokios išvesties.
Tai vadinama *apšilimo* periodu.
Priėmę priešdėlį, dabar esame
pasirengę pradėti išvesti paskesnius simbolius,
kurių kiekvienas bus grąžinamas į modelį
kaip įvestis kitame laiko žingsnyje.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Toliau nurodome priešdėlį
ir paprašome sugeneruoti 20 papildomų simbolių.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

Nors aukščiau pateikto RNN modelio įgyvendinimas nuo nulio yra pamokantis, jis nėra patogus.
Kitame skyrelyje pamatysime, kaip pasinaudoti giliojo mokymosi karkasais, kad greitai sukurtume RNN
naudodami standartines architektūras ir gautume našumo naudos,
pasikliaudami labai optimizuotomis bibliotekų funkcijomis.


## Santrauka

Galime apmokyti RNN pagrįstus kalbos modelius generuoti tekstą pagal vartotojo pateiktą teksto priešdėlį.
Paprastą RNN kalbos modelį sudaro įvesties kodavimas, RNN modeliavimas ir išvesties generavimas.
Apmokymo metu gradientų apkarpymas gali sušvelninti sprogstančių gradientų problemą, bet neišsprendžia nykstančių gradientų problemos. Eksperimente įgyvendinome paprastą RNN kalbos modelį ir apmokėme jį su gradientų apkarpymu tekstų sekose, suskaidytose simbolių lygiu. Sąlygodami pagal priešdėlį, galime naudoti kalbos modelį tikėtiniems tęsiniams generuoti, o tai pasirodo naudinga daugelyje taikymų, pvz., automatinio užbaigimo funkcijose.


## Pratimai

1. Ar įgyvendintas kalbos modelis prognozuoja kitą teksto vienetą pagal visus ankstesnius teksto vienetus iki pat pirmojo *Laiko mašinoje*?
1. Kuris hiperparametras valdo prognozavimui naudojamos istorijos ilgį?
1. Parodykite, kad vienkartinis kodavimas yra ekvivalentus skirtingo įterpinio parinkimui kiekvienam objektui.
1. Pakoreguokite hiperparametrus (pvz., epochų skaičių, paslėptųjų vienetų skaičių, laiko žingsnių skaičių minirinkinyje ir mokymosi greitį), kad pagerintumėte painumą. Kiek žemai galite nusileisti, laikydamiesi šios paprastos architektūros?
1. Pakeiskite vienkartinį kodavimą išmokstamais įterpiniais. Ar tai lemia geresnį našumą?
1. Atlikite eksperimentą ir nustatykite, kaip gerai šis kalbos modelis,
   apmokytas su *Laiko mašina*, veikia su kitomis H. G. Wellso knygomis,
   pvz., *Pasaulių karas*.
1. Atlikite kitą eksperimentą ir įvertinkite šio modelio painumą
   su kitų autorių knygomis.
1. Pakeiskite prognozavimo metodą taip, kad jis naudotų atranką,
   o ne rinktųsi labiausiai tikėtiną kitą simbolį.
    * Kas nutinka?
    * Paslinkite modelį link labiau tikėtinų išvesčių, pvz.,
    imdami imtis iš $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$, kai $\alpha > 1$.
1. Paleiskite šio skyrelio kodą be gradiento apkarpymo. Kas nutinka?
1. Pakeiskite šiame skyrelyje naudojamą aktyvacijos funkciją ReLU
   ir pakartokite šio skyrelio eksperimentus. Ar vis dar reikia gradientų apkarpymo? Kodėl?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18014)
:end_tab:
