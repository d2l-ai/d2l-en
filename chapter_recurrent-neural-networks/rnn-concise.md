# Glaustas rekurentinių neuroninių tinklų įgyvendinimas
:label:`sec_rnn-concise`

Kaip ir dauguma mūsų įgyvendinimų nuo nulio,
:numref:`sec_rnn-scratch` buvo sukurtas
siekiant suteikti įžvalgą, kaip veikia kiekvienas komponentas.
Tačiau kai RNN naudojate kasdien
arba rašote produkcinį kodą,
norėsite labiau remtis bibliotekomis,
kurios sutrumpina ir įgyvendinimo laiką
(pateikdamos bibliotekų kodą įprastiems modeliams ir funkcijoms),
ir skaičiavimo laiką
(iki kraštutinumo optimizuodamos šiuos bibliotekų įgyvendinimus).
Šiame skyrelyje parodysime, kaip įgyvendinti
tą patį kalbos modelį efektyviau,
naudojant aukšto lygio API,
kurį teikia jūsų giliojo mokymosi karkasas.
Pradedame, kaip ir anksčiau, įkeldami
*Laiko mašinos* duomenų rinkinį.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
from jax import numpy as jnp
```

## [**Modelio apibrėžimas**]

Apibrėžiame šią klasę,
naudodami aukšto lygio API
įgyvendintą RNN.

:begin_tab:`mxnet`
Konkrečiai, kad inicializuotume paslėptąją būseną,
iškviečiame nario metodą `begin_state`.
Jis grąžina sąrašą, kuriame yra
pradinė paslėptoji būsena
kiekvienam minirinkinio pavyzdžiui,
kurios forma
(paslėptųjų sluoksnių skaičius, rinkinio dydis, paslėptųjų vienetų skaičius).
Kai kuriems vėliau pristatomiems modeliams
(pvz., ilgos trumpalaikės atminties),
šiame sąraše bus ir kitos informacijos.
:end_tab:

:begin_tab:`jax`
Flax šiuo metu nesiūlo RNNCell paprastiems RNN glaustai įgyvendinti.
Yra pažangesnių RNN variantų, tokių kaip LSTM ir GRU,
kurie yra prieinami Flax `linen` API.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

Paveldėdama iš klasės `RNNLMScratch` iš :numref:`sec_rnn-scratch`,
toliau pateikta klasė `RNNLM` apibrėžia pilną RNN pagrįstą kalbos modelį.
Atkreipkite dėmesį, kad turime sukurti atskirą pilnai sujungtą išvesties sluoksnį.

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## Mokymas ir prognozavimas

Prieš apmokydami modelį, [**padarykime prognozę
su modeliu, inicializuotu atsitiktiniais svoriais.**]
Kadangi tinklo neapmokėme,
jis sugeneruos beprasmes prognozes.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Toliau [**apmokome savo modelį, pasinaudodami aukšto lygio API**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

Palyginti su :numref:`sec_rnn-scratch`,
šis modelis pasiekia panašų painumą,
bet veikia greičiau dėl optimizuotų įgyvendinimų.
Kaip ir anksčiau, galime generuoti prognozuojamus teksto vienetus,
einančius po nurodytos priešdėlio eilutės.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Santrauka

Giliojo mokymosi karkasų aukšto lygio API teikia standartinių RNN įgyvendinimus.
Šios bibliotekos padeda neeikvoti laiko standartinių modelių įgyvendinimui iš naujo.
Be to,
karkasų įgyvendinimai dažnai yra labai optimizuoti,
todėl, palyginti su įgyvendinimais nuo nulio, gaunamas reikšmingas (skaičiavimo) našumo pagerėjimas.

## Pratimai

1. Ar galite priversti RNN modelį persimokyti naudodami aukšto lygio API?
1. Įgyvendinkite :numref:`sec_sequence` autoregresinį modelį naudodami RNN.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18015)
:end_tab:
