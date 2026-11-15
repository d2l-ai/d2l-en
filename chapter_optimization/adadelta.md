# Adadelta
:label:`sec_adadelta`

Adadelta yra dar vienas AdaGrad (:numref:`sec_adagrad`) variantas. Pagrindinis skirtumas yra tas, kad jis sumažina dydį, kuriuo mokymosi greitis prisitaiko prie koordinačių. Be to, tradiciškai sakoma, kad jis neturi mokymosi greičio, nes pačią pokyčio apimtį naudoja kaip kalibravimą būsimam pokyčiui. Algoritmą pasiūlė :citet:`Zeiler.2012`. Jis gana paprastas, turint omenyje iki šiol aptartus ankstesnius algoritmus.

## Algoritmas

Trumpai tariant, Adadelta naudoja du būsenos kintamuosius: $\mathbf{s}_t$ gradiento antrojo momento nuotėkio vidurkiui saugoti ir $\Delta\mathbf{x}_t$ paties modelio parametrų pokyčio antrojo momento nuotėkio vidurkiui saugoti. Atkreipkite dėmesį, kad naudojame originalius autorių žymėjimus ir pavadinimus suderinamumui su kitomis publikacijomis ir įgyvendinimais (nėra jokios kitos tikros priežasties, kodėl reikėtų naudoti skirtingus graikiškus kintamuosius tą patį tikslą atliekančiam parametrui žymėti momentų, Adagrad, RMSProp ir Adadelta metoduose).

Štai techninės Adadelta detalės. Tarkime, kad dienos parametras yra $\rho$; gauname šiuos nuotėkio atnaujinimus, panašiai kaip :numref:`sec_rmsprop`:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Skirtumas nuo :numref:`sec_rmsprop` yra tas, kad atnaujinimus atliekame su pakartotinai sumastelintu gradientu $\mathbf{g}_t'$, t. y.

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Taigi kas yra pakartotinai sumastelintas gradientas $\mathbf{g}_t'$? Jį galime apskaičiuoti taip:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

kur $\Delta \mathbf{x}_{t-1}$ yra pakartotinai sumastelintų gradientų $\mathbf{g}_t'$ kvadratų nuotėkio vidurkis. $\Delta \mathbf{x}_{0}$ inicializuojame $0$ ir atnaujiname jį kiekviename žingsnyje su $\mathbf{g}_t'$, t. y.

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

o $\epsilon$ (maža reikšmė, pavyzdžiui, $10^{-5}$) pridedamas skaitiniam stabilumui išlaikyti.



## Įgyvendinimas

Adadelta kiekvienam kintamajam turi išlaikyti du būsenos kintamuosius: $\mathbf{s}_t$ ir $\Delta\mathbf{x}_t$. Taip gaunamas toks įgyvendinimas.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Pasirinkus $\rho = 0.9$, kiekvieno parametro atnaujinimo pusėjimo trukmė yra 10. Tai paprastai veikia gana gerai. Gauname tokį elgesį.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Glaustam įgyvendinimui tiesiog naudojame Adadelta algoritmą iš aukšto lygio API. Taip gauname šią vienos eilutės komandą daug kompaktiškesniam iškvietimui.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it is converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Santrauka

* Adadelta neturi mokymosi greičio parametro. Vietoj to mokymosi greičiui adaptuoti jis naudoja paties parametrų pokyčio spartą.
* Adadelta reikia dviejų būsenos kintamųjų gradiento antriesiems momentams ir parametrų pokyčiui saugoti.
* Adadelta naudoja nuotėkio vidurkius, kad palaikytų einamąjį atitinkamos statistikos įvertį.

## Pratimai

1. Keiskite $\rho$ reikšmę. Kas nutinka?
1. Parodykite, kaip įgyvendinti algoritmą nenaudojant $\mathbf{g}_t'$. Kodėl tai galėtų būti gera idėja?
1. Ar Adadelta tikrai neturi mokymosi greičio? Ar galite rasti optimizavimo uždavinių, kurie sugadina Adadelta?
1. Palyginkite Adadelta su Adagrad ir RMSProp ir aptarkite jų konvergavimo elgesį.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1077)
:end_tab:
