# RMSProp
:label:`sec_rmsprop`


Viena pagrindinių :numref:`sec_adagrad` problemų yra ta, kad mokymosi greitis mažėja pagal iš anksto nustatytą grafiką, efektyviai $\mathcal{O}(t^{-\frac{1}{2}})$. Nors tai paprastai tinka iškiliosioms problemoms, neiškiliosioms, tokioms kaip sutinkamos giliajame mokymesi, tai gali būti neidealu. Vis dėlto koordinatėms specifinis Adagrad prisitaikymas yra labai pageidautinas kaip pradinių sąlygų tikrinimas.

:citet:`Tieleman.Hinton.2012` pasiūlė RMSProp algoritmą kaip paprastą sprendimą, atskiriantį greičio planavimą nuo koordinatėms adaptyvių mokymosi greičių. Problema ta, kad Adagrad kaupia gradiento $\mathbf{g}_t$ kvadratus būsenos vektoriuje $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. Dėl to $\mathbf{s}_t$ be paliovos auga be ribų dėl normalizavimo nebuvimo, iš esmės tiesiškai, algoritmui konverguojant.

Vienas būdas ištaisyti šią problemą būtų naudoti $\mathbf{s}_t / t$. Pagrįstiems $\mathbf{g}_t$ skirstiniams tai konverguos. Deja, gali užtrukti labai ilgai, kol ribinis elgesys pradės turėti reikšmės, nes procedūra prisimena visą reikšmių trajektoriją. Alternatyva yra naudoti nutekėjusį vidurkį taip pat, kaip naudojome momentų metode, t. y. $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ su tam tikru parametru $\gamma > 0$. Palikus visas kitas dalis nepakeistas, gaunamas RMSProp.

## Algoritmas

Užrašykime lygtis išsamiai.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Konstanta $\epsilon > 0$ paprastai nustatoma $10^{-6}$, kad išvengtume dalybos iš nulio ar pernelyg didelių žingsnių dydžių. Turėdami šį išplėtimą, galime laisvai valdyti mokymosi greitį $\eta$ nepriklausomai nuo mastelio keitimo, taikomo kiekvienai koordinatei atskirai. Nutekėjusių vidurkių atžvilgiu galime taikyti tą pačią logiką, kaip anksčiau taikėme momentų metode. Išplėtus $\mathbf{s}_t$ apibrėžimą, gauname

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Kaip ir anksčiau :numref:`sec_momentum`, naudojame $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. Taigi svorių suma normalizuojama iki $1$, o stebėjimo pusėjimo trukmė yra $\gamma^{-1}$. Pavaizduokime svorius paskutiniams 40 laiko žingsnių su įvairiomis $\gamma$ reikšmėmis.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Įgyvendinimas nuo nulio

Kaip ir anksčiau, naudojame kvadratinę funkciją $f(\mathbf{x})=0.1x_1^2+2x_2^2$ RMSProp trajektorijai stebėti. Prisiminkite, kad :numref:`sec_adagrad`, kai naudojome Adagrad su 0,4 mokymosi greičiu, kintamieji vėlesniuose algoritmo etapuose judėjo tik labai lėtai, nes mokymosi greitis mažėjo per greitai. Kadangi $\eta$ valdomas atskirai, RMSProp taip nenutinka.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Toliau įgyvendiname RMSProp, kuris bus naudojamas giliajame tinkle. Tai lygiai taip pat paprasta.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Pradinį mokymosi greitį nustatome 0,01, o svorio narį $\gamma$ – 0,9. Tai yra, $\mathbf{s}$ vidutiniškai agreguoja paskutinius $1/(1-\gamma) = 10$ gradiento kvadrato stebėjimų.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Glaustas įgyvendinimas

Kadangi RMSProp yra gana populiarus algoritmas, jis taip pat prieinamas `Trainer` egzemplioriuje. Tereikia jį sukurti naudojant algoritmą, pavadintą `rmsprop`, ir priskiriant $\gamma$ parametrui `gamma1`.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Santrauka

* RMSProp labai panašus į Adagrad tuo, kad abu naudoja gradiento kvadratą koeficientams mastelio keitimui.
* RMSProp su momentų metodu dalijasi nutekėjusiu vidurkinimu. Tačiau RMSProp šį metodą naudoja koeficientams specifiniam pradinių sąlygų tikrinimui koreguoti.
* Praktikoje mokymosi greitį turi suplanuoti eksperimentuotojas.
* Koeficientas $\gamma$ nustato, kokia ilga yra istorija koreguojant kiekvienos koordinatės mastelį.

## Pratimai

1. Kas eksperimentiškai nutinka, jei nustatome $\gamma = 1$? Kodėl?
1. Pasukite optimizavimo uždavinį, kad minimizuotumėte $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Kas nutinka konvergavimui?
1. Išbandykite, kas nutinka RMSProp realioje mašininio mokymosi problemoje, pavyzdžiui, apmokant Fashion-MNIST. Eksperimentuokite su skirtingais mokymosi greičio koregavimo pasirinkimais.
1. Ar norėtumėte koreguoti $\gamma$ optimizavimui vykstant? Kiek RMSProp jautrus tam?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1075)
:end_tab:
