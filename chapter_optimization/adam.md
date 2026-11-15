# Adam
:label:`sec_adam`

Diskusijose, vedusiose į šį skyrelį, susidūrėme su nemažai efektyvaus optimizavimo metodų. Čia juos išsamiai apibendrinkime:

* Matėme, kad :numref:`sec_sgd` yra veiksmingesnis nei gradientinis nusileidimas sprendžiant optimizavimo uždavinius, pavyzdžiui, dėl savaiminio atsparumo perteklinei duomenų informacijai.
* Matėme, kad :numref:`sec_minibatch_sgd` suteikia reikšmingą papildomą efektyvumą dėl vektorizavimo, kai viename minirinkinyje naudojamos didesnės stebėjimų aibės. Tai yra esminis efektyvaus daugiamašinio, daugelio GPU ir apskritai lygiagretaus apdorojimo veiksnys.
* :numref:`sec_momentum` pridėjo mechanizmą praeities gradientų istorijai agreguoti, kad paspartintų konvergavimą.
* :numref:`sec_adagrad` naudojo koordinatėms specifinį mastelio keitimą, kad būtų galima gauti skaičiavimo požiūriu efektyvų prekondicionuotoją.
* :numref:`sec_rmsprop` atskyrė koordinatėms specifinį mastelio keitimą nuo mokymosi greičio koregavimo.

Adam :cite:`Kingma.Ba.2014` sujungia visus šiuos metodus į vieną efektyvų mokymosi algoritmą. Kaip ir tikėtasi, tai algoritmas, kuris tapo gana populiarus kaip vienas tvirčiausių ir veiksmingiausių giliajame mokymesi naudojamų optimizavimo algoritmų. Tačiau jis nėra be trūkumų. Visų pirma :cite:`Reddi.Kale.Kumar.2019` parodo, kad yra situacijų, kai Adam gali diverguoti dėl prastos dispersijos kontrolės. Tolesniame darbe :citet:`Zaheer.Reddi.Sachan.ea.2018` pasiūlė Adam skubų pataisymą, vadinamą Yogi, kuris sprendžia šias problemas. Apie tai vėliau. Kol kas apžvelkime Adam algoritmą.

## Algoritmas

Vienas pagrindinių Adam komponentų yra tas, kad jis naudoja eksponentiškai svertinius slankiuosius vidurkius (dar žinomus kaip nuotėkio vidurkinimas), kad gautų ir momento, ir gradiento antrojo momento įverčius. Tai yra, jis naudoja būsenos kintamuosius

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Čia $\beta_1$ ir $\beta_2$ yra neneigiami svorio parametrai. Įprasti jų pasirinkimai yra $\beta_1 = 0.9$ ir $\beta_2 = 0.999$. Tai yra, dispersijos įvertis juda *daug lėčiau* nei momento narys. Atkreipkite dėmesį, kad jei inicializuojame $\mathbf{v}_0 = \mathbf{s}_0 = 0$, pradžioje turime didelį poslinkį link mažesnių reikšmių. Tai galima ištaisyti pasinaudojus faktu, kad $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$, ir pernormuoti narius. Atitinkamai normuoti būsenos kintamieji yra

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Turėdami tinkamus įverčius, galime užrašyti atnaujinimo lygtis. Pirmiausia pakeičiame gradiento mastelį labai panašiai kaip RMSProp ir gauname

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Skirtingai nei RMSProp, mūsų atnaujinimas naudoja momentą $\hat{\mathbf{v}}_t$, o ne patį gradientą. Be to, yra nedidelis kosmetinis skirtumas, nes mastelio keitimas vyksta naudojant $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ vietoj $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Pirmasis, galima teigti, praktikoje veikia kiek geriau, todėl nukrypstama nuo RMSProp. Paprastai pasirenkame $\epsilon = 10^{-6}$ kaip gerą kompromisą tarp skaitinio stabilumo ir tikslumo.

Dabar turime visas dalis atnaujinimams apskaičiuoti. Tai šiek tiek antiklimaksinė dalis, ir turime paprastą atnaujinimą:

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Apžvelgus Adam dizainą, jo įkvėpimas aiškus. Momentas ir mastelis aiškiai matomi būsenos kintamuosiuose. Jų gana savotiškas apibrėžimas verčia šalinti narių poslinkį (tai būtų galima ištaisyti šiek tiek kitokia inicializacija ir atnaujinimo sąlyga). Antra, abiejų narių derinimas yra gana paprastas, turint RMSProp. Galiausiai aiškus mokymosi greitis $\eta$ leidžia valdyti žingsnio ilgį, kad būtų sprendžiamos konvergavimo problemos.

## Įgyvendinimas

Adam įgyvendinti nuo nulio nėra labai baisu. Patogumui laiko žingsnio skaitiklį $t$ saugome žodyne `hyperparams`. Be to, viskas paprasta.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Esame pasirengę naudoti Adam modeliui apmokyti. Naudojame mokymosi greitį $\eta = 0.01$.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Glaustesnis įgyvendinimas yra paprastas, nes `adam` yra vienas iš algoritmų, pateikiamų kaip Gluon `trainer` optimizavimo bibliotekos dalis. Todėl įgyvendinimui Gluon tereikia perduoti konfigūracijos parametrus.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Viena Adam problemų yra ta, kad jis gali nekonverguoti net iškiliosiose aplinkose, kai antrojo momento įvertis $\mathbf{s}_t$ išauga. Kaip sprendimą :citet:`Zaheer.Reddi.Sachan.ea.2018` pasiūlė patobulintą $\mathbf{s}_t$ atnaujinimą (ir inicializaciją). Kad suprastume, kas vyksta, perrašykime Adam atnaujinimą taip:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Kai tik $\mathbf{g}_t^2$ turi didelę dispersiją arba atnaujinimai reti, $\mathbf{s}_t$ gali per greitai pamiršti praeities reikšmes. Galimas šios problemos sprendimas yra $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ pakeisti $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Dabar atnaujinimo dydis nebepriklauso nuo nuokrypio dydžio. Taip gauname Yogi atnaujinimus

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Autoriai be to pataria momentą inicializuoti didesniame pradiniame rinkinyje, o ne tik pradiniu taškiniu įverčiu. Šias detales praleidžiame, nes jos nėra esminės diskusijai ir nes net be to konvergavimas išlieka gana geras.

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Santrauka

* Adam sujungia daugelio optimizavimo algoritmų ypatybes į gana tvirtą atnaujinimo taisyklę.
* Sukurtas RMSProp pagrindu, Adam taip pat naudoja EWMA minirinkinio stochastiniam gradientui.
* Adam naudoja poslinkio korekciją, kad kompensuotų lėtą pradžią vertinant momentą ir antrąjį momentą.
* Gradientams su didele dispersija galime susidurti su konvergavimo problemomis. Jas galima ištaisyti naudojant didesnius minirinkinius arba pereinant prie patobulinto $\mathbf{s}_t$ įverčio. Yogi siūlo tokią alternatyvą.

## Pratimai

1. Keiskite mokymosi greitį ir stebėkite bei analizuokite eksperimentų rezultatus.
1. Ar galite perrašyti momento ir antrojo momento atnaujinimus taip, kad nereikėtų poslinkio korekcijos?
1. Kodėl konverguojant reikia mažinti mokymosi greitį $\eta$?
1. Pabandykite sukonstruoti atvejį, kai Adam diverguoja, o Yogi konverguoja.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1078)
:end_tab:


:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1079)
:end_tab:
