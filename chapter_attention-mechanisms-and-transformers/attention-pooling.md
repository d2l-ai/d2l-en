# Dėmesio sutelkimas pagal panašumą

:label:`sec_attention-pooling`

Dabar, kai pristatėme pagrindinius dėmesio sutelkimo mechanizmo komponentus, panaudokime juos gana klasikinėje nuostatoje, būtent regresijai ir klasifikavimui per branduolinį tankio vertinimą :cite:`Nadaraya.1964,Watson.1964`. Ši atkarpa tiesiog suteikia papildomo fono: ji visiškai neprivaloma ir, jei reikia, gali būti praleista.
Iš esmės Nadaraya–Watsono įverčiai remiasi tam tikru panašumo branduoliu $\alpha(\mathbf{q}, \mathbf{k})$, susiejančiu užklausas $\mathbf{q}$ su raktais $\mathbf{k}$. Keli įprasti branduoliai yra

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

Galėtume pasirinkti dar daug kitų variantų. Plačiau apie tai žr. [Vikipedijos straipsnyje](https://en.wikipedia.org/wiki/Kernel_(statistics)) ir kaip branduolių pasirinkimas susijęs su branduoliniu tankio vertinimu, kartais dar vadinamu *Parzeno langais* :cite:`parzen1957consistent`. Visi branduoliai yra euristiniai ir juos galima derinti. Pavyzdžiui, galime keisti plotį ne tik globaliai, bet net kiekvienai koordinatei atskirai. Bet kuriuo atveju visi jie veda prie šios lygties, tinkamos ir regresijai, ir klasifikavimui:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

(Skaliarinės) regresijos atveju su stebėjimais $(\mathbf{x}_i, y_i)$ atitinkamai požymiams ir žymenims, $\mathbf{v}_i = y_i$ yra skaliarai, $\mathbf{k}_i = \mathbf{x}_i$ yra vektoriai, o užklausa $\mathbf{q}$ žymi naują vietą, kurioje reikia įvertinti $f$. (Daugelio klasių) klasifikavimo atveju naudojame $y_i$ vienkartinį kodavimą, kad gautume $\mathbf{v}_i$. Viena iš patogių šio įverčio savybių yra ta, kad jam nereikia apmokymo. Dar daugiau, jei tinkamai siaurintume branduolį didėjant duomenų kiekiui, metodas yra suderintas (angl. *consistent*) :cite:`mack1982weak`, t. y. jis konverguos į tam tikrą statistiškai optimalų sprendinį. Pradėkime nuo kai kurių branduolių tyrimo.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**Branduoliai ir duomenys**]

Visi šiame skyrelyje apibrėžti branduoliai $\alpha(\mathbf{k}, \mathbf{q})$ yra *postūmio ir sukimo invariantiški*; t. y. jei postumiame ir pasukame $\mathbf{k}$ ir $\mathbf{q}$ vienodai, $\alpha$ reikšmė nepasikeičia. Todėl paprastumo dėlei pasirenkame skaliarinius argumentus $k, q \in \mathbb{R}$ ir raktą $k = 0$ laikome atskaitos tašku. Tai duoda:

```{.python .input}
%%tab all
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

Skirtingi branduoliai atitinka skirtingas ribos (siekio) ir glotnumo sampratas. Pavyzdžiui, stačiakampis branduolys (angl. *boxcar*) atsižvelgia tik į stebėjimus, esančius ne toliau kaip $1$ atstumu (ar kitaip apibrėžtu hiperparametru), ir tai daro nesirinkdamas.

Kad pamatytume Nadaraya–Watsono vertinimą veikiant, apibrėžkime kai kuriuos apmokymo duomenis. Toliau naudojame priklausomybę

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

kur $\epsilon$ imamas iš normaliojo skirstinio su nuliniu vidurkiu ir vienetine dispersija. Ištraukiame 40 apmokymo pavyzdžių.

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Dėmesio sutelkimas per Nadaraya–Watsono regresiją**]

Turėdami duomenis ir branduolius, mums reikia tik funkcijos, apskaičiuojančios branduolinės regresijos įverčius. Atkreipkite dėmesį, kad taip pat norime gauti santykinius branduolio svorius, kad galėtume atlikti nedidelę diagnostiką. Todėl pirmiausia apskaičiuojame branduolį tarp visų apmokymo požymių (kovariančių) `x_train` ir visų validavimo požymių `x_val`. Gauname matricą, kurią vėliau normalizuojame. Padauginę ją iš apmokymo žymenų `y_train`, gauname įverčius.

Prisiminkite dėmesio sutelkimą :eqref:`eq_attention_pooling`. Tegul kiekvienas validavimo požymis yra užklausa, o kiekviena apmokymo požymio ir žymens pora yra rakto ir reikšmės pora. Todėl normalizuoti santykiniai branduolio svoriai (toliau `attention_w`) yra *dėmesio svoriai*.

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalization over keys for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

Pažiūrėkime, kokius įverčius sukuria skirtingi branduoliai.

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Pirmiausia krenta į akis tai, kad visi trys netrivialūs branduoliai (Gauso, stačiakampis ir Epanechnikovo) sukuria gana tinkamus įverčius, nelabai nutolusius nuo tikrosios funkcijos. Tik pastovusis branduolys, vedantis prie trivialaus įverčio $f(x) = \frac{1}{n} \sum_i y_i$, duoda gana nerealų rezultatą. Atidžiau patikrinkime dėmesio svorius:

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Vizualizacija aiškiai parodo, kodėl Gauso, stačiakampio ir Epanechnikovo įverčiai labai panašūs: juk jie gauti iš labai panašių dėmesio svorių, nepaisant skirtingos branduolio funkcinės formos. Tai kelia klausimą, ar taip yra visada.

## [**Dėmesio sutelkimo pritaikymas**]

Gauso branduolį galėtume pakeisti kitokio pločio branduoliu. T. y. galėtume naudoti
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$, kur $\sigma^2$ nustato branduolio plotį. Pažiūrėkime, ar tai paveikia rezultatus.

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Akivaizdu, kad kuo siauresnis branduolys, tuo mažiau glotnus įvertis. Tuo pat metu jis geriau prisitaiko prie lokalių svyravimų. Pažiūrėkime į atitinkamus dėmesio svorius.

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Kaip ir galima tikėtis, kuo siauresnis branduolys, tuo siauresnis didelių dėmesio svorių diapazonas. Taip pat akivaizdu, kad to paties pločio pasirinkimas gali būti neidealus. Iš tiesų :citet:`Silverman86` pasiūlė euristiką, priklausančią nuo lokalaus tankio. Pasiūlyta dar daug tokių „gudrybių“. Pavyzdžiui, :citet:`norelli2022asif` panaudojo panašią artimiausių kaimynų interpoliacijos techniką tarpmodaliniams vaizdų ir teksto vaizdams projektuoti.

Įžvalgus skaitytojas gali stebėtis, kodėl pateikiame šią gilią analizę metodui, kuriam daugiau nei pusė amžiaus. Pirma, tai vienas ankstyviausių šiuolaikinių dėmesio sutelkimo mechanizmų pirmtakų. Antra, jis puikiai tinka vizualizavimui. Trečia, ir tiek pat svarbu, jis parodo rankomis kurtų dėmesio sutelkimo mechanizmų ribas. Daug geresnė strategija yra *išmokti* mechanizmą, mokantis užklausų ir raktų vaizdus. Būtent tuo ir užsiimsime kituose skyreliuose.


## Santrauka

Nadaraya–Watsono branduolinė regresija yra ankstyvas dabartinių dėmesio sutelkimo mechanizmų pirmtakas.
Ją galima naudoti tiesiogiai, beveik be apmokymo ar derinimo, tiek klasifikavimui, tiek regresijai.
Dėmesio svoris priskiriamas pagal užklausos ir rakto panašumą (arba atstumą) ir pagal tai, kiek panašių stebėjimų yra turima.

## Pratimai

1. Parzeno langų tankio įverčiai yra $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$. Įrodykite, kad dvejetainiam klasifikavimui funkcija $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$, gauta Parzeno langais, yra ekvivalenti Nadaraya–Watsono klasifikavimui.
1. Įgyvendinkite stochastinį gradientinį nusileidimą, kad išmoktumėte gerą branduolių pločių reikšmę Nadaraya–Watsono regresijoje.
    1. Kas nutinka, jei tiesiog naudojate aukščiau pateiktus įverčius, kad tiesiogiai minimizuotumėte $(f(\mathbf{x_i}) - y_i)^2$? Užuomina: $y_i$ yra viena iš narių, naudojamų $f$ apskaičiuoti.
    1. Pašalinkite $(\mathbf{x}_i, y_i)$ iš $f(\mathbf{x}_i)$ įverčio ir optimizuokite branduolių pločius. Ar vis dar stebite persimokymą?
1. Tarkime, kad visi $\mathbf{x}$ guli ant vienetinės sferos, t. y. visi tenkina $\|\mathbf{x}\| = 1$. Ar galite supaprastinti narį $\|\mathbf{x} - \mathbf{x}_i\|^2$ eksponentėje? Užuomina: vėliau pamatysime, kad tai labai glaudžiai susiję su skaliarinės sandaugos dėmesio sutelkimu.
1. Prisiminkite, kad :citet:`mack1982weak` įrodė, jog Nadaraya–Watsono vertinimas yra suderintas. Kaip greitai turėtumėte mažinti dėmesio sutelkimo mechanizmo mastelį, gaunant daugiau duomenų? Pateikite šiek tiek intuicijos savo atsakymui. Ar tai priklauso nuo duomenų dimensijos? Kaip?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18026)
:end_tab:
