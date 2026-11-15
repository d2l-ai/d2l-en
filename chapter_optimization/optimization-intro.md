# Optimizavimas ir gilusis mokymasis
:label:`sec_optimization-intro`

Šiame skyrelyje aptarsime optimizavimo ir giliojo mokymosi sąryšį, taip pat optimizavimo taikymo giliajame mokymesi iššūkius.
Giliojo mokymosi uždaviniui paprastai pirmiausia apibrėžiame *nuostolio funkciją*. Turėdami nuostolio funkciją, galime naudoti optimizavimo algoritmą nuostoliui minimizuoti.
Optimizavime nuostolio funkcija dažnai vadinama optimizavimo uždavinio *tikslo funkcija*. Pagal tradiciją ir susitarimą dauguma optimizavimo algoritmų susiję su *minimizavimu*. Jei kada reikėtų maksimizuoti tikslą, yra paprastas sprendimas: tiesiog pakeiskite tikslo ženklą.

## Optimizavimo tikslas

Nors optimizavimas suteikia būdą minimizuoti giliojo mokymosi nuostolio funkciją, iš esmės optimizavimo ir giliojo mokymosi tikslai
iš principo skiriasi.
Pirmasis daugiausia susijęs su tikslo minimizavimu, o antrasis – su tinkamo modelio radimu, turint
ribotą duomenų kiekį.
:numref:`sec_generalization_basics`
išsamiai aptarėme šių dviejų tikslų skirtumą.
Pavyzdžiui,
apmokymo paklaida ir apibendrinimo paklaida paprastai skiriasi: kadangi optimizavimo algoritmo tikslo
funkcija dažniausiai yra nuostolio funkcija, pagrįsta
apmokymo duomenų rinkiniu, optimizavimo tikslas yra sumažinti apmokymo paklaidą.
Tačiau giliojo mokymosi (arba plačiau – statistinės išvados) tikslas yra
sumažinti apibendrinimo paklaidą.
Kad pasiektume antrąjį, be optimizavimo algoritmo naudojimo apmokymo paklaidai mažinti,
turime atkreipti dėmesį ir į persimokymą.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Kad iliustruotume minėtus skirtingus tikslus,
panagrinėkime
empirinę riziką ir riziką.
Kaip aprašyta
:numref:`subsec_empirical-risk-and-risk`,
empirinė rizika
yra vidutinis nuostolis
apmokymo duomenų rinkinyje,
o rizika yra tikėtinas nuostolis
visoje duomenų populiacijoje.
Toliau apibrėžiame dvi funkcijas:
rizikos funkciją `f`
ir empirinės rizikos funkciją `g`.
Tarkime, kad turime tik ribotą kiekį apmokymo duomenų.
Todėl čia `g` yra mažiau glotni nei `f`.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Žemiau pateiktas grafikas iliustruoja, kad empirinės rizikos minimumas apmokymo duomenų rinkinyje gali būti kitoje vietoje nei rizikos (apibendrinimo paklaidos) minimumas.

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Optimizavimo iššūkiai giliajame mokymesi

Šiame skyriuje daugiausia dėmesio skirsime optimizavimo algoritmų našumui minimizuojant tikslo funkciją, o ne
modelio apibendrinimo paklaidai.
:numref:`sec_linear_regression`
optimizavimo uždaviniuose skyrėme analitinius ir skaitinius sprendinius.
Giliajame mokymesi dauguma tikslo funkcijų yra
sudėtingos ir neturi analitinių sprendinių. Vietoj to turime naudoti skaitinius
optimizavimo algoritmus.
Visi šio skyriaus optimizavimo algoritmai
patenka į šią
kategoriją.

Giliojo mokymosi optimizavime yra daug iššūkių. Vieni erzinančiausių yra lokalieji minimumai, balno taškai ir nykstantys gradientai.
Pažvelkime į juos.


### Lokalieji minimumai

Bet kuriai tikslo funkcijai $f(x)$,
jei $f(x)$ reikšmė taške $x$ yra mažesnė nei $f(x)$ reikšmės bet kuriuose kituose taško $x$ aplinkos taškuose, tai $f(x)$ gali būti lokalusis minimumas.
Jei $f(x)$ reikšmė taške $x$ yra tikslo funkcijos minimumas visoje apibrėžimo srityje,
tai $f(x)$ yra globalusis minimumas.

Pavyzdžiui, turint funkciją

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

galime aproksimuoti šios funkcijos lokalųjį ir globalųjį minimumus.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Giliojo mokymosi modelių tikslo funkcija paprastai turi daug lokaliųjų optimumų.
Kai skaitinis optimizavimo uždavinio sprendinys yra arti lokaliojo optimumo, paskutinės iteracijos gautas skaitinis sprendinys gali minimizuoti tikslo funkciją tik *lokaliai*, o ne *globaliai*, nes tikslo funkcijos sprendinių gradientas artėja prie nulio arba tampa nuliu.
Tik tam tikras triukšmo kiekis gali išstumti parametrą iš lokaliojo minimumo. Iš tiesų tai viena iš naudingų
minirinkinių stochastinio gradientinio nusileidimo savybių, kai natūrali gradientų kaita tarp minirinkinių gali išstumti parametrus iš lokaliųjų minimumų.


### Balno taškai

Be lokaliųjų minimumų, balno taškai yra dar viena gradientų išnykimo priežastis. *Balno taškas* yra bet kuri vieta, kurioje visi funkcijos gradientai išnyksta, bet kuri nėra nei globalusis, nei lokalusis minimumas.
Panagrinėkime funkciją $f(x) = x^3$. Jos pirmoji ir antroji išvestinės išnyksta, kai $x=0$. Optimizavimas gali sustoti šiame taške, nors jis nėra minimumas.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Balno taškai aukštesnėse dimensijose yra dar klastingesni, kaip rodo toliau pateiktas pavyzdys. Panagrinėkime funkciją $f(x, y) = x^2 - y^2$. Jos balno taškas yra $(0, 0)$. Tai maksimumas $y$ atžvilgiu ir minimumas $x$ atžvilgiu. Be to, ji *atrodo* kaip balnas, nuo ko ši matematinė savybė ir gavo pavadinimą.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Tarkime, kad funkcijos įvestis yra $k$ dimensijų vektorius, o
išvestis – skaliaras, taigi jos Hesės matrica turės $k$ tikrinių reikšmių.
Funkcijos sprendinys
gali būti lokalusis minimumas, lokalusis maksimumas arba balno taškas
toje padėtyje, kur funkcijos gradientas lygus nuliui:

* Kai funkcijos Hesės matricos tikrinės reikšmės nulinio gradiento padėtyje yra visos teigiamos, funkcija turi lokalųjį minimumą.
* Kai funkcijos Hesės matricos tikrinės reikšmės nulinio gradiento padėtyje yra visos neigiamos, funkcija turi lokalųjį maksimumą.
* Kai funkcijos Hesės matricos tikrinės reikšmės nulinio gradiento padėtyje yra ir neigiamos, ir teigiamos, funkcija turi balno tašką.

Aukštos dimensijos uždaviniuose tikimybė, kad bent *kai kurios* tikrinės reikšmės bus neigiamos, yra gana didelė. Dėl to balno taškai tikėtinesni nei lokalieji minimumai. Kai kurias šios situacijos išimtis aptarsime kitame skyrelyje, pristatydami iškilumą. Trumpai tariant, iškiliosios funkcijos yra tos, kurių Hesės matricos tikrinės reikšmės niekada nėra neigiamos. Deja, dauguma giliojo mokymosi uždavinių nepriklauso šiai kategorijai. Vis dėlto tai puikus įrankis optimizavimo algoritmams tirti.

### Nykstantys gradientai

Tikriausiai klastingiausia susiduriama problema yra nykstantis gradientas.
Prisiminkite dažniausiai naudojamas aktyvacijos funkcijas ir jų išvestines :numref:`subsec_activation-functions`.
Pavyzdžiui, tarkime, kad norime minimizuoti funkciją $f(x) = \tanh(x)$ ir atsitiktinai pradedame nuo $x = 4$. Kaip matome, $f$ gradientas yra artimas nuliui.
Tiksliau, $f'(x) = 1 - \tanh^2(x)$, taigi $f'(4) = 0.0013$.
Todėl optimizavimas ilgą laiką stringa, kol padarome pažangą. Paaiškėja, kad tai viena iš priežasčių, kodėl giliojo mokymosi modelių apmokymas buvo gana sudėtingas prieš atsirandant ReLU aktyvacijos funkcijai.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Kaip matėme, giliojo mokymosi optimizavimas kupinas iššūkių. Laimė, yra tvirtas algoritmų rinkinys, kuris veikia gerai ir yra lengvai naudojamas net pradedantiesiems. Be to, iš tikrųjų nebūtina rasti *geriausią* sprendinį. Lokalieji optimumai ar net jų apytiksliai sprendiniai vis tiek yra labai naudingi.

## Santrauka

* Apmokymo paklaidos minimizavimas *negarantuoja*, kad rasime geriausią parametrų rinkinį apibendrinimo paklaidai minimizuoti.
* Optimizavimo uždaviniai gali turėti daug lokaliųjų minimumų.
* Uždavinys gali turėti dar daugiau balno taškų, nes paprastai uždaviniai nėra iškilieji.
* Nykstantys gradientai gali sustabdyti optimizavimą. Dažnai padeda uždavinio perparametrizavimas. Gerai inicializuoti parametrai taip pat gali būti naudingi.


## Pratimai

1. Panagrinėkime paprastą MLP su vienu paslėptuoju sluoksniu, kurio paslėptasis sluoksnis turi, tarkime, $d$ dimensijų, ir viena išvestimi. Įrodykite, kad bet kuriam lokaliajam minimumui yra bent $d!$ ekvivalentiškų sprendinių, kurie elgiasi identiškai.
1. Tarkime, kad turime simetrinę atsitiktinę matricą $\mathbf{M}$, kurios elementai
   $M_{ij} = M_{ji}$ yra atsitiktinai gaunami iš tam tikro tikimybių skirstinio
   $p_{ij}$. Be to, tarkime, kad $p_{ij}(x) = p_{ij}(-x)$, t. y. kad
   skirstinys yra simetrinis (išsamiau žr., pvz., :citet:`Wigner.1958`).
    1. Įrodykite, kad tikrinių reikšmių skirstinys taip pat yra simetrinis. Tai yra, bet kuriam tikriniam vektoriui $\mathbf{v}$ tikimybė, kad su juo susijusi tikrinė reikšmė $\lambda$ tenkina $P(\lambda > 0) = P(\lambda < 0)$.
    1. Kodėl aukščiau pateiktas teiginys *neįrodo*, kad $P(\lambda > 0) = 0.5$?
1. Kokius kitus su giliojo mokymosi optimizavimu susijusius iššūkius galite įsivaizduoti?
1. Tarkime, kad norite išlaikyti (tikrą) rutulį ant (tikro) balno.
    1. Kodėl tai sunku?
    1. Ar galite šį efektą išnaudoti ir optimizavimo algoritmams?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/489)
:end_tab:
