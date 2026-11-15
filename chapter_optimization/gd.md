# Gradientinis nusileidimas
:label:`sec_gd`

Šiame skyrelyje pristatysime pagrindines *gradientinio nusileidimo* sąvokas.
Nors giliajame mokymesi jis naudojamas retai tiesiogiai, gradientinio nusileidimo supratimas yra raktas į stochastinio gradientinio nusileidimo algoritmų supratimą.
Pavyzdžiui, optimizavimo uždavinys gali diverguoti dėl pernelyg didelio mokymosi greičio. Šį reiškinį galima pastebėti jau gradientiniame nusileidime. Panašiai pradinių sąlygų tikrinimas yra įprastas gradientinio nusileidimo metodas ir perkeliamas į pažangesnius algoritmus.
Pradėkime nuo paprasto ypatingo atvejo.


## Vienmatis gradientinis nusileidimas

Gradientinis nusileidimas vienoje dimensijoje yra puikus pavyzdys paaiškinti, kodėl gradientinio nusileidimo algoritmas gali sumažinti tikslo funkcijos reikšmę. Panagrinėkime tam tikrą tolydžiai diferencijuojamą realiųjų reikšmių funkciją $f: \mathbb{R} \rightarrow \mathbb{R}$. Naudodami Teiloro skleidinį gauname

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Tai yra, pirmos eilės aproksimacijoje $f(x+\epsilon)$ nusako funkcijos reikšmė $f(x)$ ir pirmoji išvestinė $f'(x)$ taške $x$. Nėra nepagrįsta daryti prielaidą, kad mažiems $\epsilon$ judėjimas neigiamo gradiento kryptimi sumažins $f$. Kad viskas būtų paprasta, pasirenkame fiksuotą žingsnio dydį $\eta > 0$ ir parenkame $\epsilon = -\eta f'(x)$. Įstatę tai į aukščiau pateiktą Teiloro skleidinį, gauname

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Jei išvestinė $f'(x) \neq 0$ neišnyksta, darome pažangą, nes $\eta f'^2(x)>0$. Be to, visada galime pasirinkti pakankamai mažą $\eta$, kad aukštesnės eilės nariai taptų nesvarbūs. Taigi gauname

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Tai reiškia, kad jei naudojame

$$x \leftarrow x - \eta f'(x)$$

$x$ iteracijoms, funkcijos $f(x)$ reikšmė gali mažėti. Todėl gradientiniame nusileidime pirmiausia pasirenkame pradinę reikšmę $x$ ir konstantą $\eta > 0$, o paskui jas naudojame nuolat iteruodami $x$, kol pasiekiama sustojimo sąlyga, pavyzdžiui, kai gradiento dydis $|f'(x)|$ yra pakankamai mažas arba iteracijų skaičius pasiekė tam tikrą reikšmę.

Paprastumo dėlei pasirenkame tikslo funkciją $f(x)=x^2$, kad iliustruotume, kaip įgyvendinti gradientinį nusileidimą. Nors žinome, kad $x=0$ yra $f(x)$ minimizavimo sprendinys, vis tiek naudojame šią paprastą funkciją stebėti, kaip kinta $x$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Toliau kaip pradinę reikšmę naudojame $x=10$ ir tarkime, kad $\eta=0.2$. Naudodami gradientinį nusileidimą $x$ iteruoti 10 kartų, matome, kad galiausiai $x$ reikšmė artėja prie optimalaus sprendinio.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

Optimizavimo per $x$ eigą galima nubraižyti taip.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Mokymosi greitis
:label:`subsec_gd-learningrate`

Mokymosi greitį $\eta$ gali nustatyti algoritmo kūrėjas. Jei naudojame pernelyg mažą mokymosi greitį, $x$ atnaujinamas labai lėtai, todėl reikia daugiau iteracijų geresniam sprendiniui gauti. Kad parodytume, kas nutinka tokiu atveju, panagrinėkime pažangą tame pačiame optimizavimo uždavinyje, kai $\eta = 0.05$. Kaip matome, net po 10 žingsnių vis dar esame labai toli nuo optimalaus sprendinio.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Ir atvirkščiai, jei naudojame pernelyg didelį mokymosi greitį, $\left|\eta f'(x)\right|$ gali būti per didelis pirmos eilės Teiloro skleidinio formulei. Tai yra, narys $\mathcal{O}(\eta^2 f'^2(x))$ :eqref:`gd-taylor-2` gali tapti reikšmingas. Tokiu atveju negalime garantuoti, kad $x$ iteravimas sumažins $f(x)$ reikšmę. Pavyzdžiui, kai mokymosi greitį nustatome $\eta=1.1$, $x$ peršoka optimalų sprendinį $x=0$ ir palaipsniui diverguoja.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Lokalieji minimumai

Kad iliustruotume, kas nutinka neiškiliosioms funkcijoms, panagrinėkime atvejį $f(x) = x \cdot \cos(cx)$ su tam tikra konstanta $c$. Ši funkcija turi be galo daug lokaliųjų minimumų. Priklausomai nuo mokymosi greičio pasirinkimo ir nuo to, kiek gerai sąlygotas uždavinys, galime baigti viename iš daugelio sprendinių. Toliau pateiktas pavyzdys iliustruoja, kaip (nerealiai) didelis mokymosi greitis lems prastą lokalųjį minimumą.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Daugiamatis gradientinis nusileidimas

Dabar, kai geriau intuityviai suprantame vienmatį atvejį, panagrinėkime situaciją, kai $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. Tai yra, tikslo funkcija $f: \mathbb{R}^d \to \mathbb{R}$ atvaizduoja vektorius į skaliarus. Atitinkamai jos gradientas taip pat yra daugiamatis. Tai vektorius, sudarytas iš $d$ dalinių išvestinių:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Kiekvienas gradiento dalinės išvestinės elementas $\partial f(\mathbf{x})/\partial x_i$ nurodo $f$ kitimo spartą taške $\mathbf{x}$ įvesties $x_i$ atžvilgiu. Kaip ir anksčiau vienmačiu atveju, galime naudoti atitinkamą daugelio kintamųjų funkcijų Teiloro aproksimaciją, kad gautume supratimą, ką turėtume daryti. Visų pirma turime, kad

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Kitaip tariant, iki antros eilės narių $\boldsymbol{\epsilon}$ atžvilgiu staigiausio nusileidimo kryptį nusako neigiamas gradientas $-\nabla f(\mathbf{x})$. Pasirinkus tinkamą mokymosi greitį $\eta > 0$, gaunamas prototipinis gradientinio nusileidimo algoritmas:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Kad pamatytume, kaip algoritmas elgiasi praktikoje, sukonstruokime tikslo funkciją $f(\mathbf{x})=x_1^2+2x_2^2$, kurios įvestis yra dvimatis vektorius $\mathbf{x} = [x_1, x_2]^\top$, o išvestis – skaliaras. Gradientas yra $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Stebėsime $\mathbf{x}$ trajektoriją gradientinio nusileidimo metu, pradedant nuo pradinės padėties $[-5, -2]$.

Pirmiausia mums reikia dar dviejų pagalbinių funkcijų. Pirmoji naudoja atnaujinimo funkciją ir taiko ją 20 kartų pradinei reikšmei. Antroji pagalbinė funkcija vizualizuoja $\mathbf{x}$ trajektoriją.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Toliau stebime optimizavimo kintamojo $\mathbf{x}$ trajektoriją, kai mokymosi greitis $\eta = 0.1$. Matome, kad po 20 žingsnių $\mathbf{x}$ reikšmė artėja prie savo minimumo $[0, 0]$. Pažanga gana tvarkinga, nors ir gana lėta.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Adaptyvieji metodai

Kaip matėme :numref:`subsec_gd-learningrate`, mokymosi greitį $\eta$ pasirinkti „tiksliai tinkamą“ yra sudėtinga. Jei pasirenkame per mažą, darome mažai pažangos. Jei pasirenkame per didelį, sprendinys svyruoja, o blogiausiu atveju gali net diverguoti. Ką, jei galėtume nustatyti $\eta$ automatiškai arba išvis atsikratytume poreikio rinktis mokymosi greitį?
Antros eilės metodai, kurie žiūri ne tik į tikslo funkcijos reikšmę ir gradientą,
bet ir į jos *kreivumą*, gali padėti šiuo atveju. Nors šių metodų negalima tiesiogiai taikyti giliajam mokymuisi dėl skaičiavimo kainos, jie suteikia naudingos intuicijos, kaip projektuoti pažangius optimizavimo algoritmus, imituojančius daugelį toliau aprašytų algoritmų pageidaujamų savybių.


### Niutono metodas

Peržiūrėdami kokios nors funkcijos $f: \mathbb{R}^d \rightarrow \mathbb{R}$ Teiloro skleidinį, nematome priežasties sustoti po pirmojo nario. Iš tikrųjų galime jį užrašyti kaip

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Kad išvengtume sudėtingo žymėjimo, apibrėžiame $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$ kaip $f$ Hesės matricą, kuri yra $d \times d$ matrica. Mažiems $d$ ir paprastiems uždaviniams $\mathbf{H}$ lengva apskaičiuoti. Tačiau giliesiems neuroniniams tinklams $\mathbf{H}$ gali būti nepriimtinai didelė dėl $\mathcal{O}(d^2)$ elementų saugojimo kainos. Be to, ją gali būti pernelyg brangu apskaičiuoti atgaliniu sklidimu. Kol kas nekreipkime dėmesio į tokius svarstymus ir pažiūrėkime, koks algoritmas gautųsi.

Juk $f$ minimumas tenkina $\nabla f = 0$.
Laikydamiesi diferencialinio skaičiavimo taisyklių :numref:`subsec_calculus-grad`,
išvestinę :eqref:`gd-hot-taylor` $\boldsymbol{\epsilon}$ atžvilgiu ir ignoruodami aukštesnės eilės narius, gauname

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Tai yra, kaip optimizavimo uždavinio dalį turime invertuoti Hesės matricą $\mathbf{H}$.

Kaip paprastas pavyzdys, kai $f(x) = \frac{1}{2} x^2$, turime $\nabla f(x) = x$ ir $\mathbf{H} = 1$. Taigi bet kuriam $x$ gauname $\epsilon = -x$. Kitaip tariant, *vieno* žingsnio pakanka tobulai konverguoti, nereikia jokio koregavimo! Deja, čia mums šiek tiek pasisekė: Teiloro skleidinys buvo tikslus, nes $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$.

Pažiūrėkime, kas nutinka kituose uždaviniuose.
Turint iškilią hiperbolinio kosinuso funkciją $f(x) = \cosh(cx)$ su tam tikra konstanta $c$, matome, kad
globalusis minimumas taške $x=0$ pasiekiamas
po kelių iteracijų.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Dabar panagrinėkime *neiškiliąją* funkciją, pavyzdžiui, $f(x) = x \cos(c x)$ su tam tikra konstanta $c$. Juk atkreipkite dėmesį, kad Niutono metode galiausiai dalijame iš Hesės matricos. Tai reiškia, kad jei antroji išvestinė yra *neigiama*, galime eiti *didėjančios* $f$ reikšmės kryptimi.
Tai lemtinga algoritmo yda.
Pažiūrėkime, kas nutinka praktikoje.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Tai nepavyko spektakliškai. Kaip tai ištaisyti? Vienas būdas būtų „ištaisyti“ Hesės matricą vietoj jos paėmus jos absoliutinę reikšmę. Kita strategija yra grąžinti mokymosi greitį. Atrodo, kad tai paneigia tikslą, bet ne visai. Turėdami antros eilės informaciją, galime būti atsargūs, kai kreivumas didelis, ir daryti ilgesnius žingsnius, kai tikslo funkcija plokštesnė.
Pažiūrėkime, kaip tai veikia su kiek mažesniu mokymosi greičiu, tarkime, $\eta = 0.5$. Kaip matome, turime gana efektyvų algoritmą.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Konvergavimo analizė

Analizuojame tik Niutono metodo konvergavimo greitį kokiai nors iškiliajai ir tris kartus diferencijuojamai tikslo funkcijai $f$, kurios antroji išvestinė nelygi nuliui, t. y. $f'' > 0$. Daugiamatis įrodymas yra paprastas toliau pateikto vienmačio argumento plėtinys ir praleidžiamas, nes jis mums daug nepadeda intuicijos požiūriu.

Pažymėkime $x^{(k)}$ reikšmę $x$ $k$-ojoje iteracijoje ir tegul $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$ yra atstumas nuo optimumo $k$-ojoje iteracijoje. Pagal Teiloro skleidinį sąlygą $f'(x^*) = 0$ galima užrašyti kaip

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

kuri galioja tam tikram $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$. Padalijus aukščiau pateiktą skleidinį iš $f''(x^{(k)})$, gauname

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Prisiminkite, kad turime atnaujinimą $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$.
Įstatę šią atnaujinimo lygtį ir paėmę abiejų pusių absoliutinę reikšmę, turime

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Todėl, kai tik esame srityje, kurioje $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ aprėžta, turime kvadratiškai mažėjančią paklaidą

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$


Pastebėkime, kad optimizavimo tyrėjai tai vadina *tiesiniu* konvergavimu, o tokia sąlyga kaip $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ būtų vadinama *pastovia* konvergavimo sparta.
Atkreipkite dėmesį, kad ši analizė turi keletą išlygų.
Pirma, iš tikrųjų neturime didelės garantijos, kada pasieksime greito konvergavimo sritį. Vietoj to žinome tik tai, kad kai ją pasieksime, konvergavimas bus labai greitas. Antra, ši analizė reikalauja, kad $f$ būtų gerai besielgianti iki aukštesnės eilės išvestinių. Tai susiveda į tai, kad $f$ neturėtų jokių „netikėtų“ savybių, susijusių su tuo, kaip ji gali keisti savo reikšmes.



### Pradinių sąlygų tikrinimas

Visai nenuostabu, kad pilnos Hesės matricos skaičiavimas ir saugojimas yra labai brangus. Todėl pageidautina rasti alternatyvų. Vienas būdas pagerinti padėtį yra *pradinių sąlygų tikrinimas*. Jis vengia skaičiuoti visą Hesės matricą ir apskaičiuoja tik *įstrižainės* elementus. Tai veda prie tokios formos atnaujinimo algoritmų:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


Nors tai nėra visai taip gerai kaip pilnas Niutono metodas, vis tiek tai daug geriau nei jo nenaudoti.
Kad pamatytume, kodėl tai gali būti gera idėja, panagrinėkime situaciją, kai vienas kintamasis žymi aukštį milimetrais, o kitas – aukštį kilometrais. Darant prielaidą, kad abiem natūralus mastelis yra metrai, turime baisų parametrizacijų neatitikimą. Laimė, pradinių sąlygų tikrinimas tai pašalina. Efektyviai pradinių sąlygų tikrinimas su gradientiniu nusileidimu reiškia skirtingo mokymosi greičio pasirinkimą kiekvienam kintamajam (vektoriaus $\mathbf{x}$ koordinatei).
Kaip pamatysime vėliau, pradinių sąlygų tikrinimas lemia dalį naujovių stochastinio gradientinio nusileidimo optimizavimo algoritmuose.


### Gradientinis nusileidimas su tiesine paieška

Viena pagrindinių gradientinio nusileidimo problemų yra ta, kad galime peršokti tikslą arba padaryti nepakankamą pažangą. Paprastas problemos sprendimas yra naudoti tiesinę paiešką kartu su gradientiniu nusileidimu. Tai yra, naudojame kryptį, kurią nurodo $\nabla f(\mathbf{x})$, o tada atliekame dvejetainę paiešką, kuris mokymosi greitis $\eta$ minimizuoja $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$.

Šis algoritmas konverguoja sparčiai (analizės ir įrodymo žr., pvz., :citet:`Boyd.Vandenberghe.2004`). Tačiau giliojo mokymosi tikslais tai nėra visai įgyvendinama, nes kiekvienam tiesinės paieškos žingsniui reikėtų įvertinti tikslo funkciją visame duomenų rinkinyje. Tai pernelyg brangu.

## Santrauka

* Mokymosi greičiai svarbūs. Per dideli – diverguojame, per maži – nedarome pažangos.
* Gradientinis nusileidimas gali įstrigti lokaliuosiuose minimumuose.
* Didelėse dimensijose mokymosi greičio koregavimas yra sudėtingas.
* Pradinių sąlygų tikrinimas gali padėti koreguoti mastelį.
* Niutono metodas yra daug greitesnis, kai jis pradeda tinkamai veikti iškiliuosiuose uždaviniuose.
* Būkite atsargūs naudodami Niutono metodą be jokių koregavimų neiškiliesiems uždaviniams.

## Pratimai

1. Eksperimentuokite su skirtingais mokymosi greičiais ir tikslo funkcijomis gradientiniam nusileidimui.
1. Įgyvendinkite tiesinę paiešką iškiliajai funkcijai minimizuoti intervale $[a, b]$.
    1. Ar dvejetainei paieškai reikia išvestinių, t. y. nuspręsti, ar rinktis $[a, (a+b)/2]$, ar $[(a+b)/2, b]$.
    1. Kokia sparti yra algoritmo konvergavimo sparta?
    1. Įgyvendinkite algoritmą ir pritaikykite jį $\log (\exp(x) + \exp(-2x -3))$ minimizavimui.
1. Suprojektuokite tikslo funkciją, apibrėžtą $\mathbb{R}^2$, kuriai gradientinis nusileidimas yra nepaprastai lėtas. Užuomina: skirtingas koordinates keiskite skirtingu masteliu.
1. Įgyvendinkite lengvą Niutono metodo versiją naudodami pradinių sąlygų tikrinimą:
    1. Kaip pradinių sąlygų tikrinimą naudokite įstrižainės Hesės matricą.
    1. Naudokite jos absoliutines reikšmes, o ne tikrąsias (galbūt su ženklu) reikšmes.
    1. Pritaikykite tai aukščiau pateiktam uždaviniui.
1. Pritaikykite aukščiau pateiktą algoritmą daugeliui tikslo funkcijų (iškiliųjų ar ne). Kas nutinka, jei pasukate koordinates $45$ laipsniais?

[Diskusijos](https://discuss.d2l.ai/t/351)
