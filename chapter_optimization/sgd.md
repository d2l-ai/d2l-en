# Stochastinis gradientinis nusileidimas
:label:`sec_sgd`

Ankstesniuose skyriuose apmokymo procedūroje nuolat naudojome stochastinį gradientinį nusileidimą, tačiau nepaaiškinome, kodėl jis veikia.
Kad tai nušviestume,
:numref:`sec_gd`
ką tik aprašėme pagrindinius gradientinio nusileidimo principus.
Šiame skyrelyje toliau išsamiau aptarsime
*stochastinį gradientinį nusileidimą*.

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
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## Stochastiniai gradiento atnaujinimai

Giliajame mokymesi tikslo funkcija paprastai yra kiekvieno apmokymo duomenų rinkinio pavyzdžio nuostolio funkcijų vidurkis.
Turėdami $n$ pavyzdžių apmokymo duomenų rinkinį,
tarkime, kad $f_i(\mathbf{x})$ yra nuostolio funkcija
indekso $i$ apmokymo pavyzdžiui,
kur $\mathbf{x}$ yra parametrų vektorius.
Tada gauname tikslo funkciją

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

Tikslo funkcijos gradientas taške $\mathbf{x}$ apskaičiuojamas taip:

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Jei naudojamas gradientinis nusileidimas, kiekvienos nepriklausomo kintamojo iteracijos skaičiavimo kaina yra $\mathcal{O}(n)$, kuri auga tiesiškai su $n$. Todėl kai apmokymo duomenų rinkinys didesnis, gradientinio nusileidimo kaina kiekvienai iteracijai bus didesnė.

Stochastinis gradientinis nusileidimas (SGD) sumažina skaičiavimo kainą kiekvienoje iteracijoje. Kiekvienoje stochastinio gradientinio nusileidimo iteracijoje atsitiktinai tolygiai atrenkame duomenų pavyzdžių indeksą $i\in\{1,\ldots, n\}$ ir apskaičiuojame gradientą $\nabla f_i(\mathbf{x})$, kad atnaujintume $\mathbf{x}$:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

kur $\eta$ yra mokymosi greitis. Matome, kad skaičiavimo kaina kiekvienai iteracijai sumažėja nuo gradientinio nusileidimo $\mathcal{O}(n)$ iki pastovios $\mathcal{O}(1)$. Be to, norime pabrėžti, kad stochastinis gradientas $\nabla f_i(\mathbf{x})$ yra nepaslinktas viso gradiento $\nabla f(\mathbf{x})$ įvertis, nes

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Tai reiškia, kad vidutiniškai stochastinis gradientas yra geras gradiento įvertis.

Dabar palyginsime jį su gradientiniu nusileidimu, prie gradiento pridėdami atsitiktinį triukšmą su nuliniu vidurkiu ir vienetine dispersija, kad imituotume stochastinį gradientinį nusileidimą.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Kaip matome, kintamųjų trajektorija stochastiniame gradientiniame nusileidime yra daug triukšmingesnė nei ta, kurią stebėjome gradientiniame nusileidime :numref:`sec_gd`. Tai lemia stochastinė gradiento prigimtis. Tai yra, net kai atsiduriame netoli minimumo, vis tiek veikia momentinio gradiento per $\eta \nabla f_i(\mathbf{x})$ įnešama neapibrėžtis. Net po 50 žingsnių kokybė vis dar nėra tokia gera. Dar blogiau, ji nepagerės ir po papildomų žingsnių (skatiname eksperimentuoti su didesniu žingsnių skaičiumi, kad tai patvirtintumėte). Tai palieka vienintelę alternatyvą: keisti mokymosi greitį $\eta$. Tačiau jei pasirinksime per mažą, iš pradžių nepadarysime jokios prasmingos pažangos. Kita vertus, jei pasirinksime per didelį, negausime gero sprendinio, kaip matyti aukščiau. Vienintelis būdas išspręsti šiuos prieštaringus tikslus yra *dinamiškai* mažinti mokymosi greitį optimizavimui vykstant.

Tai taip pat priežastis, kodėl į `sgd` žingsnio funkciją pridėjome mokymosi greičio funkciją `lr`. Aukščiau pateiktame pavyzdyje bet kokios mokymosi greičio planavimo funkcijos „miega“, nes susijusią funkciją `lr` nustatėme pastovią.

## Dinaminis mokymosi greitis

Pakeitus $\eta$ nuo laiko priklausančiu mokymosi greičiu $\eta(t)$, padidėja optimizavimo algoritmo konvergavimo valdymo sudėtingumas. Visų pirma turime išsiaiškinti, kaip greitai $\eta$ turėtų mažėti. Jei per greitai, sustosime optimizuoti per anksti. Jei mažiname per lėtai, per daug laiko gaištame optimizavimui. Toliau pateikiamos kelios pagrindinės strategijos, naudojamos koreguojant $\eta$ laikui bėgant (pažangesnes strategijas aptarsime vėliau):

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

Pirmajame *dalimis pastovaus* mokymosi greičio scenarijuje mokymosi greitį mažiname, pavyzdžiui, kai tik optimizavimo pažanga sustoja. Tai įprasta giliųjų tinklų apmokymo strategija. Arba galime jį mažinti daug agresyviau *eksponentiniu mažėjimu*. Deja, tai dažnai lemia pernelyg ankstyvą sustojimą, kol algoritmas dar nesukonvergavo. Populiarus pasirinkimas yra *polinominis mažėjimas* su $\alpha = 0.5$. Iškiliojo optimizavimo atveju yra nemažai įrodymų, rodančių, kad tokia sparta elgiasi gerai.

Pažiūrėkime, kaip eksponentinis mažėjimas atrodo praktikoje.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Kaip ir tikėtasi, parametrų dispersija gerokai sumažėja. Tačiau tai daroma ant nekonvergavimo į optimalų sprendinį $\mathbf{x} = (0, 0)$ aukos. Net po 1000 iteracijos žingsnių vis dar esame labai toli nuo optimalaus sprendinio. Iš tiesų algoritmas visai nekonverguoja. Kita vertus, jei naudojame polinominį mažėjimą, kai mokymosi greitis mažėja atvirkščiai proporcingai žingsnių skaičiaus kvadratinei šakniai, konvergavimas pagerėja jau po 50 žingsnių.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Yra daug daugiau mokymosi greičio nustatymo pasirinkimų. Pavyzdžiui, galėtume pradėti nuo mažo greičio, paskui greitai jį padidinti ir vėl mažinti, nors ir lėčiau. Galėtume net kaitalioti mažesnius ir didesnius mokymosi greičius. Yra didelė tokių grafikų įvairovė. Kol kas sutelkime dėmesį į mokymosi greičio grafikus, kuriems galima išsami teorinė analizė, t. y. į mokymosi greičius iškiliojoje aplinkoje. Bendriems neiškiliesiems uždaviniams labai sunku gauti prasmingas konvergavimo garantijas, nes apskritai netiesinių neiškiliųjų uždavinių minimizavimas yra NP sudėtingas. Apžvalgos ieškokite, pvz., puikiuose Tibshirani 2015 m. [paskaitų užrašuose](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf).



## Konvergavimo analizė iškiliosioms tikslo funkcijoms

Toliau pateikta iškiliųjų tikslo funkcijų stochastinio gradientinio nusileidimo konvergavimo analizė
yra neprivaloma ir daugiausia padeda geriau perprasti problemą.
Apsiribosime viena paprasčiausių įrodymų :cite:`Nesterov.Vial.2000`.
Egzistuoja gerokai pažangesnės įrodymo technikos, pavyzdžiui, kai tikslo funkcija yra ypač gerai besielgianti.


Tarkime, kad tikslo funkcija $f(\boldsymbol{\xi}, \mathbf{x})$ yra iškiloji $\mathbf{x}$ atžvilgiu
visiems $\boldsymbol{\xi}$.
Konkrečiau,
nagrinėkime stochastinio gradientinio nusileidimo atnaujinimą:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

kur $f(\boldsymbol{\xi}_t, \mathbf{x})$
yra tikslo funkcija
apmokymo pavyzdžiui $\boldsymbol{\xi}_t$,
paimtam iš tam tikro skirstinio
žingsnyje $t$, o $\mathbf{x}$ yra modelio parametras.
Pažymėkime

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

tikėtiną riziką, o $R^*$ – jos minimumą $\mathbf{x}$ atžvilgiu. Galiausiai tegul $\mathbf{x}^*$ yra minimizuotojas (darome prielaidą, kad jis egzistuoja srityje, kurioje apibrėžtas $\mathbf{x}$). Šiuo atveju galime sekti atstumą tarp dabartinio parametro $\mathbf{x}_t$ laiko momentu $t$ ir rizikos minimizuotojo $\mathbf{x}^*$ ir stebėti, ar jis laikui bėgant gerėja:

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Darome prielaidą, kad stochastinio gradiento $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ $\ell_2$ norma yra aprėžta kokia nors konstanta $L$, taigi turime, kad

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


Mus daugiausia domina, kaip atstumas tarp $\mathbf{x}_t$ ir $\mathbf{x}^*$ kinta *vidutiniškai*. Iš tikrųjų, konkrečiai žingsnių sekai atstumas gali ir padidėti, priklausomai nuo to, kokį $\boldsymbol{\xi}_t$ sutinkame. Todėl turime aprėžti skaliarinę sandaugą.
Kadangi bet kuriai iškiliajai funkcijai $f$ galioja
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$
visiems $\mathbf{x}$ ir $\mathbf{y}$,
iš iškilumo turime

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

Įstatę abi nelygybes :eqref:`eq_sgd-L` ir :eqref:`eq_sgd-f-xi-xstar` į :eqref:`eq_sgd-xt+1-xstar`, gauname atstumo tarp parametrų laiko momentu $t+1$ rėžį:

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Tai reiškia, kad darome pažangą, kol skirtumas tarp dabartinio nuostolio ir optimalaus nuostolio viršija $\eta_t L^2/2$. Kadangi šis skirtumas neišvengiamai konverguoja į nulį, iš to išplaukia, kad mokymosi greitis $\eta_t$ taip pat turi *nykti*.

Toliau skaičiuojame :eqref:`eqref_sgd-xt-diff` vidurkius. Gauname

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Paskutinis žingsnis yra nelygybių sumavimas $t \in \{1, \ldots, T\}$. Kadangi suma teleskopinė, ir praleidę apatinį narį, gauname

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

Atkreipkite dėmesį, kad pasinaudojome tuo, kad $\mathbf{x}_1$ yra duotas, todėl vidurkį galima atmesti. Galiausiai apibrėžkime

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Kadangi

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

pagal Jenseno nelygybę (nustačius $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ :eqref:`eq_jensens-inequality`) ir $R$ iškilumą, išplaukia, kad $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, taigi

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Įstatę tai į nelygybę :eqref:`eq_sgd-x1-xstar`, gauname rėžį

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

kur $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ yra atstumo tarp pradinio parametrų pasirinkimo ir galutinio rezultato rėžis. Trumpai tariant, konvergavimo greitis priklauso nuo to, kaip
aprėžta stochastinio gradiento norma ($L$) ir kaip toli nuo optimumo yra pradinė parametro reikšmė ($r$). Atkreipkite dėmesį, kad rėžis pateiktas $\bar{\mathbf{x}}$, o ne $\mathbf{x}_T$ atžvilgiu. Taip yra todėl, kad $\bar{\mathbf{x}}$ yra išglostyta optimizavimo trajektorijos versija.
Kai tik žinomi $r, L$ ir $T$, galime pasirinkti mokymosi greitį $\eta = r/(L \sqrt{T})$. Tai duoda viršutinį rėžį $rL/\sqrt{T}$. Tai yra, į optimalų sprendinį konverguojame sparta $\mathcal{O}(1/\sqrt{T})$.





## Stochastiniai gradientai ir baigtinės imtys

Iki šiol gana laisvai kalbėjome apie stochastinį gradientinį nusileidimą. Teigėme, kad iš tam tikro skirstinio $p(x, y)$ atrenkame pavyzdžius $x_i$, paprastai su žymenimis $y_i$, ir tai naudojame modelio parametrams tam tikru būdu atnaujinti. Visų pirma baigtiniam imties dydžiui tiesiog teigėme, kad diskretusis skirstinys $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$
su tam tikromis funkcijomis $\delta_{x_i}$ ir $\delta_{y_i}$
leidžia atlikti stochastinį gradientinį nusileidimą jo atžvilgiu.

Tačiau iš tikrųjų taip nedarėme. Šio skyrelio pavyzdžiuose tiesiog pridėjome triukšmo prie kitaip nestochastinio gradiento, t. y. apsimetėme, kad turime poras $(x_i, y_i)$. Paaiškėja, kad čia tai pagrįsta (išsamų aptarimą žr. pratimuose). Labiau nerimą kelia tai, kad visuose ankstesniuose aptarimuose taip aiškiai nedarėme. Vietoj to iteravome per visus pavyzdžius *lygiai vieną kartą*. Kad pamatytume, kodėl tai geriau, panagrinėkime priešingą atvejį, būtent kad iš diskrečiojo skirstinio atrenkame $n$ stebėjimų *su grąžinimu*. Tikimybė atsitiktinai pasirinkti elementą $i$ yra $1/n$. Taigi tikimybė jį pasirinkti *bent* kartą yra

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Panašus samprotavimas rodo, kad tikimybė pasirinkti tam tikrą pavyzdį (t. y. apmokymo pavyzdį) *lygiai vieną kartą* yra

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Atranka su grąžinimu lemia padidėjusią dispersiją ir sumažėjusį duomenų efektyvumą, palyginti su atranka *be grąžinimo*. Todėl praktikoje atliekame pastarąją (ir tai yra numatytasis pasirinkimas visoje šioje knygoje). Galiausiai atkreipkite dėmesį, kad pakartotiniai apmokymo duomenų rinkinio perėjimai jį apeina *skirtinga* atsitiktine tvarka.


## Santrauka

* Iškiliesiems uždaviniams galime įrodyti, kad esant plačiam mokymosi greičių pasirinkimui stochastinis gradientinis nusileidimas konverguos į optimalų sprendinį.
* Giliajam mokymuisi taip paprastai nėra. Tačiau iškiliųjų uždavinių analizė suteikia naudingų įžvalgų, kaip žiūrėti į optimizavimą, būtent: palaipsniui mažinti mokymosi greitį, nors ir ne per greitai.
* Problemų kyla, kai mokymosi greitis yra per mažas arba per didelis. Praktikoje tinkamas mokymosi greitis dažnai randamas tik po kelių eksperimentų.
* Kai apmokymo duomenų rinkinyje yra daugiau pavyzdžių, kiekvienos gradientinio nusileidimo iteracijos skaičiavimas kainuoja daugiau, todėl tokiais atvejais pirmenybė teikiama stochastiniam gradientiniam nusileidimui.
* Stochastinio gradientinio nusileidimo optimalumo garantijų neiškiliuoju atveju apskritai nėra, nes tikrintinų lokaliųjų minimumų skaičius gali būti eksponentinis.




## Pratimai

1. Eksperimentuokite su skirtingais stochastinio gradientinio nusileidimo mokymosi greičio grafikais ir skirtingu iteracijų skaičiumi. Visų pirma nubraižykite atstumą nuo optimalaus sprendinio $(0, 0)$ kaip iteracijų skaičiaus funkciją.
1. Įrodykite, kad funkcijai $f(x_1, x_2) = x_1^2 + 2 x_2^2$ normaliojo triukšmo pridėjimas prie gradiento yra ekvivalentus nuostolio funkcijos $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ minimizavimui, kur $\mathbf{x}$ imamas iš normaliojo skirstinio.
1. Palyginkite stochastinio gradientinio nusileidimo konvergavimą, kai atrenkate iš $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ su grąžinimu ir kai atrenkate be grąžinimo.
1. Kaip pakeistumėte stochastinio gradientinio nusileidimo sprendiklį, jei kuris nors gradientas (tiksliau, kuri nors su juo susijusi koordinatė) nuolat būtų didesnis nei visi kiti gradientai?
1. Tarkime, kad $f(x) = x^2 (1 + \sin x)$. Kiek lokaliųjų minimumų turi $f$? Ar galite pakeisti $f$ taip, kad ją minimizuojant reikėtų įvertinti visus lokaliuosius minimumus?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1067)
:end_tab:
