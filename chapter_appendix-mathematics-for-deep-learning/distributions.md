# Skirstiniai
:label:`sec_distributions`

Dabar, kai išmokome dirbti su tikimybėmis tiek diskrečiuoju, tiek tolydžiuoju atveju, susipažinkime su keliais dažnai sutinkamais skirstiniais.  Priklausomai nuo mašininio mokymo srities, gali prireikti išmanyti daug daugiau jų, o kai kuriose giliojo mokymosi srityse galbūt apskritai nė vieno.  Vis dėlto tai gera pagrindinė žinotina sąrašo dalis.  Pirmiausia importuokime keletą įprastų bibliotekų.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## Bernulio skirstinys

Tai paprasčiausias paprastai sutinkamas atsitiktinis dydis.  Šis atsitiktinis dydis užkoduoja monetos metimą, kuris duoda $1$ su tikimybe $p$ ir $0$ su tikimybe $1-p$.  Jei turime atsitiktinį dydį $X$ su šiuo skirstiniu, rašysime

$$
X \sim \textrm{Bernoulli}(p).
$$

Pasiskirstymo funkcija yra 

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

Tikimybių masės funkcija pavaizduota toliau.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Dabar nubraižykime pasiskirstymo funkciją :eqref:`eq_bernoulli_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

Jei $X \sim \textrm{Bernoulli}(p)$, tai:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

Bet kokios formos masyvą iš Bernulio atsitiktinio dydžio galime generuoti taip.

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## Diskretusis tolygusis skirstinys

Kitas dažnai sutinkamas atsitiktinis dydis yra diskretusis tolygusis.  Šioje diskusijoje darysime prielaidą, kad jis apibrėžtas sveikuosiuose skaičiuose $\{1, 2, \ldots, n\}$, tačiau galima laisvai pasirinkti bet kurią kitą reikšmių aibę.  Žodžio *tolygus* prasmė šiame kontekste yra ta, kad kiekviena galima reikšmė yra vienodai tikėtina.  Kiekvienos reikšmės $i \in \{1, 2, 3, \ldots, n\}$ tikimybė yra $p_i = \frac{1}{n}$.  Atsitiktinį dydį $X$ su šiuo skirstiniu žymėsime

$$
X \sim U(n).
$$

Pasiskirstymo funkcija yra 

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

Pirmiausia nubraižykime tikimybių masės funkciją.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Dabar nubraižykime pasiskirstymo funkciją :eqref:`eq_discrete_uniform_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Jei $X \sim U(n)$, tai:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

Bet kokios formos masyvą iš diskrečiojo tolygaus atsitiktinio dydžio galime generuoti taip.

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## Tolydusis tolygusis skirstinys

Toliau aptarkime tolydųjį tolygųjį skirstinį. Šio atsitiktinio dydžio idėja yra ta, kad jei diskrečiajame tolygiajame skirstinyje didiname $n$, o tada jį perskalėjame, kad tilptų į intervalą $[a, b]$, artėsime prie tolydaus atsitiktinio dydžio, kuris tiesiog parenka savavališką reikšmę iš $[a, b]$, visas su vienoda tikimybe.  Šį skirstinį žymėsime

$$
X \sim U(a, b).
$$

Tikimybės tankio funkcija yra 

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

Pasiskirstymo funkcija yra 

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

Pirmiausia nubraižykime tikimybės tankio funkciją :eqref:`eq_cont_uniform_pdf`.

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

Dabar nubraižykime pasiskirstymo funkciją :eqref:`eq_cont_uniform_cdf`.

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Jei $X \sim U(a, b)$, tai:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

Bet kokios formos masyvą iš tolygaus atsitiktinio dydžio galime generuoti taip.  Atkreipkite dėmesį, kad pagal numatytuosius nustatymus generuojama iš $U(0,1)$, todėl jei norime kito intervalo, turime perskalėti.

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## Binominis skirstinys

Šiek tiek pasunkinkime ir panagrinėkime *binominį* atsitiktinį dydį.  Šis atsitiktinis dydis atsiranda atlikus $n$ nepriklausomų eksperimentų, kurių kiekvienas pavyksta su tikimybe $p$, ir paklausus, kiek sėkmių tikimės pamatyti.

Išreikškime tai matematiškai.  Kiekvienas eksperimentas yra nepriklausomas atsitiktinis dydis $X_i$, kuriame $1$ koduosime sėkmę, o $0$ – nesėkmę.  Kadangi kiekvienas yra nepriklausomas monetos metimas, sėkmingas su tikimybe $p$, galime sakyti, kad $X_i \sim \textrm{Bernoulli}(p)$.  Tada binominis atsitiktinis dydis yra

$$
X = \sum_{i=1}^n X_i.
$$

Šiuo atveju rašysime

$$
X \sim \textrm{Binomial}(n, p).
$$

Kad gautume pasiskirstymo funkciją, turime pastebėti, kad tiksliai $k$ sėkmių gali įvykti $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ būdais, kurių kiekvieno tikimybė yra $p^k(1-p)^{n-k}$.  Taigi pasiskirstymo funkcija yra

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

Pirmiausia nubraižykime tikimybių masės funkciją.

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Dabar nubraižykime pasiskirstymo funkciją :eqref:`eq_binomial_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Jei $X \sim \textrm{Binomial}(n, p)$, tai:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

Tai išplaukia iš vidurkio tiesiškumo $n$ Bernulio atsitiktinių dydžių sumai ir fakto, kad nepriklausomų atsitiktinių dydžių sumos dispersija yra dispersijų suma. Tai galima generuoti taip.

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## Puasono skirstinys
Dabar atlikime minties eksperimentą.  Stovime autobusų stotelėje ir norime sužinoti, kiek autobusų atvyks per kitą minutę.  Pradėkime nuo $X^{(1)} \sim \textrm{Bernoulli}(p)$, kuris yra tiesiog tikimybė, kad autobusas atvyks per vienos minutės langą.  Stotelėms toli nuo miesto centro tai gali būti gana gera aproksimacija.  Galbūt niekada nepamatysime daugiau nei vieno autobuso per minutę.

Tačiau jei esame judrioje vietoje, galima ar net tikėtina, kad atvyks du autobusai.  Tai galime modeliuoti padalydami savo atsitiktinį dydį į dvi dalis – pirmąsias 30 sekundžių arba antrąsias 30 sekundžių.  Šiuo atveju galime rašyti

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

kur $X^{(2)}$ yra visa suma, o $X^{(2)}_i \sim \textrm{Bernoulli}(p/2)$.  Tada visas skirstinys yra $X^{(2)} \sim \textrm{Binomial}(2, p/2)$.

Kodėl čia sustoti?  Toliau skaidykime tą minutę į $n$ dalių.  Pagal tą patį samprotavimą kaip aukščiau matome, kad

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

Panagrinėkime šiuos atsitiktinius dydžius.  Iš ankstesnio skyriaus žinome, kad :eqref:`eq_eq_poisson_approx` turi vidurkį $\mu_{X^{(n)}} = n(p/n) = p$ ir dispersiją $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$.  Jei $n \rightarrow \infty$, matome, kad šie skaičiai stabilizuojasi iki $\mu_{X^{(\infty)}} = p$ ir dispersijos $\sigma_{X^{(\infty)}}^2 = p$.  Tai rodo, kad *galėtų būti* koks nors atsitiktinis dydis, kurį galime apibrėžti šioje begalinio skaidymo riboje.  

Tai neturėtų pernelyg stebinti, nes realiame pasaulyje galime tiesiog suskaičiuoti autobusų atvykimus, tačiau gera matyti, kad mūsų matematinis modelis yra gerai apibrėžtas.  Šią diskusiją galima formalizuoti kaip *retų įvykių dėsnį*.

Atidžiai sekdami šį samprotavimą, galime prieiti prie tokio modelio.  Sakysime, kad $X \sim \textrm{Poisson}(\lambda)$, jei tai atsitiktinis dydis, įgyjantis reikšmes $\{0,1,2, \ldots\}$ su tikimybe

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

Reikšmė $\lambda > 0$ žinoma kaip *intensyvumas* (arba *formos* parametras) ir žymi vidutinį atvykimų skaičių, kurio tikimės per vieną laiko vienetą.  

Susumavę šią tikimybių masės funkciją, galime gauti pasiskirstymo funkciją.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

Pirmiausia nubraižykime tikimybių masės funkciją :eqref:`eq_poisson_mass`.

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Dabar nubraižykime pasiskirstymo funkciją :eqref:`eq_poisson_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Kaip matėme aukščiau, vidurkiai ir dispersijos yra ypač glaustos.  Jei $X \sim \textrm{Poisson}(\lambda)$, tai:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

Tai galima generuoti taip.

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gauso skirstinys
Dabar išbandykime kitokį, bet susijusį eksperimentą.  Tarkime, vėl atliekame $n$ nepriklausomų $\textrm{Bernoulli}(p)$ matavimų $X_i$.  Šių suma pasiskirsčiusi kaip $X^{(n)} \sim \textrm{Binomial}(n, p)$.  Užuot ėmę ribą, kai $n$ didėja, o $p$ mažėja, fiksuokime $p$ ir leiskime $n \rightarrow \infty$.  Šiuo atveju $\mu_{X^{(n)}} = np \rightarrow \infty$ ir $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, todėl nėra pagrindo manyti, kad ši riba būtų gerai apibrėžta.

Tačiau ne viskas prarasta!  Tiesiog padarykime, kad vidurkis ir dispersija būtų tvarkingi, apibrėždami

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

Galima pamatyti, kad tai turi vidurkį nulį ir dispersiją vienas, todėl galima pagrįstai tikėti, kad jis konverguos į kokį nors ribinį skirstinį.  Jei nubraižysime, kaip šie skirstiniai atrodo, dar labiau įsitikinsime, kad tai veiks.

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

Vienas pastebėjimas: palyginti su Puasono atveju, dabar dalijame iš standartinio nuokrypio, o tai reiškia, kad galimus rezultatus spaudžiame į vis mažesnes sritis.  Tai rodo, kad mūsų riba nebebus diskreti, o tolydi.

Išvedimas, kas čia vyksta, nepatenka į šio dokumento aprėptį, tačiau *centrinė ribinė teorema* teigia, kad kai $n \rightarrow \infty$, tai duos Gauso skirstinį (arba kartais normalųjį skirstinį).  Aiškiau, bet kuriems $a, b$:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

kur sakome, kad atsitiktinis dydis pasiskirstęs normaliai su duotu vidurkiu $\mu$ ir dispersija $\sigma^2$, rašoma $X \sim \mathcal{N}(\mu, \sigma^2)$, jei $X$ turi tankį

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

Pirmiausia nubraižykime tikimybės tankio funkciją :eqref:`eq_gaussian_pdf`.

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

Dabar nubraižykime pasiskirstymo funkciją.  Tai nepatenka į šio priedo aprėptį, tačiau Gauso pasiskirstymo funkcija neturi uždaros formos formulės per elementaresnes funkcijas.  Naudosime `erf`, kuri suteikia būdą šį integralą apskaičiuoti skaitiškai.

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Budrūs skaitytojai atpažins kai kuriuos šiuos narius.  Iš tiesų su šiuo integralu susidūrėme :numref:`sec_integral_calculus`.  Iš tiesų mums reikia būtent to skaičiavimo, kad pamatytume, jog šio $p_X(x)$ pilnasis plotas lygus vienam, taigi tai tinkamas tankis.

Mūsų pasirinkimas dirbti su monetos metimais padarė skaičiavimus trumpesnius, tačiau nieko iš esmės nepriklausė nuo šio pasirinkimo.  Iš tiesų, jei imsime bet kokį nepriklausomų vienodai pasiskirsčiusių atsitiktinių dydžių $X_i$ rinkinį ir sudarysime

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

Tada

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

bus apytiksliai Gauso.  Kad tai veiktų, reikia papildomų sąlygų, dažniausiai $E[X^4] < \infty$, tačiau filosofija aiški.

Centrinė ribinė teorema yra priežastis, kodėl Gauso skirstinys yra esminis tikimybių teorijoje, statistikoje ir mašininiame mokyme.  Kai tik galime pasakyti, kad kažkas, ką išmatavome, yra daugelio mažų nepriklausomų indėlių suma, galime daryti prielaidą, kad matuojamas dydis bus artimas Gauso.  

Gauso skirstinys turi daugybę žavių savybių, ir norėtume aptarti dar vieną.  Gauso skirstinys yra vadinamasis *didžiausios entropijos skirstinys*.  Entropiją išsamiau nagrinėsime :numref:`sec_information_theory`, tačiau šiuo metu pakanka žinoti, kad tai atsitiktinumo matas.  Griežtai matematiniu požiūriu Gauso skirstinį galime laikyti *atsitiktiniausiu* atsitiktinio dydžio pasirinkimu su fiksuotu vidurkiu ir dispersija.  Taigi, jei žinome, kad mūsų atsitiktinis dydis turi tam tikrą vidurkį ir dispersiją, Gauso skirstinys yra tam tikra prasme konservatyviausias skirstinio pasirinkimas, kurį galime padaryti.

Užbaigdami skyrių prisiminkime, kad jei $X \sim \mathcal{N}(\mu, \sigma^2)$, tai:

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

Iš Gauso (arba standartinio normaliojo) skirstinio galime generuoti, kaip parodyta toliau.

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## Eksponentinė šeima
:label:`subsec_exponential_family`

Viena visiems aukščiau išvardytiems skirstiniams bendra savybė yra ta, kad jie visi 
priklauso vadinamajai *eksponentinei šeimai*. Eksponentinė šeima 
yra skirstinių aibė, kurių tankį galima išreikšti tokia 
forma:

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

Kadangi šis apibrėžimas gali būti šiek tiek subtilus, panagrinėkime jį atidžiai.  

Pirma, $h(\mathbf{x})$ žinoma kaip *pagrindinis matas* arba 
*bazinis matas*.  Tai galima laikyti pradiniu mato pasirinkimu, kurį 
modifikuojame savo eksponentiniu svoriu.  

Antra, turime vektorių $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in
\mathbb{R}^l$, vadinamą *natūraliaisiais parametrais* arba *kanoniniais parametrais*.  Jie
apibrėžia, kaip bus modifikuotas bazinis matas.  Natūralieji parametrai patenka
į naują matą imant šių parametrų skaliarinę sandaugą su
kokia nors funkcija $T(\cdot)$ nuo $\mathbf{x}= (x_1, x_2, ..., x_n) \in
\mathbb{R}^n$ ir keliant eksponente. Vektorius $T(\mathbf{x})= (T_1(\mathbf{x}),
T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ 
vadinamas $\boldsymbol{\eta}$ *pakankamosiomis statistikomis*. Šis pavadinimas naudojamas todėl, kad 
informacija, atvaizduojama $T(\mathbf{x})$, yra pakankama tikimybės tankiui 
apskaičiuoti ir jokios kitos informacijos iš imties $\mathbf{x}$ 
nereikia.

Trečia, turime $A(\boldsymbol{\eta})$, vadinamą *kumuliantų 
funkcija*, kuri užtikrina, kad aukščiau pateiktas skirstinys :eqref:`eq_exp_pdf` 
integruojasi į vienetą, t. y.,

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

Kad būtų konkretu, panagrinėkime Gauso skirstinį. Darydami prielaidą, kad $\mathbf{x}$ yra 
vienmatis kintamasis, matėme, kad jo tankis yra

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

Tai atitinka eksponentinės šeimos apibrėžimą su:

* *pagrindiniu matu*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *natūraliaisiais parametrais*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *pakankamosiomis statistikomis*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, ir
* *kumuliantų funkcija*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

Verta pažymėti, kad tikslus kiekvieno iš aukščiau pateiktų narių pasirinkimas yra 
šiek tiek savavališkas.  Iš tiesų svarbu, kad skirstinį būtų galima 
išreikšti šia forma, o ne pati tiksli forma.

Kaip užsiminėme :numref:`subsec_softmax_and_derivatives`, plačiai naudojamas 
metodas yra daryti prielaidą, kad galutinė išvestis $\mathbf{y}$ seka 
eksponentinės šeimos skirstinį. Eksponentinė šeima yra dažna ir 
galinga skirstinių šeima, dažnai sutinkama mašininiame mokyme.


## Santrauka
* Bernulio atsitiktiniais dydžiais galima modeliuoti įvykius su rezultatu taip/ne.
* Diskretieji tolygieji skirstiniai modeliuoja pasirinkimą iš baigtinės galimybių aibės.
* Tolydieji tolygieji skirstiniai renkasi iš intervalo.
* Binominiai skirstiniai modeliuoja Bernulio atsitiktinių dydžių seriją ir skaičiuoja sėkmių skaičių.
* Puasono atsitiktiniai dydžiai modeliuoja retų įvykių atsiradimą.
* Gauso atsitiktiniai dydžiai modeliuoja didelio skaičiaus nepriklausomų atsitiktinių dydžių sudėjimo rezultatą.
* Visi aukščiau išvardyti skirstiniai priklauso eksponentinei šeimai.

## Pratimai

1. Koks yra atsitiktinio dydžio, kuris yra dviejų nepriklausomų binominių atsitiktinių dydžių $X, Y \sim \textrm{Binomial}(16, 1/2)$ skirtumas $X-Y$, standartinis nuokrypis?
2. Jei paimsime Puasono atsitiktinį dydį $X \sim \textrm{Poisson}(\lambda)$ ir nagrinėsime $(X - \lambda)/\sqrt{\lambda}$, kai $\lambda \rightarrow \infty$, galime parodyti, kad jis tampa apytiksliai Gauso.  Kodėl tai prasminga?
3. Kokia yra dviejų diskrečiųjų tolygiųjų atsitiktinių dydžių su $n$ elementų sumos tikimybių masės funkcija?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1099)
:end_tab:
