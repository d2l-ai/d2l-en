```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Automatinis diferencijavimas
:label:`sec_autograd`

Prisiminkite iš :numref:`sec_calculus`,
kad išvestinių skaičiavimas yra lemiamas žingsnis
visuose optimizavimo algoritmuose,
kuriuos naudosime giliesiems tinklams apmokyti.
Nors skaičiavimai paprasti,
atlikti juos rankomis gali būti nuobodu ir lengva suklysti,
o šios problemos tik didėja,
kai mūsų modeliai tampa sudėtingesni.

Laimei, visi šiuolaikiniai giliojo mokymosi karkasai
šį darbą perima iš mūsų,
siūlydami *automatinį diferencijavimą*
(dažnai sutrumpintai *autograd*).
Kai duomenis perduodame per kiekvieną paskesnę funkciją,
karkasas sukuria *skaičiavimo grafą*,
kuris seka, kaip kiekviena reikšmė priklauso nuo kitų.
Išvestinėms apskaičiuoti
automatinis diferencijavimas
eina šiuo grafu atgal,
taikydamas grandinės taisyklę.
Skaičiavimo algoritmas, taikantis grandinės taisyklę
tokiu būdu, vadinamas *atgaliniu sklidimu*.

Nors autograd bibliotekos pastarąjį dešimtmetį
tapo karštu rūpesčiu,
jos turi ilgą istoriją.
Tiesą sakant, ankstyviausios nuorodos į autograd
siekia daugiau nei pusę amžiaus :cite:`Wengert.1964`.
Pagrindinės šiuolaikinio atgalinio sklidimo idėjos
siekia 1980 m. daktaro disertaciją :cite:`Speelpenning.1980`
ir buvo toliau plėtojamos devintojo dešimtmečio pabaigoje :cite:`Griewank.1989`.
Nors atgalinis sklidimas tapo numatytuoju gradientų
skaičiavimo metodu, jis nėra vienintelis pasirinkimas.
Pavyzdžiui, programavimo kalba Julia naudoja
sklidimą pirmyn :cite:`Revels.Lubin.Papamarkou.2016`.
Prieš nagrinėdami metodus,
pirmiausia įvaldykime autograd paketą.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Paprasta funkcija

Tarkime, kad mus domina
(**funkcijos
$y = 2\mathbf{x}^{\top}\mathbf{x}$
diferencijavimas
stulpelio vektoriaus $\mathbf{x}$ atžvilgiu.**)
Pradžioje priskiriame `x` pradinę reikšmę.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**Prieš apskaičiuodami $y$ gradientą
$\mathbf{x}$ atžvilgiu,
mums reikia vietos jam saugoti.**]
Paprastai vengiame skirstyti naują atmintį
kiekvieną kartą skaičiuodami išvestinę,
nes giliajam mokymuisi reikia
paeiliui skaičiuoti išvestines
tų pačių parametrų atžvilgiu
labai daug kartų,
ir mums gali pritrūkti atminties.
Atkreipkite dėmesį, kad skaliarinę reikšmę turinčios funkcijos gradientas
vektoriaus $\mathbf{x}$ atžvilgiu
yra vektorinė reikšmė, turinti
tą pačią formą kaip $\mathbf{x}$.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**Dabar apskaičiuojame savo funkciją nuo `x` ir rezultatą priskiriame `y`.**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**Dabar galime apskaičiuoti `y` gradientą
`x` atžvilgiu**] iškviesdami
jo metodą `backward`.
Tada gradientą galime pasiekti
per `x` atributą `grad`.
:end_tab:

:begin_tab:`pytorch`
[**Dabar galime apskaičiuoti `y` gradientą
`x` atžvilgiu**] iškviesdami
jo metodą `backward`.
Tada gradientą galime pasiekti
per `x` atributą `grad`.
:end_tab:

:begin_tab:`tensorflow`
[**Dabar galime apskaičiuoti `y` gradientą
`x` atžvilgiu**] iškviesdami
metodą `gradient`.
:end_tab:

:begin_tab:`jax`
[**Dabar galime apskaičiuoti `y` gradientą
`x` atžvilgiu**] perduodami per
transformaciją `grad`.
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

(**Jau žinome, kad funkcijos $y = 2\mathbf{x}^{\top}\mathbf{x}$
gradientas $\mathbf{x}$ atžvilgiu turėtų būti $4\mathbf{x}$.**)
Dabar galime patikrinti, ar automatinis gradiento skaičiavimas
ir laukiamas rezultatas sutampa.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Dabar apskaičiuokime
kitą funkciją nuo `x`
ir raskime jos gradientą.**]
Atkreipkite dėmesį, kad MXNet iš naujo nustato gradiento buferį
kiekvieną kartą, kai įrašome naują gradientą.
:end_tab:

:begin_tab:`pytorch`
[**Dabar apskaičiuokime
kitą funkciją nuo `x`
ir raskime jos gradientą.**]
Atkreipkite dėmesį, kad PyTorch automatiškai
nenustato iš naujo gradiento buferio,
kai įrašome naują gradientą.
Vietoj to naujas gradientas
pridedamas prie jau saugomo gradiento.
Toks elgesys praverčia,
kai norime optimizuoti kelių tikslo funkcijų
sumą.
Gradiento buferiui nustatyti iš naujo
galime iškviesti `x.grad.zero_()` taip:
:end_tab:

:begin_tab:`tensorflow`
[**Dabar apskaičiuokime
kitą funkciją nuo `x`
ir raskime jos gradientą.**]
Atkreipkite dėmesį, kad TensorFlow iš naujo nustato gradiento buferį
kiekvieną kartą, kai įrašome naują gradientą.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## Atgalinis sklidimas neskaliariniams kintamiesiems

Kai `y` yra vektorius,
natūraliausias `y` išvestinės
vektoriaus `x` atžvilgiu vaizdavimas
yra matrica, vadinama *Jakobio matrica*,
kurioje yra kiekvienos `y` komponentės
dalinės išvestinės
kiekvienos `x` komponentės atžvilgiu.
Panašiai aukštesnės eilės `y` ir `x` atveju
diferencijavimo rezultatas gali būti dar aukštesnės eilės tenzorius.

Nors Jakobio matricos pasirodo kai kuriuose
pažangiuose mašininio mokymosi metoduose,
dažniau norime susumuoti
kiekvienos `y` komponentės gradientus
pagal visą vektorių `x`,
gaudami vektorių, kurio forma tokia pati kaip `x`.
Pavyzdžiui, dažnai turime vektorių,
vaizduojantį nuostolių funkcijos reikšmę,
apskaičiuotą atskirai kiekvienam pavyzdžiui
apmokymo pavyzdžių *minirinkinyje*.
Čia tiesiog norime (**susumuoti gradientus,
apskaičiuotus atskirai kiekvienam pavyzdžiui**).

:begin_tab:`mxnet`
MXNet šią problemą sprendžia redukuodamas visus tenzorius į skaliarus
susumavimu prieš skaičiuodamas gradientą.
Kitaip tariant, užuot grąžinęs Jakobio matricą
$\partial_{\mathbf{x}} \mathbf{y}$,
jis grąžina sumos gradientą
$\partial_{\mathbf{x}} \sum_i y_i$.
:end_tab:

:begin_tab:`pytorch`
Kadangi giliojo mokymosi karkasai skirtingai
interpretuoja
neskaliarinių tenzorių gradientus,
PyTorch imasi tam tikrų žingsnių, kad išvengtų painiavos.
Iškvietus `backward` neskaliariniam dydžiui, kyla klaida,
nebent nurodome PyTorch, kaip redukuoti objektą į skaliarą.
Formaliau, turime pateikti kokį nors vektorių $\mathbf{v}$,
kad `backward` apskaičiuotų
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$,
o ne $\partial_{\mathbf{x}} \mathbf{y}$.
Ši kita dalis gali būti painoka,
bet dėl priežasčių, kurios paaiškės vėliau,
šis argumentas (vaizduojantis $\mathbf{v}$) vadinamas `gradient`.
Išsamesnio aprašymo ieškokite Yang Zhang
[„Medium" įraše](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29).
:end_tab:

:begin_tab:`tensorflow`
Pagal numatytuosius nustatymus TensorFlow grąžina sumos gradientą.
Kitaip tariant, užuot grąžinęs
Jakobio matricą $\partial_{\mathbf{x}} \mathbf{y}$,
jis grąžina sumos gradientą
$\partial_{\mathbf{x}} \sum_i y_i$.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## Skaičiavimo atsiejimas

Kartais norime [**kai kuriuos skaičiavimus perkelti
už įrašyto skaičiavimo grafo ribų.**]
Pavyzdžiui, tarkime, naudojame įvestį
sukurti kai kuriems pagalbiniams tarpiniams nariams,
kurių gradiento skaičiuoti nenorime.
Šiuo atveju turime *atsieti*
atitinkamą skaičiavimo grafą
nuo galutinio rezultato.
Šis žaislinis pavyzdys tai paaiškina:
tarkime, turime `z = x * y` ir `y = x * x`,
bet norime sutelkti dėmesį į *tiesioginę* `x` įtaką `z`,
o ne per `y` perduodamą įtaką.
Šiuo atveju galime sukurti naują kintamąjį `u`,
turintį tą pačią reikšmę kaip `y`,
bet kurio *kilmė* (kaip jis buvo sukurtas)
ištrinta.
Taigi `u` grafe neturi protėvių,
ir gradientai neteka per `u` į `x`.
Pavyzdžiui, apskaičiavus `z = x * u` gradientą,
gausime rezultatą `u`
(o ne `3 * x * x`, kaip galėjote
tikėtis, nes `z = x * x * x`).

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

Atkreipkite dėmesį, kad nors ši procedūra
atsieja `y` protėvius
nuo grafo, vedančio į `z`,
skaičiavimo grafas, vedantis į `y`,
išlieka, todėl galime apskaičiuoti
`y` gradientą `x` atžvilgiu.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## Gradientai ir Python valdymo srautas

Iki šiol apžvelgėme atvejus, kai kelias nuo įvesties iki išvesties
buvo aiškiai apibrėžtas funkcija, pvz., `z = x * x * x`.
Programavimas suteikia daug daugiau laisvės, kaip skaičiuojame rezultatus.
Pavyzdžiui, galime padaryti, kad jie priklausytų nuo pagalbinių kintamųjų
arba sąlygotų pasirinkimus tarpiniais rezultatais.
Vienas automatinio diferencijavimo privalumas yra tas,
kad [**net jei**] skaičiavimo grafo kūrimui
(**funkcijos reikėjo praeiti Python valdymo srauto labirintą**)
(pvz., sąlygas, ciklus ir savavališkus funkcijų iškvietimus),
(**vis tiek galime apskaičiuoti gauto kintamojo gradientą.**)
Tam iliustruoti panagrinėkime šį kodo fragmentą,
kuriame `while` ciklo iteracijų skaičius
ir `if` sakinio įvertinimas
abu priklauso nuo įvesties `a` reikšmės.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Toliau iškviečiame šią funkciją, kaip įvestį perduodami atsitiktinę reikšmę.
Kadangi įvestis yra atsitiktinis dydis,
nežinome, kokią formą
įgis skaičiavimo grafas.
Tačiau kai tik vykdome `f(a)`
su konkrečia įvestimi, realizuojame
konkretų skaičiavimo grafą
ir vėliau galime paleisti `backward`.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

Nors mūsų funkcija `f` demonstravimo tikslais yra kiek dirbtinė,
jos priklausomybė nuo įvesties gana paprasta:
tai *tiesinė* `a` funkcija
su dalimis apibrėžtu mastelio koeficientu.
Todėl `f(a) / a` yra pastovių įrašų vektorius,
ir, be to, `f(a) / a` turi sutapti
su `f(a)` gradientu `a` atžvilgiu.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

Dinaminis valdymo srautas giliajame mokyme labai dažnas.
Pavyzdžiui, apdorojant tekstą, skaičiavimo grafas
priklauso nuo įvesties ilgio.
Šiais atvejais automatinis diferencijavimas
tampa gyvybiškai svarbus statistiniam modeliavimui,
nes gradiento neįmanoma apskaičiuoti *a priori*.

## Diskusija

Dabar paragavote automatinio diferencijavimo galios.
Bibliotekų, skaičiuojančių išvestines
ir automatiškai, ir efektyviai, kūrimas
labai padidino giliojo mokymosi praktikų produktyvumą,
išlaisvindamas juos, kad galėtų sutelkti dėmesį į mažiau mechaniškus darbus.
Be to, autograd leidžia projektuoti didžiulius modelius,
kuriems gradientų skaičiavimas pieštuku ir popieriumi
užimtų neįmanomai daug laiko.
Įdomu tai, kad nors autograd naudojame modeliams *optimizuoti*
(statistine prasme),
pačių autograd bibliotekų *optimizavimas*
(skaičiavimo prasme)
yra turtinga tema,
labai svarbi karkasų kūrėjams.
Čia kompiliatorių ir grafų manipuliavimo priemonės
naudojamos rezultatams apskaičiuoti
patogiausiu ir atmintį tausojančiausiu būdu.

Kol kas pabandykite prisiminti šiuos pagrindus: (i) prie tų kintamųjų, kurių atžvilgiu norime išvestinių, pridėkite gradientus; (ii) įrašykite tikslinės reikšmės skaičiavimą; (iii) vykdykite atgalinio sklidimo funkciją; ir (iv) pasiekite gautą gradientą.


## Pratimai

1. Kodėl antrąją išvestinę apskaičiuoti daug brangiau nei pirmąją?
1. Paleidę atgalinio sklidimo funkciją, nedelsdami paleiskite ją dar kartą ir pažiūrėkite, kas nutinka. Ištirkite.
1. Valdymo srauto pavyzdyje, kuriame skaičiuojame `d` išvestinę `a` atžvilgiu, kas nutiktų, jei kintamąjį `a` pakeistume atsitiktiniu vektoriumi ar matrica? Tuomet skaičiavimo `f(a)` rezultatas nebėra skaliaras. Kas nutinka rezultatui? Kaip tai išanalizuoti?
1. Tegul $f(x) = \sin(x)$. Nubraižykite $f$ ir jos išvestinės $f'$ grafikus. Nenaudokite fakto, kad $f'(x) = \cos(x)$, o rezultatą gaukite naudodami automatinį diferencijavimą.
1. Tegul $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Užrašykite priklausomybės grafą, sekantį rezultatus nuo $x$ iki $f(x)$.
1. Naudodami grandinės taisyklę apskaičiuokite minėtos funkcijos išvestinę $\frac{df}{dx}$, kiekvieną narį išdėstydami anksčiau sukonstruotame priklausomybės grafe.
1. Turint grafą ir tarpinius išvestinių rezultatus, skaičiuojant gradientą yra keletas variantų. Įvertinkite rezultatą vieną kartą pradėdami nuo $x$ iki $f$ ir vieną kartą nuo $f$ sekdami atgal iki $x$. Kelias nuo $x$ iki $f$ paprastai vadinamas *tiesioginiu diferencijavimu* (angl. *forward differentiation*), o kelias nuo $f$ iki $x$ – atgaliniu diferencijavimu.
1. Kada galėtumėte norėti naudoti tiesioginį, o kada atgalinį diferencijavimą? Užuomina: apsvarstykite reikalingų tarpinių duomenų kiekį, galimybę lygiagretinti žingsnius ir susijusių matricų bei vektorių dydį.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17970)
:end_tab:
