# Adagrad
:label:`sec_adagrad`

Pradėkime nagrinėdami mokymosi uždavinius su požymiais, kurie pasitaiko retai.


## Reti požymiai ir mokymosi greičiai

Įsivaizduokite, kad apmokome kalbos modelį. Kad pasiektume gerą tikslumą, paprastai norime mažinti mokymosi greitį tęsiant apmokymą, dažniausiai $\mathcal{O}(t^{-\frac{1}{2}})$ arba lėtesne sparta. Dabar panagrinėkime modelį, apmokomą su retais požymiais, t. y. požymiais, kurie pasitaiko tik retai. Tai įprasta natūraliajai kalbai, pavyzdžiui, daug mažiau tikėtina, kad pamatysime žodį *prekondicionavimas* nei *mokymasis*. Tačiau tai taip pat įprasta kitose srityse, tokiose kaip skaičiuojamoji reklama ir personalizuotas kolaboratyvusis filtravimas. Juk yra daug dalykų, kurie domina tik nedidelį skaičių žmonių.

Su retais požymiais susiję parametrai gauna prasmingus atnaujinimus tik tada, kai šie požymiai pasitaiko. Esant mažėjančiam mokymosi greičiui, galime atsidurti situacijoje, kai dažnų požymių parametrai gana greitai konverguoja į savo optimalias reikšmes, o retų požymių atveju mums dar trūksta pakankamai dažnų stebėjimų, kad būtų galima nustatyti jų optimalias reikšmes. Kitaip tariant, mokymosi greitis dažniems požymiams mažėja per lėtai arba retiems – per greitai.

Galimas šios problemos taisymo būdas būtų skaičiuoti, kiek kartų matome tam tikrą požymį, ir naudoti tai kaip laikrodį mokymosi greičiams koreguoti. Tai yra, užuot pasirinkę $\eta = \frac{\eta_0}{\sqrt{t + c}}$ formos mokymosi greitį, galėtume naudoti $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$. Čia $s(i, t)$ skaičiuoja požymio $i$ nenulinių reikšmių skaičių, kurį stebėjome iki momento $t$. Tai iš tikrųjų gana lengva įgyvendinti be reikšmingų pridėtinių sąnaudų. Tačiau tai nepasiteisina, kai neturime tikro retumo, o tik duomenis, kuriuose gradientai dažnai labai maži ir tik retai dideli. Juk neaišku, kur brėžti ribą tarp to, kas laikoma stebėtu požymiu, o kas ne.

:citet:`Duchi.Hazan.Singer.2011` Adagrad tai sprendžia pakeisdamas gana grubų skaitiklį $s(i, t)$ anksčiau stebėtų gradientų kvadratų agregatu. Visų pirma mokymosi greičiui koreguoti jis naudoja $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$. Tai turi du privalumus: pirma, nebereikia nuspręsti, kada gradientas yra pakankamai didelis. Antra, jis automatiškai keičia mastelį pagal gradientų dydį. Koordinatės, kurios įprastai atitinka didelius gradientus, gerokai sumažinamos, o kitos su mažais gradientais sulaukia daug švelnesnio elgesio. Praktikoje tai veda prie labai veiksmingos skaičiuojamosios reklamos ir panašių uždavinių optimizavimo procedūros. Tačiau tai paslepia kai kuriuos papildomus Adagrad privalumus, kuriuos geriausia suprasti prekondicionavimo kontekste.


## Prekondicionavimas

Iškilieji optimizavimo uždaviniai tinka algoritmų savybėms analizuoti. Juk daugumai neiškiliųjų uždavinių sunku išvesti prasmingas teorines garantijas, tačiau *intuicija* ir *įžvalgos* dažnai išlieka. Panagrinėkime uždavinį minimizuoti $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

Kaip matėme :numref:`sec_momentum`, šį uždavinį galima perrašyti per jo tikrinį skaidinį $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ ir gauti daug paprastesnį uždavinį, kuriame kiekvieną koordinatę galima spręsti atskirai:

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Čia naudojome $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$, taigi ir $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$. Modifikuoto uždavinio minimizuotojas yra $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$, o minimali reikšmė $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Tai daug lengviau apskaičiuoti, nes $\boldsymbol{\Lambda}$ yra įstrižaininė matrica, kurioje yra $\mathbf{Q}$ tikrinės reikšmės.

Jei $\mathbf{c}$ truputį sutrikdytume, tikėtume, kad $f$ minimizuotojas pasikeis tik nežymiai. Deja, taip nėra. Nors nežymūs $\mathbf{c}$ pokyčiai lemia tokius pat nežymius $\bar{\mathbf{c}}$ pokyčius, to negalima pasakyti apie $f$ (ir atitinkamai $\bar{f}$) minimizuotoją. Kai tikrinės reikšmės $\boldsymbol{\Lambda}_i$ didelės, matysime tik mažus $\bar{x}_i$ ir $\bar{f}$ minimumo pokyčius. Ir atvirkščiai, esant mažoms $\boldsymbol{\Lambda}_i$, $\bar{x}_i$ pokyčiai gali būti dramatiški. Didžiausios ir mažiausios tikrinės reikšmės santykis vadinamas optimizavimo uždavinio sąlygotumo skaičiumi.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Jei sąlygotumo skaičius $\kappa$ didelis, optimizavimo uždavinį sunku tiksliai išspręsti. Turime pasirūpinti, kad teisingai gautume didelį reikšmių dinaminį diapazoną. Mūsų analizė veda į akivaizdų, nors ir kiek naivų klausimą: ar negalėtume tiesiog „ištaisyti“ problemos iškreipdami erdvę taip, kad visos tikrinės reikšmės būtų $1$? Teoriškai tai gana lengva: reikia tik $\mathbf{Q}$ tikrinių reikšmių ir vektorių, kad uždavinį perskaičiuotume iš $\mathbf{x}$ į uždavinį su $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. Naujoje koordinačių sistemoje $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ galima supaprastinti iki $\|\mathbf{z}\|^2$. Deja, tai gana nepraktiškas pasiūlymas. Tikrinių reikšmių ir vektorių skaičiavimas apskritai yra *daug brangesnis* nei paties uždavinio sprendimas.

Nors tikslus tikrinių reikšmių skaičiavimas gali būti brangus, jų spėjimas ir apskaičiavimas bent apytiksliai gali būti daug geriau nei visiškai nieko nedaryti. Visų pirma galėtume naudoti $\mathbf{Q}$ įstrižainės elementus ir atitinkamai pakeisti mastelį. Tai *daug* pigiau nei tikrinių reikšmių skaičiavimas.

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Šiuo atveju turime $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ ir konkrečiai $\tilde{\mathbf{Q}}_{ii} = 1$ visiems $i$. Daugeliu atvejų tai gerokai supaprastina sąlygotumo skaičių. Pavyzdžiui, anksčiau aptartais atvejais tai visiškai pašalintų nagrinėjamą problemą, nes uždavinys yra lygiagretus ašims.

Deja, susiduriame su dar viena problema: giliajame mokymesi paprastai net neturime prieigos prie tikslo funkcijos antrosios išvestinės: kai $\mathbf{x} \in \mathbb{R}^d$, antrajai išvestinei net minirinkinyje apskaičiuoti gali prireikti $\mathcal{O}(d^2)$ atminties ir darbo, todėl praktiškai tai neįmanoma. Išradinga Adagrad idėja yra naudoti šios sunkiai pasiekiamos Hesės matricos įstrižainės pakaitalą, kurį palyginti pigu apskaičiuoti ir kuris yra veiksmingas – pačio gradiento dydį.

Kad suprastume, kodėl tai veikia, pažvelkime į $\bar{f}(\bar{\mathbf{x}})$. Turime, kad

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

kur $\bar{\mathbf{x}}_0$ yra $\bar{f}$ minimizuotojas. Taigi gradiento dydis priklauso ir nuo $\boldsymbol{\Lambda}$, ir nuo atstumo iki optimumo. Jei $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ nesikeistų, to užtektų. Juk tokiu atveju gradiento dydžio $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ pakanka. Kadangi AdaGrad yra stochastinio gradientinio nusileidimo algoritmas, matysime nenulinės dispersijos gradientus net optimume. Todėl galime saugiai naudoti gradientų dispersiją kaip pigų Hesės matricos mastelio pakaitalą. Išsami analizė peržengia šio skyrelio ribas (tai užimtų kelis puslapius). Išsamiau skaitytojus nukreipiame į :cite:`Duchi.Hazan.Singer.2011`.

## Algoritmas

Formalizuokime aukščiau pateiktą diskusiją. Kintamąjį $\mathbf{s}_t$ naudojame praeities gradientų dispersijai kaupti taip.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Čia operacijos taikomos kiekvienai koordinatei atskirai. Tai yra, $\mathbf{v}^2$ turi elementus $v_i^2$. Panašiai $\frac{1}{\sqrt{v}}$ turi elementus $\frac{1}{\sqrt{v_i}}$, o $\mathbf{u} \cdot \mathbf{v}$ turi elementus $u_i v_i$. Kaip ir anksčiau, $\eta$ yra mokymosi greitis, o $\epsilon$ yra adityvinė konstanta, užtikrinanti, kad nedalytume iš $0$. Galiausiai inicializuojame $\mathbf{s}_0 = \mathbf{0}$.

Kaip ir momento metodo atveju, turime sekti pagalbinį kintamąjį, šiuo atveju, kad kiekvienai koordinatei būtų galima turėti atskirą mokymosi greitį. Tai reikšmingai nepadidina Adagrad kainos, palyginti su SGD, nes pagrindinės sąnaudos paprastai yra skaičiuoti $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ ir jo išvestinę.

Atkreipkite dėmesį, kad kaupiant gradientų kvadratus $\mathbf{s}_t$ reiškia, kad $\mathbf{s}_t$ auga iš esmės tiesiškai (praktikoje šiek tiek lėčiau nei tiesiškai, nes gradientai pradžioje mažėja). Tai lemia $\mathcal{O}(t^{-\frac{1}{2}})$ mokymosi greitį, nors ir koreguojamą kiekvienai koordinatei atskirai. Iškiliesiems uždaviniams tai visiškai tinka. Tačiau giliajame mokymesi galbūt norėtume mokymosi greitį mažinti gerokai lėčiau. Tai paskatino sukurti nemažai Adagrad variantų, kuriuos aptarsime vėlesniuose skyreliuose. Kol kas pažiūrėkime, kaip jis elgiasi kvadratiniame iškiliajame uždavinyje. Naudojame tą patį uždavinį kaip anksčiau:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Adagrad įgyvendinsime naudodami tą patį mokymosi greitį kaip anksčiau, t. y. $\eta = 0.4$. Kaip matome, nepriklausomo kintamojo iteracinė trajektorija yra glotnesnė. Tačiau dėl kaupiamojo $\boldsymbol{s}_t$ poveikio mokymosi greitis nuolat mažėja, todėl nepriklausomas kintamasis vėlesniuose iteracijų etapuose juda nebe taip daug.

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Padidinus mokymosi greitį iki $2$, matome daug geresnį elgesį. Tai jau rodo, kad mokymosi greičio mažėjimas gali būti gana agresyvus net be triukšmo atveju, todėl turime užtikrinti, kad parametrai tinkamai konverguotų.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Įgyvendinimas nuo nulio

Kaip ir momentų metodas, Adagrad turi išlaikyti būsenos kintamąjį tokios pat formos kaip parametrai.

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Palyginti su eksperimentu :numref:`sec_minibatch_sgd`, modeliui apmokyti naudojame
didesnį mokymosi greitį.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Glaustas įgyvendinimas

Naudodami algoritmo `adagrad` egzempliorių `Trainer`, galime iškviesti Adagrad algoritmą Gluon.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Santrauka

* Adagrad dinamiškai mažina mokymosi greitį kiekvienai koordinatei atskirai.
* Jis naudoja gradiento dydį kaip priemonę koreguoti, kaip greitai pasiekiama pažanga: koordinatės su dideliais gradientais kompensuojamos mažesniu mokymosi greičiu.
* Tikslios antrosios išvestinės skaičiavimas giliojo mokymosi uždaviniuose paprastai neįmanomas dėl atminties ir skaičiavimo apribojimų. Gradientas gali būti naudingas pakaitalas.
* Jei optimizavimo uždavinys turi gana netolygią struktūrą, Adagrad gali padėti sušvelninti iškraipymą.
* Adagrad ypač veiksmingas retiems požymiams, kai retai pasitaikantiems nariams mokymosi greitį reikia mažinti lėčiau.
* Giliojo mokymosi uždaviniuose Adagrad kartais gali būti pernelyg agresyvus mažindamas mokymosi greitį. Strategijas tam sušvelninti aptarsime :numref:`sec_adam` kontekste.

## Pratimai

1. Įrodykite, kad ortogonaliajai matricai $\mathbf{U}$ ir vektoriui $\mathbf{c}$ galioja: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Kodėl tai reiškia, kad sutrikdymų dydis nepasikeičia po ortogonaliojo kintamųjų keitimo?
1. Išbandykite Adagrad $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ ir taip pat tikslo funkcijai, pasuktai 45 laipsniais, t. y. $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Ar jis elgiasi kitaip?
1. Įrodykite [Geršgorino apskritimų teoremą](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem), kuri teigia, kad matricos $\mathbf{M}$ tikrinės reikšmės $\lambda_i$ tenkina $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ bent vienam $j$ pasirinkimui.
1. Ką Geršgorino teorema mums sako apie įstrižaininiu būdu prekondicionuotos matricos $\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$ tikrines reikšmes?
1. Išbandykite Adagrad tinkamam giliajam tinklui, pavyzdžiui, :numref:`sec_lenet` taikomam Fashion-MNIST.
1. Kaip reikėtų modifikuoti Adagrad, kad mokymosi greičio mažėjimas būtų ne toks agresyvus?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1073)
:end_tab:
