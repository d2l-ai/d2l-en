# Momentas
:label:`sec_momentum`

:numref:`sec_sgd` apžvelgėme, kas nutinka atliekant stochastinį gradientinį nusileidimą, t. y. optimizuojant, kai prieinamas tik triukšmingas gradiento variantas. Visų pirma pastebėjome, kad triukšmingiems gradientams turime būti ypač atsargūs rinkdamiesi mokymosi greitį esant triukšmui. Jei jį sumažiname per greitai, konvergavimas sustoja. Jei esame per nuolaidūs, nekonverguojame į pakankamai gerą sprendinį, nes triukšmas toliau veda mus nuo optimumo.

## Pagrindai

Šiame skyrelyje nagrinėsime veiksmingesnius optimizavimo algoritmus, ypač tam tikriems optimizavimo uždavinių tipams, kurie įprasti praktikoje.


### Nuotėkio vidurkiai

Ankstesniame skyrelyje aptarėme minirinkinių SGD kaip skaičiavimo spartinimo priemonę. Tai taip pat turėjo malonų šalutinį poveikį: gradientų vidurkinimas sumažino dispersiją. Minirinkinio stochastinį gradientinį nusileidimą galima apskaičiuoti pagal:

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Kad žymėjimas būtų paprastas, čia naudojome $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ kaip stochastinį gradientinį nusileidimą pavyzdžiui $i$, naudojant svorius, atnaujintus laiko momentu $t-1$.
Būtų gerai, jei dispersijos mažinimo poveikiu galėtume pasinaudoti ir už gradientų vidurkinimo minirinkinyje ribų. Vienas būdas tai pasiekti yra gradiento skaičiavimą pakeisti „nuotėkio vidurkiu“:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

kuriam nors $\beta \in (0, 1)$. Tai efektyviai pakeičia momentinį gradientą tokiu, kuris suvidurkintas iš kelių *praeities* gradientų. $\mathbf{v}$ vadinamas *greičiu*. Jis kaupia praeities gradientus panašiai, kaip sunkus rutulys, riedantis tikslo funkcijos reljefu, integruoja praeities jėgas. Kad geriau pamatytume, kas vyksta, išplėskime $\mathbf{v}_t$ rekursyviai į

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

Didelis $\beta$ reiškia tolimo nuotolio vidurkį, o mažas $\beta$ – tik nedidelę pataisą, palyginti su gradientiniu metodu. Naujas gradiento pakaitalas nebenurodo staigiausio nusileidimo krypties konkrečiame pavyzdyje, o veikiau praeities gradientų svertinio vidurkio kryptį. Tai leidžia mums pasiekti didžiąją dalį vidurkinimo rinkinyje naudos neapskaičiuojant jame gradientų. Prie šios vidurkinimo procedūros dar išsamiau grįšime vėliau.

Aukščiau pateiktas samprotavimas tapo pagrindu tam, kas dabar vadinama *pagreitintais* gradientiniais metodais, pavyzdžiui, gradientais su momentu. Jie turi papildomą privalumą – yra daug veiksmingesni tais atvejais, kai optimizavimo uždavinys yra blogai sąlygotas (t. y. kai kai kuriomis kryptimis pažanga daug lėtesnė nei kitomis, primenant siaurą kanjoną). Be to, jie leidžia vidurkinti vėlesnius gradientus ir gauti stabilesnes nusileidimo kryptis. Iš tiesų pagreitinimo aspektas net be triukšmo iškiliesiems uždaviniams yra viena pagrindinių priežasčių, kodėl momentas veikia ir kodėl jis veikia taip gerai.

Kaip ir galima tikėtis, dėl savo veiksmingumo momentas yra gerai ištirta tema giliojo mokymosi optimizavime ir už jo ribų. Žr., pvz., puikų :citet:`Goh.2017` [aiškinamąjį straipsnį](https://distill.pub/2017/momentum/), kuriame pateikiama išsami analizė ir interaktyvi animacija. Jį pasiūlė :citet:`Polyak.1964`. :citet:`Nesterov.2018` pateikia išsamų teorinį aptarimą iškiliojo optimizavimo kontekste. Momento nauda giliajame mokymesi žinoma jau seniai. Žr., pvz., :citet:`Sutskever.Martens.Dahl.ea.2013` aptarimą.

### Blogai sąlygotas uždavinys

Kad geriau suprastume geometrines momentų metodo savybes, grįžtame prie gradientinio nusileidimo, nors ir su gerokai nemalonesne tikslo funkcija. Prisiminkite, kad :numref:`sec_gd` naudojome $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, t. y. vidutiniškai iškreiptą elipsoidinę tikslo funkciją. Šią funkciją iškreipiame dar labiau ištempdami ją $x_1$ kryptimi:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Kaip ir anksčiau, $f$ minimumas yra $(0, 0)$. Ši funkcija yra *labai* plokščia $x_1$ kryptimi. Pažiūrėkime, kas nutinka, kai su šia nauja funkcija atliekame gradientinį nusileidimą kaip anksčiau. Pasirenkame mokymosi greitį $0.4$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Pagal konstrukciją, gradientas $x_2$ kryptimi yra *daug* didesnis ir kinta daug greičiau nei horizontalia $x_1$ kryptimi. Taigi esame įstrigę tarp dviejų nepageidaujamų pasirinkimų: jei pasirenkame mažą mokymosi greitį, užtikriname, kad sprendinys nediverguos $x_2$ kryptimi, bet gauname lėtą konvergavimą $x_1$ kryptimi. Ir atvirkščiai, su dideliu mokymosi greičiu greitai judame $x_1$ kryptimi, bet diverguojame $x_2$ kryptimi. Toliau pateiktas pavyzdys iliustruoja, kas nutinka net po nežymaus mokymosi greičio padidinimo nuo $0.4$ iki $0.6$. Konvergavimas $x_1$ kryptimi pagerėja, bet bendra sprendinio kokybė yra daug prastesnė.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Momentų metodas

Momentų metodas leidžia išspręsti aukščiau aprašytą gradientinio nusileidimo problemą. Žiūrėdami į aukščiau pateiktą optimizavimo trasą, galime nujausti, kad praeities gradientų vidurkinimas veiktų gerai. Juk $x_1$ kryptimi tai sukaups gerai sulygiuotus gradientus, taip padidindamas atstumą, kurį įveikiame kiekvienu žingsniu. Ir atvirkščiai, $x_2$ kryptimi, kur gradientai svyruoja, agreguotas gradientas sumažins žingsnio dydį dėl vienas kitą naikinančių svyravimų.
Naudojant $\mathbf{v}_t$ vietoj gradiento $\mathbf{g}_t$, gaunamos šios atnaujinimo lygtys:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Atkreipkite dėmesį, kad kai $\beta = 0$, gauname įprastą gradientinį nusileidimą. Prieš gilindamiesi į matematines savybes, trumpai pažvelkime, kaip algoritmas elgiasi praktikoje.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Kaip matome, net su tuo pačiu mokymosi greičiu, kurį naudojome anksčiau, momentas vis tiek gerai konverguoja. Pažiūrėkime, kas nutinka sumažinus momento parametrą. Perpus sumažinus iki $\beta = 0.25$, gauname trajektoriją, kuri vos konverguoja. Vis dėlto tai daug geriau nei be momento (kai sprendinys diverguoja).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Atkreipkite dėmesį, kad momentą galime derinti su stochastiniu gradientiniu nusileidimu ir ypač su minirinkinių stochastiniu gradientiniu nusileidimu. Vienintelis pokytis yra tas, kad tuomet gradientus $\mathbf{g}_{t, t-1}$ pakeičiame $\mathbf{g}_t$. Galiausiai patogumui inicializuojame $\mathbf{v}_0 = 0$ laiko momentu $t=0$. Pažiūrėkime, ką nuotėkio vidurkinimas iš tikrųjų daro atnaujinimams.

### Efektyvus pavyzdžio svoris

Prisiminkite, kad $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Riboje nariai susumuojami į $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Kitaip tariant, užuot darę $\eta$ dydžio žingsnį gradientiniame ar stochastiniame gradientiniame nusileidime, darome $\frac{\eta}{1-\beta}$ dydžio žingsnį, kartu gaudami potencialiai daug geriau besielgiančią nusileidimo kryptį. Tai du privalumai viename. Kad iliustruotume, kaip svoriai elgiasi esant skirtingoms $\beta$ reikšmėms, panagrinėkite žemiau pateiktą diagramą.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Praktiniai eksperimentai

Pažiūrėkime, kaip momentas veikia praktikoje, t. y. kai naudojamas tinkamo optimizatoriaus kontekste. Tam mums reikia kiek labiau mastelio didėjimui pritaikyto įgyvendinimo.

### Įgyvendinimas nuo nulio

Palyginti su (minirinkinių) stochastiniu gradientiniu nusileidimu, momentų metodas turi išlaikyti pagalbinių kintamųjų aibę, t. y. greitį. Jis turi tokią pat formą kaip gradientai (ir optimizavimo uždavinio kintamieji). Toliau pateiktame įgyvendinime šiuos kintamuosius vadiname `states`.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Pažiūrėkime, kaip tai veikia praktikoje.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Kai momento hiperparametrą `momentum` padidiname iki 0,9, tai atitinka gerokai didesnį efektyvų imties dydį $\frac{1}{1 - 0.9} = 10$. Mokymosi greitį truputį sumažiname iki $0.01$, kad išlaikytume padėtį kontroliuojamą.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

Dar labiau sumažinus mokymosi greitį, sprendžiamos neglotnių optimizavimo uždavinių problemos. Nustačius $0.005$, gaunamos geros konvergavimo savybės.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Glaustas įgyvendinimas

Gluon beveik nieko nereikia daryti, nes standartiniame `sgd` sprendiklyje momentas jau įtaisytas. Nustačius atitinkamus parametrus, gaunama labai panaši trajektorija.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Teorinė analizė

Iki šiol dvimatis pavyzdys $f(x) = 0.1 x_1^2 + 2 x_2^2$ atrodė gana dirbtinis. Dabar pamatysime, kad jis iš tikrųjų gana reprezentatyvus uždavinių, su kuriais galime susidurti, tipams, bent jau minimizuojant iškiliąsias kvadratines tikslo funkcijas.

### Kvadratinės iškilosios funkcijos

Panagrinėkime funkciją

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Tai bendroji kvadratinė funkcija. Teigiamai apibrėžtoms matricoms $\mathbf{Q} \succ 0$, t. y. matricoms su teigiamomis tikrinėmis reikšmėmis, ji turi minimizuotoją $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ su minimalia reikšme $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Taigi galime perrašyti $h$ kaip

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradientas yra $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Tai yra, jis lygus atstumui tarp $\mathbf{x}$ ir minimizuotojo, padaugintam iš $\mathbf{Q}$. Todėl ir greitis yra narių $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ tiesinis darinys.

Kadangi $\mathbf{Q}$ yra teigiamai apibrėžta, ją galima išskaidyti į jos tikrinę sistemą pagal $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ su ortogonaliąja (sukimo) matrica $\mathbf{O}$ ir įstrižaine teigiamų tikrinių reikšmių matrica $\boldsymbol{\Lambda}$. Tai leidžia pakeisti kintamuosius iš $\mathbf{x}$ į $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ ir gauti daug paprastesnę išraišką:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Čia $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Kadangi $\mathbf{O}$ yra tik ortogonalioji matrica, tai reikšmingai nesutrikdo gradientų. Išreikštas per $\mathbf{z}$, gradientinis nusileidimas tampa

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Svarbus šios išraiškos faktas yra tas, kad gradientinis nusileidimas *nemaišo* skirtingų tikrinių poerdvių. Tai yra, išreikštas per $\mathbf{Q}$ tikrinę sistemą, optimizavimo uždavinys vyksta koordinatė po koordinatės. Tai galioja ir

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Tai darydami, ką tik įrodėme tokią teoremą: gradientinis nusileidimas su momentu ir be jo iškiliajai kvadratinei funkcijai skaidomas į optimizavimą koordinatė po koordinatės kvadratinės matricos tikrinių vektorių kryptimis.

### Skaliarinės funkcijos

Turėdami aukščiau pateiktą rezultatą, pažiūrėkime, kas nutinka minimizuojant funkciją $f(x) = \frac{\lambda}{2} x^2$. Gradientiniam nusileidimui turime

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Kai tik $|1 - \eta \lambda| < 1$, šis optimizavimas konverguoja eksponentiniu greičiu, nes po $t$ žingsnių turime $x_t = (1 - \eta \lambda)^t x_0$. Tai rodo, kaip konvergavimo greitis iš pradžių gerėja didinant mokymosi greitį $\eta$, kol $\eta \lambda = 1$. Toliau viskas diverguoja, o kai $\eta \lambda > 2$, optimizavimo uždavinys diverguoja.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Kad išanalizuotume konvergavimą momento atveju, pradedame perrašydami atnaujinimo lygtis dviem skaliarais: vienu $x$ ir vienu greičiui $v$. Taip gauname:

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Konvergavimo elgesį lemiančiai $2 \times 2$ matricai žymėti naudojome $\mathbf{R}$. Po $t$ žingsnių pradinis pasirinkimas $[v_0, x_0]$ tampa $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Taigi konvergavimo greitį lemia $\mathbf{R}$ tikrinės reikšmės. Puikią animaciją rasite :citet:`Goh.2017` [Distill įraše](https://distill.pub/2017/momentum/), o išsamią analizę – :citet:`Flammarion.Bach.2015`. Galima įrodyti, kad kai $0 < \eta \lambda < 2 + 2 \beta$, greitis konverguoja. Tai platesnis galimų parametrų diapazonas nei $0 < \eta \lambda < 2$ gradientiniam nusileidimui. Tai taip pat leidžia manyti, kad apskritai pageidautinos didelės $\beta$ reikšmės. Tolesnės detalės reikalauja nemažai techninių smulkmenų, todėl rekomenduojame norintiems susipažinti su originaliomis publikacijomis.

## Santrauka

* Momentas pakeičia gradientus nuotėkio vidurkiu iš praeities gradientų. Tai gerokai pagreitina konvergavimą.
* Jis pageidautinas tiek be triukšmo gradientiniam nusileidimui, tiek (triukšmingam) stochastiniam gradientiniam nusileidimui.
* Momentas apsaugo nuo optimizavimo proceso sustojimo, kuris daug labiau tikėtinas stochastiniam gradientiniam nusileidimui.
* Efektyvus gradientų skaičius yra $\frac{1}{1-\beta}$ dėl eksponentinio praeities duomenų svorio mažinimo.
* Iškiliųjų kvadratinių uždavinių atveju tai galima išsamiai analizuoti aiškiai.
* Įgyvendinimas gana paprastas, bet reikia saugoti papildomą būsenos vektorių (greitį $\mathbf{v}$).

## Pratimai

1. Naudokite kitus momento hiperparametrų ir mokymosi greičių derinius ir stebėkite bei analizuokite skirtingus eksperimentų rezultatus.
1. Išbandykite gradientinį nusileidimą ir momentą kvadratiniam uždaviniui, kuriame turite kelias tikrines reikšmes, t. y. $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, pvz., $\lambda_i = 2^{-i}$. Nubraižykite, kaip mažėja $x$ reikšmės esant inicializacijai $x_i = 1$.
1. Išveskite $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$ minimalią reikšmę ir minimizuotoją.
1. Kas pasikeičia atliekant stochastinį gradientinį nusileidimą su momentu? Kas nutinka naudojant minirinkinių stochastinį gradientinį nusileidimą su momentu? Eksperimentuokite su parametrais.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1071)
:end_tab:
