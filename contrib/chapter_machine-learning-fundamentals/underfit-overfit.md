```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Nepakankamas mokymasis ir persimokymas 
:label:`sec_polynomial`

Šiame skyriuje išbandome kai kurias anksčiau matytas sąvokas. Kad viskas būtų paprasta, kaip žaislinį pavyzdį naudojame polinominę regresiją.

```{.python .input  n=3}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import math
```

```{.python .input  n=5}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

### Duomenų rinkinio generavimas

Pirmiausia mums reikia duomenų. Turėdami $x$, mokymo ir testavimo duomenų [**žymėms generuoti naudosime šį kubinį polinomą**]:

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

Triukšmo narys $\epsilon$ paklūsta normaliajam skirstiniui
su vidurkiu 0 ir standartiniu nuokrypiu 0,1.
Optimizuojant paprastai norime vengti
labai didelių gradientų ar nuostolių reikšmių.
Todėl *požymiai*
perskalėjami iš $x^i$ į $\frac{x^i}{i!}$.
Tai leidžia išvengti labai didelių reikšmių didelėms laipsnio rodiklio reikšmėms $i$.
Sugeneruosime po 100 pavyzdžių mokymo ir testavimo rinkiniui.

```{.python .input  n=6}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()        
        p, n = max(3, self.num_inputs), num_train + num_val
        w = d2l.tensor([1.2, -3.4, 5.6] + [0]*(p-3))
        if tab.selected('mxnet') or tab.selected('pytorch'):
            x = d2l.randn(n, 1)
            noise = d2l.randn(n, 1) * 0.1
        if tab.selected('tensorflow'):
            x = d2l.normal((n, 1))
            noise = d2l.normal((n, 1)) * 0.1
        X = d2l.concat([x ** (i+1) / math.gamma(i+2) for i in range(p)], 1)
        self.y = d2l.matmul(X, d2l.reshape(w, (-1, 1))) + noise
        self.X = X[:,:num_inputs]
        
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

Dar kartą primename, kad `poly_features` saugomi vienanariai
perskalėjami gama funkcija,
kur $\Gamma(n)=(n-1)!$.
[**Pažiūrėkite į pirmuosius 2 pavyzdžius**] iš sugeneruoto duomenų rinkinio.
Reikšmė 1 techniškai yra požymis,
būtent pastovusis požymis, atitinkantis poslinkį.

### [**Trečios eilės polinominės funkcijos pritaikymas (normalus atvejis)**]

Pradėsime naudodami trečios eilės polinominę funkciją, kuri yra tokios pat eilės kaip duomenų generavimo funkcija.
Rezultatai rodo, kad šio modelio mokymo ir testavimo nuostolius galima efektyviai sumažinti.
Išmoktieji modelio parametrai taip pat yra artimi tikrosioms reikšmėms $w = [1{,}2; -3{,}4; 5{,}6], b=5$.

```{.python .input  n=7}
%%tab all
def train(p):
    if tab.selected('mxnet') or tab.selected('tensorflow'):
        model = d2l.LinearRegression(lr=0.01)
    if tab.selected('pytorch'):
        model = d2l.LinearRegression(p, lr=0.01)
    model.board.ylim = [1, 1e2]
    data = Data(200, 200, p, 20)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    print(model.get_w_b())
    
train(p=3)
```

### [**Tiesinės funkcijos pritaikymas (nepakankamas mokymasis)**]

Dar kartą pažiūrėkime į tiesinės funkcijos pritaikymą.
Po sumažėjimo ankstyvosiose epochose
sunku toliau mažinti
šio modelio mokymo nuostolius.
Kai baigiama paskutinė epochos iteracija,
mokymo nuostoliai vis dar dideli.
Kai tiesiniai modeliai naudojami netiesiniams dėsningumams
(kaip čia trečios eilės polinominė funkcija) pritaikyti,
jie linkę nepakankamai išmokti.

```{.python .input  n=8}
%%tab all
train(p=1)
```

### [**Aukštesnės eilės polinominės funkcijos pritaikymas (persimokymas)**]

Dabar pabandykime apmokyti modelį
naudodami per aukšto laipsnio polinomą.
Čia nepakanka duomenų, kad būtų išmokta, jog
aukštesnio laipsnio koeficientai turėtų būti artimi nuliui.
Todėl mūsų pernelyg sudėtingas modelis
yra toks jautrus, kad jį veikia
mokymo duomenų triukšmas.
Nors mokymo nuostolius galima efektyviai sumažinti,
testavimo nuostoliai vis tiek yra gerokai didesni.
Tai rodo, kad
sudėtingas modelis persimoko duomenims.

```{.python .input  n=9}
%%tab all
train(p=10)
```

Tolesniuose skyriuose toliau
aptarsime persimokymo problemas
ir jų sprendimo būdus,
pavyzdžiui, svorių nykimą ir atsitiktinį praretinimą.


## Santrauka

* Kadangi generalizacijos paklaidos negalima įvertinti pagal mokymo paklaidą, vien mokymo paklaidos minimizavimas nebūtinai reiškia generalizacijos paklaidos sumažėjimą. Mašininio mokymo modeliai turi būti atsargūs ir saugotis persimokymo, kad generalizacijos paklaida būtų kuo mažesnė.
* Validavimo aibę galima naudoti modelio parinkimui, jei ji nenaudojama pernelyg laisvai.
* Nepakankamas mokymasis reiškia, kad modelis negali sumažinti mokymo paklaidos. Kai mokymo paklaida yra daug mažesnė už validavimo paklaidą, vyksta persimokymas.
* Turėtume pasirinkti tinkamai sudėtingą modelį ir vengti nepakankamo mokymo pavyzdžių skaičiaus.


## Pratimai

1. Ar galite tiksliai išspręsti polinominės regresijos uždavinį? Užuomina: naudokite tiesinę algebrą.
1. Panagrinėkite polinomų modelio parinkimą:
    1. Nubraižykite mokymo nuostolius, priklausomai nuo modelio sudėtingumo (polinomo laipsnio). Ką pastebite? Kokio laipsnio polinomo reikia, kad mokymo nuostoliai sumažėtų iki 0?
    1. Šiuo atveju nubraižykite testavimo nuostolius.
    1. Sugeneruokite tą patį grafiką kaip duomenų kiekio funkciją.
1. Kas nutinka, jei atsisakote polinominių požymių $x^i$ normalizavimo ($1/i!$)? Ar galite tai ištaisyti kitu būdu?
1. Ar kada nors galite tikėtis nulinės generalizacijos paklaidos?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/234)
:end_tab:
