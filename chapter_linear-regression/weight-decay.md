```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Svorių nykimas
:label:`sec_weight_decay`

Kai jau apibūdinome persimokymo problemą,
galime pristatyti pirmąjį savo *reguliarizavimo* metodą.
Prisiminkite, kad persimokymą visada galime sumažinti
surinkdami daugiau apmokymo duomenų.
Tačiau tai gali būti brangu, imti daug laiko
arba visiškai nuo mūsų nepriklausyti,
todėl trumpuoju laikotarpiu tai neįmanoma.
Kol kas galime laikyti, kad jau turime
tiek aukštos kokybės duomenų, kiek leidžia mūsų ištekliai,
ir sutelkti dėmesį į turimas priemones,
kai duomenų rinkinys laikomas duotu.

Prisiminkite, kad savo polinominės regresijos pavyzdyje
(:numref:`subsec_polynomial-curve-fitting`)
galėjome apriboti savo modelio talpą,
keisdami pritaikomo polinomo
laipsnį.
Iš tiesų požymių skaičiaus ribojimas
yra populiarus persimokymo mažinimo būdas.
Tačiau paprastas požymių atmetimas
gali būti pernelyg grubi priemonė.
Pasilikę prie polinominės regresijos
pavyzdžio, panagrinėkime, kas gali nutikti
su didelės dimensijos įvestimi.
Natūralūs polinomų apibendrinimai
daugiamačiams duomenims vadinami *monomais*,
kurie yra tiesiog kintamųjų laipsnių sandaugos.
Monomo laipsnis yra laipsnių suma.
Pavyzdžiui, $x_1^2 x_2$ ir $x_3 x_5^2$
yra abu 3 laipsnio monomai.

Atkreipkite dėmesį, kad $d$ laipsnio narių skaičius
sparčiai didėja, augant $d$.
Turint $k$ kintamųjų, $d$ laipsnio monomų
skaičius yra ${k - 1 + d} \choose {k - 1}$.
Net nedideli laipsnio pokyčiai, tarkime, nuo $2$ iki $3$,
dramatiškai padidina mūsų modelio sudėtingumą.
Todėl dažnai mums reikia smulkesnės priemonės
funkcijos sudėtingumui reguliuoti.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
```

## Normos ir svorių nykimas

(**Užuot tiesiogiai keitęs parametrų skaičių,
*svorių nykimas* veikia apribodamas reikšmes,
kurias gali įgyti parametrai.**)
Už giliojo mokymosi ssluoksnių ribų dažniau vadinamas
$\ell_2$ reguliarizavimu,
kai optimizuojamas minirinkinio stochastiniu gradientiniu nusileidimu,
svorių nykimas galbūt yra plačiausiai naudojamas metodas
parametriniams mašininio mokymosi modeliams reguliarizuoti.
Šio metodo pagrindas yra pagrindinė intuicija,
kad tarp visų funkcijų $f$
funkcija $f = 0$
(priskirianti reikšmę $0$ visoms įvestims)
tam tikra prasme yra *paprasčiausia*,
ir kad funkcijos sudėtingumą galime išmatuoti
jos parametrų atstumu nuo nulio.
Bet kaip tiksliai turėtume matuoti
atstumą tarp funkcijos ir nulio?
Vienintelio teisingo atsakymo nėra.
Tiesą sakant, ištisos matematikos šakos,
įskaitant funkcinės analizės dalis
ir Banacho erdvių teoriją,
skirtos tokiems klausimams spręsti.

Viena paprasta interpretacija galėtų būti
matuoti tiesinės funkcijos
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ sudėtingumą
kokia nors jos svorių vektoriaus norma, pvz., $\| \mathbf{w} \|^2$.
Prisiminkite, kad $\ell_2$ ir $\ell_1$ normas,
kurios yra bendresnės $\ell_p$ normos atskiri atvejai,
pristatėme :numref:`subsec_lin-algebra-norms`.
Dažniausias būdas užtikrinti mažą svorių vektorių
yra pridėti jo normą kaip baudos narį
prie nuostolių minimizavimo uždavinio.
Taigi savo pradinį tikslą,
*minimizuoti prognozės nuostolius apmokymo žymenyse*,
pakeičiame nauju tikslu,
*minimizuoti prognozės nuostolių ir baudos nario sumą*.
Dabar, jei mūsų svorių vektorius pernelyg išauga,
mūsų mokymosi algoritmas gali sutelkti dėmesį
į svorių normos $\| \mathbf{w} \|^2$ minimizavimą,
o ne į apmokymo paklaidos minimizavimą.
Būtent to ir norime.
Kad viską parodytume kode,
atgaivinsime savo ankstesnį pavyzdį
iš :numref:`sec_linear_regression` apie tiesinę regresiją.
Ten mūsų nuostoliai buvo

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Prisiminkite, kad $\mathbf{x}^{(i)}$ yra požymiai,
$y^{(i)}$ yra bet kurio duomenų pavyzdžio $i$ žymuo, o $(\mathbf{w}, b)$
yra atitinkamai svorių ir poslinkio parametrai.
Kad nubaustume svorių vektoriaus dydį,
turime kaip nors pridėti $\| \mathbf{w} \|^2$ prie nuostolių funkcijos,
bet kaip modelis turėtų kompromisiškai suderinti
standartinius nuostolius su šia nauja adityviąja bauda?
Praktikoje šį kompromisą apibūdiname
*reguliarizavimo konstanta* $\lambda$,
neneigiamu hiperparametru,
kurį parenkame naudodami validavimo duomenis:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


Kai $\lambda = 0$, atkuriame savo pradinę nuostolių funkciją.
Kai $\lambda > 0$, apribojame $\| \mathbf{w} \|$ dydį.
Dalijame iš $2$ pagal susitarimą:
kai imame kvadratinės funkcijos išvestinę,
$2$ ir $1/2$ susiprastina, užtikrindami, kad atnaujinimo
išraiška atrodytų gražiai ir paprastai.
Atidus skaitytojas galėtų susimąstyti, kodėl dirbame su kvadratine
norma, o ne su standartine norma (t. y. Euklido atstumu).
Taip darome dėl skaičiavimo patogumo.
Pakėlę $\ell_2$ normą kvadratu, atsikratome kvadratinės šaknies,
ir lieka kiekvienos svorių vektoriaus komponentės kvadratų suma.
Dėl to baudos išvestinę lengva apskaičiuoti:
išvestinių suma lygi sumos išvestinei.


Be to, galite paklausti, kodėl apskritai dirbame su $\ell_2$ norma,
o ne, tarkime, su $\ell_1$ norma.
Iš tiesų kiti pasirinkimai yra tinkami ir
populiarūs visoje statistikoje.
Nors $\ell_2$ reguliarizuoti tiesiniai modeliai sudaro
klasikinį *kalnagūbrio regresijos* (angl. *ridge regression*) algoritmą,
$\ell_1$ reguliarizuota tiesinė regresija
yra panašiai fundamentalus statistikos metodas,
populiariai žinomas kaip *Lasso regresija*.
Viena priežasčių dirbti su $\ell_2$ norma yra ta,
kad ji skiria neproporcingai didelę baudą
dideliems svorių vektoriaus komponentams.
Tai paslenka mūsų mokymosi algoritmą
link modelių, kurie svorį paskirsto tolygiai
tarp didesnio požymių skaičiaus.
Praktikoje tai gali padaryti juos atsparesnius
vieno kintamojo matavimo paklaidai.
Priešingai, $\ell_1$ baudos veda prie modelių,
kurie sutelkia svorius nedidelėje požymių aibėje,
kitus svorius nustatydami lygius nuliui.
Tai suteikia mums efektyvų *požymių atrankos* metodą,
kuris gali būti pageidautinas dėl kitų priežasčių.
Pavyzdžiui, jei mūsų modelis remiasi tik keliais požymiais,
tada mums gali nereikėti rinkti, saugoti ar perduoti duomenų
apie kitus (atmestus) požymius.

Naudodami tą patį žymėjimą kaip :eqref:`eq_linreg_batch_update`,
minirinkinio stochastinis gradientinis nusileidimas
$\ell_2$ reguliarizuotai regresijai atnaujina taip:

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Kaip ir anksčiau, atnaujiname $\mathbf{w}$ pagal dydį,
kuriuo mūsų įvertis skiriasi nuo stebinio.
Tačiau taip pat mažiname $\mathbf{w}$ dydį link nulio.
Todėl metodas kartais vadinamas „svorių nykimu":
turint tik baudos narį,
mūsų optimizavimo algoritmas kiekviename apmokymo žingsnyje
*sumažina* svorį.
Skirtingai nuo požymių atrankos,
svorių nykimas suteikia mums mechanizmą nuolat reguliuoti funkcijos sudėtingumą.
Mažesnės $\lambda$ reikšmės atitinka
mažiau suvaržytą $\mathbf{w}$,
o didesnės $\lambda$ reikšmės
$\mathbf{w}$ varžo žymiai stipriau.
Ar įtraukiame atitinkamą poslinkio baudą $b^2$,
gali skirtis skirtinguose įgyvendinimuose
ir gali skirtis skirtinguose neuroninio tinklo sluoksniuose.
Dažnai poslinkio nario nereguliarizuojame.
Be to,
nors $\ell_2$ reguliarizavimas gali būti neekvivalentus svorių nykimui kitiems optimizavimo algoritmams,
reguliarizavimo idėja
mažinant svorių dydį
vis tiek išlieka teisinga.

## Didelės dimensijos tiesinė regresija

Svorių nykimo teikiamą naudą galime iliustruoti
paprastu sintetiniu pavyzdžiu.

Pirmiausia [**sugeneruojame duomenis kaip anksčiau**]:

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Šiame sintetiniame duomenų rinkinyje mūsų žymuo gaunamas
iš pagrindinės tiesinės įvesčių funkcijos,
iškraipytos Gauso triukšmu
su nuliniu vidurkiu ir standartiniu nuokrypiu 0,01.
Iliustracijos tikslais
galime išryškinti persimokymo poveikį,
padidindami savo uždavinio dimensiją iki $d = 200$
ir dirbdami su nedidele apmokymo aibe, turinčia tik 20 pavyzdžių.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Įgyvendinimas nuo nulio

Dabar pabandykime įgyvendinti svorių nykimą nuo nulio.
Kadangi mūsų optimizatorius yra
minirinkinio stochastinis gradientinis nusileidimas,
tereikia pridėti kvadratinę $\ell_2$ baudą
prie pradinės nuostolių funkcijos.

### (**$\ell_2$ normos baudos apibrėžimas**)

Galbūt patogiausias būdas įgyvendinti šią baudą
yra pakelti visus narius kvadratu vietoje ir juos susumuoti.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Modelio apibrėžimas

Galutiniame modelyje
tiesinė regresija ir kvadratiniai nuostoliai nepasikeitė nuo :numref:`sec_linear_scratch`,
todėl tiesiog apibrėšime klasės `d2l.LinearRegressionScratch` poklasį. Vienintelis pakeitimas čia yra tas, kad mūsų nuostoliai dabar apima baudos narį.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

Toliau pateiktas kodas pritaiko mūsų modelį apmokymo aibėje su 20 pavyzdžių ir įvertina jį validavimo aibėje su 100 pavyzdžių.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**Apmokymas be reguliarizavimo**]

Dabar paleidžiame šį kodą su `lambd = 0`,
išjungdami svorių nykimą.
Atkreipkite dėmesį, kad smarkiai persimokome:
apmokymo paklaida mažėja, bet validavimo paklaida nemažėja –
vadovėlinis persimokymo atvejis.

```{.python .input}
%%tab all
train_scratch(0)
```

### [**Svorių nykimo naudojimas**]

Žemiau paleidžiame su dideliu svorių nykimu.
Atkreipkite dėmesį, kad apmokymo paklaida didėja,
bet validavimo paklaida mažėja.
Būtent šio poveikio
ir tikimės iš reguliarizavimo.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**Glaustas įgyvendinimas**]

Kadangi svorių nykimas yra visur paplitęs
neuroninių tinklų optimizavime,
giliojo mokymosi karkasas padaro jį ypač patogų,
integruodamas svorių nykimą į patį optimizavimo algoritmą,
kad būtų lengva naudoti kartu su bet kuria nuostolių funkcija.
Be to, ši integracija turi skaičiavimo naudą:
ji leidžia pasitelkti įgyvendinimo gudrybes svorių nykimui pridėti prie algoritmo
be jokių papildomų skaičiavimo sąnaudų.
Kadangi atnaujinimo svorių nykimo dalis
priklauso tik nuo dabartinės kiekvieno parametro reikšmės,
optimizatorius vis tiek turi vieną kartą paliesti kiekvieną parametrą.

:begin_tab:`mxnet`
Žemiau nurodome
svorių nykimo hiperparametrą tiesiogiai
per `wd`, kurdami savo `Trainer`.
Pagal numatytuosius nustatymus Gluon sumažina ir
svorius, ir poslinkius vienu metu.
Atkreipkite dėmesį, kad hiperparametras `wd`
bus padaugintas iš `wd_mult`,
atnaujinant modelio parametrus.
Taigi, jei `wd_mult` nustatysime lygų nuliui,
poslinkio parametras $b$ nenyks.
:end_tab:

:begin_tab:`pytorch`
Žemiau nurodome
svorių nykimo hiperparametrą tiesiogiai
per `weight_decay`, kurdami savo optimizatorių.
Pagal numatytuosius nustatymus PyTorch sumažina ir
svorius, ir poslinkius vienu metu, bet
galime sukonfigūruoti optimizatorių skirtingus parametrus tvarkyti
pagal skirtingas politikas.
Čia `weight_decay` nustatome tik
svoriams (parametrams `net.weight`), todėl
poslinkis (parametras `net.bias`) nenyks.
:end_tab:

:begin_tab:`tensorflow`
Žemiau sukuriame $\ell_2$ reguliarizatorių su
svorių nykimo hiperparametru `wd` ir pritaikome jį sluoksnio svoriams
per argumentą `kernel_regularizer`.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**Grafikas atrodo panašiai kaip tada,
kai svorių nykimą įgyvendinome nuo nulio**].
Tačiau ši versija veikia greičiau
ir ją lengviau įgyvendinti,
o šie privalumai taps ryškesni,
kai spręsite didesnius uždavinius
ir šis darbas taps rutina.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

Kol kas palietėme vieną sampratą apie tai,
kas sudaro paprastą tiesinę funkciją.
Tačiau net paprastoms netiesinėms funkcijoms situacija gali būti gerokai sudėtingesnė. Tam parodyti
[atkuriančiojo branduolio Hilberto erdvės (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) sąvoka
leidžia taikyti tiesinėms funkcijoms
pristatytus įrankius netiesiniame kontekste.
Deja, RKHS pagrįsti algoritmai
dažniausiai prastai skalėja dideliems, didelės dimensijos duomenims.
Šioje knygoje dažnai taikysime įprastą euristiką,
pagal kurią svorių nykimas taikomas
visiems gilaus tinklo sluoksniams.

## Santrauka

Reguliarizavimas yra įprastas persimokymo sprendimo būdas. Klasikiniai reguliarizavimo metodai prideda baudos narį prie nuostolių funkcijos (apmokymo metu), kad sumažintų išmokto modelio sudėtingumą.
Vienas konkretus pasirinkimas modeliui išlaikyti paprastą yra $\ell_2$ bauda. Tai veda prie svorių nykimo minirinkinio stochastinio gradientinio nusileidimo algoritmo atnaujinimo žingsniuose.
Praktikoje svorių nykimo funkcionalumas pateikiamas giliojo mokymosi karkasų optimizatoriuose.
Skirtingos parametrų aibės tame pačiame apmokymo cikle gali turėti skirtingą atnaujinimo elgesį.



## Pratimai

1. Eksperimentuokite su $\lambda$ reikšme šiame skyrelyje nagrinėjamame vertinimo uždavinyje. Nubraižykite apmokymo ir validavimo tikslumą kaip $\lambda$ funkciją. Ką pastebite?
1. Naudodami validavimo aibę raskite optimalią $\lambda$ reikšmę. Ar tikrai tai optimali reikšmė? Ar tai svarbu?
1. Kaip atrodytų atnaujinimo lygtys, jei vietoj $\|\mathbf{w}\|^2$ kaip baudą pasirinktume $\sum_i |w_i|$ ($\ell_1$ reguliarizavimas)?
1. Žinome, kad $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Ar galite rasti panašią lygtį matricoms (žr. Frobenijaus normą :numref:`subsec_lin-algebra-norms`)?
1. Peržiūrėkite ryšį tarp apmokymo paklaidos ir apibendrinimo paklaidos. Be svorių nykimo, didesnio apmokymo ir tinkamo sudėtingumo modelio naudojimo, kokie dar būdai galėtų padėti spręsti persimokymo problemą?
1. Bajeso statistikoje naudojame apriorinės tikimybės ir tikėtinumo sandaugą aposteriorinei gauti per $P(w \mid x) \propto P(x \mid w) P(w)$. Kaip galite susieti $P(w)$ su reguliarizavimu?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17979)
:end_tab:
