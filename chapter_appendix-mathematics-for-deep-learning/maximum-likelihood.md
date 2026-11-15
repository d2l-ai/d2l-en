# Didžiausias tikėtinumas
:label:`sec_maximum_likelihood`

Vienas dažniausiai mašininiame mokyme sutinkamų mąstymo būdų yra didžiausio tikėtinumo požiūris.  Tai samprata, kad dirbant su tikimybiniu modeliu su nežinomais parametrais, tikriausi yra tie parametrai, kurie duomenims suteikia didžiausią tikimybę.

## Didžiausio tikėtinumo principas

Tai turi Bajeso interpretaciją, apie kurią naudinga pagalvoti.  Tarkime, turime modelį su parametrais $\boldsymbol{\theta}$ ir duomenų pavyzdžių rinkinį $X$.  Konkretumo dėlei galime įsivaizduoti, kad $\boldsymbol{\theta}$ yra viena reikšmė, nusakanti tikimybę, kad metant monetą iškris herbas, o $X$ yra nepriklausomų monetos metimų seka.  Šį pavyzdį išsamiai nagrinėsime vėliau.

Jei norime rasti tikriausią savo modelio parametrų reikšmę, tai reiškia, kad norime rasti

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

Pagal Bajeso taisyklę tai yra tas pats, kas

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

Išraiška $P(X)$ – nuo parametrų nepriklausanti duomenų sugeneravimo tikimybė – visai nepriklauso nuo $\boldsymbol{\theta}$, todėl ją galima praleisti nekeičiant geriausio $\boldsymbol{\theta}$ pasirinkimo.  Panašiai galime daryti prielaidą, kad neturime jokios apriorinės nuostatos, kuri parametrų aibė yra geresnė už kitas, todėl galime teigti, kad $P(\boldsymbol{\theta})$ taip pat nepriklauso nuo teta!  Tai, pavyzdžiui, prasminga mūsų monetos metimo pavyzdyje, kur tikimybė, kad iškris herbas, gali būti bet kuri reikšmė iš $[0,1]$ be jokio išankstinio įsitikinimo, ar moneta sąžininga, ar ne (dažnai vadinama *neinformatyviuoju apriorinių skirstiniu*).  Taigi matome, kad pritaikę Bajeso taisyklę gauname, kad geriausias $\boldsymbol{\theta}$ pasirinkimas yra $\boldsymbol{\theta}$ didžiausio tikėtinumo įvertinys:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Pagal įprastą terminologiją duomenų tikimybė, kai duoti parametrai ($P(X \mid \boldsymbol{\theta})$), vadinama *tikėtinumu* (angl. *likelihood*).

### Konkretus pavyzdys

Pažiūrėkime, kaip tai veikia konkrečiame pavyzdyje.  Tarkime, turime vieną parametrą $\theta$, nusakantį tikimybę, kad metant monetą iškris herbas.  Tada tikimybė, kad iškris skaičius, yra $1-\theta$, todėl jei mūsų stebimi duomenys $X$ yra seka su $n_H$ herbų ir $n_T$ skaičių, galime pasinaudoti tuo, kad nepriklausomos tikimybės dauginamos, ir gauti, kad

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

Jei metame $13$ monetų ir gauname seką „HHHTHTTHHHHHT“, kurioje $n_H = 9$ ir $n_T = 4$, matome, kad tai yra

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Vienas gražus šio pavyzdžio dalykas tas, kad atsakymą žinome iš anksto.  Iš tiesų, jei žodžiu sakytume: „Metiau 13 monetų ir 9 iškrito herbu, koks geriausias spėjimas apie tikimybę, kad moneta iškris herbu?“, kiekvienas teisingai atspėtų $9/13$.  Didžiausio tikėtinumo metodas suteiks mums būdą šį skaičių gauti iš pirmųjų principų taip, kad jis apibendrintų daug sudėtingesnes situacijas.

Mūsų pavyzdyje $P(X \mid \theta)$ grafikas yra toks:

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

Jo didžiausia reikšmė yra kažkur netoli mūsų tikėtos $9/13 \approx 0{,}7\ldots$.  Norėdami pamatyti, ar tai tiksliai ten, galime pasitelkti matematinę analizę.  Atkreipkite dėmesį, kad maksimume funkcijos gradientas yra lygus nuliui.  Taigi didžiausio tikėtinumo įvertinį :eqref:`eq_max_like` galime rasti suradę $\theta$ reikšmes, kuriose išvestinė lygi nuliui, ir išsirinkę tą, kuri duoda didžiausią tikimybę.  Skaičiuojame:

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Tai turi tris sprendinius: $0$, $1$ ir $9/13$.  Pirmieji du akivaizdžiai yra minimumai, o ne maksimumai, nes mūsų sekai suteikia tikimybę $0$.  Paskutinė reikšmė *nesuteikia* mūsų sekai nulinės tikimybės, todėl turi būti didžiausio tikėtinumo įvertinys $\hat \theta = 9/13$.

## Skaitinis optimizavimas ir neigiamas logaritminis tikėtinumas

Ankstesnis pavyzdys gražus, bet kas, jei turime milijardus parametrų ir duomenų pavyzdžių?

Pirmiausia pastebėkime, kad jei darome prielaidą, jog visi duomenų pavyzdžiai yra nepriklausomi, paties tikėtinumo praktiškai nebegalime nagrinėti, nes jis yra daugelio tikimybių sandauga.  Iš tiesų kiekviena tikimybė yra iš $[0,1]$, tarkime, paprastai apie $1/2$, o sandauga $(1/2)^{1000000000}$ yra gerokai žemiau mašininio tikslumo.  Tiesiogiai su tuo dirbti negalime.  

Tačiau prisiminkite, kad logaritmas sandaugas paverčia sumomis, tada

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

Šis skaičius puikiai telpa net į vieno tikslumo $32$ bitų slankiojo kablelio skaičių.  Taigi turėtume nagrinėti *logaritminį tikėtinumą*, kuris yra

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

Kadangi funkcija $x \mapsto \log(x)$ yra didėjanti, tikėtinumo maksimizavimas yra tas pats, kas logaritminio tikėtinumo maksimizavimas.  Iš tiesų :numref:`sec_naive_bayes` pamatysime šį samprotavimą taikomą dirbant su konkrečiu naiviojo Bajeso klasifikatoriaus pavyzdžiu.

Dažnai dirbame su nuostolių funkcijomis, kurias norime minimizuoti.  Didžiausią tikėtinumą galime paversti nuostolių minimizavimu paėmę $-\log(P(X \mid \boldsymbol{\theta}))$, tai yra *neigiamą logaritminį tikėtinumą*.

Tai iliustruodami panagrinėkime ankstesnį monetos metimo uždavinį ir apsimeskime, kad nežinome uždaros formos sprendinio.  Galime apskaičiuoti, kad

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Tai galima užrašyti kodu ir laisvai optimizuoti net milijardams monetos metimų.

```{.python .input}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

Skaitinis patogumas nėra vienintelė priežastis, kodėl žmonės mėgsta naudoti neigiamą logaritminį tikėtinumą. Yra dar keletas priežasčių, kodėl jis yra labiau tinkamas.



Antra priežastis, kodėl nagrinėjame logaritminį tikėtinumą, – supaprastintas matematinės analizės taisyklių taikymas. Kaip aptarta aukščiau, dėl nepriklausomumo prielaidų dauguma mašininiame mokyme sutinkamų tikimybių yra pavienių tikimybių sandaugos.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Tai reiškia, kad jei tiesiogiai taikytume sandaugos taisyklę išvestinei apskaičiuoti, gautume

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Tam reikia $n(n-1)$ daugybų ir $(n-1)$ sudėčių, todėl laikas yra proporcingas įvesties kvadratui!  Pakankamai išradingai sugrupavus narius tai galima sumažinti iki tiesinio laiko, bet tam reikia pamąstyti.  Neigiamam logaritminiam tikėtinumui turime kitaip

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

kas duoda

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Tam reikia tik $n$ dalybų ir $n-1$ sudėčių, todėl laikas yra tiesinis įvesties atžvilgiu.

Trečia ir paskutinė priežastis nagrinėti neigiamą logaritminį tikėtinumą yra ryšys su informacijos teorija, kurią išsamiai aptarsime :numref:`sec_information_theory`.  Tai griežta matematinė teorija, suteikianti būdą matuoti informacijos kiekį arba atsitiktinumą atsitiktiniame dydyje.  Pagrindinis tos srities tyrimo objektas yra entropija, kuri yra

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

ir matuoja šaltinio atsitiktinumą. Atkreipkite dėmesį, kad tai yra ne kas kita, kaip vidutinė $-\log$ tikimybė, todėl jei imame savo neigiamą logaritminį tikėtinumą ir padalijame iš duomenų pavyzdžių skaičiaus, gauname entropijos giminaitį, žinomą kaip kryžminė entropija.  Vien šios teorinės interpretacijos pakaktų, kad būtų verta pateikti vidutinį neigiamą logaritminį tikėtinumą duomenų rinkinyje kaip modelio našumo matą.

## Didžiausias tikėtinumas tolydiesiems kintamiesiems

Viskas, ką iki šiol darėme, darė prielaidą, kad dirbame su diskrečiaisiais atsitiktiniais dydžiais, bet kas, jei norime dirbti su tolydžiaisiais?

Trumpai tariant, nieko nesikeičia, tik visas tikimybes pakeičiame tikimybės tankiu.  Prisimindami, kad tankius žymime mažąja raide $p$, tai reiškia, kad, pavyzdžiui, dabar sakome

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

Kyla klausimas: „Kodėl tai leistina?“  Juk tankius įvedėme todėl, kad tikimybės gauti konkrečius rezultatus buvo lygios nuliui, tad ar duomenų sugeneravimo tikimybė, esant bet kokiems parametrams, nėra lygi nuliui?

Iš tiesų taip ir yra, o suprasti, kodėl galime pereiti prie tankių, reiškia išsiaiškinti, kas nutinka su epsilonais.

Pirmiausia iš naujo apibrėžkime savo tikslą.  Tarkime, kad tolydiesiems atsitiktiniams dydžiams nebenorime skaičiuoti tikimybės gauti tiksliai tinkamą reikšmę, o tik atitikti iki kokio nors intervalo $\epsilon$.  Supaprastinimo dėlei darome prielaidą, kad mūsų duomenys yra pakartotiniai vienodai pasiskirsčiusių atsitiktinių dydžių $X_1, \ldots, X_N$ stebiniai $x_1, \ldots, x_N$.  Kaip matėme anksčiau, tai galima užrašyti kaip

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Taigi, jei iš to imsime neigiamus logaritmus, gausime

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Jei nagrinėsime šią išraišką, vienintelė vieta, kur pasirodo $\epsilon$, yra adityvioji konstanta $-N\log(\epsilon)$.  Ji visai nepriklauso nuo parametrų $\boldsymbol{\theta}$, todėl optimalus $\boldsymbol{\theta}$ pasirinkimas nepriklauso nuo mūsų $\epsilon$ pasirinkimo!  Ar reikalautume keturių skaitmenų, ar keturių šimtų, geriausias $\boldsymbol{\theta}$ pasirinkimas išlieka tas pats, todėl epsilon galime laisvai atmesti ir matyti, kad optimizuoti norime

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Taigi matome, kad didžiausio tikėtinumo požiūris gali veikti su tolydžiaisiais atsitiktiniais dydžiais taip pat lengvai kaip su diskrečiaisiais, tikimybes pakeitus tikimybių tankiais.

## Santrauka
* Didžiausio tikėtinumo principas teigia, kad geriausiai duotą duomenų rinkinį atitinkantis modelis yra tas, kuris duomenis sugeneruoja su didžiausia tikimybe.
* Dažnai žmonės vietoj to dirba su neigiamu logaritminiu tikėtinumu dėl įvairių priežasčių: skaitinio stabilumo, sandaugų pavertimo sumomis (ir dėl to supaprastėjančių gradientų skaičiavimų) bei teorinių ryšių su informacijos teorija.
* Nors paprasčiausia tai pagrįsti diskrečiuoju atveju, tai galima laisvai apibendrinti ir tolydžiajam atvejui maksimizuojant duomenų taškams priskirtą tikimybės tankį.

## Pratimai
1. Tarkime, žinote, kad neneigiamas atsitiktinis dydis turi tankį $\alpha e^{-\alpha x}$ kažkokiai reikšmei $\alpha>0$.  Gaunate vieną atsitiktinio dydžio stebinį – skaičių $3$.  Koks yra $\alpha$ didžiausio tikėtinumo įvertinys?
2. Tarkime, turite duomenų rinkinį imčių $\{x_i\}_{i=1}^N$, ištrauktų iš Gauso skirstinio su nežinomu vidurkiu, bet dispersija $1$.  Koks yra vidurkio didžiausio tikėtinumo įvertinys?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1097)
:end_tab:
