# Kalbos modeliai
:label:`sec_language-model`

:numref:`sec_text-sequence` matėme, kaip tekstų sekas atvaizduoti teksto vienetais, kuriuos galima laikyti diskrečių stebėjimų, pavyzdžiui, žodžių ar simbolių, seka. Tarkime, kad teksto sekos ilgio $T$ teksto vienetai yra $x_1, x_2, \ldots, x_T$.
*Kalbos modelių* tikslas
yra įvertinti visos sekos jungtinę tikimybę:

$$P(x_1, x_2, \ldots, x_T),$$

kur galima taikyti statistinius įrankius
iš :numref:`sec_sequence`.

Kalbos modeliai yra nepaprastai naudingi. Pavyzdžiui, idealus kalbos modelis turėtų pats generuoti natūralų tekstą, tiesiog po vieną imdamas teksto vienetą $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Visiškai kitaip nei beždžionė, naudojanti rašomąją mašinėlę, visas tokio modelio sugeneruotas tekstas atrodytų kaip natūrali kalba, pvz., anglų kalbos tekstas. Be to, to pakaktų prasmingam dialogui generuoti, tiesiog sąlygojant tekstą pagal ankstesnius dialogo fragmentus.
Akivaizdu, kad vis dar esame labai toli nuo tokios sistemos sukūrimo, nes jai reikėtų *suprasti* tekstą, o ne tik generuoti gramatiškai prasmingą turinį.

Nepaisant to, kalbos modeliai labai pravarto net ir ribota forma.
Pavyzdžiui, frazės „to recognize speech“ ir „to wreck a nice beach“ skamba labai panašiai.
Tai gali sukelti dviprasmiškumą kalbos atpažinime,
kuris lengvai išsprendžiamas per kalbos modelį, atmetantį antrą variantą kaip nepagrįstą.
Lygiai taip pat dokumentų santraukų algoritme verta žinoti, kad „dog bites man“ pasitaiko daug dažniau nei „man bites dog“, o „I want to eat grandma“ yra gana šokiruojantis teiginys, o „I want to eat, grandma“ – daug nekaltesnis.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Kalbos modelių mokymasis

Akivaizdus klausimas, kaip turėtume modeliuoti dokumentą ar net teksto vienetų seką.
Tarkime, kad tekstinius duomenis skaidome žodžių lygiu.
Pradėkime nuo pagrindinių tikimybių taisyklių taikymo:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Pavyzdžiui,
teksto sekos, kurioje yra keturi žodžiai, tikimybė būtų tokia:

$$\begin{aligned}&P(\textrm{deep}, \textrm{learning}, \textrm{is}, \textrm{fun}) \\
=&P(\textrm{deep}) P(\textrm{learning}  \mid  \textrm{deep}) P(\textrm{is}  \mid  \textrm{deep}, \textrm{learning}) P(\textrm{fun}  \mid  \textrm{deep}, \textrm{learning}, \textrm{is}).\end{aligned}$$

### Markovo modeliai ir $n$-gramos
:label:`subsec_markov-models-and-n-grams`

Iš :numref:`sec_sequence` sekų modelių analizių
pritaikykime Markovo modelius kalbos modeliavimui.
Sekų skirstinys tenkina pirmosios eilės Markovo savybę, jei $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. Aukštesnės eilės atitinka ilgesnes priklausomybes. Tai veda prie keleto aproksimacijų, kurias galėtume taikyti sekai modeliuoti:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Tikimybių formulės, apimančios vieną, du ir tris kintamuosius, paprastai vadinamos atitinkamai *unigramų*, *bigramų* ir *trigramų* modeliais.
Kad apskaičiuotume kalbos modelį, turime apskaičiuoti
žodžių tikimybę ir žodžio sąlyginę tikimybę,
žinant kelis ankstesnius žodžius.
Atkreipkite dėmesį, kad
tokios tikimybės yra
kalbos modelio parametrai.



### Žodžių dažnis

Čia
darome prielaidą, kad mokymo duomenų rinkinys yra didelis teksto tekstynas, pavyzdžiui, visi
Vikipedijos įrašai, [Gutenbergo projektas](https://en.wikipedia.org/wiki/Project_Gutenberg)
ir visas internete paskelbtas tekstas.
Žodžių tikimybę galima apskaičiuoti pagal santykinį
duoto žodžio dažnį mokymo duomenų rinkinyje.
Pavyzdžiui, įvertį $\hat{P}(\textrm{deep})$ galima apskaičiuoti kaip
bet kurio sakinio, prasidedančio žodžiu „deep“, tikimybę. Šiek tiek
mažiau tikslus būdas būtų suskaičiuoti visus žodžio
„deep“ pasikartojimus ir padalyti iš bendro žodžių skaičiaus
tekstyne.
Tai veikia gana gerai, ypač dažniems
žodžiams. Eidami toliau, galėtume bandyti įvertinti

$$\hat{P}(\textrm{learning} \mid \textrm{deep}) = \frac{n(\textrm{deep, learning})}{n(\textrm{deep})},$$

kur $n(x)$ ir $n(x, x')$ yra pavienių žodžių ir
iš eilės einančių žodžių porų pasikartojimų skaičius.
Deja,
žodžių poros tikimybę
įvertinti šiek tiek sunkiau, nes
„deep learning“ pasikartojimų yra daug mažiau.
Visų pirma kai kuriems neįprastiems žodžių deriniams gali būti sunku
rasti pakankamai pasikartojimų tiksliems įverčiams gauti.
Kaip rodo empiriniai rezultatai :numref:`subsec_natural-lang-stat`,
situacija pablogėja trijų žodžių ir didesniems deriniams.
Bus daug tikėtinų trijų žodžių derinių, kurių greičiausiai nematysime savo duomenų rinkinyje.
Jei nepateiksime kokio nors sprendimo tokiems žodžių deriniams priskirti nenulinį skaičių, negalėsime jų naudoti kalbos modelyje. Jei duomenų rinkinys mažas arba žodžiai labai reti, galime nerasti nė vieno jų.

### Laplaso glotninimas

Įprasta strategija – atlikti tam tikrą *Laplaso glotninimo* formą.
Sprendimas yra
pridėti nedidelę konstantą prie visų skaičių.
Pažymėkime $n$ bendrą žodžių skaičių
mokymo rinkinyje,
o $m$ – unikalių žodžių skaičių.
Šis sprendimas padeda su pavieniais žodžiais, pvz., per

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Čia $\epsilon_1,\epsilon_2$ ir $\epsilon_3$ yra hiperparametrai.
Paimkime $\epsilon_1$ kaip pavyzdį:
kai $\epsilon_1 = 0$, glotninimas netaikomas;
kai $\epsilon_1$ artėja prie teigiamos begalybės,
$\hat{P}(x)$ artėja prie tolygiosios tikimybės $1/m$.
Tai gana primityvus variantas to, ką gali
pasiekti kitos technikos :cite:`Wood.Gasthaus.Archambeau.ea.2011`.


Deja, tokie modeliai gana greitai tampa nepatogūs
dėl šių priežasčių.
Pirma,
kaip aptarta :numref:`subsec_natural-lang-stat`,
daug $n$-gramų pasitaiko labai retai,
todėl Laplaso glotninimas gana netinka kalbos modeliavimui.
Antra, turime saugoti visus skaičius.
Trečia, tai visiškai ignoruoja žodžių prasmę. Pavyzdžiui,
„cat“ ir „feline“ turėtų pasitaikyti susijusiuose kontekstuose.
Tokius modelius labai sunku pritaikyti papildomiems kontekstams,
o giliuoju mokymusi pagrįsti kalbos modeliai puikiai tinka
tai įvertinti.
Galiausiai ilgos žodžių
sekos beveik tikrai yra naujos, todėl modelis, kuris tiesiog
skaičiuoja anksčiau matytų žodžių sekų dažnį, tikrai veiks prastai.
Todėl likusioje skyriaus dalyje sutelkiame dėmesį į neuroninių tinklų naudojimą kalbos modeliavimui.


## Painumas
:label:`subsec_perplexity`

Toliau aptarkime, kaip matuoti kalbos modelio kokybę, kurią vėliau naudosime savo modeliams vertinti kituose skyreliuose.
Vienas būdas – patikrinti, kiek tekstas stebina.
Geras kalbos modelis gali labai tiksliai prognozuoti teksto vienetus, kurie ateina toliau.
Panagrinėkite šiuos frazės „It is raining“ tęsinius, pasiūlytus skirtingų kalbos modelių:

1. „It is raining outside“
1. „It is raining banana tree“
1. „It is raining piouw;kcj pwepoiut“

Kokybės požiūriu 1 pavyzdys akivaizdžiai geriausias. Žodžiai yra prasmingi ir logiškai nuoseklūs.
Nors jis gali ir nelabai tiksliai atspindėti, kuris žodis semantiškai seka („in San Francisco“ ir „in winter“ būtų buvę visiškai pagrįsti tęsiniai), modelis geba užfiksuoti, kokio tipo žodis seka.
2 pavyzdys yra gerokai blogesnis, nes sukuria beprasmį tęsinį. Vis dėlto bent jau modelis išmoko rašyti žodžius ir tam tikro laipsnio koreliaciją tarp žodžių. Galiausiai 3 pavyzdys rodo prastai apmokytą modelį, kuris tinkamai neatitinka duomenų.

Modelio kokybę galėtume matuoti skaičiuodami sekos tikėtinumą.
Deja, tai skaičius, kurį sunku suprasti ir sunku palyginti.
Juk trumpesnės sekos pasitaiko kur kas tikėtiniau nei ilgesnės,
todėl modelio vertinimas pagal Tolstojaus šedevrą
*Karas ir taika* neišvengiamai duos daug mažesnį tikėtinumą nei, pavyzdžiui, pagal Saint-Exupéry apysaką *Mažasis princas*. Trūksta vidurkio atitikmens.

Čia praverčia informacijos teorija.
Entropiją, nuostabą ir kryžminę entropiją apibrėžėme,
kai pristatėme softmax regresiją
(:numref:`subsec_info_theory_basics`).
Jei norime suglaudinti tekstą, galime klausti apie
kito teksto vieneto prognozavimą, turint dabartinį teksto vienetų rinkinį.
Geresnis kalbos modelis turėtų leisti tiksliau prognozuoti kitą teksto vienetą.
Taigi jis turėtų leisti sunaudoti mažiau bitų sekai suglaudinti.
Todėl galime jį matuoti kryžminės entropijos nuostoliu, vidurkintu
per visus sekos $n$ teksto vienetų:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

kur $P$ pateikia kalbos modelis, o $x_t$ yra faktinis teksto vienetas, stebėtas sekoje laiko žingsniu $t$.
Tai daro skirtingo ilgio dokumentų našumą palyginamą. Dėl istorinių priežasčių natūraliosios kalbos apdorojimo mokslininkai mieliau naudoja dydį, vadinamą *painumu* (angl. *perplexity*). Trumpai tariant, tai :eqref:`eq_avg_ce_for_lm` eksponentė:

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

Painumą geriausia suprasti kaip realių pasirinkimų, kuriuos turime nuspręsdami, kurį teksto vienetą rinktis toliau, skaičiaus geometrinio vidurkio atvirkštinį dydį. Pažvelkime į keletą atvejų:

* Geriausiu atveju modelis visada tobulai įvertina tikslinio teksto vieneto tikimybę kaip 1. Šiuo atveju modelio painumas yra 1.
* Blogiausiu atveju modelis visada prognozuoja tikslinio teksto vieneto tikimybę kaip 0. Šioje situacijoje painumas yra teigiama begalybė.
* Bazinėje linijoje modelis prognozuoja tolygųjį skirstinį per visus turimus žodyno teksto vienetus. Šiuo atveju painumas lygus unikalių žodyno teksto vienetų skaičiui. Iš tikrųjų, jei saugotume seką be jokio glaudinimo, tai būtų geriausia, ką galėtume padaryti ją koduodami. Todėl tai suteikia netrivialią viršutinę ribą, kurią turi įveikti bet kuris naudingas modelis.

## Sekų skaidymas
:label:`subsec_partitioning-seqs`

Kalbos modelius projektuosime naudodami neuroninius tinklus
ir naudosime painumą, kad įvertintume,
kaip gerai modelis
prognozuoja kitą teksto vienetą, turint dabartinį teksto vienetų rinkinį
teksto sekose.
Prieš pristatydami modelį,
tarkime, kad jis
vienu metu apdoroja minirinkinį iš iš anksto nustatyto ilgio sekų.
Dabar klausimas, kaip [**atsitiktinai skaityti įvesties sekų ir tikslinių sekų minirinkinius**].


Tarkime, kad duomenų rinkinys yra $T$ teksto vienetų indeksų seka `corpus`.
Suskaidysime ją
į posekius, kur kiekvienas posekis turi $n$ teksto vienetų (laiko žingsnių).
Kad kiekvienai epochai iteruotume per
(beveik) visus viso duomenų rinkinio teksto vienetus
ir gautume visus galimus ilgio $n$ posekius,
galime įvesti atsitiktinumą.
Konkrečiau,
kiekvienos epochos pradžioje
atmetame pirmuosius $d$ teksto vienetų,
kur $d\in [0,n)$ atsitiktinai imamas tolygiai.
Likusi sekos dalis
tada skaidoma
į $m=\lfloor (T-d)/n \rfloor$ posekių.
Pažymėkime $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$ ilgio $n$ posekį, prasidedantį nuo teksto vieneto $x_t$ laiko žingsniu $t$.
Gauti $m$ suskaidytų posekių yra
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}.$
Kiekvienas posekis bus naudojamas kaip įvesties seka kalbos modeliui.


Kalbos modeliavimui
tikslas yra prognozuoti kitą teksto vienetą pagal iki šiol matytus teksto vienetus; todėl tikslai (žymenys) yra originali seka, pastumta vienu teksto vienetu.
Bet kurios įvesties sekos $\mathbf x_t$ tikslinė seka
yra $\mathbf x_{t+1}$, kurios ilgis $n$.

![Penkių įvesties sekų ir tikslinių sekų porų gavimas iš suskaidytų ilgio 5 posekių.](../img/lang-model-data.svg)
:label:`fig_lang_model_data`

:numref:`fig_lang_model_data` parodytas penkių įvesties sekų ir tikslinių sekų porų gavimo pavyzdys, kai $n=5$ ir $d=2$.

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = d2l.tensor([corpus[i:i+num_steps+1] 
                        for i in range(len(corpus)-num_steps)])
    self.X, self.Y = array[:,:-1], array[:,1:]
```

Kalbos modeliams apmokyti
atsitiktinai imsime
įvesties sekų ir tikslinių sekų poras
minirinkiniuose.
Šis duomenų įkėliklis kiekvieną kartą atsitiktinai sugeneruoja minirinkinį iš duomenų rinkinio.
Argumentas `batch_size` nurodo posekių pavyzdžių skaičių kiekviename minirinkinyje,
o `num_steps` yra posekio ilgis teksto vienetais.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

Kaip matome toliau,
tikslinių sekų minirinkinį
galima gauti
pastūmus įvesties sekas
vienu teksto vienetu.

```{.python .input  n=7}
%%tab all
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
```

## Santrauka ir diskusija

Kalbos modeliai įvertina teksto sekos jungtinę tikimybę. Ilgoms sekoms $n$-gramos yra patogus modelis, nes sutrumpina priklausomybę. Tačiau yra daug struktūros, bet nepakanka dažnio, kad efektyviai susidorotume su retais žodžių deriniais per Laplaso glotninimą. Todėl vėlesniuose skyreliuose sutelksime dėmesį į neuroninį kalbos modeliavimą.
Kalbos modeliams apmokyti galime atsitiktinai imti įvesties sekų ir tikslinių sekų poras minirinkiniuose. Po apmokymo naudosime painumą kalbos modelio kokybei matuoti.

Kalbos modelius galima didinti didinant duomenų dydį, modelio dydį ir apmokymo skaičiavimų kiekį. Dideli kalbos modeliai gali atlikti norimas užduotis prognozuodami išvesties tekstą pagal įvesties teksto nurodymus. Kaip aptarsime vėliau (pvz., :numref:`sec_large-pretraining-transformers`),
šiuo metu
dideli kalbos modeliai sudaro pažangiausių įvairių užduočių sistemų pagrindą.


## Pratimai

1. Tarkime, mokymo duomenų rinkinyje yra 100 000 žodžių. Kiek žodžių dažnių ir kelių gretimų žodžių dažnių turi saugoti keturgramė?
1. Kaip modeliuotumėte dialogą?
1. Kokių kitų ilgų sekų duomenų skaitymo metodų galite sugalvoti?
1. Apsvarstykite mūsų metodą atmesti tolygiai atsitiktinį skaičių pirmųjų teksto vienetų kiekvienos epochos pradžioje.
    1. Ar tai tikrai lemia visiškai tolygų sekų skirstinį dokumente?
    1. Ką reikėtų padaryti, kad viskas būtų dar tolygiau?
1. Jei norime, kad sekos pavyzdys būtų pilnas sakinys, kokią problemą tai sukelia minirinkinių atrankoje? Kaip ją galime ištaisyti?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18012)
:end_tab:
