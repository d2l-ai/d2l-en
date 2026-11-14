```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Tikimybės ir statistika
:label:`sec_prob`

Vienaip ar kitaip,
mašininis mokymasis yra visas apie neapibrėžtumą.
Prižiūrimame mokyme norime prognozuoti
kažką nežinomo (*tikslinį dydį*),
kai žinoma kažkas kita (*požymiai*).
Priklausomai nuo tikslo,
galime bandyti prognozuoti
labiausiai tikėtiną tikslinio dydžio reikšmę.
Arba galime prognozuoti reikšmę su mažiausiu
tikėtinu atstumu nuo tikslinio dydžio.
O kartais norime ne tik
prognozuoti konkrečią reikšmę,
bet ir *kiekybiškai įvertinti savo neapibrėžtumą*.
Pavyzdžiui, kai duoti kai kurie požymiai,
apibūdinantys pacientą,
galime norėti žinoti, *kiek tikėtina*, kad jis
patirs infarktą per ateinančius metus.
Neprižiūrimame mokyme
neapibrėžtumas dažnai mums rūpi.
Norint nustatyti, ar matavimų rinkinys yra anomalus,
naudinga žinoti, kiek tikėtina
stebėti reikšmes dominančioje populiacijoje.
Be to, skatinamajame mokyme
norime sukurti agentus,
kurie veiktų išmaniai įvairiose aplinkose.
Tam reikia samprotauti apie tai,
kaip gali keistis aplinka
ir kokių atlygių galima tikėtis
reaguojant į kiekvieną iš turimų veiksmų.

*Tikimybių teorija* yra matematikos šaka,
susijusi su samprotavimu esant neapibrėžtumui.
Turėdami tikimybinį kokio nors proceso modelį,
galime samprotauti apie įvairių įvykių tikėtinumą.
Tikimybių naudojimas
pasikartojančių įvykių (pvz., monetos metimų)
dažniams aprašyti
yra gana nekontroversiškas.
Tiesą sakant, *dažninės* (frekventistinės) mokyklos mokslininkai laikosi
tokios tikimybės interpretacijos,
kuri taikoma *tik* tokiems pasikartojantiems įvykiams.
Priešingai, *Bajeso* mokyklos mokslininkai
tikimybių kalbą vartoja plačiau,
kad formalizuotų samprotavimą esant neapibrėžtumui.
Bajeso tikimybę apibūdina
dvi unikalios savybės:
(i) įsitikinimo laipsnių priskyrimas
nepasikartojantiems įvykiams,
pvz., kokia yra *tikimybė*,
kad užtvanka sugrius?;
ir (ii) subjektyvumas. Nors Bajeso
tikimybė pateikia nedviprasmiškas taisykles,
kaip reikėtų atnaujinti savo įsitikinimus
atsižvelgiant į naujus įrodymus,
ji leidžia skirtingiems asmenims
pradėti nuo skirtingų *apriorinių* įsitikinimų.
*Statistika* padeda mums samprotauti atgaline kryptimi,
pradedant duomenų rinkimu ir organizavimu
ir einant atgal prie išvadų,
kurias galime padaryti apie procesą,
sukūrusį duomenis.
Kai tik analizuojame duomenų rinkinį, ieškodami dėsningumų,
kurie, tikimės, galėtų apibūdinti platesnę populiaciją,
taikome statistinį mąstymą.
Daugybė kursų, specialybių, disertacijų, karjerų, katedrų,
bendrovių ir institucijų buvo skirta
tikimybių teorijos ir statistikos studijoms.
Nors šis skyrelis tik paliečia paviršių,
suteiksime pagrindą,
kurio jums reikia pradėti kurti modelius.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## Paprastas pavyzdys: monetos metimas

Įsivaizduokite, kad planuojame mesti monetą
ir norime kiekybiškai įvertinti, kiek tikėtina,
kad iškris herbas (o ne skaičius).
Jei moneta yra *teisinga*,
abi baigtys
(herbas ir skaičius)
yra vienodai tikėtinos.
Be to, jei planuojame mesti monetą $n$ kartų,
herbo iškritimų dalis,
kurios *tikimės*,
turėtų tiksliai sutapti
su *tikėtina* skaičiaus iškritimų dalimi.
Vienas intuityvus būdas tai pamatyti
yra simetrija:
kiekvienai galimai baigčiai
su $n_\textrm{h}$ herbų ir $n_\textrm{t} = (n - n_\textrm{h})$ skaičių
yra vienodai tikėtina baigtis
su $n_\textrm{t}$ herbų ir $n_\textrm{h}$ skaičių.
Atkreipkite dėmesį, kad tai įmanoma tik tada,
jei vidutiniškai tikimės, kad
$1/2$ metimų baigsis herbu,
o $1/2$ – skaičiumi.
Žinoma, jei atliksite šį eksperimentą
daug kartų po $n=1000000$ metimų,
galbūt niekada nepamatysite bandymo,
kuriame $n_\textrm{h} = n_\textrm{t}$ tiksliai.


Formaliai dydis $1/2$ vadinamas *tikimybe*,
ir čia jis atspindi tikrumą,
su kuriuo bet kuris metimas
baigsis herbu.
Tikimybės priskiria įvertį tarp $0$ ir $1$
dominančioms baigtims, vadinamoms *įvykiais*.
Čia dominantis įvykis yra $\textrm{herbas}$,
ir atitinkamą tikimybę žymime $P(\textrm{herbas})$.
Tikimybė $1$ rodo absoliutų tikrumą
(įsivaizduokite triukšmingą monetą, kurios abi pusės yra herbai),
o tikimybė $0$ rodo negalimumą
(pvz., jei abi pusės būtų skaičiai).
Dažniai $n_\textrm{h}/n$ ir $n_\textrm{t}/n$ nėra tikimybės,
o veikiau *statistikos*.
Tikimybės yra *teoriniai* dydžiai,
slypintys duomenis generuojančio proceso pagrinde.
Čia tikimybė $1/2$
yra pačios monetos savybė.
Priešingai, statistikos yra *empiriniai* dydžiai,
apskaičiuojami kaip stebimų duomenų funkcijos.
Mūsų susidomėjimas tikimybiniais ir statistiniais dydžiais
yra neatskiriamai susipynęs.
Dažnai kuriame specialias statistikas, vadinamas *įvertiniais*,
kurie, turint duomenų rinkinį, pateikia
modelio parametrų, pvz., tikimybių, *įverčius*.
Be to, kai tie įvertiniai tenkina
gerą savybę, vadinamą *pagrįstumu* (angl. *consistency*),
mūsų įverčiai konverguos
prie atitinkamos tikimybės.
Savo ruožtu šios išvestos tikimybės
nusako tikėtinas statistines
tos pačios populiacijos duomenų savybes,
su kuriais galime susidurti ateityje.

Tarkime, kad užtikome tikrą monetą,
kurios tikros $P(\textrm{herbas})$
nežinome.
Kad ištirtume šį dydį
statistiniais metodais,
turime (i) surinkti kiek duomenų;
ir (ii) sukurti įvertinį.
Duomenų gavimas čia lengvas:
galime mesti monetą daug kartų
ir užrašyti visas baigtis.
Formaliai realizacijų paėmimas
iš kokio nors pagrindinio atsitiktinio proceso
vadinamas *atranka*.
Kaip galbūt nuspėjote,
vienas natūralus įvertinys
yra stebėtų *herbų* skaičiaus
ir bendro metimų skaičiaus santykis.

Dabar tarkime, kad moneta iš tiesų buvo teisinga,
t. y. $P(\textrm{herbas}) = 0.5$.
Teisingos monetos metimams imituoti
galime iškviesti bet kurį atsitiktinių skaičių generatorių.
Yra keletas paprastų būdų atrinkti
įvykio su tikimybe $0.5$ imtis.
Pavyzdžiui, Python `random.random`
pateikia skaičius intervale $[0,1]$,
kur tikimybė patekti
į bet kurį subintervalą $[a, b] \subset [0,1]$
lygi $b-a$.
Taigi galime gauti `0` ir `1`, kiekvieną su tikimybe `0.5`,
tikrindami, ar grąžintas slankiojo kablelio skaičius didesnis nei `0.5`:

```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

Bendriau, galime imituoti kelis ėmimus
iš bet kurio kintamojo su baigtiniu skaičiumi
galimų baigčių
(pvz., monetos metimas ar kauliuko metimas)
iškviesdami daugianarę funkciją (angl. *multinomial*),
nustatydami pirmąjį argumentą
kaip ėmimų skaičių,
o antrąjį – kaip tikimybių sąrašą,
susietą su kiekviena galima baigtimi.
Kad imituotume dešimt teisingos monetos metimų,
priskiriame tikimybių vektorių `[0.5, 0.5]`,
interpretuodami indeksą 0 kaip herbą,
o indeksą 1 – kaip skaičių.
Funkcija grąžina vektorių,
kurio ilgis lygus galimų baigčių
skaičiui (čia 2),
kur pirmoji komponentė nurodo
herbų pasirodymų skaičių,
o antroji komponentė – skaičių pasirodymų skaičių.

```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab jax
fair_probs = [0.5, 0.5]
# jax.random does not have multinomial distribution implemented
np.random.multinomial(100, fair_probs)
```

Kiekvieną kartą paleidus šį atrankos procesą,
gausite naują atsitiktinę reikšmę,
kuri gali skirtis nuo ankstesnės baigties.
Padalijus iš metimų skaičiaus,
gauname kiekvienos baigties *dažnį*
mūsų duomenyse.
Atkreipkite dėmesį, kad šie dažniai,
kaip ir tikimybės,
kurias jie turi įvertinti,
sudaro sumą $1$.

```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab jax
np.random.multinomial(100, fair_probs) / 100
```

Čia, nors mūsų imituojama moneta yra teisinga
(patys nustatėme tikimybes `[0.5, 0.5]`),
herbų ir skaičių skaičiai gali nesutapti.
Taip yra todėl, kad atrinkome tik gana nedidelį imčių skaičių.
Jei patys nebūtume įgyvendinę simuliacijos
ir matytume tik rezultatą,
kaip žinotume, ar moneta šiek tiek neteisinga,
ar galimas nuokrypis nuo $1/2$ buvo
tik mažos imties dydžio artefaktas?
Pažiūrėkime, kas nutinka, kai imituojame 10 000 metimų.

```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

Paprastai pasikartojančių įvykių (pvz., monetos metimų) vidurkiams,
didėjant kartojimų skaičiui,
mūsų įverčiai garantuotai konverguoja
prie tikrųjų pagrindinių tikimybių.
Matematinė šio reiškinio formuluotė
vadinama *didžiųjų skaičių dėsniu*,
o *centrinė ribinė teorema*
sako, kad daugeliu atvejų,
didėjant imties dydžiui $n$,
šios paklaidos turėtų mažėti
sparta $(1/\sqrt{n})$.
Sudarykime daugiau intuicijos tyrinėdami,
kaip kinta mūsų įvertis, didinant
metimų skaičių nuo 1 iki 10 000.

```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Kiekviena ištisinė kreivė atitinka vieną iš dviejų monetos reikšmių
ir rodo mūsų įvertintą tikimybę, kad moneta iškris ta reikšme
po kiekvienos eksperimentų grupės.
Punktyrinė juoda linija rodo tikrąją pagrindinę tikimybę.
Gaunant daugiau duomenų atliekant daugiau eksperimentų,
kreivės konverguoja link tikrosios tikimybės.
Galbūt jau pradedate matyti kai kurių sudėtingesnių klausimų,
kurie užima statistikus, kontūrus:
kaip greitai vyksta ši konvergencija?
Jei jau išbandėme daug monetų,
pagamintų toje pačioje gamykloje,
kaip galėtume įtraukti šią informaciją?

##  Formalesnis nagrinėjimas

Jau nuėjome gana toli: suformulavome
tikimybinį modelį,
sugeneravome sintetinius duomenis,
paleidome statistinį įvertinį,
empiriškai įvertinome konvergenciją
ir pateikėme paklaidos metrikas (patikrinome nuokrypį).
Tačiau norint nueiti daug toliau,
turėsime būti tikslesni.


Kai kalbame apie atsitiktinumą,
galimų baigčių aibę žymime $\mathcal{S}$
ir vadiname ją *baigčių erdve*.
Čia kiekvienas elementas yra skirtinga galima *baigtis*.
Vienos monetos metimo atveju
$\mathcal{S} = \{\textrm{herbas}, \textrm{skaičius}\}$.
Vienam kauliukui $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
Metant dvi monetas, galimos baigtys yra
$\{(\textrm{herbas}, \textrm{herbas}), (\textrm{herbas}, \textrm{skaičius}), (\textrm{skaičius}, \textrm{herbas}),  (\textrm{skaičius}, \textrm{skaičius})\}$.
*Įvykiai* yra baigčių erdvės poaibiai.
Pavyzdžiui, įvykis „pirmasis monetos metimas baigiasi herbu"
atitinka aibę $\{(\textrm{herbas}, \textrm{herbas}), (\textrm{herbas}, \textrm{skaičius})\}$.
Kai atsitiktinio eksperimento baigtis $z$ tenkina
$z \in \mathcal{A}$, įvykis $\mathcal{A}$ įvyko.
Vienam kauliuko metimui galime apibrėžti įvykius
„iškrito $5$" ($\mathcal{A} = \{5\}$)
ir „iškrito nelyginis skaičius"  ($\mathcal{B} = \{1, 3, 5\}$).
Šiuo atveju, jei kauliukas parodė $5$,
sakytume, kad įvyko ir $\mathcal{A}$, ir $\mathcal{B}$.
Kita vertus, jei $z = 3$,
tai $\mathcal{A}$ neįvyko,
bet $\mathcal{B}$ įvyko.


*Tikimybės* funkcija atvaizduoja įvykius
į realiąsias reikšmes ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
Įvykio $\mathcal{A}$ tikimybė, žymima $P(\mathcal{A})$,
duotoje baigčių erdvėje $\mathcal{S}$
turi šias savybes:
* Bet kurio įvykio $\mathcal{A}$ tikimybė yra neneigiamas realusis skaičius, t. y. $P(\mathcal{A}) \geq 0$;
* Visos baigčių erdvės tikimybė yra $1$, t. y. $P(\mathcal{S}) = 1$;
* Bet kuriai suskaičiuojamai sekai įvykių $\mathcal{A}_1, \mathcal{A}_2, \ldots$, kurie yra *nesutaikomi* (t. y. $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ visiems $i \neq j$), tikimybė, kad įvyks bet kuris iš jų, lygi jų individualių tikimybių sumai, t. y. $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Šias tikimybių teorijos aksiomas,
pasiūlytas :citet:`Kolmogorov.1933`,
galima taikyti greitai išvedant daugelį svarbių pasekmių.
Pavyzdžiui, iš karto išplaukia,
kad bet kurio įvykio $\mathcal{A}$
*arba* jo priešingo įvykio $\mathcal{A}'$ tikimybė yra 1
(nes $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Taip pat galime įrodyti, kad $P(\emptyset) = 0$,
nes $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Vadinasi, bet kurio įvykio $\mathcal{A}$
*ir* jo priešingo įvykio $\mathcal{A}'$ vienu metu įvykimo tikimybė
yra $P(\mathcal{A} \cap \mathcal{A}') = 0$.
Neformaliai tai reiškia, kad negalimi įvykiai
įvyksta su nuline tikimybe.



## Atsitiktiniai dydžiai

Kai kalbėjome apie tokius įvykius kaip kauliuko metimas,
kuriame iškrenta nelyginis skaičius, arba pirmasis monetos metimas, baigęsis herbu,
rėmėmės *atsitiktinio dydžio* idėja.
Formaliai atsitiktiniai dydžiai yra atvaizdžiai
iš pagrindinės baigčių erdvės
į (galbūt daugelio) reikšmių aibę.
Galite susimąstyti, kuo atsitiktinis dydis
skiriasi nuo baigčių erdvės,
nes abu yra baigčių rinkiniai.
Svarbu, kad atsitiktiniai dydžiai gali būti daug grubesni
nei pradinė baigčių erdvė.
Galime apibrėžti dvejetainį atsitiktinį dydį, pvz., „didesnis nei 0,5",
net kai pagrindinė baigčių erdvė yra begalinė,
pavyzdžiui, atkarpos tarp $0$ ir $1$ taškai.
Be to, keli atsitiktiniai dydžiai
gali dalytis ta pačia pagrindine baigčių erdve.
Pavyzdžiui, „ar suveikia namų signalizacija"
ir „ar mano namuose buvo apiplėšimas"
yra dvejetainiai atsitiktiniai dydžiai,
turintys bendrą pagrindinę baigčių erdvę.
Todėl žinodami vieno atsitiktinio dydžio reikšmę,
galime sužinoti ką nors apie tikėtiną kito atsitiktinio dydžio reikšmę.
Žinodami, kad signalizacija suveikė,
galime įtarti, kad namai greičiausiai buvo apiplėšti.


Kiekviena atsitiktinio dydžio įgyjama reikšmė atitinka
pagrindinės baigčių erdvės poaibį.
Taigi atvejis, kai atsitiktinis dydis $X$
įgyja reikšmę $v$, žymimas $X=v$, yra *įvykis*,
o $P(X=v)$ žymi jo tikimybę.
Kartais šis žymėjimas tampa nepatogus,
ir kai kontekstas aiškus, galime žymėjimą vartoti laisviau.
Pavyzdžiui, $P(X)$ galime vartoti kalbėdami apskritai
apie $X$ *skirstinį*, t. y.
funkciją, nusakančią tikimybę,
kad $X$ įgyja bet kurią duotą reikšmę.
Kitais atvejais rašome tokias išraiškas
kaip $P(X,Y) = P(X) P(Y)$
kaip sutrumpinimą teiginiui, kuris
teisingas visoms reikšmėms,
kurias gali įgyti atsitiktiniai dydžiai $X$ ir $Y$, t. y.
visiems $i,j$ teisinga, kad $P(X=i \textrm{ ir } Y=j) = P(X=i)P(Y=j)$.
Dar kitais atvejais žymėjimą vartojame laisvai rašydami
$P(v)$, kai atsitiktinis dydis aiškus iš konteksto.
Kadangi tikimybių teorijoje įvykis yra baigčių aibė iš baigčių erdvės,
galime nurodyti reikšmių intervalą, kurį įgyja atsitiktinis dydis.
Pavyzdžiui, $P(1 \leq X \leq 3)$ žymi įvykio $\{1 \leq X \leq 3\}$ tikimybę.


Atkreipkite dėmesį, kad yra subtilus skirtumas
tarp *diskrečiųjų* atsitiktinių dydžių,
pvz., monetos metimų ar kauliuko metimų,
ir *tolydžiųjų*,
pvz., atsitiktinai iš populiacijos
atrinkto žmogaus svorio ir ūgio.
Šiuo atveju retai iš tikrųjų rūpi
kieno nors tikslus ūgis.
Be to, jei matuotume pakankamai tiksliai,
rastume, kad nėra dviejų planetos žmonių,
turinčių visiškai tą patį ūgį.
Tiesą sakant, pakankamai tiksliai matuodami
niekada neturėtumėte to paties ūgio
atsikėlę ir eidami miegoti.
Beveik nėra prasmės klausti apie
tikslią tikimybę, kad kažkas
yra 1,801392782910287192 metro ūgio.
Vietoj to dažniausiai svarbiau galėti pasakyti,
ar kieno nors ūgis patenka į tam tikrą intervalą,
tarkime, tarp 1,79 ir 1,81 metro.
Šiais atvejais dirbame su tikimybių *tankiais*.
Ūgis lygiai 1,80 metro
neturi tikimybės, bet turi nenulinį tankį.
Kad apskaičiuotume intervalui priskirtą tikimybę,
turime paimti tankio *integralą*
per tą intervalą.

## Keli atsitiktiniai dydžiai

Galbūt pastebėjote, kad net ankstesnio skyrelio
neįveikėme nepateikę teiginių apie sąveikas
tarp kelių atsitiktinių dydžių
(prisiminkite, kad $P(X,Y) = P(X) P(Y)$).
Didžioji dalis mašininio mokymosi
susijusi su tokiais ryšiais.
Čia baigčių erdvė būtų
dominanti populiacija,
tarkime, klientai, atliekantys sandorius su įmone,
internete esančios nuotraukos
arba biologams žinomi baltymai.
Kiekvienas atsitiktinis dydis atspindėtų
(nežinomą) skirtingo atributo reikšmę.
Kai tik iš populiacijos atrenkame individą,
stebime kiekvieno atsitiktinio dydžio realizaciją.
Kadangi atsitiktinių dydžių įgyjamos reikšmės
atitinka baigčių erdvės poaibius,
kurie gali dalinai ar visiškai persikloti
arba būti visiškai nesikertantys,
žinodami vieno atsitiktinio dydžio reikšmę
galime atnaujinti savo įsitikinimus apie tai,
kurios kito atsitiktinio dydžio reikšmės yra tikėtinos.
Jei pacientas ateina į ligoninę
ir pastebime, kad jam
sunku kvėpuoti
ir jis prarado uoslę,
manome, kad jam tikėtiniau
yra COVID-19, nei manytume,
jei jis neturėtų kvėpavimo sunkumų
ir jo uoslė būtų visiškai normali.


Dirbdami su keliais atsitiktiniais dydžiais,
galime sudaryti įvykius, atitinkančius
kiekvieną reikšmių derinį,
kurį kintamieji gali įgyti kartu.
Tikimybės funkcija, priskirianti
tikimybes kiekvienam iš šių derinių
(pvz., $A=a$ ir $B=b$),
vadinama *jungtinės tikimybės* funkcija
ir tiesiog grąžina tikimybę, priskirtą
atitinkamų baigčių erdvės poaibių
sankirtai.
*Jungtinė tikimybė*, priskirta įvykiui,
kai atsitiktiniai dydžiai $A$ ir $B$
įgyja reikšmes atitinkamai $a$ ir $b$,
žymima $P(A = a, B = b)$,
kur kablelis reiškia „ir".
Atkreipkite dėmesį, kad bet kurioms reikšmėms $a$ ir $b$
teisinga, kad

$$P(A=a, B=b) \leq P(A=a) \textrm{ and } P(A=a, B=b) \leq P(B = b),$$

nes kad įvyktų $A=a$ ir $B=b$,
turi įvykti $A=a$ *ir* turi įvykti $B=b$.
Įdomu tai, kad jungtinė tikimybė
pasako viską, ką galime žinoti apie šiuos
atsitiktinius dydžius tikimybine prasme,
ir gali būti naudojama daugeliui kitų
naudingų dydžių išvesti, įskaitant individualių
skirstinių $P(A)$ ir $P(B)$ atkūrimą.
Kad atkurtume $P(A=a)$, tiesiog susumuojame
$P(A=a, B=v)$ per visas reikšmes $v$,
kurias gali įgyti atsitiktinis dydis $B$:
$P(A=a) = \sum_v P(A=a, B=v)$.


Santykis $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
pasirodo esąs be galo svarbus.
Jis vadinamas *sąlygine tikimybe*
ir žymimas simboliu „$\mid$":

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

Jis nusako naują tikimybę,
susijusią su įvykiu $B=b$,
kai sąlygojame faktu, kad įvyko $A=a$.
Šią sąlyginę tikimybę galime laikyti
dėmesio apribojimu tik tuo baigčių erdvės poaibiu,
kuris susijęs su $A=a$,
ir tada pernormavimu, kad
visų tikimybių suma būtų 1.
Sąlyginės tikimybės iš tikrųjų yra
tiesiog įprastos tikimybės,
todėl jos tenkina visas aksiomas,
jei visus narius sąlygojame
tuo pačiu įvykiu ir taip
apribojame dėmesį ta pačia baigčių erdve.
Pavyzdžiui, nesikertantiems įvykiams
$\mathcal{B}$ ir $\mathcal{B}'$ turime, kad
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.


Naudodami sąlyginių tikimybių apibrėžimą,
galime išvesti garsųjį rezultatą, vadinamą *Bajeso teorema*.
Pagal konstrukciją turime $P(A, B) = P(B\mid A) P(A)$
ir $P(A, B) = P(A\mid B) P(B)$.
Sujungę abi lygtis gauname
$P(B\mid A) P(A) = P(A\mid B) P(B)$, taigi

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$






Ši paprasta lygtis turi gilių pasekmių, nes
ji leidžia apversti sąlygojimo tvarką.
Jei mokame įvertinti $P(B\mid A)$, $P(A)$ ir $P(B)$,
tada galime įvertinti $P(A\mid B)$.
Dažnai lengviau tiesiogiai įvertinti vieną narį,
bet ne kitą, ir čia į pagalbą gali ateiti Bajeso teorema.
Pavyzdžiui, jei žinome simptomų paplitimą sergant tam tikra liga
ir atitinkamai bendrą ligos bei simptomų paplitimą,
galime nustatyti, kiek tikėtina, kad žmogus
serga ta liga pagal jo simptomus.
Kai kuriais atvejais galime neturėti tiesioginės prieigos prie $P(B)$,
pavyzdžiui, simptomų paplitimo.
Tokiu atveju praverčia supaprastinta Bajeso teoremos versija:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Kadangi žinome, kad $P(A \mid B)$ turi būti normuota iki $1$, t. y. $\sum_a P(A=a \mid B) = 1$,
galime ją naudoti apskaičiuoti

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

Bajeso statistikoje stebėtoją įsivaizduojame
kaip turintį tam tikrus (subjektyvius) apriorinius įsitikinimus
apie turimų hipotezių tikėtinumą,
užkoduotus *apriorinėje* tikimybėje $P(H)$,
ir *tikėtinumo funkciją*, nusakančią, kiek tikėtina
stebėti bet kurią surinktų įrodymų reikšmę
kiekvienai klasės hipotezei $P(E \mid H)$.
Bajeso teorema tuomet interpretuojama kaip nurodanti,
kaip atnaujinti pradinę *apriorinę* tikimybę $P(H)$
atsižvelgiant į turimus įrodymus $E$,
kad gautume *aposteriorinius* įsitikinimus
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$.
Neformaliai tai galima išreikšti taip:
„aposteriorinė lygi apriorinei, padaugintai iš tikėtinumo ir padalytai iš įrodymų".
Kadangi įrodymų tikimybė $P(E)$ vienoda visoms hipotezėms,
pakanka tiesiog normuoti per hipotezes.

Atkreipkite dėmesį, kad $\sum_a P(A=a \mid B) = 1$ taip pat leidžia mums *marginalizuoti* atsitiktinius dydžius. Tai yra, galime išmesti kintamuosius iš jungtinio skirstinio, pvz., $P(A, B)$. Juk turime, kad

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

Nepriklausomumas yra dar viena iš esmės svarbi sąvoka,
sudaranti daugelio svarbių statistikos idėjų
pagrindą.
Trumpai tariant, du kintamieji yra *nepriklausomi*,
jei sąlygojimas $A$ reikšme nepakeičia
su $B$ susieto tikimybių skirstinio
ir atvirkščiai.
Formaliau, nepriklausomumas, žymimas $A \perp B$,
reikalauja, kad $P(A \mid B) = P(A)$, ir todėl
$P(A,B) = P(A \mid B) P(B) = P(A) P(B)$.
Nepriklausomumas dažnai yra tinkama prielaida.
Pavyzdžiui, jei atsitiktinis dydis $A$
atspindi vienos teisingos monetos metimo baigtį,
o atsitiktinis dydis $B$
atspindi kitos monetos metimo baigtį,
tai žinojimas, ar $A$ baigėsi herbu,
neturėtų daryti įtakos tikimybei,
kad $B$ baigsis herbu.


Nepriklausomumas ypač naudingas, kai jis galioja tarp iš eilės
imamų duomenų imčių iš kokio nors pagrindinio skirstinio
(leidžiantis daryti tvirtas statistines išvadas)
arba kai galioja tarp įvairių mūsų duomenų kintamųjų,
leisdamas dirbti su paprastesniais modeliais,
užkoduojančiais šią nepriklausomumo struktūrą.
Kita vertus, priklausomybių tarp atsitiktinių dydžių įvertinimas
dažnai yra pats mokymosi tikslas.
Mums rūpi įvertinti ligos tikimybę pagal simptomus
būtent todėl, kad manome,
jog ligos ir simptomai *nėra* nepriklausomi.


Atkreipkite dėmesį, kad kadangi sąlyginės tikimybės yra tikros tikimybės,
nepriklausomumo ir priklausomumo sąvokos taikomos ir joms.
Du atsitiktiniai dydžiai $A$ ir $B$ yra *sąlygiškai nepriklausomi*
duotam trečiam kintamajam $C$ tada ir tik tada, kai $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Įdomu tai, kad du kintamieji gali būti nepriklausomi apskritai,
bet tapti priklausomi sąlygojant trečiuoju.
Taip dažnai nutinka, kai du atsitiktiniai dydžiai $A$ ir $B$
yra tam tikro trečiojo kintamojo $C$ priežastys.
Pavyzdžiui, kaulų lūžiai ir plaučių vėžys gali būti nepriklausomi
bendroje populiacijoje, bet jei sąlygosime buvimu ligoninėje,
galime rasti, kad kaulų lūžiai neigiamai koreliuoja su plaučių vėžiu.
Taip yra todėl, kad kaulo lūžis *paaiškina*, kodėl žmogus yra ligoninėje,
ir taip sumažina tikimybę, kad jis hospitalizuotas dėl plaučių vėžio.


Ir atvirkščiai, du priklausomi atsitiktiniai dydžiai
gali tapti nepriklausomi sąlygojant trečiuoju.
Tai dažnai nutinka, kai du kitaip nesusiję įvykiai
turi bendrą priežastį.
Batų dydis ir skaitymo lygis yra stipriai koreliuoti
tarp pradinių klasių mokinių,
bet ši koreliacija išnyksta, jei sąlygojame amžiumi.



## Pavyzdys
:label:`subsec_probability_hiv_app`

Išbandykime savo įgūdžius.
Tarkime, kad gydytojas pacientui atlieka ŽIV testą.
Šis testas yra gana tikslus ir suklysta tik su 1 % tikimybe,
kai pacientas sveikas, bet nustatomas kaip sergantis,
t. y. sveikų pacientų testas yra teigiamas 1 % atvejų.
Be to, testas niekada nepraleidžia ŽIV, jei pacientas iš tikrųjų juo serga.
Naudojame $D_1 \in \{0, 1\}$ diagnozei žymėti
($0$, jei neigiama, ir $1$, jei teigiama),
o $H \in \{0, 1\}$ – ŽIV statusui žymėti.

| Sąlyginė tikimybė | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

Atkreipkite dėmesį, kad stulpelių sumos visos lygios 1 (bet eilučių sumos ne),
nes tai sąlyginės tikimybės.
Apskaičiuokime tikimybę, kad pacientas serga ŽIV,
jei testas pasirodo teigiamas, t. y. $P(H = 1 \mid D_1 = 1)$.
Intuityviai tai priklausys nuo to, kaip dažna yra liga,
nes tai veikia klaidingų aliarmų skaičių.
Tarkime, kad populiacijoje liga gana reta, pvz., $P(H=1) = 0.0015$.
Kad pritaikytume Bajeso teoremą, turime taikyti marginalizavimą
nustatydami

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Tai veda prie

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

Kitaip tariant, tikimybė, kad pacientas iš tikrųjų serga ŽIV,
yra tik 13,06 %,
nepaisant to, kad testas gana tikslus.
Kaip matome, tikimybė gali prieštarauti intuicijai.
Ką turėtų daryti pacientas, gavęs tokią bauginančią žinią?
Greičiausiai pacientas paprašytų gydytojo
atlikti kitą testą, kad gautų aiškumo.
Antrasis testas turi kitas charakteristikas
ir nėra toks geras kaip pirmasis.

| Sąlyginė tikimybė | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

Deja, antrasis testas taip pat pasirodo teigiamas.
Apskaičiuokime reikalingas tikimybes, kad galėtume pasitelkti Bajeso teoremą,
darydami sąlyginio nepriklausomumo prielaidą:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

Dabar galime taikyti marginalizavimą, kad gautume tikimybę,
jog abu testai pasirodo teigiami:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Galiausiai tikimybė, kad pacientas serga ŽIV, kai abu testai teigiami, yra

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

Tai yra, antrasis testas leido įgyti daug didesnį pasitikėjimą, kad ne viskas gerai.
Nepaisant to, kad antrasis testas yra gerokai mažiau tikslus nei pirmasis,
jis vis tiek žymiai pagerino mūsų įvertį.
Prielaida, kad abu testai yra sąlygiškai nepriklausomi vienas nuo kito,
buvo esminė mūsų gebėjimui gauti tikslesnį įvertį.
Panagrinėkime kraštutinį atvejį, kai tą patį testą atliekame du kartus.
Tokioje situacijoje abu kartus tikėtumės to paties rezultato,
todėl pakartojus tą patį testą papildomos informacijos negauname.
Atidus skaitytojas galėjo pastebėti, kad diagnozė elgėsi
kaip klasifikatorius, slepiantis akyse,
kai mūsų gebėjimas nuspręsti, ar pacientas sveikas,
didėja gaunant daugiau požymių (testų rezultatų).


## Vidurkiai

Priimant sprendimus dažnai reikia ne tik atsižvelgti
į atskiriems įvykiams priskirtas tikimybes,
bet ir sujungti jas į naudingus agregatus,
galinčius mums nurodyti kryptį.
Pavyzdžiui, kai atsitiktiniai dydžiai įgyja tolydžias skaliarines reikšmes,
dažnai norime žinoti, kokios reikšmės *vidutiniškai* tikėtis.
Šis dydis formaliai vadinamas *vidurkiu* (matematine viltimi).
Jei investuojame,
pirmasis dominantis dydis
gali būti grąža, kurios galime tikėtis,
suvidurkinus per visas galimas baigtis
(ir svertinus atitinkamomis tikimybėmis).
Pavyzdžiui, tarkime, kad su 50 % tikimybe
investicija gali visiškai žlugti,
su 40 % tikimybe ji gali duoti 2$\times$ grąžą,
o su 10 % tikimybe – 10$\times$ grąžą.
Kad apskaičiuotume tikėtiną grąžą,
susumuojame visas grąžas, kiekvieną padaugindami
iš tikimybės, kad ji pasitaikys.
Tai duoda vidurkį
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$.
Taigi tikėtina grąža yra 1,8$\times$.


Apskritai atsitiktinio dydžio $X$ *vidurkis* (arba matematinė viltis)
apibrėžiamas taip:

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Panašiai tankių atveju gauname $E[X] = \int x \;dp(x)$.
Kartais mus domina tam tikros $x$ funkcijos
vidurkis.
Šiuos vidurkius galime apskaičiuoti taip:

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ and } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

atitinkamai diskrečiosioms tikimybėms ir tankiams.
Grįždami prie aukščiau pateikto investavimo pavyzdžio,
$f$ gali būti su grąža susijusi *naudingumo funkcija*
(laimė).
Elgesio ekonomistai jau seniai pastebėjo,
kad žmonės pinigų praradimui priskiria didesnį nenaudingumą,
nei naudingumas, gaunamas uždirbus vieną dolerį
palyginti su jų pradiniu lygiu.
Be to, pinigų vertė paprastai yra mažiau nei tiesinė.
Turėti 100 tūkst. dolerių, o ne nulį dolerių,
gali nulemti, ar sumokėsite nuomą,
gerai maitinsitės ir gausite kokybišką sveikatos priežiūrą,
ar kentėsite benamystę.
Kita vertus, nauda iš
200 tūkst., palyginti su 100 tūkst., yra mažiau dramatiška.
Tokie samprotavimai paskatino klišę,
kad „pinigų naudingumas yra logaritminis".


Jei naudingumas, susijęs su visišku praradimu, būtų $-1$,
o naudingumai, susiję su grąžomis $1$, $2$ ir $10$,
būtų atitinkamai $1$, $2$ ir $4$,
tada tikėtina investavimo laimė
būtų $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
(tikėtinas 30 % naudingumo praradimas).
Jei tai tikrai būtų jūsų naudingumo funkcija,
gal geriausia būtų laikyti pinigus banke.

Priimant finansinius sprendimus
taip pat galime norėti įvertinti,
kiek investicija yra *rizikinga*.
Čia mums rūpi ne tik tikėtina reikšmė,
bet ir tai, kiek faktinės reikšmės linkusios *svyruoti*
aplink šią reikšmę.
Atkreipkite dėmesį, kad negalime tiesiog paimti
skirtumo tarp faktinės ir tikėtinos reikšmių
vidurkio.
Taip yra todėl, kad skirtumo vidurkis
yra vidurkių skirtumas,
t. y. $E[X - E[X]] = E[X] - E[E[X]] = 0$.
Tačiau galime nagrinėti
bet kurios neneigiamos šio skirtumo funkcijos vidurkį.
Atsitiktinio dydžio *dispersija* apskaičiuojama nagrinėjant
*kvadratinių* skirtumų vidurkį:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Čia lygybė išplaukia išskleidus
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
ir paėmus kiekvieno nario vidurkį.
Dispersijos kvadratinė šaknis yra kitas
naudingas dydis, vadinamas *standartiniu nuokrypiu*.
Nors jis ir dispersija
perteikia tą pačią informaciją (bet kurį galima apskaičiuoti iš kito),
standartinis nuokrypis turi gerą savybę:
jis išreiškiamas tais pačiais vienetais
kaip ir pradinis dydis,
kurį atvaizduoja atsitiktinis dydis.

Galiausiai atsitiktinio dydžio funkcijos
dispersija
apibrėžiama analogiškai:

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

Grįždami prie savo investavimo pavyzdžio,
dabar galime apskaičiuoti investicijos dispersiją.
Ji lygi $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$.
Visais atžvilgiais tai rizikinga investicija.
Atkreipkite dėmesį, kad pagal matematinę konvenciją vidurkis ir dispersija
dažnai žymimi $\mu$ ir $\sigma^2$.
Ypač taip yra tada, kai juos naudojame
Gauso skirstinio parametrams nusakyti.

Taip pat, kaip pristatėme vidurkius
ir dispersiją *skaliariniams* atsitiktiniams dydžiams,
galime tai padaryti ir vektorinėms reikšmėms.
Vidurkiai paprasti, nes galime juos taikyti elementas po elemento.
Pavyzdžiui, $\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
turi koordinates $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
*Kovariacijos* sudėtingesnės.
Jas apibrėžiame imdami atsitiktinių dydžių ir jų vidurkio
skirtumo *išorinės sandaugos* vidurkį:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

Ši matrica $\boldsymbol{\Sigma}$ vadinama kovariacijų matrica.
Lengvas būdas pamatyti jos poveikį yra panagrinėti kokį nors vektorių $\mathbf{v}$,
kurio dydis toks pat kaip $\mathbf{x}$.
Gauname, kad

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

Taigi $\boldsymbol{\Sigma}$ leidžia mums apskaičiuoti dispersiją
bet kuriai tiesinei $\mathbf{x}$ funkcijai
atliekant paprastą matricų daugybą.
Nediagonaliniai elementai nusako, kiek koordinatės yra koreliuotos:
reikšmė 0 reiškia, kad koreliacijos nėra,
o didesnė teigiama reikšmė
reiškia, kad jos stipriau koreliuotos.



## Diskusija

Mašininiame mokyme yra daug dalykų, dėl kurių galime būti neapibrėžti!
Galime nebūti tikri dėl žymens reikšmės, kai duota įvestis.
Galime nebūti tikri dėl įvertintos parametro reikšmės.
Galime net nebūti tikri, ar naudojimo metu gaunami duomenys
apskritai yra iš to paties skirstinio kaip apmokymo duomenys.

*Aleatoriniu neapibrėžtumu* vadiname neapibrėžtumą,
kuris yra būdingas pačiam uždaviniui
ir atsiranda dėl tikro atsitiktinumo,
kurio nepaaiškina stebimi kintamieji.
*Epistemiu neapibrėžtumu* vadiname neapibrėžtumą
dėl modelio parametrų, t. y. tokį neapibrėžtumą,
kurį galime tikėtis sumažinti surinkdami daugiau duomenų.
Galime turėti epistemį neapibrėžtumą
dėl tikimybės,
kad moneta iškris herbu,
bet net kai žinome šią tikimybę,
lieka aleatorinis neapibrėžtumas
dėl bet kurio būsimo metimo baigties.
Nesvarbu, kiek ilgai stebėtume, kaip kas nors meta teisingą monetą,
niekada nebūsime daugiau ar mažiau nei 50 % tikri,
kad kitas metimas baigsis herbu.
Šie terminai kilę iš mechaninio modeliavimo
(apžvalgą apie šį [neapibrėžtumo kiekybinio įvertinimo](https://en.wikipedia.org/wiki/Uncertainty_quantification) aspektą žr. :citet:`Der-Kiureghian.Ditlevsen.2009`).
Tačiau verta pažymėti, kad šie terminai yra šiek tiek netiksliai vartojami.
Žodis *epistemis* nurodo viską, kas susiję su *žiniomis*,
todėl filosofine prasme visas neapibrėžtumas yra epistemis.


Matėme, kad duomenų atranka iš kokio nors nežinomo tikimybių skirstinio
gali suteikti informacijos, kurią galima panaudoti
duomenis generuojančio skirstinio parametrams įvertinti.
Vis dėlto sparta, kuria tai įmanoma, gali būti gana lėta.
Monetos metimo pavyzdyje (ir daugelyje kitų)
geriau nei kurti įvertinius,
konverguojančius sparta $1/\sqrt{n}$, negalime,
kur $n$ yra imties dydis (pvz., metimų skaičius).
Tai reiškia, kad pereidami nuo 10 iki 1000 stebinių (dažniausiai lengvai pasiekiamas uždavinys)
matome dešimteriopą neapibrėžtumo sumažėjimą,
o kiti 1000 stebinių padeda palyginti mažai,
suteikdami tik 1,41 karto sumažėjimą.
Tai nuolatinė mašininio mokymosi savybė:
nors dažnai lengvai pasiekiama pažanga,
tolesnei pažangai reikia labai daug duomenų
ir dažnai su jais – milžiniško skaičiavimų kiekio.
Empirinę šio fakto apžvalgą didelio masto kalbos modeliams žr. :citet:`Revels.Lubin.Papamarkou.2016`.

Taip pat patobulinome savo kalbą ir įrankius statistiniam modeliavimui.
Šio proceso metu susipažinome su sąlyginėmis tikimybėmis
ir viena svarbiausių statistikos lygčių – Bajeso teorema.
Tai efektyvi priemonė atskirti duomenų perteikiamą informaciją
per tikėtinumo narį $P(B \mid A)$, nusakantį,
kaip gerai stebiniai $B$ atitinka parametrų pasirinkimą $A$,
ir apriorinę tikimybę $P(A)$, valdančią, kiek tikėtinas
buvo konkretus $A$ pasirinkimas iš pat pradžių.
Visų pirma matėme, kaip šią taisyklę galima taikyti
diagnozių tikimybėms priskirti,
atsižvelgiant į testo efektyvumą *ir*
pačios ligos paplitimą (t. y. mūsų apriorinę tikimybę).

Galiausiai pristatėme pirmąjį netrivialių klausimų rinkinį
apie konkretaus tikimybių skirstinio poveikį,
būtent vidurkius ir dispersijas.
Nors tikimybių skirstiniui yra daug daugiau nei tik tiesinių ir kvadratinių
vidurkių,
šie du jau suteikia nemažai žinių
apie galimą skirstinio elgesį.
Pavyzdžiui, [Čebyšovo nelygybė](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
teigia, kad $P(|X - \mu| \geq k \sigma) \leq 1/k^2$,
kur $\mu$ yra vidurkis, $\sigma^2$ – skirstinio dispersija,
o $k > 1$ – mūsų pasirinktas pasikliovimo parametras.
Ji sako, kad skirstinio imtys su bent 50 % tikimybe
patenka į intervalą $[-\sqrt{2} \sigma, \sqrt{2} \sigma]$,
kurio centras yra vidurkis.




## Pratimai

1. Pateikite pavyzdį, kai daugiau duomenų stebėjimas gali sumažinti baigties neapibrėžtumą iki savavališkai mažo lygio.
1. Pateikite pavyzdį, kai daugiau duomenų stebėjimas sumažina neapibrėžtumą tik iki tam tikro taško, o toliau nebe. Paaiškinkite, kodėl taip yra ir kur tikitės šio taško.
1. Empiriškai parodėme konvergavimą prie vidurkio monetos metimo atveju. Apskaičiuokite tikimybės, kad pamatysime herbą, įverčio dispersiją, kai paimta $n$ imčių.
    1. Kaip dispersija kinta priklausomai nuo stebinių skaičiaus?
    1. Naudodami Čebyšovo nelygybę apribokite nuokrypį nuo vidurkio.
    1. Kaip tai susiję su centrine ribine teorema?
1. Tarkime, kad paimame $m$ imčių $x_i$ iš tikimybių skirstinio, kurio vidurkis nulis, o dispersija vienetas. Apskaičiuokite vidurkius $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Ar galime taikyti Čebyšovo nelygybę kiekvienam $z_m$ nepriklausomai? Kodėl ne?
1. Turint du įvykius su tikimybėmis $P(\mathcal{A})$ ir $P(\mathcal{B})$, apskaičiuokite viršutinę ir apatinę $P(\mathcal{A} \cup \mathcal{B})$ ir $P(\mathcal{A} \cap \mathcal{B})$ ribas. Užuomina: pavaizduokite situaciją [Veno diagrama](https://en.wikipedia.org/wiki/Venn_diagram).
1. Tarkime, turime atsitiktinių dydžių seką, tarkime, $A$, $B$ ir $C$, kur $B$ priklauso tik nuo $A$, o $C$ – tik nuo $B$. Ar galite supaprastinti jungtinę tikimybę $P(A, B, C)$? Užuomina: tai [Markovo grandinė](https://en.wikipedia.org/wiki/Markov_chain).
1. :numref:`subsec_probability_hiv_app` pavyzdyje tarkite, kad dviejų testų rezultatai nėra nepriklausomi. Visų pirma tarkite, kad kiekvieno testo atskirai klaidingai teigiamų rezultatų dažnis yra 10 %, o klaidingai neigiamų – 1 %. Tai yra, tarkite, kad $P(D =1 \mid H=0) = 0.1$ ir $P(D = 0 \mid H=1) = 0.01$. Be to, tarkite, kad esant $H = 1$ (užsikrėtęs) testų rezultatai yra sąlygiškai nepriklausomi, t. y. $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$, bet sveikų pacientų rezultatai susieti per $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Remdamiesi iki šiol turima informacija, išveskite $D_1$ ir $D_2$ jungtinių tikimybių lentelę, kai $H=0$.
    1. Išveskite tikimybę, kad pacientas serga ($H=1$), kai vienas testas pasirodo teigiamas. Galite laikyti, kad pradinė tikimybė $P(H=1) = 0.0015$ tokia pati kaip anksčiau.
    1. Išveskite tikimybę, kad pacientas serga ($H=1$), kai abu testai pasirodo teigiami.
1. Tarkime, kad esate investicinio banko turto valdytojas ir galite rinktis akcijas $s_i$, į kurias investuoti. Jūsų portfelio svorių suma turi būti $1$, o kiekvienos akcijos svoris yra $\alpha_i$. Akcijų vidutinė grąža yra $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$, o kovariacija – $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Apskaičiuokite tikėtiną duoto portfelio $\boldsymbol{\alpha}$ grąžą.
    1. Jei norėtumėte maksimizuoti portfelio grąžą, kaip turėtumėte pasirinkti investicijas?
    1. Apskaičiuokite portfelio *dispersiją*.
    1. Suformuluokite optimizavimo uždavinį maksimizuoti grąžą, išlaikant dispersiją apribotą viršutine riba. Tai Nobelio premija apdovanotas [Markowitzo portfelis](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013`. Jam išspręsti reikės kvadratinio programavimo sprendiklio – tai gerokai už šios knygos ribų.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/198)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17971)
:end_tab:
