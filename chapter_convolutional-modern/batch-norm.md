```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Rinkinio normalizacija
:label:`sec_batch_norm`

Giliuosius neuroninius tinklus apmokyti sunku.
Priversti juos konverguoti per pagrįstą laiką gali būti keblu.
Šiame skyrelyje aprašome *rinkinio normalizaciją* – populiarią ir veiksmingą techniką,
nuosekliai pagreitinančią giliųjų tinklų konvergavimą :cite:`Ioffe.Szegedy.2015`.
Kartu su liekamaisiais blokais, aptariamais vėliau :numref:`sec_resnet`, rinkinio normalizacija
leido praktikams įprastai apmokyti tinklus, turinčius daugiau nei 100 sluoksnių.
Antrinė (atsitiktinė) rinkinio normalizacijos nauda slypi jai būdingoje reguliarizacijoje.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Giliųjų tinklų apmokymas

Dirbdami su duomenimis, prieš apmokymą dažnai juos iš anksto apdorojame.
Duomenų išankstinio apdorojimo pasirinkimai dažnai lemia milžinišką galutinių rezultatų skirtumą.
Prisiminkite, kaip MLP taikėme būsto kainoms prognozuoti (:numref:`sec_kaggle_house`).
Pirmasis mūsų žingsnis dirbant su realiais duomenimis
buvo standartizuoti įvesties požymius, kad jie turėtų
nulinį vidurkį $\boldsymbol{\mu} = 0$ ir vienetinę dispersiją $\boldsymbol{\Sigma} = \boldsymbol{1}$ per kelis stebėjimus :cite:`friedman1987exploratory`, dažnai perskalinant pastarąją taip, kad įstrižainė būtų vienetinė, t. y. $\Sigma_{ii} = 1$.
Kita strategija yra perskalinti vektorius iki vienetinio ilgio, galbūt su nuliniu vidurkiu *kiekvienam stebėjimui*.
Tai gali veikti gerai, pvz., erdvinių jutiklių duomenims. Šios išankstinio apdorojimo technikos ir daugelis kitų
naudingos, kad vertinimo uždavinys išliktų gerai kontroliuojamas.
Požymių atrankos ir išskyrimo apžvalgą žr., pavyzdžiui, :citet:`guyon2008feature` straipsnyje.
Vektorių standartizavimas turi ir malonų šalutinį poveikį – ribojamas jiems taikomų funkcijų sudėtingumas. Pavyzdžiui, garsioji spindulio ir paraštės riba :cite:`Vapnik95` atraminių vektorių mašinose ir perceptrono konvergavimo teorema :cite:`Novikoff62` remiasi apriboto normos įvestimis.

Intuityviai ši standartizacija gerai dera su mūsų optimizavimo algoritmais,
nes parametrus *a priori* patalpina panašiu mastu.
Todėl natūralu klausti, ar atitinkamas normalizavimo žingsnis *giliojo* tinklo viduje
nebūtų naudingas. Nors būtent taip samprotaujant nebuvo išrasta rinkinio normalizacija :cite:`Ioffe.Szegedy.2015`, tai naudingas būdas suprasti ją ir jos giminaitę – sluoksnio normalizaciją :cite:`Ba.Kiros.Hinton.2016` – vieningoje sistemoje.

Antra, tipiniam MLP ar CNN apmokymo metu
tarpinių sluoksnių kintamieji
(pvz., afininės transformacijos išvestys MLP)
gali įgyti labai skirtingo dydžio reikšmes:
ir išilgai sluoksnių nuo įvesties iki išvesties, ir tarp to paties sluoksnio vienetų,
ir laikui bėgant dėl mūsų modelio parametrų atnaujinimų.
Rinkinio normalizacijos išradėjai neformaliai postulavo,
kad šis tokių kintamųjų skirstinio dreifas gali trukdyti tinklo konvergavimui.
Intuityviai galime spėti, kad jei vieno
sluoksnio kintamųjų aktyvacijos yra 100 kartų didesnės nei kito sluoksnio,
tai gali reikalauti kompensuojamųjų mokymosi greičio pakeitimų. Adaptyvūs sprendikliai,
tokie kaip AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` ar Distributed Shampoo :cite:`anil2020scalable`, siekia tai išspręsti optimizavimo požiūriu, pvz., pridėdami antros eilės metodų aspektų.
Alternatyva yra užkirsti kelią problemos atsiradimui paprasčiausiai adaptyvia normalizacija.

Trečia, gilesni tinklai yra sudėtingi ir labiau linkę persimokyti.
Tai reiškia, kad reguliarizacija tampa kritiškesnė. Įprasta reguliarizacijos technika yra triukšmo
įterpimas. Tai žinoma jau seniai, pvz., kalbant apie triukšmo įterpimą į
įvestis :cite:`Bishop.1995`. Tai taip pat yra dropout pagrindas :numref:`sec_dropout`. Paaiškėja, gana atsitiktinai, kad rinkinio normalizacija suteikia visas tris naudas: išankstinį apdorojimą, skaitinį stabilumą ir reguliarizaciją.

Rinkinio normalizacija taikoma atskiriems sluoksniams arba, pasirinktinai, jiems visiems:
kiekvienoje apmokymo iteracijoje
pirmiausia normalizuojame (rinkinio normalizacijos) įvestis,
atimdami jų vidurkį ir
dalydami iš jų standartinio nuokrypio,
kur abu įvertinami pagal dabartinio minirinkinio statistikas.
Toliau pritaikome mastelio koeficientą ir postūmį, kad atkurtume prarastus laisvės
laipsnius. Būtent dėl šios *normalizacijos*, pagrįstos *rinkinio* statistikomis,
*rinkinio normalizacija* ir gavo savo pavadinimą.

Atkreipkite dėmesį, kad jei bandytume taikyti rinkinio normalizaciją su 1 dydžio minirinkiniais,
nieko negalėtume išmokti.
Taip yra todėl, kad atėmus vidurkius,
kiekvienas paslėptasis vienetas įgytų reikšmę 0.
Kaip galite nuspėti, kadangi rinkinio normalizacijai skiriame visą skyrelį,
su pakankamai dideliais minirinkiniais šis metodas pasirodo veiksmingas ir stabilus.
Viena išvada čia yra ta, kad taikant rinkinio normalizaciją
rinkinio dydžio pasirinkimas yra
dar svarbesnis nei be rinkinio normalizacijos, arba bent jau
reikia tinkamo kalibravimo, nes galime koreguoti rinkinio dydį.

Pažymėkime $\mathcal{B}$ minirinkinį, o $\mathbf{x} \in \mathcal{B}$ – rinkinio normalizacijos ($\textrm{BN}$) įvestį.
Šiuo atveju rinkinio normalizacija apibrėžiama taip:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm`
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ yra minirinkinio $\mathcal{B}$ imties vidurkis,
o $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ – imties standartinis nuokrypis.
Pritaikius standartizaciją,
gautas minirinkinys
turi nulinį vidurkį ir vienetinę dispersiją.
Vienetinės dispersijos pasirinkimas
(o ne kokio nors kito stebuklingo skaičiaus) yra savavališkas. Šį laisvės laipsnį atkuriame
įtraukdami elementų lygmens
*mastelio parametrą* $\boldsymbol{\gamma}$ ir *postūmio parametrą* $\boldsymbol{\beta}$,
kurių forma tokia pati kaip $\mathbf{x}$. Abu yra parametrai, kuriuos
reikia išmokti modelio apmokymo metu.

Tarpinių sluoksnių kintamųjų dydžiai
apmokymo metu negali išsiskirti,
nes rinkinio normalizacija aktyviai centruoja ir perskalina juos atgal
iki nurodyto vidurkio ir dydžio (per $\hat{\boldsymbol{\mu}}_\mathcal{B}$ ir ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
Praktinė patirtis patvirtina, kad, kaip užsiminėme aptardami požymių perskalinimą, rinkinio normalizacija leidžia naudoti agresyvesnius mokymosi greičius.
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ ir ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ :eqref:`eq_batchnorm` apskaičiuojame taip:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Atkreipkite dėmesį, kad prie dispersijos įverčio pridedame mažą konstantą $\epsilon > 0$,
kad niekada nebandytume dalyti iš nulio,
net tais atvejais, kai empirinis dispersijos įvertis gali būti labai mažas ar išnykti.
Įverčiai $\hat{\boldsymbol{\mu}}_\mathcal{B}$ ir ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ neutralizuoja mastelio problemą
naudodami triukšmingus vidurkio ir dispersijos įverčius.
Galite pamanyti, kad šis triukšmingumas turėtų būti problema.
Priešingai, jis iš tikrųjų yra naudingas.

Tai pasirodo esanti pasikartojanti giliojo mokymosi tema.
Dėl priežasčių, kurios teoriškai dar nėra gerai apibūdintos,
įvairūs triukšmo šaltiniai optimizavime
dažnai lemia greitesnį apmokymą ir mažesnį persimokymą:
atrodo, kad ši variacija veikia kaip reguliarizacijos forma.
:citet:`Teye.Azizpour.Smith.2018` ir :citet:`Luo.Wang.Shao.ea.2018`
susiejo rinkinio normalizacijos savybes atitinkamai su Bajeso apriorais ir baudomis.
Visų pirma tai šiek tiek nušviečia mįslę,
kodėl rinkinio normalizacija geriausiai veikia vidutinio dydžio minirinkiniams, 50–100 diapazone.
Atrodo, kad šis konkretus minirinkinio dydis įterpia būtent „tinkamą triukšmo kiekį“ kiekviename sluoksnyje, tiek mastelio požiūriu per $\hat{\boldsymbol{\sigma}}$, tiek postūmio per $\hat{\boldsymbol{\mu}}$: didesnis
minirinkinys mažiau reguliarizuoja dėl stabilesnių įverčių, o mažyčiai minirinkiniai
sunaikina naudingą signalą dėl didelės dispersijos. Toliau tyrinėjant šią kryptį, svarstant alternatyvius
išankstinio apdorojimo ir filtravimo tipus, galbūt bus rasta kitų veiksmingų reguliarizacijos tipų.

Fiksavus apmokytą modelį, galite pamanyti,
kad mieliau naudotume visą duomenų rinkinį
vidurkiui ir dispersijai įvertinti.
Baigus apmokymą, kodėl norėtume,
kad tas pats vaizdas būtų klasifikuojamas skirtingai,
priklausomai nuo rinkinio, kuriame jis atsitiktinai atsiduria?
Apmokymo metu toks tikslus skaičiavimas neįmanomas,
nes tarpiniai kintamieji
visiems duomenų pavyzdžiams
keičiasi kaskart, kai atnaujiname modelį.
Tačiau kai modelis apmokytas,
galime apskaičiuoti kiekvieno sluoksnio kintamųjų vidurkius ir dispersijas
pagal visą duomenų rinkinį.
Iš tiesų tai yra standartinė praktika
modeliams, naudojantiems rinkinio normalizaciją;
todėl rinkinio normalizacijos sluoksniai veikia skirtingai
*apmokymo režimu* (normalizuojant pagal minirinkinio statistikas)
nei *prognozavimo režimu* (normalizuojant pagal duomenų rinkinio statistikas).
Šia forma jie primena dropout reguliarizacijos elgesį :numref:`sec_dropout`,
kur triukšmas įterpiamas tik apmokymo metu.


## Rinkinio normalizacijos sluoksniai

Rinkinio normalizacijos įgyvendinimai pilnai sujungtiems sluoksniams
ir konvoliuciniams sluoksniams šiek tiek skiriasi.
Vienas esminis rinkinio normalizacijos ir kitų sluoksnių skirtumas
yra tas, kad kadangi pirmoji veikia su visu minirinkiniu vienu metu,
negalime tiesiog ignoruoti rinkinio dimensijos,
kaip darėme anksčiau pristatydami kitus sluoksnius.

### Pilnai sujungti sluoksniai

Taikydami rinkinio normalizaciją pilnai sujungtiems sluoksniams,
:citet:`Ioffe.Szegedy.2015` savo originaliame straipsnyje įterpė rinkinio normalizaciją po afininės transformacijos
ir *prieš* netiesinę aktyvacijos funkciją. Vėlesni taikymai eksperimentavo
įterpdami rinkinio normalizaciją iškart *po* aktyvacijos funkcijų.
Pažymėję pilnai sujungto sluoksnio įvestį $\mathbf{x}$,
afininę transformaciją
$\mathbf{W}\mathbf{x} + \mathbf{b}$ (su svorių parametru $\mathbf{W}$ ir poslinkio parametru $\mathbf{b}$),
o aktyvacijos funkciją $\phi$,
rinkinio normalizaciją naudojančio
pilnai sujungto sluoksnio išvesties $\mathbf{h}$ skaičiavimą galime išreikšti taip:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Prisiminkite, kad vidurkis ir dispersija apskaičiuojami
pagal *tą patį* minirinkinį,
kuriam taikoma transformacija.

### Konvoliuciniai sluoksniai

Panašiai su konvoliuciniais sluoksniais
rinkinio normalizaciją galime taikyti po konvoliucijos,
bet prieš netiesinę aktyvacijos funkciją. Pagrindinis skirtumas nuo rinkinio normalizacijos
pilnai sujungtuose sluoksniuose yra tas, kad operaciją taikome kiekvienam kanalui atskirai
*per visas vietas*. Tai suderinama su mūsų postūmio invariantiškumo prielaida, kuri
paskatino konvoliucijas: darėme prielaidą, kad konkreti šablono vieta
vaizde nėra kritiška supratimo tikslais.

Tarkime, kad mūsų minirinkiniuose yra $m$ pavyzdžių
ir kad kiekvienam kanalui
konvoliucijos išvesties aukštis yra $p$, o plotis – $q$.
Konvoliuciniams sluoksniams kiekvieną rinkinio normalizaciją atliekame
vienu metu per $m \cdot p \cdot q$ elementų kiekviename išvesties kanale.
Taigi skaičiuodami vidurkį ir dispersiją surenkame reikšmes per visas erdvines vietas
ir todėl
tą patį vidurkį ir dispersiją
duotame kanale
taikome kiekvienos erdvinės vietos reikšmei normalizuoti.
Kiekvienas kanalas turi savo mastelio ir postūmio parametrus,
abu yra skaliarai.

### Sluoksnio normalizacija
:label:`subsec_layer-normalization-in-bn`

Atkreipkite dėmesį, kad konvoliucijų kontekste rinkinio normalizacija yra gerai apibrėžta net
1 dydžio minirinkiniams: juk turime visas vaizdo vietas, per kurias galime vidurkinti. Todėl
vidurkis ir dispersija yra gerai apibrėžti, net jei tik vieno stebėjimo ribose. Ši svarstymo kryptis
paskatino :citet:`Ba.Kiros.Hinton.2016` pristatyti *sluoksnio normalizacijos* sąvoką. Ji veikia lygiai kaip
rinkinio normalizacija, tik taikoma po vieną stebėjimą. Todėl ir postūmis, ir mastelio koeficientas yra skaliarai. $n$ dimensijų vektoriui $\mathbf{x}$ sluoksnio normalizacija pateikiama taip:

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

kur mastelis ir postūmis taikomi koeficientų lygmeniu
ir pateikiami taip:

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

Kaip ir anksčiau, pridedame mažą postūmį $\epsilon > 0$, kad išvengtume dalybos iš nulio. Vienas pagrindinių sluoksnio normalizacijos naudojimo privalumų yra tas, kad ji apsaugo nuo divergencijos. Juk, nepaisant $\epsilon$, sluoksnio normalizacijos išvestis nepriklauso nuo mastelio. Tai yra, turime $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ bet kuriam $\alpha \neq 0$. Tai tampa lygybe, kai $|\alpha| \to \infty$ (apytikslė lygybė atsiranda dėl dispersijos postūmio $\epsilon$).

Kitas sluoksnio normalizacijos privalumas yra tas, kad ji nepriklauso nuo minirinkinio dydžio. Ji taip pat nepriklauso nuo to, ar esame apmokymo, ar testavimo režime. Kitaip tariant, tai tiesiog deterministinė transformacija, standartizuojanti aktyvacijas iki nurodyto mastelio. Tai gali būti labai naudinga užkertant kelią divergencijai optimizavime. Tolesnių detalių praleidžiame ir rekomenduojame norintiems perskaityti originalų straipsnį.

### Rinkinio normalizacija prognozavimo metu

Kaip minėjome anksčiau, rinkinio normalizacija paprastai veikia skirtingai
apmokymo režimu nei prognozavimo režimu.
Pirma, imties vidurkio ir imties dispersijos triukšmas,
atsirandantis juos vertinant pagal minirinkinius,
nebėra pageidautinas, kai tik apmokome modelį.
Antra, galbūt neturime prabangos
skaičiuoti rinkinio normalizacijos statistikų kiekvienam rinkiniui.
Pavyzdžiui,
gali tekti taikyti savo modelį darant po vieną prognozę.

Paprastai po apmokymo naudojame visą duomenų rinkinį
stabiliems kintamųjų statistikų įverčiams apskaičiuoti
ir tada juos fiksuojame prognozavimo metu.
Taigi rinkinio normalizacija apmokymo metu veikia kitaip nei testavimo metu.
Prisiminkite, kad dropout taip pat pasižymi šia savybe.

## (**Įgyvendinimas nuo nulio**)

Kad pamatytume, kaip rinkinio normalizacija veikia praktikoje, toliau ją įgyvendiname nuo nulio.

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        # `linen.Module.variables` have a `value` attribute containing the array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

Dabar galime [**sukurti tinkamą `BatchNorm` sluoksnį.**]
Mūsų sluoksnis saugos tinkamus
mastelio `gamma` ir postūmio `beta` parametrus,
kuriuos abu atnaujinsime apmokymo metu.
Be to, mūsų sluoksnis saugos
vidurkių ir dispersijų slenkamuosius vidurkius
vėlesniam naudojimui modelio prognozavimo metu.

Nepaisydami algoritminių detalių,
atkreipkite dėmesį į dizaino šabloną, slypintį mūsų sluoksnio įgyvendinimo pagrinde.
Paprastai matematiką apibrėžiame atskiroje funkcijoje, tarkime, `batch_norm`.
Tada šią funkcionalumą integruojame į pasirinktinį sluoksnį,
kurio kodas daugiausia sprendžia apskaitos klausimus,
tokius kaip duomenų perkėlimas į tinkamą įrenginio kontekstą,
reikalingų kintamųjų paskyrimas ir inicializavimas,
slenkamųjų vidurkių (čia vidurkio ir dispersijos) sekimas ir pan.
Šis šablonas leidžia aiškiai atskirti matematiką nuo šablonų kodo.
Taip pat atkreipkite dėmesį, kad patogumo dėlei
čia nesirūpinome automatiniu įvesties formos nustatymu;
todėl turime visur nurodyti požymių skaičių.
Šiuo metu visi šiuolaikiniai giliojo mokymosi karkasai siūlo automatinį dydžio ir formos nustatymą
aukšto lygio rinkinio normalizacijos API (praktikoje naudosime būtent jį).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

Naudojome `momentum`, kad valdytume ankstesnių vidurkio ir dispersijos įverčių agregavimą. Tai šiek tiek klaidingas pavadinimas, nes jis neturi nieko bendra su optimizavimo *judesio* nariu. Vis dėlto tai plačiai priimtas šio nario pavadinimas, ir paisydami API pavadinimų konvencijos savo kode naudojame tą patį kintamojo pavadinimą.

## [**LeNet su rinkinio normalizacija**]

Kad pamatytume, kaip `BatchNorm` taikyti kontekste,
toliau pritaikome jį tradiciniam LeNet modeliui (:numref:`sec_lenet`).
Prisiminkite, kad rinkinio normalizacija taikoma
po konvoliucinių ar pilnai sujungtų sluoksnių,
bet prieš atitinkamas aktyvacijos funkcijas.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
Kadangi `BatchNorm` sluoksniai turi apskaičiuoti rinkinio statistikas
(vidurkį ir dispersiją), Flax seka žodyną `batch_stats`, atnaujindamas
jį su kiekvienu minirinkiniu. Tokias kolekcijas kaip `batch_stats` galima saugoti
`TrainState` objekte (klasėje `d2l.Trainer`, apibrėžtoje
:numref:`oo-design-training`) kaip atributą, o modelio tiesioginio sklidimo metu
jos turėtų būti perduodamos argumentui `mutable`, kad Flax grąžintų pakeistus
kintamuosius.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

Kaip ir anksčiau, [**apmokysime savo tinklą su Fashion-MNIST duomenų rinkiniu**].
Šis kodas praktiškai identiškas tam, kurį naudojome pirmą kartą apmokydami LeNet.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

Pažvelkime į [**mastelio parametrą `gamma`
ir postūmio parametrą `beta`**], išmoktus
pirmajame rinkinio normalizacijos sluoksnyje.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**Glaustas įgyvendinimas**]

Palyginti su klase `BatchNorm`,
kurią ką tik patys apibrėžėme,
galime tiesiogiai naudoti klasę `BatchNorm`, apibrėžtą giliojo mokymosi karkaso aukšto lygio API.
Kodas atrodo praktiškai identiškas
mūsų aukščiau pateiktam įgyvendinimui, tik nebereikia nurodyti papildomų argumentų, kad dimensijos būtų teisingos.

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

Toliau [**naudojame tuos pačius hiperparametrus modeliui apmokyti.**]
Atkreipkite dėmesį, kad, kaip įprasta, aukšto lygio API variantas veikia gerokai greičiau,
nes jo kodas sukompiliuotas į C++ ar CUDA,
o mūsų pasirinktinį įgyvendinimą turi interpretuoti Python.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Diskusija

Intuityviai manoma, kad rinkinio normalizacija
daro optimizavimo kraštovaizdį glotnesnį.
Tačiau turime būti atsargūs, skirdami
spekuliatyvias intuicijas nuo tikrų paaiškinimų
reiškiniams, kuriuos stebime apmokydami giliuosius modelius.
Prisiminkite, kad net nežinome, kodėl paprastesni
gilieji neuroniniai tinklai (MLP ir įprasti CNN)
apskritai gerai apibendrina.
Net su dropout ir svorių silpnėjimu
jie lieka tokie lankstūs, kad jų gebėjimui apibendrinti nematytus duomenis
tikriausiai reikia gerokai rafinuotesnių mokymosi teorijos apibendrinimo garantijų.

Originalus straipsnis, siūlantis rinkinio normalizaciją :cite:`Ioffe.Szegedy.2015`, be galingo ir naudingo įrankio pristatymo,
pateikė paaiškinimą, kodėl ji veikia:
mažindama *vidinį kovariacijų poslinkį* (angl. *internal covariate shift*).
Tikriausiai *vidiniu kovariacijų poslinkiu* jie
turėjo omenyje kažką panašaus į aukščiau išreikštą intuiciją –
sampratą, kad kintamųjų reikšmių skirstinys keičiasi
apmokymo eigoje.
Tačiau su šiuo paaiškinimu buvo dvi problemos:
i) Šis dreifas labai skiriasi nuo *kovariacijų poslinkio*,
todėl pavadinimas yra klaidingas. Jei kas, jis artimesnis konceptų dreifui.
ii) Paaiškinimas siūlo nepakankamai apibrėžtą intuiciją,
bet palieka klausimą, *kodėl būtent ši technika veikia*,
atvirą ir reikalaujantį griežto paaiškinimo.
Šioje knygoje siekiame perteikti intuicijas, kuriomis praktikai
vadovaujasi kurdami giliuosius neuroninius tinklus.
Tačiau manome, kad svarbu
atskirti šias vedančias intuicijas
nuo nusistovėjusių mokslinių faktų.
Galiausiai, kai įvaldysite šią medžiagą
ir pradėsite rašyti savo mokslinius straipsnius,
norėsite aiškiai atriboti
techninius teiginius nuo nuojautų.

Po rinkinio normalizacijos sėkmės
jos paaiškinimas *vidiniu kovariacijų poslinkiu*
ne kartą iškildavo debatuose techninėje literatūroje
ir platesniame diskurse apie tai, kaip pristatyti mašininio mokymosi tyrimus.
Įsimintinoje kalboje, pasakytoje priimant Laiko išbandymo apdovanojimą
2017 m. NeurIPS konferencijoje,
Ali Rahimi naudojo *vidinį kovariacijų poslinkį*
kaip pagrindinį argumentą, lyginantį
šiuolaikinę giliojo mokymosi praktiką su alchemija.
Vėliau pavyzdys buvo išsamiai aptartas
pozicijos dokumente, nubrėžiančiame
nerimą keliančias mašininio mokymosi tendencijas :cite:`Lipton.Steinhardt.2018`.
Kiti autoriai
pasiūlė alternatyvių rinkinio normalizacijos sėkmės paaiškinimų,
kai kurie :cite:`Santurkar.Tsipras.Ilyas.ea.2018`
teigia, kad rinkinio normalizacijos sėkmė pasiekiama nepaisant elgesio,
kuris tam tikra prasme yra priešingas teigiamam originaliame straipsnyje.


Pažymime, kad *vidinis kovariacijų poslinkis*
nėra labiau vertas kritikos nei bet kuris iš
tūkstančių panašiai neaiškių teiginių,
kasmet pateikiamų techninėje mašininio mokymosi literatūroje.
Tikėtina, kad jo rezonansas kaip šių debatų pagrindinio taško
kyla iš plataus atpažįstamumo tikslinei auditorijai.
Rinkinio normalizacija pasirodė esanti nepakeičiamas metodas,
taikomas beveik visuose diegiamuose vaizdų klasifikatoriuose,
o straipsnis, pristatęs šią techniką, sulaukė
dešimčių tūkstančių citavimų. Tačiau spėjame, kad vadovaujantys reguliarizacijos principai
per triukšmo įterpimą, pagreitinimas per perskalinimą ir galiausiai išankstinis apdorojimas
gali paskatinti ir kitus sluoksnių bei technikų išradimus ateityje.

Praktiškesniu požiūriu verta prisiminti kelis rinkinio normalizacijos aspektus:

* Modelio apmokymo metu rinkinio normalizacija nuolat koreguoja tarpinę
  tinklo išvestį, naudodama minirinkinio vidurkį ir standartinį nuokrypį, kad
  tarpinės išvesties reikšmės kiekviename visų neuroninio tinklo sluoksnių lygyje būtų stabilesnės.
* Rinkinio normalizacija šiek tiek skiriasi pilnai sujungtiems ir konvoliuciniams sluoksniams. Iš tikrųjų
  konvoliuciniams sluoksniams kartais kaip alternatyvą galima naudoti sluoksnio normalizaciją.
* Kaip ir dropout sluoksnis, rinkinio normalizacijos sluoksniai veikia skirtingai
  apmokymo ir prognozavimo režimais.
* Rinkinio normalizacija naudinga reguliarizacijai ir konvergavimui optimizavime gerinti. Priešingai,
  originali motyvacija – sumažinti vidinį kovariacijų poslinkį – atrodo nėra pagrįstas paaiškinimas.
* Norėdami gauti tvirtesnių modelių, mažiau jautrių įvesties trikdžiams, apsvarstykite rinkinio normalizacijos pašalinimą :cite:`wang2022removing`.

## Pratimai

1. Ar prieš rinkinio normalizaciją turėtume pašalinti poslinkio parametrą iš pilnai sujungto sluoksnio ar konvoliucinio sluoksnio? Kodėl?
1. Palyginkite LeNet mokymosi greičius su rinkinio normalizacija ir be jos.
    1. Nubraižykite validavimo tikslumo augimą.
    1. Kokį didelį mokymosi greitį galite pasirinkti, kol optimizavimas sutrinka abiem atvejais?
1. Ar mums reikia rinkinio normalizacijos kiekviename sluoksnyje? Eksperimentuokite su tuo.
1. Įgyvendinkite „lengvą“ rinkinio normalizacijos versiją, kuri pašalina tik vidurkį, arba, alternatyviai, tokią, kuri
   pašalina tik dispersiją. Kaip ji elgiasi?
1. Užfiksuokite parametrus `beta` ir `gamma`. Stebėkite ir išanalizuokite rezultatus.
1. Ar galite pakeisti dropout rinkinio normalizacija? Kaip pasikeičia elgesys?
1. Tyrimų idėjos: sugalvokite kitų normalizacijos transformacijų, kurias galite taikyti:
    1. Ar galite pritaikyti tikimybinę integralinę transformaciją?
    1. Ar galite naudoti pilno rango kovariacijos įvertį? Kodėl tikriausiai neturėtumėte to daryti?
    1. Ar galite naudoti kitus kompaktiškus matricų variantus (blokinės įstrižainės, mažo poslinkio rango, Monarch ir kt.)?
    1. Ar suretinimo glaudinimas veikia kaip reguliarizatorius?
    1. Ar yra kitų projekcijų (pvz., iškilojo kūgio, simetrijos grupei specifinių transformacijų), kurias galite naudoti?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18005)
:end_tab:
