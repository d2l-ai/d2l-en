# Tikrinės skaidos
:label:`sec_eigendecompositions`

Tikrinės reikšmės dažnai yra viena naudingiausių sąvokų,
su kuriomis susidursime studijuodami tiesinę algebrą,
tačiau pradedančiajam lengva nepastebėti jų svarbos.
Toliau pristatome tikrinę skaidą ir
bandome perteikti, kodėl ji tokia svarbi.

Tarkime, turime matricą $A$ su šiais elementais:

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

Jei $A$ pritaikome bet kuriam vektoriui $\mathbf{v} = [x, y]^\top$,
gauname vektorių $\mathbf{A}\mathbf{v} = [2x, -y]^\top$.
Tai turi intuityvų aiškinimą:
ištempti vektorių dvigubai $x$ kryptimi,
o tada apversti jį $y$ kryptimi.

Vis dėlto yra *kai kurių* vektorių, kuriems kai kas išlieka nepakitę.
Būtent $[1, 0]^\top$ pereina į $[2, 0]^\top$,
o $[0, 1]^\top$ – į $[0, -1]^\top$.
Šie vektoriai lieka toje pačioje tiesėje,
o vienintelis pokytis – matrica juos ištempia
atitinkamai $2$ ir $-1$ kartus.
Tokius vektorius vadiname *tikriniais vektoriais*,
o koeficientą, kiek jie ištempiami, – *tikrinėmis reikšmėmis*.

Bendru atveju, jei galime rasti skaičių $\lambda$
ir vektorių $\mathbf{v}$, kad

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

sakome, kad $\mathbf{v}$ yra matricos $A$ tikrinis vektorius, o $\lambda$ – tikrinė reikšmė.

## Tikrinių reikšmių radimas
Išsiaiškinkime, kaip jas rasti. Iš abiejų pusių atėmę $\lambda \mathbf{v}$
ir iškėlę vektorių prieš skliaustus,
matome, kad aukščiau pateikta lygybė yra ekvivalenti:

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

Kad galiotų :eqref:`eq_eigvalue_der`, matome, kad $(\mathbf{A} - \lambda \mathbf{I})$
turi kurią nors kryptį suspausti iki nulio,
todėl ji nėra apverčiama, taigi determinantas lygus nuliui.
Taigi *tikrines reikšmes* galime rasti
išsiaiškinę, su kuriuo $\lambda$ yra $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$.
Radę tikrines reikšmes, galime išspręsti
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
ir rasti atitinkamus *tikrinius vektorius*.

### Pavyzdys
Pažiūrėkime tai su sudėtingesne matrica

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

Jei nagrinėtume $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$,
matytume, kad tai ekvivalentu daugianario lygčiai
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$.
Taigi dvi tikrinės reikšmės yra $4$ ir $1$.
Norėdami rasti atitinkamus vektorius, turime išspręsti

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

Tai galime išspręsti su vektoriais atitinkamai $[1, -1]^\top$ ir $[1, 2]^\top$.

Tai galime patikrinti kodu naudodami įtaisytąją `numpy.linalg.eig` procedūrą.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

Atkreipkite dėmesį, kad `numpy` normalizuoja tikrinius vektorius iki vienetinio ilgio,
o mes savuosius ėmėme bet kokio ilgio.
Be to, ženklas parenkamas laisvai.
Vis dėlto apskaičiuoti vektoriai yra lygiagretūs
tiems, kuriuos radome ranka su tomis pačiomis tikrinėmis reikšmėmis.

## Matricų skaida
Pratęskime ankstesnį pavyzdį dar vienu žingsniu.  Tegu

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

yra matrica, kurios stulpeliai yra matricos $\mathbf{A}$ tikriniai vektoriai. Tegu

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

yra matrica, kurios įstrižainėje yra atitinkamos tikrinės reikšmės.
Tada tikrinių reikšmių ir tikrinių vektorių apibrėžimas mums sako, kad

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

Matrica $W$ yra apverčiama, todėl abi puses galime dešinėje padauginti iš $W^{-1}$
ir gauti, kad galime užrašyti

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

Kitame skyriuje pamatysime keletą gražių tokios skaidos pasekmių,
bet kol kas pakanka žinoti, kad tokia skaida
egzistuos tol, kol galime rasti pilną
tiesiškai nepriklausomų tikrinių vektorių rinkinį (kad $W$ būtų apverčiama).

## Veiksmai su tikrinėmis skaidomis
Viena gera tikrinių skaidų :eqref:`eq_eig_decomp` savybė yra ta, kad
daugelį įprastų veiksmų galime aiškiai užrašyti
per tikrinę skaidą. Pirmas pavyzdys:

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Tai rodo, kad bet kuriam teigiamam matricos laipsniui
tikrinė skaida gaunama tiesiog pakėlus tikrines reikšmes tuo pačiu laipsniu.
Tą patį galima parodyti ir neigiamiems laipsniams,
todėl jei norime matricą apversti, pakanka nagrinėti

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

arba, kitaip tariant, tiesiog apversti kiekvieną tikrinę reikšmę.
Tai veiks tol, kol kiekviena tikrinė reikšmė nelygi nuliui,
taigi matome, kad apverčiamumas yra tas pats, kas nebuvimas nulinių tikrinių reikšmių.

Iš tiesų, papildomas darbas gali parodyti, kad jei $\lambda_1, \ldots, \lambda_n$
yra matricos tikrinės reikšmės, tai tos matricos determinantas yra

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

t. y. visų tikrinių reikšmių sandauga.
Tai intuityviai suprantama, nes kad ir kaip $\mathbf{W}$ tempia,
$W^{-1}$ tai atšaukia, todėl galiausiai vienintelis tempimas vyksta
dauginant iš įstrižaininės matricos $\boldsymbol{\Sigma}$,
kuri tūrius ištempia įstrižainės elementų sandauga.

Galiausiai prisiminkite, kad rangas buvo didžiausias
tiesiškai nepriklausomų matricos stulpelių skaičius.
Atidžiai išnagrinėję tikrinę skaidą,
matome, kad rangas yra lygus
nenulinių matricos $\mathbf{A}$ tikrinių reikšmių skaičiui.

Pavyzdžių galėtų būti ir daugiau, tačiau, tikėkimės, mintis aiški:
tikrinė skaida gali supaprastinti daugelį tiesinės algebros skaičiavimų
ir yra pagrindinė operacija, slypinti už daugelio skaitinių algoritmų
ir didelės dalies analizės, kurią atliekame tiesinėje algebroje.

## Simetrinių matricų tikrinės skaidos
Ne visada įmanoma rasti pakankamai tiesiškai nepriklausomų tikrinių vektorių,
kad aukščiau aprašytas procesas veiktų. Pavyzdžiui, matrica

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

turi tik vieną tikrinį vektorių – $(1, 0)^\top$.
Tokioms matricoms nagrinėti reikia sudėtingesnių metodų,
nei galime aprėpti (pavyzdžiui, Žordano normaliosios formos ar singuliarinių reikšmių skaidos).
Dažnai turėsime susiaurinti dėmesį iki matricų,
kurioms galime garantuoti pilno tikrinių vektorių rinkinio egzistavimą.

Dažniausiai sutinkama šeima – *simetrinės matricos*,
t. y. matricos, kurioms $\mathbf{A} = \mathbf{A}^\top$.
Šiuo atveju galime imti $W$ kaip *ortogonaliąją matricą* – matricą, kurios stulpeliai yra vienetinio ilgio vektoriai, statmeni vienas kitam, ir kuriai
$\mathbf{W}^\top = \mathbf{W}^{-1}$ – o visos tikrinės reikšmės bus realios.
Taigi šiuo ypatingu atveju :eqref:`eq_eig_decomp` galime užrašyti kaip

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Geršgorino skritulių teorema
Apie tikrines reikšmes dažnai sunku samprotauti intuityviai.
Pateikus savavališką matricą, mažai ką galima pasakyti
apie jos tikrines reikšmes jų neskaičiuojant.
Vis dėlto yra viena teorema, kuri leidžia lengvai gerai įvertinti,
jei didžiausios reikšmės yra įstrižainėje.

Tegu $\mathbf{A} = (a_{ij})$ yra bet kuri kvadratinė matrica ($n\times n$).
Apibrėšime $r_i = \sum_{j \neq i} |a_{ij}|$.
Tegu $\mathcal{D}_i$ žymi skritulį kompleksinėje plokštumoje
su centru $a_{ii}$ ir spinduliu $r_i$.
Tada kiekviena $\mathbf{A}$ tikrinė reikšmė yra viename iš $\mathcal{D}_i$.

Tai gali būti sunku iškart suvokti, todėl pažiūrėkime pavyzdį.
Nagrinėkime matricą:

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Turime $r_1 = 0{,}3$, $r_2 = 0{,}6$, $r_3 = 0{,}8$ ir $r_4 = 0{,}9$.
Matrica simetrinė, todėl visos tikrinės reikšmės yra realios.
Tai reiškia, kad visos mūsų tikrinės reikšmės bus kuriame nors iš intervalų

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


Skaitinis skaičiavimas rodo,
kad tikrinės reikšmės yra apytiksliai $0{,}99$, $2{,}97$, $4{,}95$, $9{,}08$,
visos patogiai tilpusios pateiktuose intervaluose.

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

Šitaip tikrines reikšmes galima apytiksliai įvertinti,
ir įverčiai bus gana tikslūs tuo atveju,
kai įstrižainė yra
gerokai didesnė už visus kitus elementus.

Tai smulkmena, bet tokioje sudėtingoje
ir subtilioje temoje kaip tikrinė skaida
gera įgyti bet kokį intuityvų supratimą.

## Naudingas taikymas: iteruotų atvaizdžių augimas

Dabar, kai iš principo suprantame, kas yra tikriniai vektoriai,
pažiūrėkime, kaip jie gali padėti giliai suprasti
problemą, esminę neuroninių tinklų elgsenai: tinkamą svorių inicializavimą.

### Tikriniai vektoriai kaip ilgalaikė elgsena

Visapusiškas giliųjų neuroninių tinklų
inicializavimo matematinis tyrimas nepatenka į šio teksto aprėptį,
tačiau čia galime pamatyti supaprastintą versiją ir suprasti,
kaip tikrinės reikšmės gali padėti suprasti, kaip šie modeliai veikia.
Kaip žinome, neuroniniai tinklai veikia pereidami
tarp tiesinių transformacijų sluoksnių ir netiesinių operacijų.
Supaprastinimo dėlei čia darysime prielaidą, kad netiesiškumo nėra
ir kad transformacija yra viena kartojama matricos operacija $A$,
todėl mūsų modelio išvestis yra

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

Kai šie modeliai inicializuojami, $A$ imama kaip
atsitiktinė matrica su Gauso elementais, taigi sukurkime tokią.
Konkrečiai, pradedame nuo $5 \times 5$ matricos su Gauso skirstiniu, kurio vidurkis nulis, o dispersija vienas.

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Elgsena su atsitiktiniais duomenimis
Supaprastinimo dėlei savo žaisliniame modelyje
darysime prielaidą, kad duomenų vektorius, kurį paduodame, $\mathbf{v}_{in}$,
yra atsitiktinis penkiamatis Gauso vektorius.
Pagalvokime, ko norime, kad įvyktų.
Kontekstui pagalvokime apie bendrą MM uždavinį,
kuriame bandome įvesties duomenis, pavyzdžiui, paveikslėlį, paversti prognoze,
pavyzdžiui, tikimybe, kad paveikslėlyje yra katė.
Jei pakartotinis $\mathbf{A}$ taikymas
atsitiktinį vektorių ištempia iki labai didelio ilgio,
tai nedideli įvesties pokyčiai bus sustiprinti
iki didelių išvesties pokyčių – menki įvesties paveikslėlio pakeitimai
lemtų labai skirtingas prognozes.
Tai neatrodo teisinga!

Kita vertus, jei $\mathbf{A}$ atsitiktinius vektorius sutrumpina,
tai pereinant daugelį sluoksnių vektorius iš esmės susitrauks iki nieko,
ir išvestis nebepriklausys nuo įvesties. Tai taip pat akivaizdžiai neteisinga!

Turime eiti siaura riba tarp augimo ir slopimo,
kad užtikrintume, jog išvestis keičiasi priklausomai nuo įvesties, bet nedaug!

Pažiūrėkime, kas įvyksta, kai savo matricą $\mathbf{A}$ pakartotinai dauginame
iš atsitiktinio įvesties vektoriaus ir stebime normą.

```{.python .input}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Norma auga nevaldomai!
Iš tiesų, jei paimsime dalmenų sąrašą, pamatysime dėsningumą.

```{.python .input}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

Pažiūrėję į paskutinę aukščiau pateikto skaičiavimo dalį,
matome, kad atsitiktinis vektorius ištempiamas `1.974459321485[...]` kartų,
kur galinė dalis šiek tiek kinta,
tačiau tempimo koeficientas yra stabilus.

### Ryšys su tikriniais vektoriais

Matėme, kad tikriniai vektoriai ir tikrinės reikšmės atitinka
tai, kiek kažkas ištempiama,
tačiau tai buvo konkretiems vektoriams ir konkrečiam tempimui.
Pažiūrėkime, kokie jie yra $\mathbf{A}$.
Nedidelė išlyga: pasirodo, kad norint juos pamatyti visus,
reikės pereiti prie kompleksinių skaičių.
Galite į juos žiūrėti kaip į tempimus ir sukimus.
Paėmę kompleksinio skaičiaus normą
(realiosios ir menamosios dalių kvadratų sumos kvadratinę šaknį),
galime išmatuoti tempimo koeficientą. Taip pat juos surikiuokime.

```{.python .input}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### Pastebėjimas

Čia matome kai ką netikėto:
tas skaičius, kurį anksčiau nustatėme kaip
mūsų matricos $\mathbf{A}$, pritaikytos atsitiktiniam vektoriui,
ilgalaikį tempimą, yra *tiksliai*
(tikslumu iki trylikos skaitmenų po kablelio!)
didžiausia $\mathbf{A}$ tikrinė reikšmė.
Tai akivaizdžiai nėra sutapimas!

Tačiau jei dabar pagalvosime, kas vyksta geometriškai,
tai pradeda atrodyti prasminga. Panagrinėkime atsitiktinį vektorių.
Šis atsitiktinis vektorius šiek tiek rodo į visas puses,
taigi, visų pirma, jis bent kiek rodo
ta pačia kryptimi kaip ir $\mathbf{A}$ tikrinis vektorius,
atitinkantis didžiausią tikrinę reikšmę.
Tai tiek svarbu, kad jie vadinami
*pagrindine tikrine reikšme* ir *pagrindiniu tikriniu vektoriumi*.
Pritaikius $\mathbf{A}$, mūsų atsitiktinis vektorius
ištempiamas visomis įmanomomis kryptimis,
atitinkančiomis visus įmanomus tikrinius vektorius,
tačiau labiausiai jis ištempiamas
šio pagrindinio tikrinio vektoriaus kryptimi.
Tai reiškia, kad pritaikius $A$,
mūsų atsitiktinis vektorius tampa ilgesnis ir rodo kryptimi,
artimesne pagrindiniam tikriniam vektoriui.
Matricą taikant daug kartų, sutapimas su pagrindiniu tikriniu vektoriumi vis didėja, kol
praktiškai mūsų atsitiktinis vektorius yra paverčiamas
pagrindiniu tikriniu vektoriumi!
Iš tiesų šis algoritmas yra pagrindas
to, kas vadinama *laipsnių iteracija* (angl. *power iteration*),
skirta didžiausiai matricos tikrinei reikšmei ir tikriniam vektoriui rasti. Išsamiau žr., pavyzdžiui, :cite:`Golub.Van-Loan.1996`.

### Normalizavimo taisymas

Iš aukščiau pateiktų svarstymų padarėme išvadą,
kad nenorime, jog atsitiktinis vektorius būtų ištempiamas ar suspaudžiamas,
norime, kad atsitiktiniai vektoriai viso proceso metu išliktų maždaug to paties dydžio.
Tam dabar perskalėjame savo matricą šia pagrindine tikrine reikšme,
kad didžiausia tikrinė reikšmė dabar būtų tiesiog vienas.
Pažiūrėkime, kas šiuo atveju įvyksta.

```{.python .input}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Taip pat galime kaip anksčiau nubraižyti santykį tarp iš eilės einančių normų ir pamatyti, kad jis iš tiesų stabilizuojasi.

```{.python .input}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Diskusija

Dabar matome būtent tai, ko tikėjomės!
Matricas normalizavę pagrindine tikrine reikšme,
matome, kad atsitiktiniai duomenys nebesprogsta kaip anksčiau,
o galiausiai nusistovi ties tam tikra reikšme.
Būtų gerai tokius dalykus galėti padaryti iš pirmųjų principų,
ir pasirodo, kad jei gilinsimės į jų matematiką,
galime pamatyti, kad didžiausia didelės atsitiktinės matricos
su nepriklausomais elementais (Gauso skirstinys, vidurkis nulis,
dispersija vienas) tikrinė reikšmė vidutiniškai yra maždaug $\sqrt{n}$,
arba mūsų atveju $\sqrt{5} \approx 2{,}2$,
dėl žavaus fakto, žinomo kaip *apskritimo dėsnis* :cite:`Ginibre.1965`.
Parodyta, kad atsitiktinių matricų tikrinių reikšmių (ir susijusio objekto, vadinamo singuliarinėmis reikšmėmis) ryšys
yra glaudžiai susijęs su tinkamu neuroninių tinklų inicializavimu, kaip aptarta :citet:`Pennington.Schoenholz.Ganguli.2017` ir vėlesniuose darbuose.

## Santrauka
* Tikriniai vektoriai yra vektoriai, kuriuos matrica ištempia nekeisdama krypties.
* Tikrinės reikšmės yra dydis, kuriuo tikriniai vektoriai ištempiami pritaikius matricą.
* Matricos tikrinė skaida leidžia daugelį veiksmų suvesti į veiksmus su tikrinėmis reikšmėmis.
* Geršgorino skritulių teorema gali pateikti apytikres matricos tikrinių reikšmių vertes.
* Iteruotų matricos laipsnių elgsena pirmiausia priklauso nuo didžiausios tikrinės reikšmės dydžio.  Šis supratimas turi daug taikymų neuroninių tinklų inicializavimo teorijoje.

## Pratimai
1. Kokios yra šios matricos tikrinės reikšmės ir tikriniai vektoriai?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  Kokios yra šios matricos tikrinės reikšmės ir tikriniai vektoriai ir kas šiame pavyzdyje keista, palyginti su ankstesniu?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Neskaičiuodami tikrinių reikšmių, pasakykite, ar įmanoma, kad mažiausia šios matricos tikrinė reikšmė būtų mažesnė nei $0{,}5$? *Pastaba*: šį uždavinį galima išspręsti mintimis.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1087)
:end_tab:
