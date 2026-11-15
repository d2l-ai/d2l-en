# Žodžių įterpiniai su globaliaisiais vektoriais (GloVe)
:label:`sec_glove`


Žodžių ir žodžių bendras pasirodymas
konteksto languose
gali nešti gausią semantinę informaciją.
Pavyzdžiui,
dideliame tekstyne
žodis „solid“ (kietas) yra
labiau linkęs pasirodyti kartu
su „ice“ (ledas) nei su „steam“ (garai),
o žodis „gas“ (dujos)
tikriausiai pasirodo kartu su „steam“
dažniau nei su „ice“.
Be to,
tokių bendrų pasirodymų
globalią tekstyno statistiką
galima apskaičiuoti iš anksto:
tai gali lemti efektyvesnį apmokymą.
Kad žodžių įterpiniams
pasinaudotume viso tekstyno statistine
informacija,
pirmiausia grįžkime prie
skip-gram modelio :numref:`subsec_skip-gram`,
bet interpretuokime jį
naudodami globalią tekstyno statistiką,
tokią kaip bendrų pasirodymų skaičiai.

## Skip-gram su globalia tekstyno statistika
:label:`subsec_skipgram-global`

Pažymėję $q_{ij}$
skip-gram modelio sąlyginę tikimybę
$P(w_j\mid w_i)$
žodžio $w_j$ turint žodį $w_i$,
turime

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

kur
bet kuriam indeksui $i$
vektoriai $\mathbf{v}_i$ ir $\mathbf{u}_i$
vaizduoja žodį $w_i$
atitinkamai kaip centrinį ir konteksto žodį,
o $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$
yra žodyno indeksų aibė.

Panagrinėkime žodį $w_i$,
kuris gali pasirodyti kelis kartus
tekstyne.
Visame tekstyne
visi konteksto žodžiai,
kai $w_i$ laikomas jų centriniu žodžiu,
sudaro žodžių indeksų *daugiaaibę* (angl. *multiset*) $\mathcal{C}_i$,
*kuri leidžia to paties elemento kelis egzempliorius*.
Bet kuriam elementui
jo egzempliorių skaičius vadinamas jo *kartotinumu*.
Kad iliustruotume pavyzdžiu,
tarkime, kad žodis $w_i$ tekstyne pasirodo du kartus,
o konteksto žodžių indeksai,
kai $w_i$ yra jų centrinis žodis,
dviejuose konteksto languose
yra
$k, j, m, k$ ir $k, l, k, j$.
Taigi daugiaaibė $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$, kur
elementų $j, k, l, m$
kartotinumai yra atitinkamai 2, 4, 1, 1.

Dabar elemento $j$ kartotinumą
daugiaaibėje $\mathcal{C}_i$ pažymėkime $x_{ij}$.
Tai yra globalus
žodžio $w_j$ (kaip konteksto žodžio)
ir žodžio $w_i$ (kaip centrinio žodžio)
bendro pasirodymo tame pačiame konteksto lange
visame tekstyne skaičius.
Naudojant tokią globalią tekstyno statistiką,
skip-gram modelio nuostolių funkcija
ekvivalenti

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

Toliau pažymime
$x_i$
visų konteksto žodžių skaičių
konteksto languose,
kuriuose $w_i$ pasirodo kaip jų centrinis žodis,
kuris ekvivalentus $|\mathcal{C}_i|$.
Tegul $p_{ij}$
yra sąlyginė tikimybė
$x_{ij}/x_i$ generuoti
konteksto žodį $w_j$ turint centrinį žodį $w_i$;
:eqref:`eq_skipgram-x_ij`
galima perrašyti kaip

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij` išraiška $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ apskaičiuoja
globalios tekstyno statistikos
sąlyginio pasiskirstymo $p_{ij}$
ir
modelio prognozių
sąlyginio pasiskirstymo $q_{ij}$
kryžminę entropiją.
Šis nuostolis
taip pat sverlamas $x_i$, kaip paaiškinta aukščiau.
Minimizuojant nuostolių funkciją
:eqref:`eq_skipgram-p_ij`,
prognozuojamas sąlyginis pasiskirstymas
priartės prie
sąlyginio pasiskirstymo
iš globalios tekstyno statistikos.


Nors kryžminės entropijos nuostolių funkcija
dažnai naudojama
tikimybinių pasiskirstymų atstumui matuoti,
čia ji gali būti ne pats geriausias pasirinkimas.
Viena vertus, kaip minėjome :numref:`sec_approx_train`,
tinkamo $q_{ij}$ normavimo sąnaudos
lemia sumą per visą žodyną,
kuri gali būti skaičiavimo požiūriu brangi.
Kita vertus,
didelis kiekis retų
įvykių iš didelio tekstyno
kryžminės entropijos nuostolio dažnai modeliuojamas
priskiriant
per didelį svorį.

## GloVe modelis

Atsižvelgiant į tai,
*GloVe* modelis atlieka tris skip-gram modelio pakeitimus,
paremtus kvadratiniu nuostoliu :cite:`Pennington.Socher.Manning.2014`:

1. Naudoja kintamuosius $p'_{ij}=x_{ij}$ ir $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$,
kurie nėra tikimybiniai pasiskirstymai,
ir abiejų logaritmus, todėl kvadratinio nuostolio narys yra $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Prideda du skaliarinius modelio parametrus kiekvienam žodžiui $w_i$: centrinio žodžio poslinkį $b_i$ ir konteksto žodžio poslinkį $c_i$.
3. Kiekvieno nuostolio nario svorį pakeičia svorio funkcija $h(x_{ij})$, kur $h(x)$ didėja intervale $[0, 1]$.

Sudėjus viską kartu, GloVe apmokymas yra šios nuostolių funkcijos minimizavimas:

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

Svorio funkcijai siūlomas pasirinkimas yra:
$h(x) = (x/c) ^\alpha$ (pvz., $\alpha = 0{,}75$), jei $x < c$ (pvz., $c = 100$); kitaip $h(x) = 1$.
Šiuo atveju,
kadangi $h(0)=0$,
bet kurio $x_{ij}=0$ kvadratinio nuostolio narį
galima praleisti skaičiavimo efektyvumui.
Pavyzdžiui,
naudojant minirinkinių stochastinį gradientinį nusileidimą apmokymui,
kiekvienoje iteracijoje
atsitiktinai atrenkame *nenulinių* $x_{ij}$ minirinkinį
gradientams apskaičiuoti
ir modelio parametrams atnaujinti.
Atkreipkite dėmesį, kad šie nenuliniai $x_{ij}$ yra iš anksto apskaičiuota
globali tekstyno statistika;
todėl modelis vadinamas GloVe –
*Global Vectors* (globalieji vektoriai).

Reikia pabrėžti, kad
jei žodis $w_i$ pasirodo žodžio $w_j$ konteksto lange,
tai *ir atvirkščiai*.
Todėl $x_{ij}=x_{ji}$.
Skirtingai nei word2vec,
kuris pritaiko asimetrinę sąlyginę tikimybę
$p_{ij}$,
GloVe pritaiko simetrinį $\log \, x_{ij}$.
Todėl bet kurio žodžio centrinio žodžio vektorius ir
konteksto žodžio vektorius GloVe modelyje yra matematiškai ekvivalentūs.
Tačiau praktikoje dėl skirtingų inicializavimo reikšmių
tas pats žodis po apmokymo šiuose dviejuose vektoriuose vis tiek gali gauti skirtingas reikšmes:
GloVe juos susumuoja kaip išvesties vektorių.



## GloVe interpretavimas iš bendro pasirodymo tikimybių santykio


GloVe modelį taip pat galime interpretuoti iš kitos perspektyvos.
Naudodami tuos pačius žymėjimus
:numref:`subsec_skipgram-global`,
tegul $p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$ yra sąlyginė tikimybė generuoti konteksto žodį $w_j$ turint $w_i$ kaip centrinį žodį tekstyne.
:numref:`tab_glove`
išvardyta keletas bendro pasirodymo tikimybių
turint žodžius „ice“ ir „steam“
bei jų santykiai, pagrįsti statistika iš didelio tekstyno.


:Žodžių ir žodžių bendro pasirodymo tikimybės bei jų santykiai iš didelio tekstyno (adaptuota iš 1 lentelės :citet:`Pennington.Socher.Manning.2014`)
:label:`tab_glove`

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|



Iš :numref:`tab_glove` galime pastebėti:

* Žodžiui $w_k$, kuris susijęs su „ice“, bet nesusijęs su „steam“, pavyzdžiui, $w_k=\textrm{solid}$, tikimės didesnio bendro pasirodymo tikimybių santykio, pavyzdžiui, 8,9.
* Žodžiui $w_k$, kuris susijęs su „steam“, bet nesusijęs su „ice“, pavyzdžiui, $w_k=\textrm{gas}$, tikimės mažesnio bendro pasirodymo tikimybių santykio, pavyzdžiui, 0,085.
* Žodžiui $w_k$, kuris susijęs ir su „ice“, ir su „steam“, pavyzdžiui, $w_k=\textrm{water}$, tikimės bendro pasirodymo tikimybių santykio, artimo 1, pavyzdžiui, 1,36.
* Žodžiui $w_k$, kuris nesusijęs nei su „ice“, nei su „steam“, pavyzdžiui, $w_k=\textrm{fashion}$, tikimės bendro pasirodymo tikimybių santykio, artimo 1, pavyzdžiui, 0,96.




Matyti, kad bendro pasirodymo tikimybių santykis
gali intuityviai išreikšti
žodžių tarpusavio sąryšį.
Todėl galime suprojektuoti
trijų žodžių vektorių funkciją,
kuri pritaikytų šį santykį.
Bendro pasirodymo tikimybių santykiui
${p_{ij}}/{p_{ik}}$,
kai $w_i$ yra centrinis žodis,
o $w_j$ ir $w_k$ yra konteksto žodžiai,
norime pritaikyti šį santykį
naudodami kokią nors funkciją $f$:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

Iš daugelio galimų $f$ dizainų
toliau pasirenkame tik vieną pagrįstą variantą.
Kadangi bendro pasirodymo tikimybių santykis
yra skaliaras,
reikalaujame, kad
$f$ būtų skaliarinė funkcija, pavyzdžiui,
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$.
Sukeitus žodžių indeksus
$j$ ir $k$ :eqref:`eq_glove-f`,
turi galioti
$f(x)f(-x)=1$,
todėl viena galimybė yra $f(x)=\exp(x)$,
t. y.

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Dabar pasirinkime
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$,
kur $\alpha$ yra konstanta.
Kadangi $p_{ij}=x_{ij}/x_i$, suėmę abiejų pusių logaritmus gauname $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$.
Papildomus poslinkio narius galime naudoti $- \log\, \alpha + \log\, x_i$ pritaikyti, pavyzdžiui, centrinio žodžio poslinkį $b_i$ ir konteksto žodžio poslinkį $c_j$:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

Matuodami :eqref:`eq_glove-square` kvadratinę paklaidą su svoriais,
gauname GloVe nuostolių funkciją
:eqref:`eq_glove-loss`.



## Santrauka

* Skip-gram modelį galima interpretuoti naudojant globalią tekstyno statistiką, pavyzdžiui, žodžių ir žodžių bendro pasirodymo skaičius.
* Kryžminės entropijos nuostolis gali būti ne pats geriausias pasirinkimas dviejų tikimybinių pasiskirstymų skirtumui matuoti, ypač dideliam tekstynui. GloVe naudoja kvadratinį nuostolį iš anksto apskaičiuotai globaliai tekstyno statistikai pritaikyti.
* GloVe centrinio žodžio vektorius ir konteksto žodžio vektorius bet kuriam žodžiui yra matematiškai ekvivalentūs.
* GloVe galima interpretuoti iš žodžių ir žodžių bendro pasirodymo tikimybių santykio.


## Pratimai

1. Jei žodžiai $w_i$ ir $w_j$ pasirodo tame pačiame konteksto lange, kaip galime naudoti jų atstumą teksto sekoje, kad performuotume sąlyginės tikimybės $p_{ij}$ skaičiavimo metodą? Užuomina: žr. GloVe straipsnio 4.2 skyrių :cite:`Pennington.Socher.Manning.2014`.
1. Ar bet kurio žodžio centrinio žodžio poslinkis ir konteksto žodžio poslinkis GloVe modelyje yra matematiškai ekvivalentūs? Kodėl?


[Diskusijos](https://discuss.d2l.ai/t/385)
