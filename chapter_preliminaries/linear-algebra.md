```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Tiesinė algebra
:label:`sec_linear-algebra`

Dabar jau galime įkelti duomenų rinkinius į tenzorius
ir apdoroti šiuos tenzorius
pagrindinėmis matematinėmis operacijomis.
Norint pradėti kurti sudėtingus modelius,
mums prireiks ir keleto tiesinės algebros įrankių.
Šiame skyrelyje švelniai supažindiname
su svarbiausiomis sąvokomis,
pradedant skaliarine aritmetika
ir pereinant prie matricų daugybos.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Skaliarai


Dauguma kasdienės matematikos
susideda iš skaičių apdorojimo
po vieną.
Formaliai šias reikšmes vadiname *skaliarais*.
Pavyzdžiui, temperatūra Palo Alto
yra malonūs $72$ Farenheito laipsniai.
Jei norėtumėte konvertuoti temperatūrą į Celsijaus laipsnius,
įvertintumėte išraišką
$c = \frac{5}{9}(f - 32)$, kai $f$ lygus $72$.
Šioje lygtyje reikšmės
$5$, $9$ ir $32$ yra pastovūs skaliarai.
Kintamieji $c$ ir $f$
paprastai žymi nežinomus skaliarus.

Skaliarus žymime
paprastomis mažosiomis raidėmis
(pvz., $x$, $y$ ir $z$),
o visų (tolydžiųjų)
*realiąsias reikšmes įgyjančių* skaliarų erdvę – $\mathbb{R}$.
Patogumo dėlei praleisime
griežtas *erdvių* apibrėžtis:
tiesiog prisiminkite, kad išraiška $x \in \mathbb{R}$
yra formalus būdas pasakyti, kad $x$ yra realiąją reikšmę įgyjantis skaliaras.
Simbolis $\in$ (tariamas „priklauso")
žymi priklausymą aibei.
Pavyzdžiui, $x, y \in \{0, 1\}$
rodo, kad $x$ ir $y$ yra kintamieji,
galintys įgyti tik reikšmes $0$ arba $1$.

(**Skaliarai įgyvendinami kaip tenzoriai,
turintys tik vieną elementą.**)
Toliau priskiriame du skaliarus
ir atliekame gerai žinomas sudėties, daugybos,
dalybos ir kėlimo laipsniu operacijas.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## Vektoriai

Šiuo metu [**vektorių galite įsivaizduoti kaip fiksuoto ilgio skaliarų masyvą.**]
Kaip ir jų atitikmenis kode,
šiuos skaliarus vadiname vektoriaus *elementais*
(sinonimai: *įrašai* ir *komponentės*).
Kai vektoriai vaizduoja realių duomenų rinkinių pavyzdžius,
jų reikšmės turi tam tikrą realią reikšmę.
Pavyzdžiui, jei apmokytume modelį prognozuoti
paskolos neišmokėjimo riziką,
galėtume kiekvieną pareiškėją susieti su vektoriumi,
kurio komponentės atitinka dydžius,
tokius kaip jų pajamos, darbo stažas
ar ankstesnių neįvykdymų skaičius.
Jei tirtume infarkto riziką,
kiekvienas vektorius galėtų vaizduoti pacientą,
o jo komponentės galėtų atitikti
naujausius gyvybinius požymius, cholesterolio lygį,
kasdienio fizinio krūvio minutes ir kt.
Vektorius žymime pastorintomis mažosiomis raidėmis
(pvz., $\mathbf{x}$, $\mathbf{y}$ ir $\mathbf{z}$).

Vektoriai įgyvendinami kaip $1$-osios eilės tenzoriai.
Paprastai tokie tenzoriai gali būti savavališko ilgio,
priklausomai nuo atminties apribojimų. Dėmesio: Python, kaip ir daugumoje programavimo kalbų, vektorių indeksai prasideda nuo $0$ (angl. *zero-based indexing*, indeksavimas nuo nulio), o tiesinėje algebroje apatiniai indeksai prasideda nuo $1$ (indeksavimas nuo vieneto).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

Vektoriaus elementą galime nurodyti naudodami apatinį indeksą.
Pavyzdžiui, $x_2$ žymi antrąjį $\mathbf{x}$ elementą.
Kadangi $x_2$ yra skaliaras, jo nepastoriname.
Pagal numatytuosius nustatymus vektorius vizualizuojame
sudėdami jo elementus vertikaliai:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

Čia $x_1, \ldots, x_n$ yra vektoriaus elementai.
Vėliau atskirsime tokius *stulpelio vektorius*
ir *eilutės vektorius*, kurių elementai sudėti horizontaliai.
Prisiminkite, kad [**tenzoriaus elementus pasiekiame indeksavimu.**]

```{.python .input}
%%tab all
x[2]
```

Norėdami nurodyti, kad vektorių sudaro $n$ elementų,
rašome $\mathbf{x} \in \mathbb{R}^n$.
Formaliai $n$ vadiname vektoriaus *dimensija*.
[**Kode tai atitinka tenzoriaus ilgį**],
pasiekiamą Python integruota funkcija `len`.

```{.python .input}
%%tab all
len(x)
```

Ilgį taip pat galime pasiekti per atributą `shape`.
Forma yra kortežas, nurodantis tenzoriaus ilgį kiekviena ašimi.
(**Tenzorių, turinčių tik vieną ašį, formos turi tik vieną elementą.**)

```{.python .input}
%%tab all
x.shape
```

Dažnai žodis „dimensija" vartojamas dviprasmiškai:
tiek ašių skaičiui,
tiek ilgiui tam tikra ašimi žymėti.
Kad išvengtume šios painiavos,
ašių skaičiui žymėti vartojame terminą *eilė*,
o *dimensija* – tik
komponenčių skaičiui žymėti.


## Matricos

Kaip skaliarai yra $0$-osios eilės tenzoriai,
o vektoriai – $1$-osios eilės tenzoriai,
taip matricos yra $2$-osios eilės tenzoriai.
Matricas žymime pastorintomis didžiosiomis raidėmis
(pvz., $\mathbf{X}$, $\mathbf{Y}$ ir $\mathbf{Z}$)
ir kode vaizduojame tenzoriais su dviem ašimis.
Išraiška $\mathbf{A} \in \mathbb{R}^{m \times n}$
rodo, kad matricą $\mathbf{A}$
sudaro $m \times n$ realiąsias reikšmes įgyjančių skaliarų,
išdėstytų $m$ eilučių ir $n$ stulpelių.
Kai $m = n$, sakome, kad matrica yra *kvadratinė*.
Vizualiai bet kurią matricą galime pavaizduoti kaip lentelę.
Kad nurodytume atskirą elementą,
pridedame apatinius eilutės ir stulpelio indeksus, pvz.,
$a_{ij}$ yra reikšmė, priklausanti $\mathbf{A}$
$i$-ajai eilutei ir $j$-ajam stulpeliui:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


Kode matricą $\mathbf{A} \in \mathbb{R}^{m \times n}$
vaizduojame $2$-osios eilės tenzoriumi, kurio forma ($m$, $n$).
[**Bet kurį tinkamo dydžio $m \times n$ tenzorių galime paversti
$m \times n$ matrica,**]
nurodę norimą formą funkcijai `reshape`:

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

Kartais norime apversti ašis.
Kai sukeičiame matricos eilutes ir stulpelius,
rezultatas vadinamas jos *transponuota matrica*.
Formaliai matricos $\mathbf{A}$ transponuotą matricą
žymime $\mathbf{A}^\top$, ir jei $\mathbf{B} = \mathbf{A}^\top$,
tai $b_{ij} = a_{ji}$ visiems $i$ ir $j$.
Taigi $m \times n$ matricos transponuota matrica
yra $n \times m$ matrica:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Kode bet kurios (**matricos transponuotą matricą**) galime gauti taip:

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**Simetrinės matricos yra kvadratinių matricų poaibis,
lygių savo pačių transponuotoms matricoms:
$\mathbf{A} = \mathbf{A}^\top$.**]
Ši matrica yra simetrinė:

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

Matricos naudingos duomenų rinkiniams vaizduoti.
Paprastai eilutės atitinka atskirus įrašus,
o stulpeliai – skirtingus atributus.



## Tenzoriai

Nors su vien skaliarais, vektoriais ir matricomis
galite nueiti toli savo mašininio mokymosi kelyje,
galiausiai gali prireikti dirbti su
aukštesnės eilės [**tenzoriais**].
Tenzoriai (**suteikia mums bendrą būdą aprašyti
$n$-osios eilės masyvų plėtinius.**)
Programinius *tenzoriaus klasės* objektus vadiname „tenzoriais"
būtent todėl, kad jie taip pat gali turėti savavališką ašių skaičių.
Nors gali būti painu vartoti žodį
*tenzorius* ir matematiniam objektui,
ir jo įgyvendinimui kode,
mūsų reikšmė paprastai turėtų būti aiški iš konteksto.
Bendruosius tenzorius žymime didžiosiomis raidėmis
specialiu šriftu
(pvz., $\mathsf{X}$, $\mathsf{Y}$ ir $\mathsf{Z}$),
o jų indeksavimo mechanizmas
(pvz., $x_{ijk}$ ir $[\mathsf{X}]_{1, 2i-1, 3}$)
natūraliai išplaukia iš matricų indeksavimo.

Tenzoriai taps svarbesni,
kai pradėsime dirbti su vaizdais.
Kiekvienas vaizdas pateikiamas kaip $3$-iosios eilės tenzorius,
kurio ašys atitinka aukštį, plotį ir *kanalą*.
Kiekvienoje erdvinėje vietoje kiekvienos spalvos
(raudonos, žalios ir mėlynos) intensyvumai
sudedami kanalo ašimi.
Be to, vaizdų rinkinys
kode vaizduojamas $4$-osios eilės tenzoriumi,
kurio skirtingi vaizdai indeksuojami
pirmąja ašimi.
Aukštesnės eilės tenzoriai konstruojami, kaip ir vektoriai bei matricos,
didinant formos komponenčių skaičių.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## Pagrindinės tenzorių aritmetikos savybės

Skaliarai, vektoriai, matricos
ir aukštesnės eilės tenzoriai
turi keletą patogių savybių.
Pavyzdžiui, poelementės operacijos
sukuria išvestis, kurių forma
tokia pati kaip operandų.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

[**Dviejų matricų poelementė sandauga
vadinama jų *Adamaro sandauga***] (žymima $\odot$).
Dviejų matricų
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ Adamaro sandaugos įrašus
galime išrašyti:



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**Pridėjus skaliarą prie tenzoriaus arba padauginus skaliarą ir tenzorių**], gaunamas rezultatas,
kurio forma tokia pati kaip pradinio tenzoriaus.
Čia prie kiekvieno tenzoriaus elemento pridedamas skaliaras (arba jis dauginamas iš skaliaro).

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## Redukavimas
:label:`subsec_lin-alg-reduction`

Dažnai norime apskaičiuoti [**tenzoriaus elementų sumą.**]
Vektoriaus $\mathbf{x}$, kurio ilgis $n$, elementų sumai išreikšti
rašome $\sum_{i=1}^n x_i$. Tam yra paprasta funkcija:

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

Kad išreikštume [**savavališkos formos tenzorių elementų sumas**],
tiesiog sumuojame pagal visas jo ašis.
Pavyzdžiui, $m \times n$ matricos $\mathbf{A}$ elementų suma
galėtų būti užrašyta $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

Pagal numatytuosius nustatymus iškvietus sumos funkciją
tenzorius *redukuojamas* visomis ašimis,
galiausiai gaunamas skaliaras.
Mūsų bibliotekos taip pat leidžia [**nurodyti ašis,
kuriomis tenzorius turi būti redukuojamas.**]
Norėdami susumuoti visus elementus eilutėmis (0 ašis),
funkcijoje `sum` nurodome `axis=0`.
Kadangi įvesties matrica redukuojama 0 ašimi,
kad gautume išvesties vektorių,
šios ašies išvesties formoje nėra.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

Nurodžius `axis=1`, bus redukuojamas stulpelių matmuo (1 ašis), susumuojant visų stulpelių elementus.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Matricos redukavimas sumuojant ir eilutėmis, ir stulpeliais
yra ekvivalentus visų matricos elementų susumavimui.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**Susijęs dydis yra *vidurkis*, dar vadinamas *aritmetiniu vidurkiu*.**]
Vidurkį skaičiuojame dalydami sumą
iš bendro elementų skaičiaus.
Kadangi vidurkio skaičiavimas toks dažnas,
jam skirta atskira bibliotekos funkcija,
veikianti analogiškai kaip `sum`.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Panašiai vidurkio skaičiavimo funkcija
taip pat gali redukuoti tenzorių konkrečiomis ašimis.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Suma be redukavimo
:label:`subsec_lin-alg-non-reduction`

Kartais gali būti naudinga [**išlaikyti nepakeistą ašių skaičių**],
kviečiant sumos ar vidurkio skaičiavimo funkciją.
Tai svarbu, kai norime naudoti transliavimo mechanizmą.

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Pavyzdžiui, kadangi `sum_A` po kiekvienos eilutės susumavimo išlaiko abi ašis,
galime (**padalyti `A` iš `sum_A` naudodami transliavimą**),
kad sukurtume matricą, kurios kiekvienos eilutės suma lygi $1$.

```{.python .input}
%%tab all
A / sum_A
```

Jei norime apskaičiuoti [**`A` elementų kaupiamąją sumą tam tikra ašimi**],
tarkime, `axis=0` (eilutė po eilutės), galime iškviesti funkciją `cumsum`.
Pagal paskirtį ši funkcija neredukuoja įvesties tenzoriaus jokia ašimi.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Skaliarinės sandaugos

Iki šiol atlikome tik poelementes operacijas, sumas ir vidurkius.
Ir jei tai būtų viskas, ką galime daryti, tiesinė algebra
nenusipelnytų atskiro skyrelio.
Laimei, čia viskas tampa įdomiau.
Viena pagrindinių operacijų yra skaliarinė sandauga.
Kai duoti du vektoriai $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
jų *skaliarinė sandauga* $\mathbf{x}^\top \mathbf{y}$ (dar žinoma kaip *vidinė sandauga*, $\langle \mathbf{x}, \mathbf{y}  \rangle$)
yra tos pačios padėties elementų sandaugų suma:
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~Dviejų vektorių *skaliarinė sandauga* yra tos pačios padėties elementų sandaugų suma~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

Ekvivalenčiai, (**dviejų vektorių skaliarinę sandaugą galime apskaičiuoti
atlikdami poelementę daugybą, o po jos – sumavimą:**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

Skaliarinės sandaugos naudingos įvairiuose kontekstuose.
Pavyzdžiui, kai duotas reikšmių rinkinys,
žymimas vektoriumi $\mathbf{x}  \in \mathbb{R}^n$,
ir svorių (angl. *weights*) rinkinys, žymimas $\mathbf{w} \in \mathbb{R}^n$,
reikšmių $\mathbf{x}$ svertinė suma
pagal svorius $\mathbf{w}$
gali būti išreikšta skaliarine sandauga $\mathbf{x}^\top \mathbf{w}$.
Kai svoriai neneigiami
ir sudaro sumą $1$, t. y. $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
skaliarinė sandauga išreiškia *svertinį vidurkį*.
Normalizavus du vektorius iki vienetinio ilgio,
skaliarinės sandaugos išreiškia kampo tarp jų kosinusą.
Vėliau šiame skyrelyje formaliai pristatysime šią *ilgio* sąvoką.


## Matricos ir vektoriaus sandaugos

Dabar, kai žinome, kaip skaičiuoti skaliarines sandaugas,
galime pradėti suprasti *sandaugą*
tarp $m \times n$ matricos $\mathbf{A}$
ir $n$-mačio vektoriaus $\mathbf{x}$.
Pradžioje savo matricą
pavaizduojame jos eilučių vektoriais

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

kur kiekvienas $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
yra eilutės vektorius, vaizduojantis matricos $\mathbf{A}$ $i$-ąją eilutę.

[**Matricos ir vektoriaus sandauga $\mathbf{A}\mathbf{x}$
yra tiesiog $m$ ilgio stulpelio vektorius,
kurio $i$-asis elementas yra skaliarinė sandauga
$\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Daugybą iš matricos
$\mathbf{A}\in \mathbb{R}^{m \times n}$
galime laikyti transformacija, projektuojančia vektorius
iš $\mathbb{R}^{n}$ į $\mathbb{R}^{m}$.
Šios transformacijos nepaprastai naudingos.
Pavyzdžiui, posūkius galime pavaizduoti
kaip daugybą iš tam tikrų kvadratinių matricų.
Matricos ir vektoriaus sandaugos taip pat aprašo
pagrindinį skaičiavimą, naudojamą apskaičiuojant
kiekvieno neuroninio tinklo sluoksnio išvestis,
kai duotos ankstesnio sluoksnio išvestys.

:begin_tab:`mxnet`
Matricos ir vektoriaus sandaugai kode išreikšti
naudojame tą pačią funkciją `dot`.
Operacija nustatoma
pagal argumentų tipą.
Atkreipkite dėmesį, kad `A` stulpelių dimensija
(jos ilgis 1 ašimi)
turi sutapti su `x` dimensija (jo ilgiu).
:end_tab:

:begin_tab:`pytorch`
Matricos ir vektoriaus sandaugai kode išreikšti
naudojame funkciją `mv`.
Atkreipkite dėmesį, kad `A` stulpelių dimensija
(jos ilgis 1 ašimi)
turi sutapti su `x` dimensija (jo ilgiu).
Python turi patogų operatorių `@`,
galintį vykdyti ir matricos–vektoriaus,
ir matricos–matricos sandaugas
(priklausomai nuo argumentų).
Taigi galime rašyti `A@x`.
:end_tab:

:begin_tab:`tensorflow`
Matricos ir vektoriaus sandaugai kode išreikšti
naudojame funkciją `matvec`.
Atkreipkite dėmesį, kad `A` stulpelių dimensija
(jos ilgis 1 ašimi)
turi sutapti su `x` dimensija (jo ilgiu).
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## Matricų daugyba

Kai perprasite skaliarines sandaugas ir matricos ir vektoriaus sandaugas,
*matricų daugyba* turėtų būti paprasta.

Tarkime, turime dvi matricas
$\mathbf{A} \in \mathbb{R}^{n \times k}$
ir $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Tegul $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ žymi
eilutės vektorių, vaizduojantį matricos $\mathbf{A}$ $i$-ąją eilutę,
o $\mathbf{b}_{j} \in \mathbb{R}^k$ – stulpelio vektorių
iš matricos $\mathbf{B}$ $j$-ojo stulpelio:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


Kad sudarytume matricų sandaugą $\mathbf{C} \in \mathbb{R}^{n \times m}$,
tiesiog apskaičiuojame kiekvieną elementą $c_{ij}$
kaip skaliarinę sandaugą tarp
$\mathbf{A}$ $i$-osios eilutės
ir $\mathbf{B}$ $j$-ojo stulpelio,
t. y. $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Matricų daugybą $\mathbf{AB}$ galime laikyti
$m$ matricos ir vektoriaus sandaugų
arba $m \times n$ skaliarinių sandaugų atlikimu
ir rezultatų sujungimu į
$n \times m$ matricą.**]
Šiame fragmente
atliekame matricų `A` ir `B` daugybą.
Čia `A` yra matrica su dviem eilutėmis ir trimis stulpeliais,
o `B` – matrica su trimis eilutėmis ir keturiais stulpeliais.
Po daugybos gauname matricą su dviem eilutėmis ir keturiais stulpeliais.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

Terminas *matricų daugyba* (angl. *matrix–matrix multiplication*)
dažnai supaprastinamas iki *matricų daugybos*
ir neturėtų būti painiojamas su Adamaro sandauga.


## Normos
:label:`subsec_lin-algebra-norms`

Vieni naudingiausių tiesinės algebros operatorių yra *normos*.
Neformaliai vektoriaus norma parodo, kiek jis yra *didelis*.
Pavyzdžiui, $\ell_2$ norma matuoja
vektoriaus (Euklido) ilgį.
Čia vartojame *dydžio* sąvoką, susijusią su vektoriaus komponenčių didumu
(o ne jo dimensija).

Norma yra funkcija $\| \cdot \|$, kuri atvaizduoja vektorių
į skaliarą ir tenkina šias tris savybes:

1. Kai duotas bet kuris vektorius $\mathbf{x}$, jei vektorių (visus jo elementus)
   padauginame iš skaliaro $\alpha \in \mathbb{R}$, jo norma atitinkamai keičiasi:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Bet kuriems vektoriams $\mathbf{x}$ ir $\mathbf{y}$
   normos tenkina trikampio nelygybę:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. Vektoriaus norma yra neneigiama ir lygi nuliui tik tada, kai vektorius nulinis:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Daug funkcijų yra tinkamos normos, ir skirtingos normos
koduoja skirtingas dydžio sąvokas.
Euklido norma, kurią visi išmokome pradinės mokyklos geometrijoje
skaičiuodami statiojo trikampio įžambinę,
yra vektoriaus elementų kvadratų sumos kvadratinė šaknis.
Formaliai tai vadinama [**$\ell_2$ *norma***] ir išreiškiama

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

Metodas `norm` apskaičiuoja $\ell_2$ normą.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**$\ell_1$ norma**] taip pat dažna,
o susijęs matas vadinamas Manheteno atstumu.
Pagal apibrėžtį $\ell_1$ norma susumuoja
vektoriaus elementų absoliutines reikšmes:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

Palyginti su $\ell_2$ norma, ji mažiau jautri išskirtims.
Kad apskaičiuotume $\ell_1$ normą,
absoliutinės reikšmės operaciją
sukomponuojame su sumavimo operacija.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

Ir $\ell_2$, ir $\ell_1$ normos yra bendresnių $\ell_p$ *normų*
atskiri atvejai:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Matricų atveju reikalai sudėtingesni.
Juk matricas galima vertinti ir kaip atskirų įrašų rinkinius,
*ir* kaip objektus, veikiančius vektorius ir paverčiančius juos kitais vektoriais.
Pavyzdžiui, galime klausti, kiek ilgesnė
gali būti matricos ir vektoriaus sandauga $\mathbf{X} \mathbf{v}$
palyginti su $\mathbf{v}$.
Ši mąstymo linija veda prie vadinamosios *spektrinės* normos.
Kol kas pristatome [**kur kas lengviau apskaičiuojamą *Frobenijaus normą***],
apibrėžiamą kaip
matricos elementų kvadratų sumos
kvadratinė šaknis:

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

Frobenijaus norma elgiasi taip, tarsi
būtų matricos formos vektoriaus $\ell_2$ norma.
Iškvietus šią funkciją, bus apskaičiuota
matricos Frobenijaus norma.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

Nors nenorime per daug skubėti,
jau galime sudaryti intuiciją, kodėl šios sąvokos naudingos.
Giliajame mokyme dažnai bandome spręsti optimizavimo uždavinius:
*maksimizuoti* stebimiems duomenims priskirtą tikimybę;
*maksimizuoti* su rekomendacijų modeliu susijusias pajamas;
*minimizuoti* atstumą tarp prognozių
ir etaloninių stebėjimų;
*minimizuoti* atstumą tarp to paties asmens nuotraukų
reprezentacijų,
kartu *maksimizuojant* atstumą tarp skirtingų asmenų
nuotraukų reprezentacijų.
Šie atstumai, sudarantys
giliojo mokymosi algoritmų tikslus,
dažnai išreiškiami normomis.


## Diskusija

Šiame skyrelyje apžvelgėme visą tiesinę algebrą,
kurios jums prireiks, kad suprastumėte
didelę šiuolaikinio giliojo mokymosi dalį.
Tiesinė algebra turi daug daugiau,
ir didelė jos dalis naudinga mašininiam mokymuisi.
Pavyzdžiui, matricas galima skaidyti į daugiklius,
o šie skaidiniai gali atskleisti
žemos dimensijos struktūrą realiuose duomenų rinkiniuose.
Yra ištisų mašininio mokymosi porūšių,
kurie sutelkia dėmesį į matricų skaidinių
ir jų apibendrinimų aukštesnės eilės tenzoriams naudojimą,
kad atskleistų duomenų rinkinių struktūrą
ir spręstų prognozavimo uždavinius.
Tačiau ši knyga sutelkia dėmesį į gilųjį mokymąsi.
Ir manome, kad jums labiau norėsis
išmokti daugiau matematikos,
kai praktiškai išbandysite
mašininį mokymąsi su realiais duomenų rinkiniais.
Taigi, nors pasiliekame teisę
vėliau pristatyti daugiau matematikos,
čia šį skyrelį baigiame.

Jei norite sužinoti daugiau tiesinės algebros,
yra daug puikių knygų ir šaltinių internete.
Pažangesniam greitajam kursui apsvarstykite
:citet:`Strang.1993`, :citet:`Kolter.2008` ir :citet:`Petersen.Pedersen.ea.2008`.

Apibendrinant:

* Skaliarai, vektoriai, matricos ir tenzoriai yra
  pagrindiniai matematiniai objektai, naudojami tiesinėje algebroje,
  ir turi atitinkamai nulį, vieną, dvi ir savavališką skaičių ašių.
* Tenzorius galima atpjauti arba redukuoti nurodytomis ašimis
  atitinkamai indeksavimu arba operacijomis, tokiomis kaip `sum` ir `mean`.
* Poelementės sandaugos vadinamos Adamaro sandaugomis.
  Priešingai, skaliarinės sandaugos, matricos ir vektoriaus sandaugos bei matricų sandaugos
  nėra poelementės operacijos ir paprastai grąžina objektus,
  kurių formos skiriasi nuo operandų.
* Palyginti su Adamaro sandaugomis, matricų sandaugos
  skaičiuojamos gerokai ilgiau (kubiniu, o ne kvadratiniu laiku).
* Normos atspindi įvairias vektoriaus (ar matricos) didumo sąvokas
  ir dažnai taikomos dviejų vektorių skirtumui,
  kad būtų išmatuotas jų atstumas.
* Įprastos vektorių normos yra $\ell_1$ ir $\ell_2$ normos,
   o įprastos matricų normos – *spektrinė* ir *Frobenijaus* normos.


## Pratimai

1. Įrodykite, kad matricos transponuotos matricos transponuota matrica yra pati matrica: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Kai duotos dvi matricos $\mathbf{A}$ ir $\mathbf{B}$, parodykite, kad sudėtis ir transponavimas komutuoja: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Ar bet kurios kvadratinės matricos $\mathbf{A}$ atveju $\mathbf{A} + \mathbf{A}^\top$ visada yra simetrinė? Ar galite įrodyti rezultatą naudodami tik ankstesnių dviejų pratimų rezultatus?
1. Šiame skyrelyje apibrėžėme tenzorių `X`, kurio forma (2, 3, 4). Kokia yra `len(X)` išvestis? Parašykite savo atsakymą nerašydami jokio kodo, tada patikrinkite atsakymą kodu.
1. Ar savavališkos formos tenzoriui `X` `len(X)` visada atitinka tam tikros `X` ašies ilgį? Kokia tai ašis?
1. Paleiskite `A / A.sum(axis=1)` ir pažiūrėkite, kas nutinka. Ar galite išanalizuoti rezultatus?
1. Keliaujant tarp dviejų taškų Manheteno centre, kokį atstumą reikia įveikti koordinačių požiūriu, t. y. prospektais ir gatvėmis? Ar galima keliauti įstrižai?
1. Panagrinėkite tenzorių, kurio forma (2, 3, 4). Kokios yra sumavimo išvesčių formos 0, 1 ir 2 ašimis?
1. Pateikite tenzorių su trimis ar daugiau ašių funkcijai `linalg.norm` ir stebėkite jos išvestį. Ką ši funkcija apskaičiuoja savavališkos formos tenzoriams?
1. Panagrinėkite tris dideles matricas, tarkime, $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ ir $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, inicializuotas Gauso atsitiktiniais dydžiais. Norite apskaičiuoti sandaugą $\mathbf{A} \mathbf{B} \mathbf{C}$. Ar yra skirtumas atminties pėdsake ir greityje, priklausomai nuo to, ar skaičiuojate $(\mathbf{A} \mathbf{B}) \mathbf{C}$, ar $\mathbf{A} (\mathbf{B} \mathbf{C})$? Kodėl?
1. Panagrinėkite tris dideles matricas, tarkime, $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ ir $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Ar yra skirtumas greityje, priklausomai nuo to, ar skaičiuojate $\mathbf{A} \mathbf{B}$, ar $\mathbf{A} \mathbf{C}^\top$? Kodėl? Kas pasikeičia, jei inicializuojate $\mathbf{C} = \mathbf{B}^\top$ nekopijuodami atminties? Kodėl?
1. Panagrinėkite tris matricas, tarkime, $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Sukonstruokite tenzorių su trimis ašimis, sudėję $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. Kokia yra dimensija? Atpjaukite antrąją trečiosios ašies koordinatę, kad atkurtumėte $\mathbf{B}$. Patikrinkite, ar jūsų atsakymas teisingas.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17968)
:end_tab:
