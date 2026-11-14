```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Duomenų apdorojimas
:label:`sec_ndarray`

Kad ką nors nuveiktume,
mums reikia būdo duomenims saugoti ir apdoroti.
Paprastai su duomenimis reikia daryti dvi svarbias veiklas:
(i) juos gauti;
ir (ii) apdoroti, kai jie jau kompiuteryje.
Nėra prasmės gauti duomenis
neturint būdo jų saugoti,
todėl pradėkime praktiškai
su $n$-mačiais masyvais,
kuriuos taip pat vadiname *tenzoriais*.
Jei jau žinote mokslinių skaičiavimų paketą
NumPy, tai bus labai lengva.
Visuose šiuolaikiniuose giliojo mokymosi karkasuose
*tenzoriaus klasė* (`ndarray` MXNet,
`Tensor` PyTorch ir TensorFlow)
primena NumPy `ndarray`
su keliomis pridėtomis puikiomis savybėmis.
Pirma, tenzoriaus klasė
palaiko automatinį diferencijavimą.
Antra, ji pasitelkia GPU
skaitiniams skaičiavimams pagreitinti,
o NumPy veikia tik su CPU.
Dėl šių savybių neuroninius tinklus
lengva programuoti ir jie greitai veikia.



## Darbo pradžia

:begin_tab:`mxnet`
Pradžioje importuojame `np` (`numpy`) ir
`npx` (`numpy_extension`) modulius iš MXNet.
Čia `np` modulyje yra
NumPy palaikomos funkcijos,
o `npx` modulyje – plėtinių rinkinys,
sukurtas giliajam mokymuisi
NumPy panašioje aplinkoje įgalinti.
Naudodami tenzorius, beveik visada
iškviečiame `set_np` funkciją:
tai skirta tenzorių apdorojimo suderinamumui
su kitais MXNet komponentais.
:end_tab:

:begin_tab:`pytorch`
(**Pradžioje importuojame PyTorch biblioteką.
Atkreipkite dėmesį, kad paketo pavadinimas yra `torch`.**)
:end_tab:

:begin_tab:`tensorflow`
Pradžioje importuojame `tensorflow`.
Trumpumo dėlei praktikai
dažnai priskiria slapyvardį `tf`.
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**Tenzorius yra (galbūt daugiamatis) skaitinių reikšmių masyvas.**]
Vienmačiu atveju, t. y. kai duomenims reikia tik vienos ašies,
tenzorius vadinamas *vektoriumi*.
Su dviem ašimis tenzorius vadinamas *matrica*.
Kai ašių yra $k > 2$, specialiųjų pavadinimų atsisakome
ir objektą tiesiog vadiname $k$-*osios eilės tenzoriumi*.

:begin_tab:`mxnet`
MXNet suteikia įvairių funkcijų
naujiems tenzoriams kurti,
iš anksto užpildytiems reikšmėmis.
Pavyzdžiui, iškvietę `arange(n)`,
galime sukurti tolygiai išdėstytų reikšmių vektorių,
prasidedantį nuo 0 (imtinai)
ir pasibaigiantį ties `n` (neimtinai).
Pagal numatytuosius nustatymus intervalo dydis yra $1$.
Jei nenurodyta kitaip,
nauji tenzoriai saugomi pagrindinėje atmintyje
ir skirti skaičiavimams CPU.
:end_tab:

:begin_tab:`pytorch`
PyTorch suteikia įvairių funkcijų
naujiems tenzoriams kurti,
iš anksto užpildytiems reikšmėmis.
Pavyzdžiui, iškvietę `arange(n)`,
galime sukurti tolygiai išdėstytų reikšmių vektorių,
prasidedantį nuo 0 (imtinai)
ir pasibaigiantį ties `n` (neimtinai).
Pagal numatytuosius nustatymus intervalo dydis yra $1$.
Jei nenurodyta kitaip,
nauji tenzoriai saugomi pagrindinėje atmintyje
ir skirti skaičiavimams CPU.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow suteikia įvairių funkcijų
naujiems tenzoriams kurti,
iš anksto užpildytiems reikšmėmis.
Pavyzdžiui, iškvietę `range(n)`,
galime sukurti tolygiai išdėstytų reikšmių vektorių,
prasidedantį nuo 0 (imtinai)
ir pasibaigiantį ties `n` (neimtinai).
Pagal numatytuosius nustatymus intervalo dydis yra $1$.
Jei nenurodyta kitaip,
nauji tenzoriai saugomi pagrindinėje atmintyje
ir skirti skaičiavimams CPU.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
Kiekviena iš šių reikšmių vadinama
tenzoriaus *elementu*.
Tenzorių `x` sudaro 12 elementų.
Bendrą tenzoriaus elementų skaičių
galime sužinoti pagal jo atributą `size`.
:end_tab:

:begin_tab:`pytorch`
Kiekviena iš šių reikšmių vadinama
tenzoriaus *elementu*.
Tenzorių `x` sudaro 12 elementų.
Bendrą tenzoriaus elementų skaičių
galime sužinoti naudodami jo metodą `numel`.
:end_tab:

:begin_tab:`tensorflow`
Kiekviena iš šių reikšmių vadinama
tenzoriaus *elementu*.
Tenzorių `x` sudaro 12 elementų.
Bendrą tenzoriaus elementų skaičių
galime sužinoti naudodami funkciją `size`.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**Tenzoriaus *formą* galime sužinoti**)
(ilgį kiekviena ašimi),
patikrinę jo atributą `shape`.
Kadangi čia turime vektorių,
`shape` turi tik vieną elementą
ir sutampa su dydžiu.

```{.python .input}
%%tab all
x.shape
```

Tenzoriaus [**formą galime pakeisti
nekeisdami jo dydžio ar reikšmių**],
iškviesdami `reshape`.
Pavyzdžiui, galime paversti
savo vektorių `x`, kurio forma (12,),
matrica `X`, kurios forma (3, 4).
Naujas tenzorius išlaiko visus elementus,
bet juos perkonfigūruoja į matricą.
Atkreipkite dėmesį, kad mūsų vektoriaus elementai
išdėstomi po vieną eilutę ir todėl
`x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Atkreipkite dėmesį, kad nurodyti kiekvieną formos komponentą
`reshape` yra nereikalinga.
Kadangi jau žinome savo tenzoriaus dydį,
galime apskaičiuoti vieną formos komponentą pagal likusius.
Pavyzdžiui, kai duotas tenzorius, kurio dydis $n$,
ir tikslinė forma ($h$, $w$),
žinome, kad $w = n/h$.
Kad vienas formos komponentas būtų išvestas automatiškai,
galime įrašyti `-1` vietoj formos komponento,
kuris turi būti išvestas automatiškai.
Mūsų atveju užuot iškvietę `x.reshape(3, 4)`,
galėjome lygiai taip pat iškviesti `x.reshape(-1, 4)` arba `x.reshape(3, -1)`.

Praktikams dažnai tenka dirbti su tenzoriais,
inicializuotais taip, kad juos sudarytų vien 0 arba 1.
[**Tenzorių, kurio visi elementai lygūs 0**] (~~arba vienetui~~)
ir kurio forma (2, 3, 4), galime sukonstruoti funkcija `zeros`.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

Panašiai galime sukurti tenzorių,
kurio visi elementai lygūs 1, iškviesdami `ones`.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

Dažnai norime
[**atsitiktinai (ir nepriklausomai) imti kiekvieną elementą**]
iš duoto tikimybių skirstinio.
Pavyzdžiui, neuroninių tinklų parametrai
dažnai inicializuojami atsitiktinai.
Šis fragmentas sukuria tenzorių,
kurio elementai imami iš
standartinio Gauso (normaliojo) skirstinio,
kurio vidurkis 0, o standartinis nuokrypis 1.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

Galiausiai tenzorius galime sukonstruoti
[**nurodydami tikslią kiekvieno elemento reikšmę**],
pateikdami (galbūt įdėtuosius) Python sąrašus,
kuriuose yra skaitinių literalų.
Čia konstruojame matricą iš sąrašų sąrašo,
kur išorinis sąrašas atitinka 0 ašį,
o vidinis sąrašas – 1 ašį.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indeksavimas ir atpjovimas

Kaip ir Python sąrašų atveju,
tenzoriaus elementus galime pasiekti
indeksuodami (pradedant nuo 0).
Norėdami pasiekti elementą pagal jo padėtį
sąrašo pabaigos atžvilgiu,
galime naudoti neigiamus indeksus.
Galiausiai galime pasiekti ištisus indeksų intervalus
atpjaudami (angl. *slicing*) (pvz., `X[start:stop]`),
kur grąžinamoje reikšmėje yra
pirmasis indeksas (`start`), *bet ne paskutinis* (`stop`).
Galiausiai, kai $k$-osios eilės tenzoriui
nurodomas tik vienas indeksas (ar atpjovimas),
jis taikomas 0 ašimi.
Taigi toliau pateiktame kode
[**`[-1]` pasirenka paskutinę eilutę, o `[1:3]` –
antrąją ir trečiąją eilutes**].

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Be skaitymo, (**matricos elementus galime ir *rašyti*, nurodydami indeksus.**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow `Tensors` yra nekeičiami ir jiems negalima priskirti reikšmių.
TensorFlow `Variables` yra keičiami būsenos konteineriai, kurie palaiko
priskyrimą. Turėkite omenyje, kad TensorFlow gradientai neplinta atgal
per `Variable` priskyrimus.

Be reikšmės priskyrimo visam `Variable`, galime rašyti
`Variable` elementus nurodydami indeksus.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

Jei norime [**keliems elementams priskirti tą pačią reikšmę,
indeksavimą taikome kairėje priskyrimo operacijos pusėje.**]
Pavyzdžiui, `[:2, :]` pasiekia
pirmąją ir antrąją eilutes,
kur `:` paima visus elementus 1 ašimi (stulpeliu).
Nors aptarėme matricų indeksavimą,
tai taip pat veikia vektoriams
ir daugiau nei dviejų matmenų tenzoriams.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## Operacijos

Dabar, kai žinome, kaip konstruoti tenzorius
ir kaip skaityti jų elementus bei į juos rašyti,
galime pradėti juos apdoroti
įvairiomis matematinėmis operacijomis.
Viena naudingiausių jų grupių
yra *poelementės* operacijos.
Jos kiekvienam tenzoriaus elementui
taiko standartinę skaliarinę operaciją.
Funkcijoms, priimančioms du tenzorius kaip įvestis,
poelementės operacijos taiko kokį nors standartinį dvinarį operatorių
kiekvienai atitinkamų elementų porai.
Poelementę funkciją galime sukurti
iš bet kurios funkcijos, atvaizduojančios
skaliarą į skaliarą.

Matematiniame žymėjime tokius
*vienanarius* skaliarinius operatorius (turinčius vieną įvestį)
žymime parašu
$f: \mathbb{R} \rightarrow \mathbb{R}$.
Tai tiesiog reiškia, kad funkcija atvaizduoja
bet kurį realųjį skaičių į kokį nors kitą realųjį skaičių.
Daugumą standartinių operatorių, įskaitant vienanarius, tokius kaip $e^x$, galima taikyti poelemenčiui.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

Panašiai *dvinarius* skaliarinius operatorius,
kurie atvaizduoja realiųjų skaičių poras
į (vieną) realųjį skaičių,
žymime parašu
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Kai duoti bet kurie du *vienodos formos* vektoriai $\mathbf{u}$
ir $\mathbf{v}$ bei dvinaris operatorius $f$, galime sukurti vektorių
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$,
visiems $i$ nustatydami $c_i \gets f(u_i, v_i)$,
kur $c_i, u_i$ ir $v_i$ yra vektorių $\mathbf{c}, \mathbf{u}$ ir $\mathbf{v}$
$i$-ieji elementai.
Čia sukūrėme vektorinę reikšmę turinčią funkciją
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$,
skaliarinę funkciją *pakeldami*
į poelementę vektorinę operaciją.
Įprasti standartiniai aritmetiniai operatoriai:
sudėtis (`+`), atimtis (`-`),
daugyba (`*`), dalyba (`/`)
ir kėlimas laipsniu (`**`)
visi buvo *pakelti* į poelementes operacijas
vienodos formos bet kokios formos tenzoriams.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

Be poelementių skaičiavimų,
galime atlikti ir tiesinės algebros operacijas,
tokias kaip skaliarinės sandaugos ir matricų daugyba.
Jas išsamiau aptarsime
:numref:`sec_linear-algebra`.

Taip pat galime [***sujungti* kelis tenzorius,**]
sudėdami juos vieną po kito į didesnį.
Reikia tik pateikti tenzorių sąrašą
ir pasakyti sistemai, kuria ašimi sujungti.
Toliau pateiktas pavyzdys rodo, kas atsitinka sujungus
dvi matricas pagal eilutes (0 ašis),
o ne pagal stulpelius (1 ašis).
Matome, kad pirmosios išvesties 0 ašies ilgis ($6$)
yra dviejų įvesties tenzorių 0 ašies ilgių suma ($3 + 3$),
o antrosios išvesties 1 ašies ilgis ($8$)
yra dviejų įvesties tenzorių 1 ašies ilgių suma ($4 + 4$).

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

Kartais norime
[**sukonstruoti dvejetainį tenzorių pagal *logines išraiškas*.**]
Paimkime `X == Y` kaip pavyzdį.
Kiekvienai pozicijai `i, j`, jei `X[i, j]` ir `Y[i, j]` lygūs,
atitinkamas rezultato įrašas įgyja reikšmę `1`,
priešingu atveju – reikšmę `0`.

```{.python .input}
%%tab all
X == Y
```

[**Susumavus visus tenzoriaus elementus**], gaunamas tenzorius, turintis tik vieną elementą.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Transliavimas
:label:`subsec_broadcasting`

Dabar jau žinote, kaip atlikti
poelementes dvinares operacijas
su dviem vienodos formos tenzoriais.
Tam tikromis sąlygomis,
net kai formos skiriasi,
vis tiek galime [**atlikti poelementes dvinares operacijas
iškviesdami *transliavimo mechanizmą* (angl. *broadcasting*).**]
Transliavimas veikia pagal
šią dviejų žingsnių procedūrą:
(i) išplečiame vieną ar abu masyvus
kopijuodami elementus ašimis, kurių ilgis 1,
kad po šios transformacijos
du tenzoriai turėtų tą pačią formą;
(ii) atliekame poelementę operaciją
su gautais masyvais.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

Kadangi `a` ir `b` yra atitinkamai $3\times1$
ir $1\times2$ matricos,
jų formos nesutampa.
Transliavimas sukuria didesnę $3\times2$ matricą,
replikuodamas matricą `a` stulpeliais
ir matricą `b` eilutėmis
prieš sudedant jas poelemenčiui.

```{.python .input}
%%tab all
a + b
```

## Atminties taupymas

[**Vykdant operacijas, gali būti
paskirstoma nauja atmintis rezultatams laikyti.**]
Pavyzdžiui, jei rašome `Y = X + Y`,
panaikiname nuorodą į tenzorių, į kurį anksčiau rodė `Y`,
ir vietoj to nukreipiame `Y` į naujai paskirstytą atmintį.
Šią problemą galime pademonstruoti naudodami Python funkciją `id()`,
kuri pateikia tikslų nuorodos objekto
adresą atmintyje.
Atkreipkite dėmesį, kad po `Y = Y + X`
`id(Y)` rodo į kitą vietą.
Taip yra todėl, kad Python pirmiausia įvertina `Y + X`,
paskirstydamas naują atmintį rezultatui,
ir tada nukreipia `Y` į šią naują vietą atmintyje.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Tai gali būti nepageidautina dėl dviejų priežasčių.
Pirma, nenorime be reikalo
nuolat skirstyti atminties.
Mašininiame mokymesi dažnai turime
šimtus megabaitų parametrų
ir atnaujiname visus juos kelis kartus per sekundę.
Kai tik įmanoma, norime atlikti šiuos atnaujinimus *vietoje* (angl. *in place*).
Antra, į tuos pačius parametrus galime rodyti
iš kelių kintamųjų.
Jei neatnaujiname vietoje,
turime pasirūpinti, kad būtų atnaujintos visos šios nuorodos,
kad neatsirastų atminties nutekėjimo
arba netyčia nebūtų naudojami pasenę parametrai.

:begin_tab:`mxnet, pytorch`
Laimei, (**operacijas atlikti vietoje**) nesunku.
Operacijos rezultatą galime priskirti
anksčiau paskirstytam masyvui `Y`
naudodami atpjovimo žymėjimą: `Y[:] = <išraiška>`.
Šiai sąvokai iliustruoti
perrašome tenzoriaus `Z` reikšmes
jį inicializavę, naudodami `zeros_like`,
kad turėtų tą pačią formą kaip `Y`.
:end_tab:

:begin_tab:`tensorflow`
`Variables` yra keičiami būsenos konteineriai TensorFlow. Jie suteikia
būdą saugoti jūsų modelio parametrus.
Operacijos rezultatą galime priskirti `Variable`
naudodami `assign`.
Šiai sąvokai iliustruoti
perrašome `Variable` `Z` reikšmes
jį inicializavę, naudodami `zeros_like`,
kad turėtų tą pačią formą kaip `Y`.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**Jei `X` reikšmė nenaudojama tolesniuose skaičiavimuose,
galime naudoti ir `X[:] = X + Y` arba `X += Y`,
kad sumažintume operacijos atminties pridėtines išlaidas.**]
:end_tab:

:begin_tab:`tensorflow`
Net kai būseną nuolat saugote `Variable`,
galbūt norėsite dar labiau sumažinti atminties naudojimą vengdami perteklinio
paskirstymo tenzoriams, kurie nėra jūsų modelio parametrai.
Kadangi TensorFlow `Tensors` yra nekeičiami
ir gradientai neplinta per `Variable` priskyrimus,
TensorFlow nesuteikia aiškaus būdo atlikti
atskirą operaciją vietoje.

Tačiau TensorFlow suteikia `tf.function` dekoratorių,
kad skaičiavimas būtų apgaubtas TensorFlow grafu,
kuris prieš vykdymą kompiliuojamas ir optimizuojamas.
Tai leidžia TensorFlow pašalinti nenaudojamas reikšmes
ir pakartotinai naudoti ankstesnius paskirstymus, kurių nebereikia.
Taip sumažinamos TensorFlow skaičiavimų atminties pridėtinės išlaidos.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Konvertavimas į kitus Python objektus

:begin_tab:`mxnet, tensorflow`
[**Konvertuoti į NumPy tenzorių (`ndarray`)**] ar atvirkščiai nesudėtinga.
Konvertuotas rezultatas nesidalija atmintimi.
Šis nedidelis nepatogumas iš tikrųjų gana svarbus:
kai atliekate operacijas su CPU ar GPU,
nenorite stabdyti skaičiavimo laukdami, kol sužinosite,
ar Python paketas NumPy
nenori daryti kažko kito
su tuo pačiu atminties gabalu.
:end_tab:

:begin_tab:`pytorch`
[**Konvertuoti į NumPy tenzorių (`ndarray`)**] ar atvirkščiai nesudėtinga.
Torch tenzorius ir NumPy masyvas
dalysis pagrindine atmintimi,
todėl pakeitus vieną operacija vietoje,
pasikeis ir kitas.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

Norėdami (**konvertuoti vieno elemento tenzorių į Python skaliarą**),
galime iškviesti funkciją `item` arba Python integruotąsias funkcijas.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Santrauka

Tenzoriaus klasė yra pagrindinė sąsaja duomenims saugoti ir apdoroti giliojo mokymosi bibliotekose.
Tenzoriai suteikia įvairių galimybių, įskaitant kūrimo procedūras, indeksavimą ir atpjovimą, pagrindines matematines operacijas, transliavimą, atmintį tausojantį priskyrimą ir konvertavimą į kitus Python objektus ir iš jų.


## Pratimai

1. Paleiskite šio skyrelio kodą. Pakeiskite sąlygos sakinį `X == Y` į `X < Y` arba `X > Y` ir pažiūrėkite, kokį tenzorių gausite.
1. Pakeiskite du tenzorius, kurie transliavimo mechanizme operuoja pagal elementus, kitų formų tenzoriais, pvz., trimačiais. Ar rezultatas toks, kokio tikėjotės?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17966)
:end_tab:
