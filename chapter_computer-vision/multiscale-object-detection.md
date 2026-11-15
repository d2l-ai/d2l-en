# Daugiamastelis objektų aptikimas
:label:`sec_multiscale-object-detection`


:numref:`sec_anchor`
sugeneravome kelis inkarinius stačiakampius, kurių centras yra kiekvienas įvesties vaizdo pikselis.
Iš esmės šie inkariniai stačiakampiai
atvaizduoja
skirtingų vaizdo sričių imtis.
Tačiau
jei jie generuojami *kiekvienam* pikseliui, galime gauti per daug inkarinių stačiakampių, kuriuos reikia skaičiuoti.
Pagalvokite apie $561 \times 728$ dydžio įvesties vaizdą.
Jei kiekvienam pikseliui kaip centrui
sugeneruojami penki skirtingos formos
inkariniai stačiakampiai,
vaizde reikės pažymėti ir prognozuoti daugiau nei du milijonus inkarinių stačiakampių ($561 \times 728 \times 5$).

## Daugiamasteliai inkariniai stačiakampiai
:label:`subsec_multiscale-anchor-boxes`

Galbūt pastebėjote, kad
nesunku sumažinti vaizde esančių inkarinių stačiakampių skaičių.
Pavyzdžiui, galime tiesiog
tolygiai atrinkti nedidelę pikselių dalį
iš įvesties vaizdo
ir sugeneruoti inkarinius stačiakampius, kurių centrai yra šie pikseliai.
Be to,
skirtingais masteliais
galime generuoti skirtingą skirtingų dydžių inkarinių stačiakampių skaičių.
Intuityviai,
mažesni objektai vaizde
tikėtina, kad pasirodys dažniau nei didesni.
Pavyzdžiui,
$1 \times 1$, $1 \times 2$ ir $2 \times 2$ objektai
gali pasirodyti $2 \times 2$ vaizde
atitinkamai 4, 2 ir 1 galimais būdais.
Todėl naudodami mažesnius inkarinius stačiakampius mažesniems objektams aptikti, galime atrinkti daugiau sričių,
o didesniems objektams – mažiau sričių.

Kad parodytume, kaip generuoti inkarinius stačiakampius
keliais masteliais, nuskaitykime vaizdą.
Jo aukštis ir plotis yra atitinkamai 561 ir 728 pikseliai.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

Prisiminkite, kad :numref:`sec_conv_layer`
dvimatį konvoliucinio sluoksnio išvesties masyvą
pavadinome požymių žemėlapiu.
Apibrėždami požymių žemėlapio formą,
galime nustatyti bet kurio vaizdo tolygiai atrinktų inkarinių stačiakampių centrus.


Toliau apibrėžta funkcija `display_anchors`.
[**Generuojame inkarinius stačiakampius (`anchors`) požymių žemėlapyje (`fmap`), kiekvieną vienetą (pikselį) laikydami inkarinio stačiakampio centru.**]
Kadangi inkarinių stačiakampių (`anchors`) $(x, y)$ ašių koordinačių reikšmės
buvo padalytos iš požymių žemėlapio (`fmap`) pločio ir aukščio,
šios reikšmės yra tarp 0 ir 1,
o tai rodo santykines
inkarinių stačiakampių padėtis požymių žemėlapyje.

Kadangi inkarinių stačiakampių (`anchors`) centrai
pasiskirstę per visus požymių žemėlapio (`fmap`) vienetus,
šie centrai turi būti *tolygiai* pasiskirstę
bet kuriame įvesties vaizde
pagal jų santykines erdvines padėtis.
Konkrečiau,
turint požymių žemėlapio plotį ir aukštį atitinkamai `fmap_w` ir `fmap_h`,
ši funkcija *tolygiai* atrinks
pikselius `fmap_h` eilučių ir `fmap_w` stulpelių
bet kuriame įvesties vaizde.
Aplink šiuos tolygiai atrinktus pikselius
bus sugeneruoti
mastelio `s` (darant prielaidą, kad sąrašo `s` ilgis yra 1) ir skirtingų kraštinių santykių (`ratios`) inkariniai stačiakampiai.

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Pirmiausia [**panagrinėkime
mažų objektų aptikimą**].
Kad rodomus stačiakampius būtų lengviau atskirti, čia skirtingų centrų inkariniai stačiakampiai nepersidengia:
inkarinio stačiakampio mastelis nustatytas 0,15,
o požymių žemėlapio aukštis ir plotis – 4. Matome,
kad 4 eilučių ir 4 stulpelių inkarinių stačiakampių centrai vaizde pasiskirstę tolygiai.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Toliau [**perpus sumažiname požymių žemėlapio aukštį ir plotį ir naudojame didesnius inkarinius stačiakampius didesniems objektams aptikti**]. Kai mastelis nustatytas 0,4,
kai kurie inkariniai stačiakampiai persidengs.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Galiausiai [**dar perpus sumažiname požymių žemėlapio aukštį ir plotį ir padidiname inkarinio stačiakampio mastelį iki 0,8**]. Dabar inkarinio stačiakampio centras yra vaizdo centras.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Daugiamastelis aptikimas


Kadangi sugeneravome daugiamastelius inkarinius stačiakampius,
juos naudosime įvairaus dydžio objektams aptikti
skirtingais masteliais.
Toliau
pristatome daugiamastelį objektų aptikimo
metodą, pagrįstą CNN, kurį įgyvendinsime
:numref:`sec_ssd`.

Tam tikru masteliu
tarkime, kad turime $c$ požymių žemėlapių, kurių forma $h \times w$.
Naudodami :numref:`subsec_multiscale-anchor-boxes` metodą,
generuojame $hw$ inkarinių stačiakampių rinkinių,
kur kiekvieną rinkinį sudaro $a$ inkarinių stačiakampių su tuo pačiu centru.
Pavyzdžiui,
pirmuoju masteliu :numref:`subsec_multiscale-anchor-boxes` eksperimentuose,
turėdami dešimt (kanalų skaičius) $4 \times 4$ požymių žemėlapių,
sugeneravome 16 inkarinių stačiakampių rinkinių,
kur kiekviename rinkinyje yra 3 inkariniai stačiakampiai su tuo pačiu centru.
Toliau kiekvienas inkarinis stačiakampis pažymimas
klase ir poslinkiu pagal tikruosius apibrėžiamuosius stačiakampius. Dabartiniu masteliu objektų aptikimo modelis turi prognozuoti $hw$ inkarinių stačiakampių rinkinių klases ir poslinkius įvesties vaizde, kur skirtingi rinkiniai turi skirtingus centrus.


Tarkime, kad čia esantys $c$ požymių žemėlapių yra tarpinės išvestys, gautos
CNN tiesioginio sklidimo metu pagal įvesties vaizdą. Kadangi kiekviename požymių žemėlapyje yra $hw$ skirtingų erdvinių padėčių,
tą pačią erdvinę padėtį galima
laikyti turinčia $c$ vienetų.
Pagal
lokalios matymo srities apibrėžimą :numref:`sec_conv_layer`,
šie $c$ vienetų toje pačioje požymių žemėlapių
erdvinėje padėtyje
turi tą pačią lokalią matymo sritį įvesties vaizde:
jie vaizduoja įvesties vaizdo informaciją
toje pačioje lokalioje matymo srityje.
Todėl šiuos $c$ požymių žemėlapių vienetų
toje pačioje erdvinėje padėtyje
galime transformuoti į
$a$ inkarinių stačiakampių, sugeneruotų naudojant šią erdvinę padėtį,
klases ir poslinkius.
Iš esmės
naudojame tam tikroje lokalioje matymo srityje esančią įvesties vaizdo informaciją
prognozuoti inkarinių stačiakampių, esančių
arti tos lokalios matymo srities
įvesties vaizde, klases ir poslinkius.


Kai skirtingų sluoksnių požymių žemėlapiai
turi skirtingo dydžio lokalias matymo sritis įvesties vaizde, juos galima naudoti skirtingo dydžio objektams aptikti.
Pavyzdžiui, galime suprojektuoti neuroninį tinklą, kuriame
požymių žemėlapių vienetai, esantys arčiau išvesties sluoksnio,
turi platesnes lokalias matymo sritis,
todėl gali aptikti didesnius įvesties vaizdo objektus.

Trumpai tariant, galime pasinaudoti
giliųjų neuroninių tinklų
daugialygiais sluoksniniais vaizdų vaizdavimais
daugiamasteliam objektų aptikimui.
Kaip tai veikia, parodysime konkrečiu pavyzdžiu
:numref:`sec_ssd`.




## Santrauka

* Keliais masteliais galime generuoti skirtingo dydžio inkarinius stačiakampius, kad aptiktume skirtingo dydžio objektus.
* Apibrėždami požymių žemėlapių formą, galime nustatyti bet kurio vaizdo tolygiai atrinktų inkarinių stačiakampių centrus.
* Naudojame tam tikroje lokalioje matymo srityje esančią įvesties vaizdo informaciją, kad prognozuotume inkarinių stačiakampių, esančių arti tos lokalios matymo srities įvesties vaizde, klases ir poslinkius.
* Naudodami gilųjį mokymąsi, galime pasinaudoti jo daugialygiais sluoksniniais vaizdų vaizdavimais daugiamasteliam objektų aptikimui.


## Pratimai

1. Remiantis mūsų diskusijomis :numref:`sec_alexnet`, gilieji neuroniniai tinklai išmoksta hierarchinius vaizdų požymius su vis aukštesniu abstrakcijos lygiu. Ar daugiamasteliame objektų aptikime skirtingų mastelių požymių žemėlapiai atitinka skirtingus abstrakcijos lygius? Kodėl taip arba kodėl ne?
1. Pirmuoju masteliu (`fmap_w=4, fmap_h=4`) :numref:`subsec_multiscale-anchor-boxes` eksperimentuose sugeneruokite tolygiai pasiskirsčiusius inkarinius stačiakampius, kurie gali persidengti.
1. Duotas požymių žemėlapio kintamasis, kurio forma $1 \times c \times h \times w$, kur $c$, $h$ ir $w$ yra atitinkamai požymių žemėlapių kanalų skaičius, aukštis ir plotis. Kaip galite transformuoti šį kintamąjį į inkarinių stačiakampių klases ir poslinkius? Kokia yra išvesties forma?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1607)
:end_tab:
