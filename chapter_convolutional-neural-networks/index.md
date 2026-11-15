# Konvoliuciniai neuroniniai tinklai
:label:`chap_cnn`

Vaizdo duomenys vaizduojami kaip dvimatis pikselių tinklelis, nesvarbu, ar vaizdas
vienspalvis, ar spalvotas. Atitinkamai kiekvienas pikselis atitinka vieną
arba kelias skaitines reikšmes. Iki šiol ignoravome šią turtingą
struktūrą ir traktavome vaizdus kaip skaičių vektorius, juos *ištiesindami*, nepaisydami erdvinio ryšio tarp pikselių. Šis
labai nepatenkinamas metodas buvo būtinas, kad gautus vienmačius
vektorius galėtume perduoti per pilnai sujungtą MLP.

Kadangi šie tinklai yra nejautrūs požymių tvarkai, galėtume
gauti panašius rezultatus nepriklausomai nuo to, ar išsaugome tvarką,
atitinkančią pikselių erdvinę struktūrą, ar permutuojame
savo plano matricos stulpelius prieš pritaikydami MLP parametrus.
Idealiu atveju pasinaudotume savo išankstine žinia, kad netoliese esantys pikseliai
paprastai yra susiję vienas su kitu, kad sukurtume efektyvius modelius
mokymuisi iš vaizdo duomenų.

Šiame skyriuje pristatomi *konvoliuciniai neuroniniai tinklai* (CNN)
:cite:`LeCun.Jackel.Bottou.ea.1995` – galinga neuroninių tinklų šeima,
sukurta būtent šiam tikslui.
Architektūros, pagrįstos CNN, dabar yra visur kompiuterinio matymo srityje.
Pavyzdžiui, ImageNet rinkinyje
:cite:`Deng.Dong.Socher.ea.2009` tik konvoliucinių neuroninių
tinklų (trumpai – konvoliucinių tinklų) naudojimas suteikė reikšmingą našumo
pagerėjimą :cite:`Krizhevsky.Sutskever.Hinton.2012`.

Šiuolaikiniai CNN, kaip jie šnekamojoje kalboje vadinami, savo dizainu skolingi
biologijos, grupių teorijos įkvėpimui ir sveikai eksperimentinio
gudravimo dozei. Be savo imčių efektyvumo,
pasiekiant tikslius modelius, CNN paprastai yra skaičiavimo požiūriu efektyvūs,
nes jiems reikia mažiau parametrų nei pilnai sujungtoms
architektūroms ir nes konvoliucijas lengva lygiagretinti per
GPU branduolius :cite:`Chetlur.Woolley.Vandermersch.ea.2014`. Todėl specialistai dažnai
taiko CNN, kai tik įmanoma, ir vis dažniau jie tampa
patikimais konkurentais net užduotyse su vienmate seka
struktūra, tokiose kaip garsas :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, tekstas
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014` ir laiko eilučių analizė
:cite:`LeCun.Bengio.ea.1995`, kur tradiciškai naudojami rekurentiniai neuroniniai tinklai.
Kai kurie gudrūs CNN pritaikymai taip pat leido juos taikyti grafo struktūros duomenims :cite:`Kipf.Welling.2016` ir
rekomendacinėse sistemose.

Pirmiausia giliau panagrinėsime konvoliucinių
neuroninių tinklų motyvaciją. Po to apžvelgsime pagrindines operacijas,
sudarančias visų konvoliucinių tinklų pagrindą.
Jos apima pačius konvoliucinius sluoksnius,
smulkias detales, įskaitant papildymą ir žingsnį,
sutelkimo sluoksnius, naudojamus informacijai
iš gretimų erdvinių sričių agreguoti,
kelių kanalų naudojimą kiekviename sluoksnyje
ir kruopštų šiuolaikinių architektūrų struktūros aptarimą.
Skyrių užbaigsime pilnu veikiančiu LeNet pavyzdžiu –
pirmuoju sėkmingai pritaikytu konvoliuciniu tinklu,
gerokai prieš šiuolaikinio giliojo mokymosi iškilimą.
Kitame skyriuje gilinsimės į pilnus kai kurių populiarių ir palyginti naujų
CNN architektūrų įgyvendinimus,
kurių dizainas atspindi daugumą technikų,
dažniausiai naudojamų šiuolaikinių specialistų.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
