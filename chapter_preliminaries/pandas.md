```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Duomenų išankstinis apdorojimas
:label:`sec_pandas`

Iki šiol dirbome su sintetiniais duomenimis,
kurie buvo pateikti jau paruoštuose tenzoriuose.
Tačiau norint taikyti gilųjį mokymąsi realiame pasaulyje,
turime išgauti netvarkingus duomenis,
saugomus savavališkais formatais,
ir juos iš anksto apdoroti pagal savo poreikius.
Laimei, *pandas* [biblioteka](https://pandas.pydata.org/)
gali atlikti didžiąją dalį sunkaus darbo.
Šis skyrelis, nors ir neatstoja tinkamo *pandas* [vadovo](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html),
suteiks jums greitąjį kursą
apie kai kurias dažniausias procedūras.

## Duomenų rinkinio skaitymas

Kableliais atskirtų reikšmių (angl. *comma-separated values*, CSV) failai yra plačiai paplitę
lentelinių (skaičiuoklės tipo) duomenų saugojimui.
Juose kiekviena eilutė atitinka vieną įrašą
ir susideda iš kelių (kableliais atskirtų) laukų, pvz.,
„Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics".
Norėdami pademonstruoti, kaip įkelti CSV failus naudojant `pandas`,
(**toliau sukuriame CSV failą**) `../data/house_tiny.csv`.
Šis failas vaizduoja namų duomenų rinkinį,
kuriame kiekviena eilutė atitinka atskirą namą,
o stulpeliai – kambarių skaičių (`NumRooms`),
stogo tipą (`RoofType`) ir kainą (`Price`).

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Dabar importuokime `pandas` ir įkelkime duomenų rinkinį naudodami `read_csv`.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Duomenų paruošimas

Prižiūrimame mokyme apmokome modelius
prognozuoti nurodytą *tikslinę* reikšmę,
kai duotas tam tikras *įvesties* reikšmių rinkinys.
Pirmasis mūsų žingsnis apdorojant duomenų rinkinį
yra atskirti stulpelius, atitinkančius
įvesties ir tikslines reikšmes.
Stulpelius galime pasirinkti pagal pavadinimą arba
naudodami indeksavimą pagal sveikąją padėtį (`iloc`).

Galbūt pastebėjote, kad `pandas` pakeitė
visus CSV įrašus su reikšme `NA`
speciale `NaN` (angl. *not a number*, „ne skaičius") reikšme.
Tai taip pat gali nutikti, kai įrašas tuščias,
pvz., „3,,,270000".
Jos vadinamos *trūkstamomis reikšmėmis*
ir yra duomenų mokslo „blakės" –
nuolatinė grėsmė, su kuria susidursite
visos savo karjeros metu.
Priklausomai nuo konteksto,
trūkstamos reikšmės gali būti tvarkomos
arba *imputavimu*, arba *šalinimu*.
Imputavimas pakeičia trūkstamas reikšmes
jų reikšmių įverčiais,
o šalinimas tiesiog atmeta
tas eilutes arba tuos stulpelius,
kuriuose yra trūkstamų reikšmių.

Štai keletas dažnų imputavimo euristikų.
[**Kategoriniams įvesties laukams
`NaN` galime traktuoti kaip kategoriją.**]
Kadangi stulpelis `RoofType` įgyja reikšmes `Slate` ir `NaN`,
`pandas` gali konvertuoti šį stulpelį
į du stulpelius `RoofType_Slate` ir `RoofType_nan`.
Eilutėje, kurios stogo tipas yra `Slate`,
`RoofType_Slate` ir `RoofType_nan` reikšmės bus atitinkamai 1 ir 0.
Priešingai yra eilutei, kurios `RoofType` reikšmė trūksta.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Trūkstamoms skaitinėms reikšmėms
viena dažna euristika yra
[**pakeisti `NaN` įrašus
atitinkamo stulpelio vidutine reikšme**].

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Konvertavimas į tenzoriaus formatą

Dabar, kai [**visi `inputs` ir `targets` įrašai yra skaitiniai,
galime juos įkelti į tenzorių**] (prisiminkite :numref:`sec_ndarray`).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Diskusija

Dabar žinote, kaip padalyti duomenų stulpelius,
imputuoti trūkstamus kintamuosius
ir įkelti `pandas` duomenis į tenzorius.
:numref:`sec_kaggle_house` įgysite
dar keletą duomenų apdorojimo įgūdžių.
Nors šiame greitajame kurse viską supaprastinome,
duomenų apdorojimas gali tapti sudėtingas.
Pavyzdžiui, užuot buvę viename CSV faile,
mūsų duomenys gali būti išsklaidyti po kelis failus,
išgautus iš reliacinės duomenų bazės.
Pavyzdžiui, elektroninės prekybos taikyme
klientų adresai gali būti vienoje lentelėje,
o pirkimų duomenys – kitoje.
Be to, praktikai susiduria su daugybe duomenų tipų,
be kategorinių ir skaitinių, pavyzdžiui,
teksto eilutėmis, vaizdais,
garso duomenimis ir taškų debesimis.
Dažnai reikia pažangių įrankių ir efektyvių algoritmų,
kad duomenų apdorojimas netaptų
didžiausiu mašininio mokymosi proceso kliuviniu.
Šios problemos iškils, kai pereisime prie
kompiuterinės regos ir natūraliosios kalbos apdorojimo.
Galiausiai turime atkreipti dėmesį į duomenų kokybę.
Realaus pasaulio duomenų rinkiniuose dažnai pasitaiko
išskirčių, klaidingų jutiklių matavimų ir registravimo klaidų,
kurias reikia pašalinti prieš
pateikiant duomenis bet kuriam modeliui.
Duomenų vizualizavimo įrankiai, tokie kaip [seaborn](https://seaborn.pydata.org/),
[Bokeh](https://docs.bokeh.org/) ar [matplotlib](https://matplotlib.org/),
gali padėti jums rankiniu būdu peržiūrėti duomenis
ir susidaryti intuiciją apie
problemų, kurias gali tekti spręsti, tipą.


## Pratimai

1. Pabandykite įkelti duomenų rinkinius, pvz., Abalone iš [UCI mašininio mokymosi saugyklos](https://archive.ics.uci.edu/ml/datasets), ir ištirkite jų savybes. Kokia jų dalis turi trūkstamų reikšmių? Kokia kintamųjų dalis yra skaitiniai, kategoriniai ar tekstiniai?
1. Pabandykite indeksuoti ir pasirinkti duomenų stulpelius pagal pavadinimą, o ne pagal stulpelio numerį. Pandas dokumentacija apie [indeksavimą](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) pateikia daugiau informacijos, kaip tai padaryti.
1. Kokio dydžio duomenų rinkinį, jūsų manymu, galėtumėte taip įkelti? Kokie gali būti apribojimai? Užuomina: apsvarstykite duomenų skaitymo laiką, vaizdavimą, apdorojimą ir atminties pėdsaką. Išbandykite tai savo nešiojamajame kompiuteryje. Kas nutinka, jei tai išbandote serveryje?
1. Kaip elgtumėtės su duomenimis, turinčiais labai daug kategorijų? O jei visos kategorijų žymės unikalios? Ar pastarąsias reikėtų įtraukti?
1. Kokių pandas alternatyvų galite sugalvoti? O kaip [NumPy tenzorių įkėlimas iš failo](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Susipažinkite su [Pillow](https://python-pillow.org/), Python vaizdų biblioteka.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17967)
:end_tab:
