```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Sintetiniai regresijos duomenys
:label:`sec_synthetic-regression-data`


Mašininis mokymasis yra visas apie informacijos išgavimą iš duomenų.
Todėl galite pagalvoti, ko galėtume išmokti iš sintetinių duomenų?
Nors mums gali nerūpėti pačios savaime dėsningumai,
kuriuos patys įdiegėme į dirbtinį duomenis generuojantį modelį,
tokie duomenų rinkiniai vis tiek naudingi didaktikos tikslais:
jie padeda įvertinti mūsų mokymosi
algoritmų savybes ir patvirtinti, kad mūsų įgyvendinimai veikia kaip tikėtasi.
Pavyzdžiui, jei sukuriame duomenis, kurių teisingi parametrai žinomi *a priori*,
tada galime patikrinti, ar mūsų modelis iš tikrųjų gali juos atkurti.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## Duomenų rinkinio generavimas

Šiame pavyzdyje dėl glaustumo dirbsime mažoje dimensijoje.
Šis kodo fragmentas sugeneruoja 1000 pavyzdžių
su 2 dimensijų požymiais, paimtais
iš standartinio normaliojo skirstinio.
Gauta požymių matrica $\mathbf{X}$
priklauso $\mathbb{R}^{1000 \times 2}$.
Kiekvieną žymenį generuojame taikydami
*tikrąją* tiesinę funkciją
ir iškraipydami juos adityviuoju triukšmu $\boldsymbol{\epsilon}$,
nepriklausomai ir identiškai paimtu kiekvienam pavyzdžiui:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

Patogumo dėlei darome prielaidą, kad $\boldsymbol{\epsilon}$ paimtas
iš normaliojo skirstinio su vidurkiu $\mu= 0$
ir standartiniu nuokrypiu $\sigma = 0.01$.
Atkreipkite dėmesį, kad objektiniam projektavimui
kodą pridedame prie poklasio `d2l.DataModule` (pristatyto :numref:`oo-design-data`) metodo `__init__`.
Gera praktika leisti nustatyti bet kokius papildomus hiperparametrus.
Tai pasiekiame naudodami `save_hyperparameters()`.
`batch_size` bus nustatytas vėliau.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Toliau nustatome tikruosius parametrus $\mathbf{w} = [2, -3.4]^\top$ ir $b = 4.2$.
Vėliau galėsime palyginti savo įvertintus parametrus su šiomis *tikrosiomis* reikšmėmis.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Kiekviena `features` eilutė yra vektorius iš $\mathbb{R}^2$, o kiekviena `labels` eilutė yra skaliaras.**] Pažvelkime į pirmąjį įrašą.

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Duomenų rinkinio skaitymas

Mašininio mokymosi modelių apmokymui dažnai reikia kelis kartus peržiūrėti duomenų rinkinį,
paimant po vieną pavyzdžių minirinkinį.
Tada šie duomenys naudojami modeliui atnaujinti.
Kad parodytume, kaip tai veikia,
[**įgyvendiname metodą `get_dataloader`**],
užregistruodami jį klasėje `SyntheticRegressionData` per `add_to_class` (pristatytą :numref:`oo-design-utilities`).
Jis (**priima minirinkinio dydį, požymių matricą
ir žymenų vektorių ir sugeneruoja `batch_size` dydžio minirinkinius.**)
Taigi kiekvienas minirinkinys yra požymių ir žymenų kortežas.
Atkreipkite dėmesį, kad turime atsižvelgti į tai, ar esame apmokymo, ar validavimo režime:
pirmuoju atveju norėsime skaityti duomenis atsitiktine tvarka,
o antruoju atveju galimybė skaityti duomenis iš anksto apibrėžta tvarka
gali būti svarbi derinimo tikslais.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

Kad įgytume šiek tiek intuicijos, apžiūrėkime pirmąjį duomenų
minirinkinį. Kiekvienas požymių minirinkinys pateikia ir jo dydį, ir įvesties požymių dimensiją.
Panašiai mūsų žymenų minirinkinys turės atitinkamą formą, nurodytą `batch_size`.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Nors atrodo nekaltas, `iter(data.train_dataloader())`
iškvietimas
iliustruoja Python objektinio projektavimo galią.
Atkreipkite dėmesį, kad pridėjome metodą klasei `SyntheticRegressionData`
*po to*, kai sukūrėme objektą `data`.
Vis dėlto objektas gauna naudos iš
*ex post facto* (vėliau) pridėto klasės funkcionalumo.

Iteruojant gauname skirtingus minirinkinius,
kol išnaudojamas visas duomenų rinkinys (išbandykite tai).
Nors aukščiau įgyvendinta iteracija tinka didaktikos tikslams,
ji yra neefektyvi tokiais būdais, kurie gali sukelti problemų su realiais uždaviniais.
Pavyzdžiui, ji reikalauja, kad visus duomenis įkeltume į atmintį
ir atliktume daug atsitiktinių atminties prieigų.
Giliojo mokymosi karkase įgyvendinti įtaisytieji iteratoriai
yra gerokai efektyvesni ir gali dirbti
su tokiais šaltiniais kaip failuose saugomi duomenys,
srautu gaunami duomenys
ir duomenys, generuojami ar apdorojami skrydžio metu.
Toliau pabandykime įgyvendinti tą patį metodą naudodami įtaisytuosius iteratorius.

## Glaustas duomenų krautuvo įgyvendinimas

Užuot rašę savo iteratorių,
galime [**iškviesti esamą karkaso API duomenims įkelti.**]
Kaip ir anksčiau, mums reikia duomenų rinkinio su požymiais `X` ir žymenimis `y`.
Be to, įtaisytajame duomenų krautuve nustatome `batch_size`
ir leidžiame jam efektyviai pasirūpinti pavyzdžių maišymu.

:begin_tab:`jax`
JAX yra visas apie į NumPy panašią API su įrenginių spartinimu ir funkcinėmis
transformacijomis, todėl bent jau dabartinėje versijoje nėra duomenų įkėlimo
metodų. Su kitomis bibliotekomis jau turime puikių duomenų krautuvų,
ir JAX siūlo naudoti jas. Čia pasiimsime TensorFlow duomenų krautuvą
ir šiek tiek jį modifikuosime, kad jis veiktų su JAX.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

Naujasis duomenų krautuvas elgiasi lygiai kaip ankstesnis, tik yra efektyvesnis ir turi papildomo funkcionalumo.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Pavyzdžiui, karkaso API pateiktas duomenų krautuvas
palaiko įtaisytąjį metodą `__len__`,
todėl galime užklausti jo ilgio,
t. y. minirinkinių skaičiaus.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## Santrauka

Duomenų krautuvai yra patogus būdas abstrahuoti
duomenų įkėlimo ir manipuliavimo jais procesą.
Taip tas pats mašininio mokymosi *algoritmas*
gali apdoroti daugelį skirtingų tipų ir šaltinių duomenų,
jo nekeičiant.
Vienas gerų duomenų krautuvų dalykų yra tas,
kad juos galima komponuoti.
Pavyzdžiui, galime įkelti paveikslėlius
ir turėti paskesnio apdorojimo filtrą,
kuris juos apkarpo arba kitaip modifikuoja.
Taigi duomenų krautuvai gali būti naudojami
visam duomenų apdorojimo konvejeriui aprašyti.

Kalbant apie patį modelį, dvimatis tiesinis modelis
yra beveik paprasčiausias, su kuriuo galime susidurti.
Jis leidžia išbandyti regresijos modelių tikslumą,
nesirūpinant, kad duomenų per mažai
arba lygčių sistema yra nepakankamai apibrėžta.
Tai gerai panaudosime kitame skyrelyje.


## Pratimai

1. Kas nutiks, jei pavyzdžių skaičiaus negalima padalyti iš minirinkinio dydžio. Kaip pakeistumėte šį elgesį, nurodydami kitą argumentą naudodami karkaso API?
1. Tarkime, kad norime sugeneruoti milžinišką duomenų rinkinį, kuriame ir parametrų vektoriaus `w` dydis, ir pavyzdžių skaičius `num_examples` yra dideli.
    1. Kas nutinka, jei negalime visų duomenų laikyti atmintyje?
    1. Kaip sumaišytumėte duomenis, jei jie laikomi diske? Jūsų užduotis – sukurti *efektyvų* algoritmą, kuriam nereikia per daug atsitiktinių skaitymų ar rašymų. Užuomina: [pseudoatsitiktinių permutacijų generatoriai](https://en.wikipedia.org/wiki/Pseudorandom_permutation) leidžia sukurti permaišymą, aiškiai nesaugant permutacijų lentelės :cite:`Naor.Reingold.1999`.
1. Įgyvendinkite duomenų generatorių, kuris kuria naujus duomenis skrydžio metu kiekvieną kartą, kai iškviečiamas iteratorius.
1. Kaip sukurtumėte atsitiktinių duomenų generatorių, kuris kiekvieną kartą iškviestas sugeneruoja *tuos pačius* duomenis?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17975)
:end_tab:
