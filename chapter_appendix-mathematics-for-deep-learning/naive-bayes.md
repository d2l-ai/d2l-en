# Naivusis Bajesas
:label:`sec_naive_bayes`

Ankstesniuose skyriuose susipažinome su tikimybių teorija ir atsitiktiniais dydžiais.  Kad šią teoriją pritaikytume praktikoje, pristatykime *naivųjį Bajeso* klasifikatorių.  Jis naudoja vien tikimybinius pagrindus ir leidžia klasifikuoti skaitmenis.

Mokymasis – tai visų pirma prielaidų darymas. Jei norime klasifikuoti naują, anksčiau nematytą duomenų pavyzdį, turime daryti tam tikras prielaidas apie tai, kurie duomenų pavyzdžiai yra panašūs vienas į kitą. Naivusis Bajeso klasifikatorius – populiarus ir stebėtinai aiškus algoritmas – skaičiavimams supaprastinti daro prielaidą, kad visi požymiai yra nepriklausomi vienas nuo kito. Šiame skyriuje šį modelį pritaikysime simboliams paveikslėliuose atpažinti.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Optinis simbolių atpažinimas

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` yra vienas plačiai naudojamų duomenų rinkinių. Jame yra 60 000 mokymo paveikslėlių ir 10 000 validavimo paveikslėlių. Kiekviename paveikslėlyje pavaizduotas ranka rašytas skaitmuo nuo 0 iki 9. Uždavinys – kiekvieną paveikslėlį priskirti atitinkamam skaitmeniui.

Gluon modulyje `data.vision` yra klasė `MNIST`, kuri
automatiškai gauna duomenų rinkinį iš interneto.
Vėliau Gluon naudos jau atsisiųstą vietinę kopiją.
Nurodome, ar prašome mokymo, ar testavimo rinkinio,
parametrui `train` atitinkamai priskirdami reikšmę `True` arba `False`.
Kiekvienas paveikslėlis yra pilkų atspalvių, kurio plotis ir aukštis yra $28$, o forma ($28$,$28$,$1$). Naudojame pritaikytą transformaciją, kad pašalintume paskutinę kanalo dimensiją. Be to, duomenų rinkinyje kiekvienas pikselis pavaizduotas $8$ bitų sveikuoju skaičiumi be ženklo.  Problemai supaprastinti juos kvantizuojame į dvejetainius požymius.

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

Galime pasiekti konkretų pavyzdį, kuriame yra paveikslėlis ir atitinkama žymė.

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Mūsų pavyzdys, čia saugomas kintamajame `image`, atitinka paveikslėlį, kurio aukštis ir plotis yra $28$ pikseliai.

```{.python .input}
#@tab all
image.shape, image.dtype
```

Mūsų kodas kiekvieno paveikslėlio žymę saugo kaip skaliarą. Jo tipas yra $32$ bitų sveikasis skaičius.

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

Taip pat galime vienu metu pasiekti kelis pavyzdžius.

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Pavaizduokime šiuos pavyzdžius.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## Tikimybinis klasifikavimo modelis

Klasifikavimo užduotyje pavyzdį atvaizduojame į kategoriją. Čia pavyzdys yra pilkų atspalvių $28\times 28$ paveikslėlis, o kategorija – skaitmuo. (Išsamesnį paaiškinimą žr. :numref:`sec_softmax`.)
Vienas natūralus būdas išreikšti klasifikavimo užduotį yra tikimybinis klausimas: kokia yra tikriausia žymė, žinant požymius (t. y. paveikslėlio pikselius)? Pažymėkime $\mathbf x\in\mathbb R^d$ pavyzdžio požymius, o $y\in\mathbb R$ – žymę. Čia požymiai yra paveikslėlio pikseliai, o $2$ dimensijų paveikslėlį galime pakeisti vektoriumi, kad $d=28^2=784$, o žymės yra skaitmenys.
Žymės tikimybė, žinant požymius, yra $p(y  \mid  \mathbf{x})$. Jei galime apskaičiuoti šias tikimybes, mūsų pavyzdyje tai $p(y  \mid  \mathbf{x})$, kai $y=0, \ldots,9$, tada klasifikatorius pateiks prognozę $\hat{y}$, pateikiamą išraiška:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Deja, tam reikia įvertinti $p(y  \mid  \mathbf{x})$ kiekvienai $\mathbf{x} = x_1, ..., x_d$ reikšmei. Įsivaizduokite, kad kiekvienas požymis gali įgyti vieną iš $2$ reikšmių. Pavyzdžiui, požymis $x_1 = 1$ gali reikšti, kad tam tikrame dokumente yra žodis „obuolys“, o $x_1 = 0$ – kad jo nėra. Jei turėtume $30$ tokių dvejetainių požymių, tai reikštų, kad turime būti pasirengę klasifikuoti bet kurią iš $2^{30}$ (daugiau nei 1 milijardo!) galimų įvesties vektoriaus $\mathbf{x}$ reikšmių.

Be to, kur čia mokymasis? Jei norėdami prognozuoti atitinkamą žymę turime pamatyti kiekvieną įmanomą pavyzdį, tai iš tiesų nesimokome dėsningumo, o tik įsimename duomenų rinkinį.

## Naivusis Bajeso klasifikatorius

Laimei, padarę keletą sąlyginio nepriklausomumo prielaidų, galime įvesti tam tikrą induktyvųjį šališkumą ir sukurti modelį, gebantį apibendrinti iš palyginti kuklaus mokymo pavyzdžių rinkinio. Pradėkime nuo Bajeso teoremos ir klasifikatorių išreikškime kaip

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Atkreipkite dėmesį, kad vardiklis yra normalizuojantis narys $p(\mathbf{x})$, kuris nepriklauso nuo žymės $y$ reikšmės. Todėl turime rūpintis tik skaitiklio palyginimu esant skirtingoms $y$ reikšmėms. Net jei vardiklio apskaičiavimas pasirodytų neįveikiamas, galėtume jo nepaisyti, jei tik galėtume apskaičiuoti skaitiklį. Laimei, net jei norėtume atkurti normalizavimo konstantą, galėtume. Normalizavimo narį visada galime atkurti, nes $\sum_y p(y  \mid  \mathbf{x}) = 1$.

Dabar susitelkime į $p( \mathbf{x}  \mid  y)$. Naudodami tikimybės grandinės taisyklę, narį $p( \mathbf{x}  \mid  y)$ galime išreikšti kaip

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Pati savaime ši išraiška toliau mūsų neveda. Vis dar turime įvertinti maždaug $2^d$ parametrų. Tačiau jei darome prielaidą, kad *požymiai yra sąlygiškai nepriklausomi vienas nuo kito, žinant žymę*, tada staiga esame daug geresnėje padėtyje, nes šis narys supaprastėja iki $\prod_i p(x_i  \mid  y)$, ir gauname prognozuoklį

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Jei kiekvienam $i$ ir $y$ galime įvertinti $p(x_i=1  \mid  y)$ ir jos reikšmę išsaugoti $P_{xy}[i, y]$, čia $P_{xy}$ yra $d\times n$ matrica, kurioje $n$ yra klasių skaičius, o $y\in\{1, \ldots, n\}$, tada galime tai panaudoti ir $p(x_i = 0 \mid y)$ įvertinti, t. y.

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

Be to, kiekvienam $y$ įvertiname $p(y)$ ir išsaugome $P_y[y]$, kur $P_y$ yra $n$ ilgio vektorius. Tada bet kuriam naujam pavyzdžiui $\mathbf t = (t_1, t_2, \ldots, t_d)$ galėtume apskaičiuoti

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

bet kuriam $y$. Taigi mūsų sąlyginio nepriklausomumo prielaida modelio sudėtingumą pakeitė iš eksponentinės priklausomybės nuo požymių skaičiaus $\mathcal{O}(2^dn)$ į tiesinę, kuri yra $\mathcal{O}(dn)$.


## Mokymas

Dabar problema ta, kad nežinome $P_{xy}$ ir $P_y$. Todėl pirmiausia turime įvertinti jų reikšmes iš tam tikrų mokymo duomenų. Tai yra modelio *mokymas*. Įvertinti $P_y$ nėra labai sunku. Kadangi turime tik $10$ klasių, galime suskaičiuoti pasikartojimų skaičių $n_y$ kiekvienam skaitmeniui ir padalyti iš bendro duomenų kiekio $n$. Pavyzdžiui, jei skaitmuo 8 pasitaiko $n_8 = 5\,800$ kartų, o iš viso turime $n = 60\,000$ paveikslėlių, tikimybės įvertis yra $p(y=8) = 0{,}0967$.

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Dabar pereikime prie šiek tiek sudėtingesnio dalyko – $P_{xy}$. Kadangi pasirinkome nespalvotus paveikslėlius, $p(x_i  \mid  y)$ žymi tikimybę, kad $i$-asis pikselis klasėje $y$ yra įjungtas. Kaip ir anksčiau, galime suskaičiuoti, kiek kartų $n_{iy}$ įvyksta įvykis, ir padalyti iš bendro $y$ pasikartojimų skaičiaus, t. y. $n_y$. Tačiau yra kai kas šiek tiek nerimą keliančio: tam tikri pikseliai gali niekada nebūti juodi (pvz., gerai apkarpytuose paveikslėliuose kampiniai pikseliai gali visada būti balti). Patogus statistikų būdas su šia problema tvarkytis – prie visų pasikartojimų pridėti pseudoskaičius. Taigi vietoj $n_{iy}$ naudojame $n_{iy}+1$, o vietoj $n_y$ – $n_{y}+2$ (nes $i$-asis pikselis gali įgyti dvi reikšmes – būti juodas arba baltas). Tai dar vadinama *Laplaso glotninimu* (angl. *Laplace smoothing*).  Tai gali atrodyti ad hoc, tačiau ją galima pagrįsti Bajeso požiūriu Beta-binominiu modeliu.

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

Pavaizdavę šias $10\times 28\times 28$ tikimybes (kiekvienam pikseliui kiekvienoje klasėje) galėtume gauti gana niūriai atrodančius skaitmenis.

Dabar galime naudoti :eqref:`eq_naive_bayes_estimation` naujam paveikslėliui prognozuoti. Duotam $\mathbf x$ šios funkcijos apskaičiuoja $p(\mathbf x \mid y)p(y)$ kiekvienam $y$.

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Tai baigėsi siaubingai prastai! Norėdami išsiaiškinti kodėl, pažiūrėkime į tikimybes pikseliams. Paprastai jos yra skaičiai tarp $0{,}001$ ir $1$. Mes dauginame $784$ iš jų. Čia verta paminėti, kad šiuos skaičius skaičiuojame kompiuteriu, taigi su fiksuotu eksponentės intervalu. Įvyksta *skaitinis nepakankamumas* (angl. *numerical underflow*), t. y. dauginant visus mažus skaičius gaunamas dar mažesnis, kol jis suapvalinamas žemyn iki nulio.  Tai aptarėme kaip teorinę problemą :numref:`sec_maximum_likelihood`, bet čia šį reiškinį aiškiai matome praktikoje.

Kaip aptarta tame skyriuje, tai išsprendžiame pasinaudoję tuo, kad $\log a b = \log a + \log b$, t. y. pereiname prie logaritmų sumavimo.
Net jei tiek $a$, tiek $b$ yra maži skaičiai, logaritmų reikšmės turėtų būti tinkamame intervale.

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Kadangi logaritmas yra didėjanti funkcija, :eqref:`eq_naive_bayes_estimation` galime perrašyti kaip

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

Galime realizuoti šią stabilią versiją:

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Dabar galime patikrinti, ar prognozė teisinga.

```{.python .input}
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

Jei dabar prognozuosime kelis validavimo pavyzdžius, pamatysime, kad Bajeso
klasifikatorius veikia gana gerai.

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Galiausiai apskaičiuokime bendrą klasifikatoriaus tikslumą.

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

Šiuolaikiniai giluminiai tinklai pasiekia mažesnes nei $0{,}01$ klaidų dažnis. Palyginti prastą našumą lemia neteisingos statistinės prielaidos, kurias padarėme savo modelyje: darėme prielaidą, kad kiekvienas pikselis generuojamas *nepriklausomai*, priklausomai tik nuo žymės. Akivaizdu, kad žmonės skaitmenų taip nerašo, ir ši klaidinga prielaida lėmė mūsų pernelyg naivaus (Bajeso) klasifikatoriaus žlugimą.

## Santrauka
* Naudojant Bajeso taisyklę, klasifikatorių galima sukurti darant prielaidą, kad visi stebimi požymiai yra nepriklausomi.
* Šį klasifikatorių galima apmokyti duomenų rinkinyje skaičiuojant žymių ir pikselių reikšmių derinių pasikartojimus.
* Šis klasifikatorius dešimtmečius buvo aukso standartas tokioms užduotims kaip šlamšto aptikimas.

## Pratimai
1. Panagrinėkite duomenų rinkinį $[[0,0], [0,1], [1,0], [1,1]]$ su žymėmis, gaunamomis kaip dviejų elementų XOR: $[0,1,1,0]$.  Kokios yra naiviojo Bajeso klasifikatoriaus, sukurto šiame duomenų rinkinyje, tikimybės?  Ar jis sėkmingai klasifikuoja mūsų taškus?  Jei ne, kokios prielaidos pažeidžiamos?
1. Tarkime, kad įvertindami tikimybes nenaudojome Laplaso glotninimo, o testavimo metu atkeliavo duomenų pavyzdys, kuriame buvo reikšmė, niekada nematyta mokymo metu.  Ką pateiktų modelis?
1. Naivusis Bajeso klasifikatorius yra konkretus Bajeso tinklo pavyzdys, kuriame atsitiktinių dydžių priklausomybė užkoduojama grafo struktūra.  Nors pilna teorija neįeina į šio skyriaus aprėptį (išsamiai žr. :citet:`Koller.Friedman.2009`), paaiškinkite, kodėl leidus aiškią priklausomybę tarp dviejų XOR modelio įvesties kintamųjų, galima sukurti sėkmingą klasifikatorių.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1101)
:end_tab:
