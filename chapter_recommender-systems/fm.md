# Faktorizavimo mašinos

Faktorizavimo mašinos (angl. *factorization machines*, FM), pasiūlytos :citet:`Rendle.2010`, yra prižiūrimo mokymosi algoritmas, kurį galima naudoti klasifikavimo, regresijos ir rikiavimo užduotims. Jis greitai atkreipė dėmesį ir tapo populiariu bei įtakingu metodu prognozėms ir rekomendacijoms daryti. Ypač tai yra tiesinės regresijos modelio ir matricos faktorizavimo modelio apibendrinimas. Be to, jis primena atraminių vektorių mašinas su polinominiu branduoliu. Faktorizavimo mašinų pranašumai prieš tiesinę regresiją ir matricos faktorizavimą yra tokie: (1) jos gali modeliuoti $\chi$-krypčias kintamųjų sąveikas, kur $\chi$ yra polinomo eilė ir paprastai nustatoma lygi dviem; (2) greitas su faktorizavimo mašinomis susijęs optimizavimo algoritmas gali sumažinti polinominį skaičiavimo laiką iki tiesinio sudėtingumo, todėl jos yra itin efektyvios, ypač didelės dimensijos retoms įvestims. Dėl šių priežasčių faktorizavimo mašinos plačiai naudojamos šiuolaikinėje reklamoje ir produktų rekomendavime. Techninės detalės ir įgyvendinimai aprašyti žemiau.


## Dvikryptės faktorizavimo mašinos

Formaliai tegul $x \in \mathbb{R}^d$ žymi vieno pavyzdžio požymių vektorių, o $y$ – atitinkamą žymenį, kuris gali būti realioji reikšmė arba klasės žymuo, pavyzdžiui, dvejetainė klasė „paspaudė / nepaspaudė“. Antrojo laipsnio faktorizavimo mašinos modelis apibrėžiamas taip:

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

kur $\mathbf{w}_0 \in \mathbb{R}$ yra globalus poslinkis; $\mathbf{w} \in \mathbb{R}^d$ žymi i-ojo kintamojo svorius; $\mathbf{V} \in \mathbb{R}^{d\times k}$ atitinka požymių įterpinius; $\mathbf{v}_i$ žymi $i$-ąją $\mathbf{V}$ eilutę; $k$ yra latentinių veiksnių dimensija; $\langle\cdot, \cdot \rangle$ yra dviejų vektorių skaliarinė sandauga. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ modeliuoja $i$-ojo ir $j$-ojo požymių sąveiką. Kai kurias požymių sąveikas lengva suprasti, todėl jas gali suprojektuoti ekspertai. Tačiau dauguma kitų požymių sąveikų yra paslėptos duomenyse ir sunkiai nustatomos. Todėl automatinis požymių sąveikų modeliavimas gali gerokai sumažinti požymių inžinerijos pastangas. Akivaizdu, kad pirmieji du nariai atitinka tiesinės regresijos modelį, o paskutinis narys yra matricos faktorizavimo modelio plėtinys. Jei požymis $i$ vaizduoja elementą, o požymis $j$ – naudotoją, trečiasis narys yra būtent naudotojų ir elementų įterpinių skaliarinė sandauga. Verta pažymėti, kad FM taip pat gali būti apibendrintos aukštesnėms eilėms (laipsnis > 2). Vis dėlto skaitinis stabilumas gali susilpninti apibendrinimą.


## Efektyvus optimizavimo kriterijus

Optimizuojant faktorizavimo mašinas tiesmuku būdu, gaunamas $\mathcal{O}(kd^2)$ sudėtingumas, nes reikia apskaičiuoti visas porų sąveikas. Kad išspręstume šią neefektyvumo problemą, galime pertvarkyti trečiąjį FM narį, o tai gali gerokai sumažinti skaičiavimo sąnaudas ir lemti tiesinį laiko sudėtingumą ($\mathcal{O}(kd)$). Porų sąveikos nario pertvarkymas yra toks:

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Taikant šį pertvarkymą, modelio sudėtingumas gerokai sumažėja. Be to, retiems požymiams reikia apskaičiuoti tik nenulinius elementus, todėl bendras sudėtingumas yra tiesinis nenulinių požymių skaičiaus atžvilgiu.

FM modeliui mokyti galime naudoti MSE nuostolį regresijos užduočiai, kryžminės entropijos nuostolį klasifikavimo užduotims ir BPR nuostolį rikiavimo užduočiai. Optimizavimui tinka standartiniai optimizatoriai, tokie kaip stochastinis gradientinis nusileidimas ir Adam.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Modelio įgyvendinimas
Toliau pateiktas kodas įgyvendina faktorizavimo mašinas. Aiškiai matyti, kad FM sudaro tiesinės regresijos blokas ir efektyvus požymių sąveikos blokas. Galutiniam balui taikome sigmoidinę funkciją, nes CTR prognozavimą laikome klasifikavimo užduotimi.

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## Reklamos duomenų rinkinio įkėlimas
Internetinės reklamos duomenų rinkiniui įkelti naudojame CTR duomenų apvalkalą iš ankstesnio skyrelio.

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Modelio apmokymas
Po to apmokome modelį. Mokymosi greitis pagal numatytuosius nustatymus yra 0,02, o įterpinių dydis – 20. Modeliui apmokyti naudojamas `Adam` optimizatorius ir nuostolis `SigmoidBinaryCrossEntropyLoss`.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Santrauka

* FM yra bendra sistema, kurią galima taikyti įvairioms užduotims, tokioms kaip regresija, klasifikavimas ir rikiavimas.
* Požymių sąveika (kryžminimas) svarbi prognozavimo užduotims, o dvikryptę sąveiką galima efektyviai modeliuoti naudojant FM.

## Pratimai

* Ar galite išbandyti FM kituose duomenų rinkiniuose, tokiuose kaip Avazu, MovieLens ir Criteo?
* Keiskite įterpinių dydį, kad patikrintumėte poveikį našumui. Ar pastebite panašų dėsningumą kaip matricos faktorizavime?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/406)
:end_tab:
