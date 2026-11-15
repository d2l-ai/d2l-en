```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Konvoliucinių tinklų architektūrų projektavimas
:label:`sec_cnn-design`

Ankstesni skyreliai mus supažindino su šiuolaikiniu tinklų dizainu kompiuterinei regai. Visų aptartų darbų bendras bruožas buvo tas, kad jie labai rėmėsi mokslininkų intuicija. Daugelį architektūrų stipriai lėmė žmogaus kūrybiškumas ir daug mažiau – sistemingas dizaino erdvės, kurią siūlo gilieji tinklai, tyrinėjimas. Vis dėlto šis *tinklų inžinerijos* požiūris buvo nepaprastai sėkmingas.

Nuo tada, kai AlexNet (:numref:`sec_alexnet`)
įveikė įprastus kompiuterinės regos modelius su ImageNet,
tapo populiaru kurti labai gilius tinklus,
kraunant konvoliucijų blokus, visus suprojektuotus pagal tą patį šabloną.
Visų pirma $3 \times 3$ konvoliucijas
išpopuliarino VGG tinklai (:numref:`sec_vgg`).
NiN (:numref:`sec_nin`) parodė, kad net $1 \times 1$ konvoliucijos gali
būti naudingos pridedant lokalius netiesiškumus.
Be to, NiN išsprendė informacijos agregavimo tinklo galvoje problemą,
agreguodamas per visas vietas.
GoogLeNet (:numref:`sec_googlenet`) pridėjo kelias skirtingo konvoliucijų pločio šakas,
savo Inception bloke sujungdamas VGG ir NiN privalumus.
ResNet (:numref:`sec_resnet`)
pakeitė induktyvinį šališkumą link tapatybės atvaizdavimo (nuo $f(x) = 0$). Tai leido kurti labai gilius tinklus. Praėjus beveik dešimtmečiui, ResNet dizainas vis dar populiarus, o tai liudija apie jo kokybę. Galiausiai ResNeXt (:numref:`subsec_resnext`) pridėjo grupines konvoliucijas, pasiūlydamas geresnį kompromisą tarp parametrų ir skaičiavimų. Transformerių regai pirmtakai – suspaudimo ir sužadinimo tinklai (angl. Squeeze-and-Excitation Networks, SENet) – leidžia efektyviai perduoti informaciją tarp vietų
:cite:`Hu.Shen.Sun.2018`. Tai pasiekta apskaičiuojant globalią dėmesio funkciją kiekvienam kanalui.

Iki šiol praleidome tinklus, gautus per *neuroninių architektūrų paiešką* (angl. *neural architecture search*, NAS) :cite:`zoph2016neural,liu2018darts`. Taip pasirinkome todėl, kad jų kaina paprastai milžiniška, nes jie remiasi brutalios jėgos paieška, genetiniais algoritmais, skatinamuoju mokymusi ar kita hiperparametrų optimizavimo forma. Turint fiksuotą paieškos erdvę,
NAS naudoja paieškos strategiją, kad automatiškai parinktų
architektūrą pagal grąžinamą našumo įvertį.
NAS rezultatas
yra vienas tinklo egzempliorius. EfficientNet yra žymus šios paieškos rezultatas :cite:`tan2019efficientnet`.

Toliau aptariame idėją, gana skirtingą nuo *vienintelio geriausio tinklo* paieškos. Ji skaičiavimo požiūriu santykinai pigi, pakeliui duoda mokslinių įžvalgų ir yra gana veiksminga rezultatų kokybės požiūriu. Apžvelkime :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` strategiją *projektuoti tinklo dizaino erdves*. Strategija sujungia rankinio dizaino ir NAS stiprybes. Tai pasiekiama veikiant su *tinklų skirstiniais* ir optimizuojant skirstinius taip, kad būtų gautas geras visų tinklų šeimų našumas. Rezultatas yra *RegNet* tinklai, konkrečiai RegNetX ir RegNetY, plius kelių vadovaujančių principų rinkinys našių CNN projektavimui.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## AnyNet dizaino erdvė
:label:`subsec_the-anynet-design-space`

Toliau pateiktas aprašymas glaudžiai seka :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` samprotavimą su kai kuriais sutrumpinimais, kad tilptų į knygos apimtį.
Pradėkime nuo to, kad mums reikia šablono tyrinėjamai tinklų šeimai. Vienas iš šiame skyriuje aprašytų dizainų bendrų bruožų yra tas, kad tinklus sudaro *pradžios dalis*, *korpusas* ir *galva*. Pradžios dalis atlieka pradinį vaizdo apdorojimą, dažnai per konvoliucijas su didesniu lango dydžiu. Korpusą sudaro keli blokai, atliekantys didžiąją dalį transformacijų, reikalingų norint iš neapdorotų vaizdų gauti objektų atvaizdus. Galiausiai galva paverčia tai norimomis išvestimis, pavyzdžiui, per softmax regresorių daugiaklasei klasifikacijai.
Savo ruožtu korpusą sudaro keli etapai, veikiantys vaizdą mažėjančia skyra. Iš tikrųjų ir pradžios dalis, ir kiekvienas paskesnis etapas ketvirtadaliu sumažina erdvinę skyrą. Galiausiai kiekvieną etapą sudaro vienas ar daugiau blokų. Šis šablonas bendras visiems tinklams, nuo VGG iki ResNeXt. Iš tikrųjų bendrų AnyNet tinklų dizainui :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` naudojo ResNeXt bloką iš :numref:`fig_resnext_block`.


![AnyNet dizaino erdvė. Skaičiai $(\mathit{c}, \mathit{r})$ prie kiekvienos rodyklės nurodo kanalų skaičių $c$ ir vaizdų skyrą $\mathit{r} \times \mathit{r}$ tame taške. Iš kairės į dešinę: bendra tinklo struktūra, sudaryta iš pradžios dalies, korpuso ir galvos; korpusas, sudarytas iš keturių etapų; detali etapo struktūra; dvi alternatyvios blokų struktūros – viena be diskretizavimo dažnio mažinimo, kita perpus sumažinanti skyrą kiekvienoje dimensijoje. Dizaino pasirinkimai apima gylį $\mathit{d_i}$, išvesties kanalų skaičių $\mathit{c_i}$, grupių skaičių $\mathit{g_i}$ ir butelio kakliuko santykį $\mathit{k_i}$ bet kuriam etapui $\mathit{i}$.](../img/anynet.svg)
:label:`fig_anynet_full`

Išsamiai apžvelkime :numref:`fig_anynet_full` pateiktą struktūrą. Kaip minėta, AnyNet sudaro pradžios dalis, korpusas ir galva. Pradžios dalis kaip įvestį priima RGB vaizdus (3 kanalai), naudoja $3 \times 3$ konvoliuciją su žingsniu $2$, po kurios eina rinkinio normalizacija, kad perpus sumažintų skyrą iš $r \times r$ į $r/2 \times r/2$. Be to, ji generuoja $c_0$ kanalų, kurie tarnauja kaip korpuso įvestis.

Kadangi tinklas suprojektuotas gerai veikti su $224 \times 224 \times 3$ formos ImageNet vaizdais, korpusas tai sumažina iki $7 \times 7 \times c_4$ per 4 etapus (prisiminkite, kad $224 / 2^{1+4} = 7$), kurių kiekvienas turi galutinį žingsnį $2$. Galiausiai galva naudoja visiškai standartinį dizainą su globaliu sutelkimu vidurkinant, panašiai kaip NiN (:numref:`sec_nin`), po kurio eina pilnai sujungtas sluoksnis $n$ dimensijų vektoriui išvesti $n$ klasių klasifikacijai.

Dauguma svarbių dizaino sprendimų yra būdingi tinklo korpusui. Jis veikia etapais, kur kiekvieną etapą sudaro to paties tipo ResNeXt blokai, kaip aptarėme :numref:`subsec_resnext`. Dizainas ten vėl visiškai bendras: pradedame nuo bloko, kuris perpus sumažina skyrą naudodamas žingsnį $2$ (dešiniausias :numref:`fig_anynet_full`). Kad tai atitiktų, ResNeXt bloko liekamoji šaka turi praeiti per $1 \times 1$ konvoliuciją. Po šio bloko eina kintamas skaičius papildomų ResNeXt blokų, nekeičiančių nei skyros, nei kanalų skaičiaus. Atkreipkite dėmesį, kad įprasta dizaino praktika yra pridėti nedidelį butelio kakliuką konvoliucinių blokų dizaine.
Taigi su butelio kakliuko santykiu $k_i \geq 1$ leidžiame sau tam tikrą kanalų skaičių, $c_i/k_i$, kiekviename $i$-ojo etapo bloke (kaip rodo eksperimentai, tai nelabai veiksminga ir to reikėtų atsisakyti). Galiausiai, kadangi turime reikalų su ResNeXt blokais, taip pat turime pasirinkti grupių skaičių $g_i$ grupinėms konvoliucijoms $i$-ajame etape.

Ši iš pažiūros bendra dizaino erdvė vis dėlto suteikia mums daug parametrų: galime nustatyti bloko plotį (kanalų skaičių) $c_0, \ldots c_4$, gylį (blokų skaičių) kiekviename etape $d_1, \ldots d_4$, butelio kakliuko santykius $k_1, \ldots k_4$ ir grupių pločius (grupių skaičius) $g_1, \ldots g_4$.
Iš viso tai sudaro 17 parametrų, todėl gaunamas nepagrįstai didelis konfigūracijų, kurias būtų verta tyrinėti, skaičius. Mums reikia įrankių, kad veiksmingai sumažintume šią milžinišką dizaino erdvę. Čia ir atsiskleidžia konceptualus dizaino erdvių grožis. Prieš tai darydami, pirmiausia įgyvendinkime bendrą dizainą.

```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

Kiekvieną etapą sudaro `depth` ResNeXt blokų,
kur `num_channels` nurodo bloko plotį.
Atkreipkite dėmesį, kad pirmasis blokas perpus sumažina įvesties vaizdų aukštį ir plotį.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

Sujungę tinklo pradžios dalį, korpusą ir galvą,
užbaigiame AnyNet įgyvendinimą.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## Dizaino erdvių skirstiniai ir parametrai

Kaip ką tik aptarta :numref:`subsec_the-anynet-design-space`, dizaino erdvės parametrai yra tos dizaino erdvės tinklų hiperparametrai.
Panagrinėkime gerų parametrų nustatymo AnyNet dizaino erdvėje problemą. Galėtume bandyti rasti *vienintelį geriausią* parametrų pasirinkimą duotam skaičiavimų kiekiui (pvz., FLOPs ir skaičiavimo laikui). Jei leistume net tik *dvi* galimas kiekvieno parametro reikšmes, turėtume ištirti $2^{17} = 131072$ derinių, kad rastume geriausią sprendinį. Tai akivaizdžiai neįmanoma dėl pernelyg didelės kainos. Dar blogiau, iš šio pratimo iš tikrųjų nieko neišmokstame apie tai, kaip reikėtų projektuoti tinklą. Kitą kartą pridėję, tarkime, X etapą, postūmio operaciją ar panašiai, turėtume pradėti iš naujo. Dar blogiau, dėl apmokymo stochastiškumo (apvalinimo, maišymo, bitų klaidų) du paleidimai vargu ar duos visiškai tuos pačius rezultatus. Geresnė strategija būtų bandyti nustatyti bendras gaires, kaip turėtų būti susiję parametrų pasirinkimai. Pavyzdžiui, butelio kakliuko santykį, kanalų, blokų, grupių skaičių ar jų kitimą tarp sluoksnių idealiai turėtų valdyti paprastų taisyklių rinkinys. :citet:`radosavovic2019network` požiūris remiasi šiomis keturiomis prielaidomis:

1. Darome prielaidą, kad bendri dizaino principai iš tikrųjų egzistuoja, todėl daug šiuos reikalavimus tenkinančių tinklų turėtų siūlyti gerą našumą. Todėl *skirstinio* per tinklus nustatymas gali būti pagrįsta strategija. Kitaip tariant, darome prielaidą, kad šieno kupetoje yra daug gerų adatų.
1. Nereikia apmokyti tinklų iki konvergavimo, kad galėtume įvertinti, ar tinklas geras. Vietoj to pakanka naudoti tarpinius rezultatus kaip patikimą gaires galutiniam tikslumui. Naudojimas (apytikslių) pakaitinių dydžių tikslui optimizuoti vadinamas daugiatikslumo optimizavimu :cite:`forrester2007multi`. Todėl dizaino optimizavimas atliekamas pagal tikslumą, pasiektą po vos kelių perėjimų per duomenų rinkinį, taip gerokai sumažinant kainą.
1. Rezultatai, gauti mažesniu mastu (mažesniems tinklams), apibendrinami didesniems. Todėl optimizavimas atliekamas struktūriškai panašiems tinklams, bet su mažesniu blokų, mažiau kanalų ir pan. Tik pabaigoje reikės patikrinti, ar taip rasti tinklai taip pat siūlo gerą našumą dideliu mastu.
1. Dizaino aspektus galima apytiksliai išskaidyti, todėl galima gana nepriklausomai nustatyti jų poveikį rezultato kokybei. Kitaip tariant, optimizavimo uždavinys yra vidutiniškai lengvas.

Šios prielaidos leidžia pigiai išbandyti daug tinklų. Visų pirma galime *imti imtį* tolygiai iš konfigūracijų erdvės ir įvertinti jų našumą. Vėliau galime įvertinti parametrų pasirinkimo kokybę, apžvelgdami paklaidos / tikslumo, kurį galima pasiekti su minėtais tinklais, *skirstinį*. Pažymėkime $F(e)$ kaupiamąją skirstinio funkciją (KSF) paklaidoms, kurias daro tam tikros dizaino erdvės tinklai, ištraukti naudojant tikimybių skirstinį $p$. Tai yra,

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

Dabar mūsų tikslas yra rasti *tinklų* skirstinį $p$, kad dauguma tinklų turėtų labai mažą paklaidos lygį ir kad $p$ atrama būtų glausta. Žinoma, to tiksliai atlikti skaičiavimo požiūriu neįmanoma. Griebiamės tinklų imties $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$ (atitinkamai su paklaidomis $e_1, \ldots, e_n$) iš $p$ ir vietoj to naudojame empirinę KSF $\hat{F}(e, \mathcal{Z})$:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

Kai vieno pasirinkimų rinkinio KSF majoruoja (arba sutampa su) kitą KSF, iš to išplaukia, kad jo parametrų pasirinkimas yra geresnis (arba abejingas). Atitinkamai
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` eksperimentavo su bendru tinklo butelio kakliuko santykiu $k_i = k$ visiems tinklo etapams $i$. Taip atsikratoma trijų iš keturių butelio kakliuko santykį valdančių parametrų. Kad įvertintume, ar tai (neigiamai) veikia našumą, galima ištraukti tinklus iš apribotojo ir iš neapribotojo skirstinio ir palyginti atitinkamas KSF. Paaiškėja, kad šis apribojimas visai nepaveikia tinklų skirstinio tikslumo, kaip matyti pirmajame :numref:`fig_regnet-fig` skydelyje.
Panašiai galėtume pasirinkti tą patį grupės plotį $g_i = g$ įvairiuose tinklo etapuose. Vėlgi tai neįtakoja našumo, kaip matyti antrajame :numref:`fig_regnet-fig` skydelyje.
Abu žingsniai kartu sumažina laisvųjų parametrų skaičių šešiais.

![Dizaino erdvių paklaidų empirinių skirstinio funkcijų palyginimas. $\textrm{AnyNet}_\mathit{A}$ yra originali dizaino erdvė; $\textrm{AnyNet}_\mathit{B}$ susieja butelio kakliuko santykius, $\textrm{AnyNet}_\mathit{C}$ taip pat susieja grupių plotį, $\textrm{AnyNet}_\mathit{D}$ didina tinklo gylį per etapus. Iš kairės į dešinę: (i) butelio kakliuko santykių susiejimas neturi poveikio našumui; (ii) grupių pločių susiejimas neturi poveikio našumui; (iii) tinklo pločių (kanalų) didinimas per etapus pagerina našumą; (iv) tinklo gylių didinimas per etapus pagerina našumą. Paveikslas atkurtas :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` leidimu.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

Toliau ieškome būdų sumažinti galimų etapų pločio ir gylio pasirinkimų daugybę. Pagrįsta prielaida, kad gilėjant kanalų skaičius turėtų didėti, t. y. $c_i \geq c_{i-1}$ ($w_{i+1} \geq w_i$ pagal jų žymėjimą :numref:`fig_regnet-fig`), gaunant
$\textrm{AnyNetX}_D$. Panašiai lygiai taip pat pagrįsta manyti, kad etapams progresuojant jie turėtų gilėti, t. y. $d_i \geq d_{i-1}$, gaunant $\textrm{AnyNetX}_E$. Tai galima eksperimentiškai patikrinti atitinkamai trečiajame ir ketvirtajame :numref:`fig_regnet-fig` skydeliuose.

## RegNet

Gauta $\textrm{AnyNetX}_E$ dizaino erdvė susideda iš paprastų tinklų,
besilaikančių lengvai interpretuojamų dizaino principų:

* Bendras butelio kakliuko santykis $k_i = k$ visiems etapams $i$;
* Bendras grupės plotis $g_i = g$ visiems etapams $i$;
* Didinamas tinklo plotis per etapus: $c_{i} \leq c_{i+1}$;
* Didinamas tinklo gylis per etapus: $d_{i} \leq d_{i+1}$.

Taip lieka paskutinis pasirinkimų rinkinys: kaip parinkti konkrečias aukščiau minėtų galutinės $\textrm{AnyNetX}_E$ dizaino erdvės parametrų reikšmes. Tyrinėjant geriausiai veikiančius tinklus iš $\textrm{AnyNetX}_E$ skirstinio, galima pastebėti: tinklo plotis idealiai didėja tiesiškai su bloko indeksu per tinklą, t. y. $c_j \approx c_0 + c_a j$, kur $j$ yra bloko indeksas, o nuolydis $c_a > 0$. Kadangi skirtingą bloko plotį galime rinktis tik kiekvienam etapui, gauname dalimis pastovią funkciją, sukonstruotą taip, kad atitiktų šią priklausomybę. Be to, eksperimentai taip pat rodo, kad butelio kakliuko santykis $k = 1$ veikia geriausiai, t. y. patariama butelio kakliukų išvis nenaudoti.

Rekomenduojame norintiems skaitytojams susipažinti su tolesnėmis detalėmis, kaip projektuojami konkretūs tinklai skirtingiems skaičiavimų kiekiams, perskaitant :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`. Pavyzdžiui, veiksmingą 32 sluoksnių RegNetX variantą nusako $k = 1$ (be butelio kakliuko), $g = 16$ (grupės plotis yra 16), $c_1 = 32$ ir $c_2 = 80$ kanalų atitinkamai pirmajam ir antrajam etapui, parinkti $d_1=4$ ir $d_2=6$ blokų gylio. Nuostabi dizaino įžvalga yra ta, kad ji vis dar galioja net tiriant didesnio masto tinklus. Dar geriau, ji galioja net suspaudimo ir sužadinimo (SE) tinklų dizainams (RegNetY), turintiems globalią kanalų aktyvaciją :cite:`Hu.Shen.Sun.2018`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

Matome, kad kiekvienas RegNetX etapas palaipsniui mažina skyrą ir didina išvesties kanalus.

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## Mokymas

32 sluoksnių RegNetX apmokymas su Fashion-MNIST duomenų rinkiniu vyksta kaip ir anksčiau.

```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## Diskusija

Turėdami pageidaujamus induktyvinius šališkumus (prielaidas ar pirmenybes), tokius kaip lokalumas ir postūmio invariantiškumas (:numref:`sec_why-conv`),
regai CNN buvo dominuojančios architektūros šioje srityje. Taip buvo nuo LeNet iki tol, kol transformeriai (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` pradėjo pranokti CNN tikslumo požiūriu. Nors didelę dalį naujausios pažangos regos transformeriuose *galima* perkelti atgal į CNN :cite:`liu2022convnet`, tai įmanoma tik didesne skaičiavimo kaina. Ne mažiau svarbu, kad naujausios aparatinės įrangos optimizacijos (NVIDIA Ampere ir Hopper) tik padidino skirtumą transformerių naudai.

Verta pažymėti, kad transformeriai turi gerokai mažesnį induktyvinio šališkumo link lokalumo ir postūmio invariantiškumo laipsnį nei CNN. Tai, kad išmoktos struktūros įgijo viršų, lemia, ne mažiausia, didelių vaizdų kolekcijų, tokių kaip LAION-400m ir LAION-5B :cite:`schuhmann2022laion` su iki 5 milijardų vaizdų, prieinamumas. Gana netikėtai, kai kurie svarbesni šio konteksto darbai apima net MLP :cite:`tolstikhin2021mlp`.

Apibendrinant, regos transformeriai (:numref:`sec_vision-transformer`) šiuo metu pirmauja pagal
pažangiausią našumą didelio masto vaizdų klasifikavime,
rodydami, kad *mastelio keitimas nugali induktyvinius šališkumus* :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Tai apima didelio masto transformerių išankstinį apmokymą (:numref:`sec_large-pretraining-transformers`) su lygiagrečia dėmesio sutelkimo transformacija sau pačiam (:numref:`sec_multihead-attention`). Kviečiame skaitytojus panirti į tuos skyrius, kuriuose pateikta daug išsamesnė diskusija.

## Pratimai

1. Padidinkite etapų skaičių iki keturių. Ar galite suprojektuoti gilesnį RegNetX, kuris veiktų geriau?
1. Pašalinkite ResNeXt iš RegNet, pakeisdami ResNeXt bloką ResNet bloku. Kaip veikia jūsų naujas modelis?
1. Įgyvendinkite kelis „VioNet“ šeimos egzempliorius *pažeisdami* RegNetX dizaino principus. Kaip jie veikia? Kuris iš ($d_i$, $c_i$, $g_i$, $b_i$) yra svarbiausias veiksnys?
1. Jūsų tikslas – suprojektuoti „tobulą“ MLP. Ar galite naudoti aukščiau pristatytus dizaino principus gerų architektūrų paieškai? Ar įmanoma ekstrapoliuoti iš mažų į didelius tinklus?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/8738)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18009)
:end_tab:
