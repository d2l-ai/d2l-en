# Šiuolaikiniai konvoliuciniai neuroniniai tinklai
:label:`chap_modern_cnn`

Dabar, kai suprantame pagrindus, kaip sujungti CNN, apžvelkime
šiuolaikines CNN architektūras. Ši apžvalga neišvengiamai
yra nepilna dėl daugybės naujų įdomių dizainų, kurie nuolat
pridedami. Jų svarba kyla iš to, kad jie gali būti ne tik
tiesiogiai naudojami regos užduotims, bet ir tarnauja kaip pagrindiniai
požymių generatoriai sudėtingesnėms užduotims, tokioms kaip sekimas
:cite:`Zhang.Sun.Jiang.ea.2021`, segmentavimas :cite:`Long.Shelhamer.Darrell.2015`, objektų
aptikimas :cite:`Redmon.Farhadi.2018` ar stiliaus transformacija
:cite:`Gatys.Ecker.Bethge.2016`. Šiame skyriuje dauguma skyrelių
atitinka reikšmingą CNN architektūrą, kuri tam tikru metu
(ar tebėra) buvo bazinis modelis, ant kurio buvo kuriami daugelis tyrimų projektų ir
diegiamų sistemų. Kiekvienas iš šių tinklų trumpai buvo
dominuojanti architektūra, o daugelis buvo
[ImageNet konkurso](https://www.image-net.org/challenges/LSVRC/) nugalėtojai ar antros vietos laimėtojai;
šis konkursas nuo 2010 m. tarnavo kaip prižiūrimojo mokymosi pažangos
kompiuterinėje regoje barometras. Tik visai neseniai transformeriai pradėjo
išstumti CNN, pradedant :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` ir
vėliau Swin transformeriu :cite:`liu2021swin`. Šią raidą aptarsime vėliau
:numref:`chap_attention-and-transformers`.

Nors *gilių* neuroninių tinklų idėja yra gana paprasta (sukrauti
kartu daugybę sluoksnių), našumas gali labai skirtis
priklausomai nuo architektūros ir hiperparametrų pasirinkimo. Šiame skyriuje
aprašyti neuroniniai tinklai yra intuicijos, kelių
matematinių įžvalgų ir daugybės bandymų ir klaidų rezultatas. Šiuos
modelius pristatome chronologine tvarka, iš dalies tam, kad perteiktume istorijos pojūtį,
ir jūs galėtumėte susidaryti savo intuiciją apie tai, kur
juda sritis, ir galbūt sukurti savo architektūras. Pavyzdžiui,
šiame skyriuje aprašyta rinkinio normalizacija ir liekamosios jungtys
pasiūlė dvi populiarias idėjas giliesiems modeliams apmokyti ir kurti,
ir abi jos vėliau buvo pritaikytos ir už kompiuterinės
regos ribų.

Savo šiuolaikinių CNN apžvalgą pradedame nuo AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`,
pirmojo didelio masto tinklo, įdiegto ir įveikusio įprastus kompiuterinės
regos metodus didelio masto regos konkurse; VGG tinklo
:cite:`Simonyan.Zisserman.2014`, naudojančio daugybę
pasikartojančių elementų blokų; tinklo tinkle (NiN), kuris
konvoliuoja ištisus neuroninius tinklus pagal pleistrus per įvestis
:cite:`Lin.Chen.Yan.2013`; GoogLeNet, naudojančio tinklus su
daugiašakėmis konvoliucijomis :cite:`Szegedy.Liu.Jia.ea.2015`; liekanų neuroninio
tinklo (ResNet) :cite:`He.Zhang.Ren.ea.2016`, kuris tebėra viena
populiariausių paruoštų architektūrų kompiuterinėje regoje;
ResNeXt blokų :cite:`Xie.Girshick.Dollar.ea.2017`
retesnėms jungtims;
ir DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017` – liekamosios architektūros apibendrinimo.
Laikui bėgant buvo sukurta daug specialių efektyvių
tinklų optimizacijų, tokių kaip koordinačių postūmiai (ShiftNet) :cite:`wu2018shift`. Tai
pasiekė kulminaciją automatinėje efektyvių architektūrų paieškoje, tokioje kaip
MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`. Tai taip pat apima
pusiau automatinį :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` dizaino tyrinėjimą,
kuris atvedė prie RegNetX/Y, kuriuos aptarsime vėliau šiame skyriuje.
Šis darbas pamokantis tuo, kad siūlo kelią sujungti brutalią skaičiavimo jėgą su
eksperimentatoriaus išradingumu ieškant efektyvių dizaino erdvių. Paminėtinas
ir :citet:`liu2022convnet` darbas, nes jis rodo, kad mokymo technikos (pvz., optimizavimo algoritmai, duomenų papildymas ir reguliarizacija)
vaidina esminį vaidmenį gerinant tikslumą. Jis taip pat rodo, kad ilgai laikytos prielaidos, tokios kaip
konvoliucijos lango dydis, gali reikalauti peržiūros, atsižvelgiant į išaugusį
skaičiavimų ir duomenų kiekį. Šį ir daugelį kitų klausimų aptarsime nuosekliai šiame skyriuje.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```
