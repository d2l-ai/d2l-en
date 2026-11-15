# Optimizavimo algoritmai
:label:`chap_optimization`

Jei skaitėte knygą nuosekliai iki šios vietos, jau naudojote nemažai optimizavimo algoritmų giliojo mokymosi modeliams apmokyti.
Tai buvo įrankiai, leidę mums toliau atnaujinti modelio parametrus ir minimizuoti nuostolio funkcijos reikšmę, įvertintą apmokymo aibėje. Iš tiesų tas, kuriam pakanka optimizavimą laikyti juodąja dėže, minimizuojančia tikslo funkcijas paprastoje aplinkoje, galėtų tenkintis žinojimu, kad egzistuoja eilė šio proceso „burtažodžių“ (pavadinimų, tokių kaip „SGD“ ir „Adam“).

Tačiau norint pasiekti gerų rezultatų, reikia gilesnių žinių.
Optimizavimo algoritmai yra svarbūs giliajam mokymuisi.
Viena vertus, sudėtingo giliojo mokymosi modelio apmokymas gali trukti valandas, dienas ar net savaites.
Optimizavimo algoritmo našumas tiesiogiai veikia modelio apmokymo efektyvumą.
Kita vertus, suprasdami skirtingų optimizavimo algoritmų principus ir jų hiperparametrų vaidmenį,
galėsime tikslingai derinti hiperparametrus, kad pagerintume giliojo mokymosi modelių našumą.

Šiame skyriuje išsamiai nagrinėjame įprastus giliojo mokymosi optimizavimo algoritmus.
Beveik visos giliajame mokymesi kylančios optimizavimo problemos yra *neiškilosios*.
Vis dėlto algoritmų projektavimas ir analizė *iškiliųjų* problemų kontekste pasirodė esą labai pamokantys.
Todėl šiame skyriuje pateikiamas įvadas į iškilųjį optimizavimą ir labai paprasto stochastinio gradientinio nusileidimo algoritmo iškiliajai tikslo funkcijai įrodymas.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```
