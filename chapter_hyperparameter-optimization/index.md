# Hiperparametrų optimizavimas
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*) ir **Cedric Archambeau** (*Amazon*)

Kiekvieno mašininio mokymosi modelio našumas priklauso nuo jo hiperparametrų.
Jie valdo mokymosi algoritmą arba pagrindinio statistinio modelio struktūrą. Tačiau praktikoje nėra bendro būdo hiperparametrams parinkti. Vietoj to hiperparametrai dažnai nustatomi bandymų ir klaidų būdu arba kartais praktikų paliekami numatytosiomis reikšmėmis, o tai lemia
neoptimalų apibendrinimą.

Hiperparametrų optimizavimas suteikia sistemingą šios problemos sprendimo būdą,
formuluodamas ją kaip optimizavimo uždavinį: gera hiperparametrų aibė turėtų (bent jau)
minimizuoti patikros paklaidą. Palyginti su dauguma kitų mašininiame mokymesi
kylančių optimizavimo uždavinių, hiperparametrų optimizavimas yra įdėtinis, kur
kiekvienai iteracijai reikia apmokyti ir patikrinti mašininio mokymosi modelį.

Šiame skyriuje pirmiausia pristatysime hiperparametrų optimizavimo pagrindus. Taip pat pristatysime kai kuriuos naujausius pasiekimus, kurie pagerina
bendrą hiperparametrų optimizavimo efektyvumą išnaudodami pigiai įvertinamus
pradinės tikslo funkcijos pakaitalus. Skyriaus pabaigoje
turėtumėte gebėti taikyti pažangiausius hiperparametrų optimizavimo metodus
savo paties mašininio mokymosi algoritmo hiperparametrams optimizuoti.

```toc
:maxdepth: 2

hyperopt-intro
hyperopt-api
rs-async.md
sh-intro
sh-async
```
