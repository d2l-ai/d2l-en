# Mašininio mokymo pagrindai
:label:`chap_ml-fundamentals`

Kaip iliustruota :numref:`chap_introduction`,
gilusis mokymasis yra tik vienas iš daugelio populiarių mašininio mokymo uždavinių sprendimo metodų.
Kaip matėme mokydami
tiesines regresijas, softmax regresijas
ir daugiasluoksnius perceptronus,
optimizavimo algoritmai
mažina nuostolių funkcijos reikšmes
iteratyviai atnaujindami modelio parametrus.
Tačiau
mokydami didelės talpos modelius,
tokius kaip giliuosius neuroninius tinklus, rizikuojame, kad modelis persimokys.
Todėl pirmą kartą griežtai pristatysime
persimokymo, nepakankamo mokymosi ir modelio parinkimo sąvokas.
Kad padėtume su šiomis problemomis kovoti,
pristatysime reguliarizavimo metodus, tokius kaip svorių nykimas ir atsitiktinis praretinimas.
Atsižvelgiant į daugelį nesėkmingų mašininio mokymo *diegimų*,
būtina
atskleisti kai kuriuos įprastus rūpesčius
ir skatinti kritinį mąstymą, reikalingą tokioms situacijoms anksti aptikti, žalai sumažinti ir mašininiam mokymui atsakingai naudoti.
Visame kelyje siekiame suteikti tvirtą ne tik sąvokų,
bet ir mašininio mokymo modelių naudojimo praktikos suvokimą.
Šio skyriaus pabaigoje
tai, ką iki šiol pristatėme, pritaikome realiam atvejui: namų kainų prognozavimui.
Klausimus, susijusius su mūsų modelių skaičiavimo našumu,
mastelio keitimu ir efektyvumu, paliekame vėlesniems skyriams.

```toc
:maxdepth: 2

model-selection
underfit-overfit
```
