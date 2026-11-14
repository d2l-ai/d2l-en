# Daugiasluoksniai perceptronai
:label:`chap_perceptrons`

Šiame skyriuje pristatysime jūsų pirmąjį tikrai *gilų* tinklą.
Paprasčiausi giliejie tinklai vadinami *daugiasluoksniais perceptronais*,
ir juos sudaro keli neuronų sluoksniai,
kurių kiekvienas pilnai sujungtas su žemiau esančio sluoksnio neuronais
(iš kurių gauna įvestį)
ir aukščiau esančiais (kuriems, savo ruožtu, daro įtaką).
Nors automatinis diferencijavimas
labai supaprastina giliojo mokymosi algoritmų įgyvendinimą,
išsamiai pasigilinsime, kaip šie gradientai
apskaičiuojami giliuosiuose tinkluose.
Tada būsime
pasirengę
aptarti klausimus, susijusius su skaitiniu stabilumu ir parametrų inicializavimu,
kurie yra esminiai sėkmingam giliųjų tinklų apmokymui.
Apmokydami tokius didelės talpos modelius, rizikuojame persimokyti. Todėl vėl
nagrinėsime reguliarizavimą ir apibendrinimą
giliesiems tinklams.
Visame skyriuje siekiame
suteikti jums tvirtą ne tik sąvokų, bet ir giliųjų tinklų naudojimo praktikos supratimą.
Šio skyriaus pabaigoje pritaikome tai, ką iki šiol pristatėme, realiam atvejui: namų kainų
prognozavimui. Klausimus, susijusius su skaičiavimo našumu, mastelio keitimu ir mūsų modelių
efektyvumu, paliekame tolesniems skyriams.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```
