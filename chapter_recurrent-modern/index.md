# Šiuolaikiniai rekurentiniai neuroniniai tinklai
:label:`chap_modern_rnn`

Ankstesniame skyriuje pristatėme pagrindines idėjas,
slypinčias už rekurentinių neuroninių tinklų (RNN).
Tačiau, kaip ir su konvoliuciniais neuroniniais tinklais,
RNN architektūrose buvo daug naujovių,
kurios pasiekė kulminaciją keliuose sudėtinguose
dizainuose, praktikoje pasirodžiusiuose esant sėkmingais.
Visų pirma populiariausi dizainai
turi mechanizmus, skirtus sušvelninti liūdnai pagarsėjusį
RNN skaitinį nestabilumą,
kurio tipiški pavyzdžiai yra nykstantys ir sprogstantys gradientai.
Prisiminkite, kad :numref:`chap_rnn` su
sprogstančiais gradientais susidorojome taikydami paprastą
gradientų apkarpymo euristiką.
Nepaisant šios gudrybės veiksmingumo,
nykstančių gradientų problema lieka neišspręsta.

Šiame skyriuje pristatome pagrindines idėjas,
slypinčias už sėkmingiausių sekų RNN architektūrų,
kurios kilo iš dviejų straipsnių.
Pirmasis, *ilga trumpalaikė atmintis* (angl. *Long Short-Term Memory*) :cite:`Hochreiter.Schmidhuber.1997`,
pristato *atminties ląstelę* – skaičiavimo vienetą, pakeičiantį
tradicinius mazgus tinklo paslėptajame sluoksnyje.
Su šiomis atminties ląstelėmis tinklai geba
įveikti apmokymo sunkumus,
su kuriais susidūrė ankstesni rekurentiniai tinklai.
Intuityviai atminties ląstelė išvengia
nykstančio gradiento problemos,
išlaikydama kiekvienos atminties ląstelės vidinės būsenos reikšmes,
sklindančias rekurentine briauna, kurios svoris yra 1,
per daugelį paeiliui einančių laiko žingsnių.
Daugybinių sklendžių rinkinys padeda tinklui
nustatyti ne tik kokias įvestis leisti
į atminties būseną,
bet ir kada atminties būsenos turinys
turėtų paveikti modelio išvestį.

Antrasis straipsnis, *dvikryptis rekurentinis neuroninis tinklas* (angl. *Bidirectional Recurrent Neural Networks*) :cite:`Schuster.Paliwal.1997`,
pristato architektūrą, kurioje informacija
tiek iš ateities (vėlesnių laiko žingsnių),
tiek iš praeities (ankstesnių laiko žingsnių)
naudojama išvesčiai nustatyti
bet kuriame sekos taške.
Tai priešinga ankstesniems tinklams,
kuriuose išvestį gali paveikti tik praeities įvestis.
Dvikrypčiai RNN tapo pagrindiniu įrankiu
sekų žymėjimo užduotims natūraliosios kalbos apdorojime,
be daugybės kitų užduočių.
Laimei, šios dvi naujovės viena kitos neatmeta
ir buvo sėkmingai sujungtos fonemų klasifikavimui
:cite:`Graves.Schmidhuber.2005` ir rankraščio atpažinimui :cite:`graves2008novel`.


Pirmieji šio skyriaus skyreliai paaiškins LSTM architektūrą,
lengvesnę jos versiją, vadinamą sklendžių rekurentiniu vienetu (GRU),
pagrindines dvikrypčių RNN idėjas
ir trumpą paaiškinimą, kaip RNN sluoksniai
sukraunami kartu, kad sudarytų gilius RNN.
Vėliau nagrinėsime RNN taikymą
sekos į seką užduotyse,
pristatydami mašininį vertimą
kartu su pagrindinėmis idėjomis, tokiomis kaip *kodavimo ir dekodavimo* architektūros ir *spindulio paieška*.

```toc
:maxdepth: 2

lstm
gru
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

