# Kūrėjo vadovas
:label:`chap_computation`

Kartu su milžiniškais duomenų rinkiniais ir galinga aparatine įranga,
puikūs programiniai įrankiai atliko nepakeičiamą vaidmenį
sparčioje giliojo mokymosi pažangoje.
Pradedant nuo novatoriškos „Theano“ bibliotekos, išleistos 2007 m.,
lankstūs atvirojo kodo įrankiai leido tyrėjams
greitai kurti modelių prototipus, išvengiant pasikartojančio darbo,
pakartotinai naudojant standartinius komponentus,
bet išlaikant galimybę atlikti žemo lygio pakeitimus.
Laikui bėgant giliojo mokymosi bibliotekos išsivystė
ir siūlo vis grubesnes abstrakcijas.
Kaip puslaidininkių projektuotojai pereidavo nuo tranzistorių specifikavimo
prie loginių grandinių ir kodo rašymo,
taip neuroninių tinklų tyrėjai perėjo nuo galvojimo apie
atskirų dirbtinių neuronų elgseną
prie tinklų suvokimo sluoksnių visumos požiūriu,
ir dabar dažnai projektuoja architektūras galvodami apie daug grubesnius *blokus*.


Iki šiol pristatėme keletą pagrindinių mašininio mokymosi sąvokų,
žingsnis po žingsnio kildami iki pilnai veikiančių giliojo mokymosi modelių.
Ankstesniame skyriuje
įgyvendinome kiekvieną MLP komponentą nuo nulio
ir net parodėme, kaip pasinaudoti aukšto lygio API,
kad tuos pačius modelius sukurtume be vargo.
Kad jus taip toli ir taip greitai nuvestume, mes *pasitelkėme* bibliotekas,
bet praleidome pažangesnes detales apie tai, *kaip jos veikia*.
Šiame skyriuje atidengsime uždangą,
gilindamiesi į pagrindinius giliojo mokymosi skaičiavimo komponentus,
t. y. modelio konstravimą, parametrų prieigą ir inicializavimą,
pasirinktinių sluoksnių ir blokų projektavimą, modelių skaitymą iš disko ir rašymą į diską
bei GPU panaudojimą dramatiškam spartinimui pasiekti.
Šios įžvalgos jus iš *galutinio naudotojo* pavers *pažengusiu naudotoju*,
suteikdamos įrankius, reikalingus pasinaudoti
brandžios giliojo mokymosi bibliotekos pranašumais, išlaikant lankstumą
įgyvendinti sudėtingesnius modelius, įskaitant tuos, kuriuos išrasite patys!
Nors šiame skyriuje nepristatomi jokie nauji modeliai ar duomenų rinkiniai,
vėlesni pažangaus modeliavimo skyriai labai remiasi šiomis technikomis.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```
