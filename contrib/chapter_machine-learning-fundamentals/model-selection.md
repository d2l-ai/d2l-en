```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Modelio parinkimas
:label:`sec_model_selection`

Kaip mašininio mokymo mokslininkų,
mūsų tikslas yra atrasti *dėsningumus*.
Bet kaip galime būti tikri, kad
iš tikrųjų atradome *bendrą* dėsningumą,
o ne tiesiog įsiminėme savo duomenis?
Pavyzdžiui, įsivaizduokite, kad norėjome ieškoti
dėsningumų tarp genetinių žymenų,
siejančių pacientus su jų demencijos būkle,
kai žymės imamos iš aibės
$\{\text{demencija}, \text{lengvas pažintinis sutrikimas}, \text{sveikas}\}$.
Kadangi kiekvieno žmogaus genai jį unikaliai identifikuoja
(neįskaitant vienodų dvynių),
visą duomenų rinkinį galima įsiminti.

Nenorime, kad mūsų modelis sakytų:
*„Tai Bobas! Prisimenu jį! Jis serga demencija!“*
Priežastis paprasta.
Kai ateityje diegsime modelį,
susidursime su pacientais,
kurių modelis dar nebuvo matęs.
Mūsų prognozės bus naudingos tik tada,
jei mūsų modelis tikrai atrado *bendrą* dėsningumą.

Formaliau apibendrinant,
mūsų tikslas yra atrasti dėsningumus,
fiksuojančius dėsningumus pagrindinėje populiacijoje,
iš kurios buvo paimtas mūsų mokymo rinkinys.
Jei šioje veikloje pasiseks,
galėsime sėkmingai įvertinti riziką
net asmenims, su kuriais dar niekada nesusidūrėme.
Ši problema – kaip atrasti dėsningumus, kurie *generalizuojasi* –
yra pagrindinė mašininio mokymo problema.

Pavojus tas, kad mokydami modelius
turime prieigą tik prie nedidelės duomenų imties.
Didžiausiuose *su žymėmis* vieši paveikslėlių duomenų rinkiniuose, tokiuose kaip Imagenet :cite:`Deng.Dong.Socher.ea.2009`, yra
maždaug milijonas paveikslėlių. Neženklintos paveikslėlių kolekcijos, tokios kaip Flickr YFC100M duomenų rinkinys, gali būti gerokai didesnės – jame yra 100 milijonų paveikslėlių :cite:`Thomee.Shamma.Friedland.ea.2016`. Nors abu skaičiai atrodo dideli, jie yra maži, palyginti su visų galimų paveikslėlių, kuriuos galima nufotografuoti, tarkime, 1 megapikselio raiška, erdve.
Dar blogiau, kad dažnai turime mokytis iš vos šimtų pavyzdžių. Pavyzdžiui,
ligoninė gali turėti duomenų tik apie 100 retos ligos atvejų.
Dirbdami su baigtinėmis imtimis, rizikuojame
atrasti akivaizdžius sąryšius,
kurie pasirodo negalioja, kai surenkame daugiau duomenų.

Reiškinys, kai savo mokymo duomenims pritaikome modelį tiksliau nei pagrindiniam skirstiniui, vadinamas *persimokymu*, o metodai, naudojami kovoti su persimokymu, vadinami *reguliarizavimu*. Nors toliau pateikiamas tekstas nėra tinkamo įvado į statistinio mokymosi teoriją :cite:`Vapnik.1998,Boucheron.Bousquet.Lugosi.2005` pakaitalas, jis bent jau turėtų supažindinti jus su kai kuriais mokymosi metu atsirandančiais reiškiniais. Persimokymas iš tiesų labai paplitęs. Ankstesniuose skyriuose galbūt pastebėjote
persimokymą eksperimentuodami su Fashion-MNIST duomenų rinkiniu.
Jei eksperimento metu keitėte modelio struktūrą ar hiperparametrus, galbūt pastebėjote, kad su pakankamu skaičiumi neuronų, sluoksnių ir mokymo epochų modelis galiausiai gali pasiekti tobulą tikslumą mokymo rinkinyje, o tikslumas testavimo duomenyse prastėja.


## Mokymo paklaida ir generalizacijos paklaida

Kad šį reiškinį aptartume formaliau,
turime atskirti mokymo paklaidą ir generalizacijos paklaidą.
*Mokymo paklaida* $R_\mathrm{emp}$ yra mūsų modelio paklaida,
apskaičiuota mokymo duomenų rinkinyje,
o *generalizacijos paklaida* $R$ yra mūsų modelio paklaidos vidurkis,
jei jį taikytume begaliniam papildomų duomenų pavyzdžių srautui,
paimtam iš to paties pagrindinio duomenų skirstinio kaip ir mūsų pradinė imtis.
Jos apibrėžiamos taip:

$$R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] = \frac{1}{m} \sum_{i=1}^m l(\mathbf{x}_i, \mathbf{y}_i, f(\mathbf{x}_i))
\text{ and }
R[p, f] = E_{(\mathbf{x}, \mathbf{y}) \sim p} [l(\mathbf{x}, \mathbf{y}, f(\mathbf{x}))]$$

Problema ta, kad generalizacijos paklaidos $R$ niekada negalime apskaičiuoti tiksliai.
Taip yra todėl, kad begalinis duomenų srautas yra įsivaizduojamas objektas.
Praktikoje generalizacijos paklaidą turime *įvertinti*
taikydami savo modelį nepriklausomai testavimo aibei,
sudarytai iš atsitiktinai parinktų pavyzdžių $\mathbf{X}'$ ir žymių $\mathbf{Y}'$,
kurie buvo atskirti nuo mūsų mokymo rinkinio. Tai duoda $R_\mathrm{emp}[\mathbf{X}', \mathbf{Y}', f]$.

Šie trys minties eksperimentai
padės geriau iliustruoti šią situaciją.
Panagrinėkime studentą, besiruošiantį baigiamajam egzaminui.
Stropus studentas stengsis gerai pasitreniruoti
ir patikrinti savo gebėjimus naudodamas ankstesnių metų egzaminus.
Vis dėlto gerai išlaikyti praėjusių metų egzaminai negarantuoja,
kad jis pasirodys puikiai, kai tai bus svarbu.
Pavyzdžiui, studentas gali bandyti pasiruošti
mechaniškai įsimindamas atsakymus į egzamino klausimus.
Tam reikia įsiminti daug dalykų.
Jis gali net puikiai įsiminti praėjusių egzaminų atsakymus.
Kitas studentas gali ruoštis bandydamas suprasti
priežastis, kodėl pateikiami tam tikri atsakymai.
Nors tai paprastai gerai veikia vairuotojo pažymėjimo egzaminams,
rezultatai prasti, kai egzamino klausimų rinkinys yra
įvairesnis ir imamas iš didesnio, galbūt begalinio rinkinio.

Panašiai panagrinėkime modelį, kuris klausimams atsakyti tiesiog naudoja paieškos lentelę. Jei leistinų įvesčių aibė yra diskreti ir pakankamai maža, tai galbūt peržiūrėjus *daug* mokymo pavyzdžių šis metodas veiktų gerai. Vis dėlto šis modelis negali būti geresnis už atsitiktinį spėliojimą, susidūręs su pavyzdžiais, kurių niekada nematė.
Iš tikrųjų įvesčių erdvės yra per didelės, kad būtų galima įsiminti atsakymus į kiekvieną įsivaizduojamą įvestį. Pavyzdžiui, panagrinėkime nespalvotus $28\times28$ paveikslėlius. Jei kiekvienas pikselis gali įgyti vieną iš $256$ pilkumo reikšmių, tada galimų paveikslėlių yra $256^{784} \approx 10^{1888}$. Tai reiškia, kad mažos raiškos pilkų atspalvių miniatiūrų dydžio paveikslėlių yra gerokai daugiau nei maždaug $10^{82}$ atomų visatoje. Net jei galėtume susidurti su tokiais duomenimis, niekada negalėtume sau leisti saugoti paieškos lentelės. Šis reikalingų pavyzdžių skaičiaus sprogimas glaudžiai susijęs su dimensiškumo prakeiksmu, kai paprastos problemos tampa gana sudėtingos, kai duomenys yra aukštos dimensijos :cite:`Friedman.1997`.

Galiausiai panagrinėkime uždavinį,
kai bandoma klasifikuoti monetos metimų rezultatus (0 klasė: herbas, 1 klasė: skaičius)
pagal kokius nors galimus kontekstinius požymius.
Tarkime, kad moneta sąžininga.
Kad ir kokį algoritmą sugalvotume,
generalizacijos paklaida visada bus $\frac{1}{2}$.
Tačiau daugumai algoritmų
turėtume tikėtis, kad mokymo paklaida bus gerokai mažesnė,
priklausomai nuo traukimo sėkmės,
net jei neturėtume jokių požymių!
Panagrinėkime duomenų rinkinį {0, 1, 1, 1, 0, 1}.
Mūsų algoritmas be požymių turėtų grįžti prie to, kad visada prognozuotų
*daugumos klasę*, kuri iš mūsų ribotos imties atrodo esanti *1*.
Šiuo atveju modelis, visada prognozuojantis 1 klasę,
patirs $\frac{1}{3}$ paklaidą,
gerokai geresnę už mūsų generalizacijos paklaidą.
Didėjant duomenų kiekiui,
tikimybė, kad herbų dalis
reikšmingai nukryps nuo $\frac{1}{2}$, mažėja,
ir mūsų mokymo paklaida pradės sutapti su generalizacijos paklaida.

### Statistinio mokymosi teorija

Kadangi generalizacija yra pagrindinė mašininio mokymo problema,
turbūt nenustebsite sužinoję,
kad daugelis matematikų ir teoretikų paskyrė savo gyvenimus
formalioms teorijoms, aprašančioms šį reiškinį, kurti.
Savo [to paties vardo teoremoje](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) Glivenko ir Cantelli išvedė greitį, kuriuo mokymo paklaida
konverguoja į generalizacijos paklaidą :cite:`Glivenko.1933,Cantelli.1933`.
Straipsnių serijoje Vapnikas ir Chervonenkisas
išplėtė šią teoriją bendresnėms funkcijų klasėms
:cite:`Vapnik.Chervonenkis.1964,Vapnik.Chervonenkis.1968,Vapnik.Chervonenkis.1971,Vapnik.Chervonenkis.1981,Vapnik.Chervonenkis.1991,Vapnik.Chervonenkis.1974*1`.
Šis darbas padėjo statistinio mokymosi teorijos pamatus.

Standartiniame prižiūrimojo mokymosi nustatyme, kurį nagrinėjome
iki šiol ir kurio laikysimės didžiąją šios knygos dalį,
darome prielaidą, kad ir mokymo, ir testavimo duomenys
imami *nepriklausomai* iš *vienodų* skirstinių.
Tai paprastai vadinama *NVP prielaida* (nepriklausomai ir vienodai pasiskirstę, angl. *IID*). Ji reiškia, kad
visos imtys imamos iš to paties skirstinio. Ji taip pat reiškia, kad
žinant visas $n-1$ imčių, mums nėra lengviau prognozuoti $n$-ąją imtį,
nei prognozuoti pirmąją.

Gerai mašininio mokymo mokslininkui reikia kritiškai mąstyti,
ir jau dabar turėtumėte kvestionuoti šią prielaidą,
sugalvodami įprastų atvejų, kai ji negalioja.
Kas, jei apmokome mirtingumo rizikos prognozavimo modelį
su duomenimis, surinktais iš pacientų UCSF medicinos centre,
ir taikome jį pacientams Masačusetso bendrojoje ligoninėje?
Šie skirstiniai paprasčiausiai nėra vienodi. Tai gerai ištirta
statistikos problema :cite:`Rosenbaum.Rubin.1983`.
Be to, imtys gali būti koreliuotos laike.
Kas, jei klasifikuojame tviterio žinučių temas?
Naujienų ciklas sukurtų laikines priklausomybes
tarp aptariamų temų, pažeisdamas bet kokias nepriklausomumo prielaidas.

Kartais galime išsisukti su nedideliais NVP prielaidos pažeidimais,
ir mūsų modeliai ir toliau veiks nuostabiai gerai.
Galų gale beveik kiekvienas realaus pasaulio taikymas
apima bent nedidelį NVP prielaidos pažeidimą,
ir vis dėlto turime daug naudingų įrankių
įvairiems taikymams, tokiems kaip
veido atpažinimas,
kalbos atpažinimas ir kalbos vertimas. :cite:`Yu.1994` pateikia
kiekybinį šio elgesio įvertinimą.

Kiti pažeidimai tikrai kels problemų.
Įsivaizduokite, pavyzdžiui, kad bandome apmokyti
veido atpažinimo sistemą mokydami ją
vien su universiteto studentais
ir tada norime ją diegti kaip įrankį
geriatrijos pacientų slaugos namuose stebėjimui.
Tikriausiai tai neveiks gerai, nes universiteto studentai
atrodo gerokai kitaip nei vyresnio amžiaus žmonės.

Tolesniuose skyriuose aptarsime problemas,
atsirandančias dėl NVP prielaidos pažeidimų.
Kol kas, net laikydami NVP prielaidą savaime suprantama,
generalizacijos supratimas yra didžiulė problema.
Be to, tikslių teorinių pagrindų, galinčių paaiškinti,
kodėl gilieji neuroniniai tinklai generalizuojasi taip gerai,
išaiškinimas
vis dar kankina didžiausius mokymosi teorijos protus :cite:`Frankle.Carbin.2018,Bartlett.Montanari.Rakhlin.2021,Nagarajan.Kolter.2019,Kawaguchi.Kaelbling.Bengio.2017`.

Mokydami modelius, bandome ieškoti funkcijos,
kuri kuo geriau atitinka mokymo duomenis.
Jei funkcija tokia lanksti, kad gali užsikabinti už klaidingų dėsningumų
taip pat lengvai kaip už tikrų sąryšių,
tada ji gali veikti *per gerai*, nesukurdama modelio,
kuris gerai generalizuojasi nematytiems duomenims.
Būtent to norime išvengti arba bent kontroliuoti.
Daugelis giliojo mokymosi metodų yra euristikos ir gudrybės,
skirtos apsisaugoti nuo persimokymo (:numref:`sec_weight_decay`, :numref:`sec_dropout`, :numref:`sec_batch_norm`).

### Modelio sudėtingumas

Kai turime paprastus modelius ir gausius duomenis,
tikimės, kad generalizacijos paklaida bus panaši į mokymo paklaidą.
Kai dirbame su sudėtingesniais modeliais ir mažiau pavyzdžių,
tikimės, kad mokymo paklaida sumažės, bet generalizacijos tarpas padidės.
Kas tiksliai sudaro modelio sudėtingumą, yra sudėtingas klausimas.
Daug veiksnių lemia, ar modelis gerai generalizuosis.
Pavyzdžiui, modelis su daugiau parametrų paprastai gali būti laikomas sudėtingesniu.
Vis dėlto atkreipkite dėmesį, kad tai nebūtinai tiesa. Pavyzdžiui, branduolių metodai veikia erdvėse su begaliniu skaičiumi parametrų, tačiau jų modelio sudėtingumas yra labai gerai kontroliuojamas :cite:`Scholkopf.Smola.2002`.
Geriau apie tai galvoti taip: modelis,
kurio parametrai gali įgyti platesnį reikšmių diapazoną,
gali būti sudėtingesnis.
Dažnai su neuroniniais tinklais modelį,
kuriam reikia daugiau mokymo iteracijų, laikome sudėtingesniu,
o modelį, kuriam taikomas *ankstyvasis sustabdymas* (mažiau mokymo iteracijų), – mažiau sudėtingu :cite:`Prechelt.1998`.

Gali būti sunku palyginti sudėtingumą
iš esmės skirtingų modelių klasių atstovų
(pavyzdžiui, sprendimų medžių ir neuroninių tinklų).
Kol kas paprasta taisyklė yra gana naudinga:
modelis, galintis lengvai paaiškinti savavališkus faktus,
statistikų požiūriu yra sudėtingas,
o modelis, turintis tik ribotą išraiškos galią,
bet vis tiek sugebantis gerai paaiškinti duomenis,
greičiausiai yra arčiau tiesos :cite:`Vapnik.Levin.Le-Cun.1994`.
Filosofijoje tai glaudžiai susiję su Popperio
mokslinės teorijos falsifikuojamumo kriterijumi:
teorija yra gera, jei atitinka duomenis
ir jei yra konkrečių testų, kuriais ją galima paneigti.
Tai svarbu, nes visas statistinis vertinimas yra
*post hoc*,
t. y. vertiname po to, kai stebime faktus,
todėl esame pažeidžiami su tuo susijusios klaidos :cite:`Corfield.Scholkopf.Vapnik.2009`.
Kol kas filosofiją paliksime nuošalyje ir laikysimės apčiuopiamesnių klausimų.

Šiame skyriuje, kad suteiktume šiek tiek intuicijos,
sutelksime dėmesį į keletą veiksnių, kurie paprastai
veikia modelių klasės generalizuojamumą:

1. Derinamų parametrų skaičius. Kai derinamų parametrų skaičius, kartais vadinamas *laisvės laipsniais*, yra didelis, modeliai linkę būti jautresni persimokymui :cite:`Murata.Yoshizawa.Amari.1994`.
1. Parametrų įgyjamos reikšmės. Kai svoriai gali įgyti platesnį reikšmių diapazoną, modeliai gali būti jautresni persimokymui :cite:`Krogh.Hertz.1992`.
1. Mokymo pavyzdžių skaičius. Persimokyti duomenų rinkinyje, kuriame tėra vienas ar du pavyzdžiai, yra trivialiai lengva, net jei jūsų modelis paprastas. Tačiau persimokyti duomenų rinkinyje su milijonais pavyzdžių reikia itin lankstaus modelio :cite:`Henighan.Kaplan.Katz.ea.2020`.

## Modelio parinkimas

Mašininiame mokyme galutinį modelį paprastai parenkame
įvertinę keletą kandidatinių modelių.
Šis procesas vadinamas *modelio parinkimu*.
Kartais lyginami modeliai
iš esmės skiriasi pobūdžiu
(pavyzdžiui, sprendimų medžiai ir tiesiniai modeliai).
Kitais atvejais lyginame
tos pačios modelių klasės atstovus,
apmokytus su skirtingais hiperparametrų nustatymais.

Pavyzdžiui, su DBP
galime norėti palyginti modelius su
skirtingu paslėptųjų sluoksnių skaičiumi,
skirtingu paslėptųjų vienetų skaičiumi
ir įvairiais aktyvinimo funkcijų, taikomų kiekvienam paslėptajam sluoksniui, pasirinkimais. Ypač elegantišką
strategiją tam kompiuterinėje regoje pasiekti žr. :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.
Kad nustatytume geriausią iš savo kandidatinių modelių,
paprastai naudosime validavimo duomenų rinkinį.


### Validavimo duomenų rinkinys

Iš principo neturėtume liesti savo testavimo aibės,
kol nepasirinkome visų savo hiperparametrų.
Jei testavimo duomenis naudotume modelio parinkimo procese,
kyla rizika, kad galime persimokyti testavimo duomenims.
Tada patektume į rimtą bėdą.
Jei persimokome mokymo duomenims,
visada yra vertinimas testavimo duomenimis, kuris mus laiko sąžiningus.
Bet jei persimokytume testavimo duomenims, kaip apie tai sužinotume?
Pavyzdžiui, žr. :cite:`Ong.Smola.Williamson.ea.2005`, kur parodyta, kaip
tai gali lemti absurdiškus rezultatus net modeliams, kurių sudėtingumą
galima tiksliai kontroliuoti.

Taigi modelio parinkimui niekada neturėtume remtis testavimo duomenimis.
Ir vis dėlto modelio parinkimui negalime remtis vien mokymo duomenimis,
nes
negalime įvertinti generalizacijos paklaidos
tais pačiais duomenimis, kuriuos naudojame modeliui apmokyti.


Praktiniuose taikymuose vaizdas darosi miglotesnis.
Nors idealiu atveju testavimo duomenų imtumėmės tik vieną kartą,
kad įvertintume pačius geriausius modelius arba palygintume
nedidelį skaičių modelių tarpusavyje,
realaus pasaulio testavimo duomenys retai atmetami panaudojus tik vieną kartą.
Retai kada galime sau leisti naują testavimo aibę kiekvienam eksperimentų ciklui.
Iš tikrųjų, dešimtmečius pakartotinai naudojami lyginamieji duomenys gali turėti didelę įtaką
algoritmų vystymuisi, pavyzdžiui, [paveikslėlių klasifikavimo](https://paperswithcode.com/sota/image-classification-on-imagenet) ir [optinio simbolių atpažinimo](https://paperswithcode.com/sota/image-classification-on-mnist) srityse.

Įprasta praktika problemai „mokymas testavimo aibėje“ spręsti
yra duomenis padalyti į tris dalis,
be mokymo ir testavimo duomenų rinkinių, įtraukiant ir *validavimo aibę*.
Rezultatas – miglota praktika, kai ribos
tarp validavimo ir testavimo duomenų yra nerimą keliančiai neaiškios.
Jei aiškiai nenurodyta kitaip, šios knygos eksperimentuose
iš tikrųjų dirbame su tuo, ką teisingai reikėtų vadinti
mokymo duomenimis ir validavimo duomenimis, be tikrų testavimo aibių.
Todėl kiekviename knygos eksperimente pateikiamas tikslumas iš tikrųjų yra
validavimo tikslumas, o ne tikras testavimo aibės tikslumas.

### $K$ dalių kryžminė validacija

Kai mokymo duomenų trūksta,
galime net negalėti sau leisti atskirti
pakankamai duomenų tinkamai validavimo aibei sudaryti.
Vienas populiarus šios problemos sprendimas yra naudoti
$K$ *dalių kryžminę validaciją*.
Čia pradiniai mokymo duomenys padalijami į $K$ nesikertančių poaibių.
Tada modelio mokymas ir validavimas vykdomi $K$ kartų,
kiekvieną kartą mokant su $K-1$ poaibių ir validuojant
su kitu poaibiu (tuo, kuris tą ratą nebuvo naudotas mokymui).
Galiausiai mokymo ir validavimo paklaidos įvertinamos
suvidurkinant $K$ eksperimentų rezultatus.

## Nepakankamas mokymasis ar persimokymas?

Lygindami mokymo ir validavimo paklaidas,
turime atkreipti dėmesį į dvi įprastas situacijas.
Pirma, reikia stebėti atvejus,
kai ir mokymo, ir validavimo paklaida yra didelės,
bet tarp jų yra nedidelis tarpas.
Jei modelis negali sumažinti mokymo paklaidos,
tai gali reikšti, kad mūsų modelis yra per paprastas
(t. y. nepakankamai išraiškingas),
kad užfiksuotų dėsningumą, kurį bandome modeliuoti.
Be to, kadangi *generalizacijos tarpas*
tarp mokymo ir validavimo paklaidų yra mažas,
turime pagrindo manyti, kad galėtume išsisukti su sudėtingesniu modeliu.
Šis reiškinys žinomas kaip *nepakankamas mokymasis* (tačiau atkreipkite dėmesį, kad tai taip pat gali
reikšti, jog problema tiesiog labai sunki).

Kita vertus, kaip aptarėme aukščiau,
reikia stebėti atvejus,
kai mūsų mokymo paklaida yra gerokai mažesnė
nei validavimo paklaida, o tai rodo didelį *persimokymą*.
Atkreipkite dėmesį, kad persimokymas ne visada yra blogas dalykas.
Ypač giliajame mokyme gerai žinoma,
kad geriausi prognozavimo modeliai dažnai veikia
gerokai geriau mokymo duomenyse nei atidėtuose duomenyse.
Galiausiai mums dažniausiai labiau rūpi validavimo paklaida,
o ne tarpas tarp mokymo ir validavimo paklaidų.

Ar persimokysime, ar nepakankamai išmoksime, gali priklausyti
ir nuo mūsų modelio sudėtingumo,
ir nuo turimų mokymo duomenų rinkinių dydžio –
dviejų temų, kurias aptarsime toliau.

### Modelio sudėtingumas

Kad iliustruotume šiek tiek klasikinės intuicijos
apie persimokymą ir modelio sudėtingumą,
pateikiame pavyzdį su polinomais.
Turėdami mokymo duomenis, sudarytus iš vieno požymio $x$
ir atitinkamos realiosios reikšmės žymės $y$,
bandome rasti $d$ laipsnio polinomą

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

žymėms $y$ įvertinti.
Tai tiesiog tiesinės regresijos uždavinys,
kuriame mūsų požymiai yra $x$ laipsniai,
modelio svoriai yra $w_i$,
o poslinkis yra $w_0$, nes visiems $x$ galioja $x^0 = 1$.
Kadangi tai tiesiog tiesinės regresijos uždavinys,
kaip nuostolių funkciją galime naudoti kvadratinę paklaidą.


Aukštesnės eilės polinominė funkcija yra sudėtingesnė
už žemesnės eilės polinominę funkciją,
nes aukštesnės eilės polinomas turi daugiau parametrų
ir modelio funkcijos pasirinkimo diapazonas yra platesnis.
Esant fiksuotam mokymo duomenų rinkiniui,
aukštesnės eilės polinominės funkcijos visada turėtų
pasiekti mažesnę (blogiausiu atveju – tokią pat) mokymo paklaidą
nei žemesnio laipsnio polinomai.
Iš tikrųjų, kai kiekvienas duomenų pavyzdys turi skirtingą $x$ reikšmę,
polinominė funkcija, kurios laipsnis lygus duomenų pavyzdžių skaičiui,
gali tobulai pritaikyti mokymo rinkinį.
Ryšį tarp polinomo laipsnio
ir nepakankamo mokymosi bei persimokymo vizualizuojame :numref:`fig_capacity_error`.

![Modelio sudėtingumo įtaka nepakankamam mokymuisi ir persimokymui](../img/capacity-vs-error.svg)
:label:`fig_capacity_error`

Didžioji dalis šios intuicijos kyla iš statistinio mokymosi teorijos. Viena iš jos
suteikiamų garantijų :cite:`Vapnik.1998` yra ta, kad tarpas tarp empirinės rizikos ir tikėtinos rizikos yra apribotas

$$\Pr\left(R[p, f] - R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \epsilon\right) \geq 1-\delta
\ \text{for}\ \epsilon \geq c \sqrt{(\mathrm{VC} - \log \delta)/n}.$$

Čia $\delta > 0$ yra tikimybė, kad riba bus pažeista, o $\mathrm{VC}$ yra Vapniko ir Chervonenkiso (VC)
dimensija funkcijų aibės, kurią norime pritaikyti. Pavyzdžiui, $d$ laipsnio polinomams VC dimensija yra $d+1$. Galiausiai $c > 0$ yra konstanta, priklausanti tik nuo galimų nuostolių mastelio. Trumpai tariant, tai rodo, kad mūsų riba tampa vis laisvesnė, kai pasirenkame sudėtingesnius modelius, ir kad laisvųjų parametrų skaičius neturėtų didėti sparčiau nei didėja duomenų rinkinio dydis $n$. Išsamią diskusiją ir daug pažangesnius funkcijų sudėtingumo matavimo būdus žr. :cite:`Boucheron.Bousquet.Lugosi.2005`.

### Duomenų rinkinio dydis

Kaip jau rodo aukščiau pateikta riba, kitas svarbus aspektas, į kurį reikia atsižvelgti, yra duomenų rinkinio dydis.
Esant fiksuotam modeliui, kuo mažiau imčių turime mokymo duomenų rinkinyje,
tuo labiau tikėtina (ir tuo stipriau) susidursime su persimokymu.
Didėjant mokymo duomenų kiekiui,
generalizacijos paklaida paprastai mažėja.
Be to, apskritai daugiau duomenų niekada nekenkia.
Esant fiksuotai užduočiai ir duomenų skirstiniui, modelio sudėtingumas neturėtų
didėti sparčiau nei duomenų kiekis.
Turint daugiau duomenų, galime naudingai bandyti pritaikyti sudėtingesnį modelį.
Nesant pakankamai duomenų, paprastesnius modelius gali būti sunkiau pranokti.
Daugeliui užduočių gilusis mokymasis pranoksta tiesinius modelius
tik tada, kai turima daug tūkstančių mokymo pavyzdžių.
Iš dalies dabartinė giliojo mokymosi sėkmė
yra dėl dabartinio didžiulių duomenų rinkinių gausos,
kurią lėmė interneto įmonės, pigi saugykla, prijungti įrenginiai
ir plati ekonomikos skaitmenizacija.


## Santrauka

Šiame skyriuje nagrinėjome kai kuriuos teorinius mašininio mokymo pagrindus. Šiuos pagrindus pritaikyti šiuolaikiniam giliajam mokymui vis dar yra nebaigtas darbas. Vien mokymo paklaidos minimizavimas nebūtinai reiškia generalizacijos paklaidos sumažėjimą. Mašininio mokymo modeliai turi būti atsargūs ir saugotis persimokymo, kad generalizacijos paklaida būtų kuo mažesnė. Vis dėlto pateikėme šiek tiek pagrindinės intuicijos, kaip kontroliuoti generalizacijos paklaidą. Pavyzdžiui, galime pasitelkti validavimo aibes arba statistines ribas.

Keletas praktinių taisyklių: 1) Validavimo aibę galima naudoti modelio parinkimui, jei ji nenaudojama pernelyg laisvai. 2) Sudėtingesniam modeliui reikia daugiau duomenų, ir duomenų kiekis turėtų didėti bent jau taip pat sparčiai kaip modelio sudėtingumas. 3) Daugiau parametrų gali reikšti sudėtingesnius modelius, bet yra būdų, kai taip nėra, pavyzdžiui, kontroliuojant dydį. 4) Daugiau duomenų viską pagerina. Kol duomenys imami iš to paties skirstinio. 5) Tikrinkite savo prielaidas.

## Pratimai

1. Ar galite tiksliai išspręsti polinominės regresijos uždavinį?
1. Pateikite bent penkis pavyzdžius, kai priklausomi atsitiktiniai dydžiai daro nepatartiną uždavinio traktavimą kaip NVP duomenų.
1. Ar kada nors galite tikėtis matyti nulinę mokymo paklaidą? Kokiomis aplinkybėmis matytumėte nulinę generalizacijos paklaidą?
1. Kodėl $k$ dalių kryžminė validacija yra labai brangi skaičiavimo požiūriu?
1. Kodėl $k$ dalių kryžminės validacijos paklaidos įvertis yra paslinktas?
1. VC dimensija apibrėžiama kaip didžiausias taškų skaičius, kurį funkcijų klasės funkcija gali klasifikuoti su savavališkomis žymėmis $\{\pm 1\}$. Kodėl tai gali būti netinkamas būdas matuoti, kokia sudėtinga yra funkcijų klasė? Užuomina: kaip dėl funkcijų dydžio?
1. Vadovas duoda jums sudėtingą duomenų rinkinį, kuriame jūsų dabartinis algoritmas veikia nelabai gerai. Kaip jam pagrįstumėte, kad jums reikia daugiau duomenų? Užuomina: duomenų padidinti negalite, bet galite sumažinti.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/234)
:end_tab:
