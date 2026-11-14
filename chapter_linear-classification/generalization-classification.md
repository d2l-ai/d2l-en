# Apibendrinimas klasifikavime

:label:`chap_classification_generalization`



Iki šiol sutelkėme dėmesį į tai, kaip spręsti daugiaklasius klasifikavimo uždavinius,
apmokant (tiesinius) neuroninius tinklus su keliomis išvestimis ir softmax funkcijomis.
Interpretuodami savo modelio išvestis kaip tikimybines prognozes,
pagrindėme ir išvedėme kryžminės entropijos nuostolių funkciją,
kuri apskaičiuoja neigiamą logaritminį tikėtinumą,
kurį mūsų modelis (su fiksuotu parametrų rinkiniu)
priskiria tikriesiems žymenims.
Ir galiausiai pritaikėme šias priemones praktikoje,
pritaikydami savo modelį apmokymo aibei.
Tačiau, kaip visada, mūsų tikslas yra išmokti *bendrus dėsningumus*,
empiriškai įvertinamus anksčiau nematytuose duomenyse (testavimo aibėje).
Aukštas tikslumas apmokymo aibėje nieko nereiškia.
Kai tik kiekviena mūsų įvestis yra unikali
(ir iš tiesų taip yra daugumoje didelės dimensijos duomenų rinkinių),
galime pasiekti tobulą tikslumą apmokymo aibėje,
tiesiog įsimindami duomenų rinkinį pirmojoje apmokymo epochoje
ir vėliau ieškodami žymens, kai tik pamatome naują vaizdą.
Ir vis dėlto tikslių žymenų,
susietų su tiksliais apmokymo pavyzdžiais, įsiminimas
nepasako, kaip klasifikuoti naujus pavyzdžius.
Be tolesnių nurodymų, galime būti priversti grįžti
prie atsitiktinio spėliojimo, kai tik susiduriame su naujais pavyzdžiais.

Keletas degančių klausimų reikalauja neatidėliotino dėmesio:

1. Kiek testavimo pavyzdžių mums reikia, kad gerai įvertintume savo klasifikatorių tikslumą pagrindinėje populiacijoje?
1. Kas nutinka, jei toliau vertiname modelius tuo pačiu testavimo rinkiniu pakartotinai?
1. Kodėl turėtume tikėtis, kad mūsų tiesinių modelių pritaikymas apmokymo aibei
   veiks geriau nei mūsų naivi įsiminimo schema?


Kol :numref:`sec_generalization_basics` pristatė
persimokymo ir apibendrinimo pagrindus
tiesinės regresijos kontekste,
šis skyrius eis šiek tiek giliau,
pristatydamas kai kurias fundamentalias
statistinio mokymosi teorijos idėjas.
Pasirodo, kad dažnai galime garantuoti apibendrinimą *a priori*:
daugeliui modelių
ir bet kuriai norimai
apibendrinimo atotrūkio ribai $\epsilon$
dažnai galime nustatyti tam tikrą reikiamą imčių skaičių $n$,
kad jei mūsų apmokymo aibėje yra bent $n$
imčių, mūsų empirinė paklaida bus
$\epsilon$ ribose nuo tikrosios paklaidos,
*bet kuriam duomenis generuojančiam skirstiniui*.
Deja, taip pat paaiškėja,
kad nors tokios garantijos suteikia
gilų intelektualinių statybinių blokų rinkinį,
jos turi ribotą praktinę naudą
giliojo mokymosi specialistui.
Trumpai tariant, šios garantijos rodo,
kad užtikrinti giliųjų neuroninių tinklų apibendrinimą
*a priori*
reikia absurdiško skaičiaus pavyzdžių
(galbūt trilijonų ar daugiau),
net kai matome, kad mums rūpimose užduotyse
giliesiems neuroniniams tinklams paprastai
nepaprastai gerai sekasi apibendrinti su kur kas mažiau pavyzdžių (tūkstančiais).
Todėl giliojo mokymosi specialistai dažnai apskritai atsisako
*a priori* garantijų,
vietoj to naudodami metodus,
kurie gerai apibendrino
panašiuose uždaviniuose praeityje,
ir apibendrinimą patvirtindami *post hoc*
empiriniais vertinimais.
Kai pasieksime :numref:`chap_perceptrons`,
vėl grįšime prie apibendrinimo
ir pateiksime lengvą įvadą
į plačią mokslinę literatūrą,
kuri atsirado bandant
paaiškinti, kodėl giliesiems neuroniniams tinklams praktikoje pavyksta apibendrinti.

## Testavimo aibė

Kadangi jau pradėjome remtis testavimo aibėmis
kaip aukso standarto metodu
apibendrinimo paklaidai vertinti,
pradėkime aptardami
tokių paklaidos įverčių savybes.
Sutelkime dėmesį į fiksuotą klasifikatorių $f$,
nesirūpindami, kaip jis gautas.
Be to, tarkime, kad turime
*naują* pavyzdžių duomenų rinkinį $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$,
kuris nebuvo naudotas klasifikatoriui $f$ apmokyti.
Mūsų klasifikatoriaus $f$ *empirinė paklaida* rinkinyje $\mathcal{D}$
yra tiesiog egzempliorių dalis,
kuriems prognozė $f(\mathbf{x}^{(i)})$
nesutampa su tikruoju žymeniu $y^{(i)}$,
ir ji išreiškiama šia išraiška:

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

Priešingai, *populiacijos paklaida*
yra *tikėtina* pavyzdžių dalis
pagrindinėje populiacijoje
(tam tikras skirstinys $P(X,Y)$, apibūdinamas
tikimybės tankio funkcija $p(\mathbf{x},y)$),
kuriai mūsų klasifikatorius nesutampa
su tikruoju žymeniu:

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Nors $\epsilon(f)$ yra dydis, kuris mums iš tikrųjų rūpi,
negalime jo stebėti tiesiogiai,
kaip negalime tiesiogiai
stebėti vidutinio ūgio didelėje populiacijoje,
neišmatavę kiekvieno žmogaus.
Šį dydį galime tik įvertinti pagal imtis.
Kadangi mūsų testavimo aibė $\mathcal{D}$
statistiškai atspindi
pagrindinę populiaciją,
galime žiūrėti į $\epsilon_\mathcal{D}(f)$ kaip į statistinį
populiacijos paklaidos $\epsilon(f)$ įvertinį.
Be to, kadangi mus dominantis dydis $\epsilon(f)$
yra vidurkis (atsitiktinio dydžio $\mathbf{1}(f(X) \neq Y)$),
o atitinkamas įvertinys $\epsilon_\mathcal{D}(f)$
yra imties vidurkis,
populiacijos paklaidos įvertinimas
yra tiesiog klasikinis vidurkio vertinimo uždavinys,
kurį galbūt prisimenate iš :numref:`sec_prob`.

Svarbus klasikinis tikimybių teorijos rezultatas,
vadinamas *centrine ribine teorema*, garantuoja,
kad kai tik turime $n$ atsitiktinių imčių $a_1, ..., a_n$,
paimtų iš bet kurio skirstinio su vidurkiu $\mu$ ir standartiniu nuokrypiu $\sigma$,
tada, imčių skaičiui $n$ artėjant prie begalybės,
imties vidurkis $\hat{\mu}$ apytiksliai
linksta į normalųjį skirstinį, kurio centras
yra tikrasis vidurkis, o standartinis nuokrypis – $\sigma/\sqrt{n}$.
Jau tai mums pasako ką nors svarbaus:
didėjant pavyzdžių skaičiui,
mūsų testavimo paklaida $\epsilon_\mathcal{D}(f)$
turėtų artėti prie tikrosios paklaidos $\epsilon(f)$
sparta $\mathcal{O}(1/\sqrt{n})$.
Taigi, kad įvertintume savo testavimo paklaidą dvigubai tiksliau,
turime surinkti keturis kartus didesnę testavimo aibę.
Kad sumažintume savo testavimo paklaidą šimtą kartų,
turime surinkti dešimt tūkstančių kartų didesnę testavimo aibę.
Apskritai tokia sparta $\mathcal{O}(1/\sqrt{n})$
dažnai yra geriausia, ko galime tikėtis statistikoje.

Dabar, kai žinome ką nors apie asimptotinę spartą,
kuria mūsų testavimo paklaida $\epsilon_\mathcal{D}(f)$ konverguoja į tikrąją paklaidą $\epsilon(f)$,
galime pasigilinti į kai kurias svarbias detales.
Prisiminkite, kad dominantis atsitiktinis dydis
$\mathbf{1}(f(X) \neq Y)$
gali įgyti tik reikšmes $0$ ir $1$,
todėl yra Bernulio atsitiktinis dydis,
apibūdinamas parametru,
nurodančiu tikimybę, kad jis įgis reikšmę $1$.
Čia $1$ reiškia, kad mūsų klasifikatorius padarė klaidą,
todėl mūsų atsitiktinio dydžio parametras
iš tikrųjų yra tikrasis klaidų dažnis $\epsilon(f)$.
Bernulio dispersija $\sigma^2$
priklauso nuo jo parametro (čia $\epsilon(f)$)
pagal išraišką $\epsilon(f)(1-\epsilon(f))$.
Nors $\epsilon(f)$ iš pradžių nežinomas,
žinome, kad jis negali būti didesnis nei $1$.
Šiek tiek patyrinėjus šią funkciją,
paaiškėja, kad mūsų dispersija yra didžiausia,
kai tikrasis klaidų dažnis yra artimas $0.5$,
ir gali būti gerokai mažesnė, kai jis
artimas $0$ ar artimas $1$.
Tai mums sako, kad asimptotinis standartinis nuokrypis
mūsų įverčio $\epsilon_\mathcal{D}(f)$ paklaidai $\epsilon(f)$
(per $n$ testavimo imčių pasirinkimą)
negali būti didesnis nei $\sqrt{0.25/n}$.

Jei nepaisysime fakto, kad ši sparta apibūdina
elgesį, kai testavimo aibės dydis artėja prie begalybės,
o ne kai turime baigtines imtis,
tai mums sako, kad jei norime, kad mūsų testavimo paklaida $\epsilon_\mathcal{D}(f)$
aproksimuotų populiacijos paklaidą $\epsilon(f)$
taip, kad vienas standartinis nuokrypis atitiktų
$\pm 0.01$ intervalą,
tuomet turėtume surinkti maždaug 2500 imčių.
Jei norime, kad du standartiniai nuokrypiai tilptų
tame diapazone ir taip būtume 95 % tikri,
kad $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$,
tuomet mums reikės 10 000 imčių!

Pasirodo, tai yra daugelio populiarių mašininio mokymosi lyginamųjų rinkinių
testavimo aibių dydis.
Galbūt nustebsite sužinoję, kad kasmet skelbiama tūkstančiai
taikomojo giliojo mokymosi straipsnių,
kuriuose didelis dėmesys skiriamas klaidų dažnio pagerėjimui $0.01$ ar mažiau.
Žinoma, kai klaidų dažniai yra daug artimesni $0$,
pagerėjimas $0.01$ iš tiesų gali būti labai svarbus.


Viena erzinanti mūsų iki šiol atliktos analizės savybė yra ta,
kad ji iš tikrųjų pasako tik apie asimptotiką,
t. y. kaip ryšys tarp $\epsilon_\mathcal{D}$ ir $\epsilon$
kinta, kai mūsų imties dydis artėja prie begalybės.
Laimei, kadangi mūsų atsitiktinis dydis yra aprėžtas,
galime gauti galiojančias baigtinės imties ribas,
pritaikę Hoeffdingo (1963) nelygybę:

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Išsprendę mažiausią duomenų rinkinio dydį,
kuris leistų mums padaryti išvadą
su 95 % pasitikėjimu, kad atstumas $t$
tarp mūsų įverčio $\epsilon_\mathcal{D}(f)$
ir tikrojo klaidų dažnio $\epsilon(f)$
neviršija $0.01$,
rasite, kad reikia maždaug 15 000 pavyzdžių,
palyginti su 10 000 pavyzdžių, kuriuos siūlo
aukščiau pateikta asimptotinė analizė.
Jei gilinsitės į statistiką,
pamatysite, kad ši tendencija galioja apskritai.
Garantijos, galiojančios net baigtinėse imtyse,
paprastai yra šiek tiek konservatyvesnės.
Atkreipkite dėmesį, kad bendrame kontekste
šie skaičiai nėra labai nutolę,
o tai atspindi bendrą asimptotinės analizės naudingumą,
suteikiant mums apytikslius skaičius, net jei jie nėra
garantijos, su kuriomis galėtume eiti į teismą.

## Testavimo aibės pakartotinis naudojimas

Tam tikra prasme dabar esate pasirengę sėkmingai
atlikti empirinius mašininio mokymosi tyrimus.
Beveik visi praktiniai modeliai kuriami
ir validuojami pagal veikimą testavimo aibėje,
o jūs dabar esate testavimo aibės meistras.
Bet kuriam fiksuotam klasifikatoriui $f$
žinote, kaip įvertinti jo testavimo paklaidą $\epsilon_\mathcal{D}(f)$,
ir tiksliai žinote, ką galima (ir ko negalima)
pasakyti apie jo populiacijos paklaidą $\epsilon(f)$.

Taigi tarkime, kad pasitelkiate šias žinias
ir ruošiatės apmokyti savo pirmąjį modelį $f_1$.
Žinodami, kokio pasitikėjimo reikia
jūsų klasifikatoriaus klaidų dažnio veikimu,
taikote aukščiau pateiktą analizę, kad nustatytumėte
tinkamą pavyzdžių skaičių,
kurį skirsite testavimo aibei.
Be to, tarkime, kad rimtai priėmėte pamokas iš
:numref:`sec_generalization_basics`
ir pasirūpinote išsaugoti testavimo aibės neliečiamumą,
atlikdami visą išankstinę analizę,
hiperparametrų derinimą ir net parinkimą
tarp kelių konkuruojančių modelių architektūrų
validavimo aibėje.
Galiausiai įvertinate savo modelį $f_1$
testavimo aibėje ir pateikiate nepaslinktą
populiacijos paklaidos įvertį
su susijusiu pasikliautinuoju intervalu.

Kol kas viskas atrodo gerai.
Tačiau tą naktį pabundate 3 val. ryto
su puikia nauja modeliavimo metodo idėja.
Kitą dieną užkoduojate savo naują modelį,
suderinate jo hiperparametrus validavimo aibėje,
ir ne tik jūsų naujas modelis $f_2$ veikia,
bet ir jo klaidų dažnis atrodo gerokai mažesnis nei $f_1$.
Tačiau atradimo džiaugsmas staiga išblėsta,
kai ruošiatės galutiniam vertinimui.
Jūs neturite testavimo aibės!

Nors pradinė testavimo aibė $\mathcal{D}$
vis dar guli jūsų serveryje,
dabar susiduriate su dviem didžiulėmis problemomis.
Pirma, kai rinkote savo testavimo aibę,
reikiamą tikslumo lygį nustatėte
darydami prielaidą, kad vertinate
vieną klasifikatorių $f$.
Tačiau jei imatės vertinti kelis klasifikatorius $f_1, ..., f_k$
toje pačioje testavimo aibėje,
turite apsvarstyti klaidingo atradimo problemą.
Anksčiau galėjote būti 95 % tikri,
kad $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$
vienam klasifikatoriui $f$,
todėl klaidinančio rezultato tikimybė
buvo vos 5 %.
Su $k$ klasifikatorių rinkiniu
sunku garantuoti,
kad nėra nė vieno iš jų,
kurio testavimo aibės rezultatas klaidinantis.
Nagrinėjant 20 klasifikatorių,
galite apskritai neturėti galios
atmesti galimybės,
kad bent vienas iš jų
gavo klaidinantį įvertį.
Ši problema susijusi su daugkartiniu hipotezių tikrinimu,
kuris, nepaisant plačios statistikos literatūros,
išlieka nuolatine mokslinius tyrimus kamuojančia problema.


Jei to nepakanka jums sunerimti,
yra ypatinga priežastis nepasitikėti
rezultatais, gaunamais vėlesniuose vertinimuose.
Prisiminkite, kad mūsų testavimo aibės veikimo analizė
rėmėsi prielaida, kad klasifikatorius
buvo parinktas be jokio kontakto su testavimo aibe,
todėl galėjome žiūrėti į testavimo aibę
kaip atsitiktinai paimtą iš pagrindinės populiacijos.
Čia jūs ne tik testuojate kelias funkcijas,
vėlesnė funkcija $f_2$ buvo pasirinkta
po to, kai pamatėte $f_1$ veikimą testavimo aibėje.
Kai tik informacija iš testavimo aibės nutekėjo modeliuotojui,
ji niekada nebegali būti tikra testavimo aibe griežčiausia prasme.
Ši problema vadinama *adaptyviuoju persimokymu* ir neseniai iškilo
kaip mokymosi teoretikų ir statistikų didelio susidomėjimo tema
:cite:`dwork2015preserving`.
Laimei, nors įmanoma
nutekinti visą informaciją iš atskirtos aibės,
o teoriniai blogiausi scenarijai yra niūrūs,
šios analizės gali būti pernelyg konservatyvios.
Praktikoje pasirūpinkite sukurti tikras testavimo aibes,
konsultuokitės su jomis kuo rečiau,
atsižvelkite į daugkartinį hipotezių tikrinimą,
pateikdami pasikliautinuosius intervalus,
ir agresyviau didinkite budrumą,
kai statymai dideli, o jūsų duomenų rinkinio dydis mažas.
Vykdydami lyginamųjų užduočių seriją,
dažnai gera praktika yra turėti
kelias testavimo aibes, kad po kiekvieno rato
senąją testavimo aibę būtų galima paversti validavimo aibe.





## Statistinio mokymosi teorija

Paprastai tariant, *testavimo aibės yra viskas, ką iš tikrųjų turime*,
ir vis dėlto šis faktas atrodo keistai nepatenkinamas.
Pirma, retai turime *tikrą testavimo aibę* – nebent
patys kuriame duomenų rinkinį,
kažkas kitas tikriausiai jau įvertino
savo klasifikatorių mūsų tariamoje „testavimo aibėje".
Ir net kai turime pirmumo teisę,
netrukus pajuntame nusivylimą, norėdami
įvertinti savo vėlesnius modeliavimo bandymus
be graužiančio jausmo,
kad negalime pasitikėti savo skaičiais.
Be to, net tikra testavimo aibė gali pasakyti tik *post hoc*,
ar klasifikatorius iš tikrųjų apibendrino populiacijai,
o ne ar turime pagrindo tikėtis *a priori*,
kad jis turėtų apibendrinti.

Turėdami omenyje šiuos abejojimus,
galbūt dabar esate pakankamai paruošti
pamatyti *statistinio mokymosi teorijos* patrauklumą –
matematinės mašininio mokymosi šakos,
kurios specialistai siekia išaiškinti
fundamentalius principus, paaiškinančius,
kodėl / kada modeliai, apmokyti empiriniais duomenimis,
gali / gebės apibendrinti nematytus duomenis.
Vienas pagrindinių statistinio mokymosi tyrėjų tikslų
buvo apriboti apibendrinimo atotrūkį,
susiejant modelių klasės savybes
su imčių skaičiumi duomenų rinkinyje.

Mokymosi teoretikai siekia apriboti skirtumą
tarp *empirinės paklaidos* $\epsilon_\mathcal{S}(f_\mathcal{S})$
išmokto klasifikatoriaus $f_\mathcal{S}$,
apmokyto ir įvertinto
apmokymo aibėje $\mathcal{S}$,
ir tikrosios paklaidos $\epsilon(f_\mathcal{S})$
to paties klasifikatoriaus pagrindinėje populiacijoje.
Tai gali atrodyti panašu į vertinimo problemą,
kurią ką tik sprendėme, bet yra esminis skirtumas.
Anksčiau klasifikatorius $f$ buvo fiksuotas,
ir duomenų rinkinio mums reikėjo
tik vertinimo tikslais.
Ir iš tiesų bet kuris fiksuotas klasifikatorius apibendrina:
jo paklaida (anksčiau nematytame) duomenų rinkinyje
yra nepaslinktas populiacijos paklaidos įvertis.
Bet ką galime pasakyti, kai klasifikatorius
apmokomas ir vertinamas tame pačiame duomenų rinkinyje?
Ar kada nors galime būti tikri, kad apmokymo paklaida
bus artima testavimo paklaidai?


Tarkime, kad mūsų išmoktas klasifikatorius $f_\mathcal{S}$ turi būti parinktas
iš tam tikros iš anksto nurodytos funkcijų aibės $\mathcal{F}$.
Prisiminkite iš mūsų testavimo aibių aptarimo,
kad nors lengva įvertinti
vieno klasifikatoriaus paklaidą,
viskas tampa sudėtinga, kai pradedame
nagrinėti klasifikatorių rinkinius.
Net jei bet kurio vieno (fiksuoto) klasifikatoriaus
empirinė paklaida
su didele tikimybe bus artima jo tikrajai paklaidai,
kai nagrinėjame klasifikatorių rinkinį,
turime nerimauti dėl galimybės,
kad *vienam vieninteliam* iš jų
paklaida bus įvertinta blogai.
Nerimas tas, kad galime pasirinkti tokį klasifikatorių
ir taip rimtai nuvertinti
populiacijos paklaidą.
Be to, net tiesiniams modeliams,
kadangi jų parametrai įgyja tolydžias reikšmes,
paprastai renkamės iš
begalinės funkcijų klasės ($|\mathcal{F}| = \infty$).

Vienas ambicingas problemos sprendimas
yra sukurti analizės įrankius
tolygiajam konvergavimui įrodyti, t. y.
kad su didele tikimybe
kiekvieno klasės klasifikatoriaus $f\in\mathcal{F}$ empirinis klaidų dažnis
*vienu metu* konverguos į savo tikrąjį klaidų dažnį.
Kitaip tariant, ieškome teorinio principo,
kuris leistų teigti, kad
su tikimybe bent $1-\delta$
(kuriai nors mažai $\delta$)
jokio klasifikatoriaus klaidų dažnis $\epsilon(f)$
(tarp visų klasės $\mathcal{F}$ klasifikatorių)
nebus neteisingai įvertintas
daugiau nei kokia nors maža reikšmė $\alpha$.
Aišku, kad tokių teiginių negalime padaryti
visoms modelių klasėms $\mathcal{F}$.
Prisiminkite įsiminimo mašinų klasę,
kuri visada pasiekia empirinę paklaidą $0$,
bet niekada nepranoksta atsitiktinio spėliojimo
pagrindinėje populiacijoje.

Tam tikra prasme įsiminėjų klasė yra per daug lanksti.
Joks toks tolygiojo konvergavimo rezultatas negalėtų galioti.
Kita vertus, fiksuotas klasifikatorius yra nenaudingas – jis
apibendrina tobulai, bet netinka nei
apmokymo, nei testavimo duomenims.
Todėl centrinis mokymosi klausimas
istoriškai buvo formuluojamas kaip kompromisas
tarp lankstesnių (didesnės dispersijos) modelių klasių,
kurios geriau pritaiko apmokymo duomenis, bet rizikuoja persimokymu,
ir standesnių (didesnio poslinkio) modelių klasių,
kurios gerai apibendrina, bet rizikuoja nepakankamu mokymusi.
Pagrindinis mokymosi teorijos klausimas
buvo sukurti tinkamą
matematinę analizę, kuri kiekybiškai nustatytų,
kur modelis yra šiame spektre,
ir pateiktų susijusias garantijas.

Serijoje pamatinių darbų
Vapnikas ir Chervonenkis išplėtė
santykinių dažnių
konvergavimo teoriją
bendresnėms funkcijų klasėms
:cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91`.
Vienas pagrindinių šios krypties darbų indėlių
yra Vapniko ir Chervonenkio (VC) dimensija,
kuri matuoja (vieną) modelių klasės
sudėtingumo (lankstumo) sampratą.
Be to, vienas jų pagrindinių rezultatų apriboja
skirtumą tarp empirinės paklaidos
ir populiacijos paklaidos kaip funkciją
nuo VC dimensijos ir imčių skaičiaus:

$$P\left(R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta
\ \textrm{ for }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$

Čia $\delta > 0$ yra tikimybė, kad riba pažeidžiama,
$\alpha$ yra viršutinė apibendrinimo atotrūkio riba,
o $n$ yra duomenų rinkinio dydis.
Galiausiai $c > 0$ yra konstanta, priklausanti
tik nuo galimų nuostolių masto.
Vienas ribos panaudojimas galėtų būti įstatyti norimas
$\delta$ ir $\alpha$ reikšmes,
kad nustatytume, kiek imčių surinkti.
VC dimensija kiekybiškai nusako didžiausią
duomenų taškų skaičių, kuriems galime priskirti
bet kokį savavališką (dvejetainį) žymėjimą
ir kiekvienam rasti klasėje tokį modelį $f$,
kuris sutampa su tuo žymėjimu.
Pavyzdžiui, tiesiniai modeliai $d$ dimensijų įvestims
turi VC dimensiją $d+1$.
Nesunku suprasti, kad tiesė gali priskirti
bet kokį galimą žymėjimą trims taškams dvimatėje erdvėje,
bet ne keturiems.
Deja, teorija linkusi būti
pernelyg pesimistiška sudėtingesniems modeliams,
ir šiai garantijai gauti paprastai reikia
gerokai daugiau pavyzdžių, nei iš tikrųjų reikia
norimam klaidų dažniui pasiekti.
Atkreipkite dėmesį ir tai, kad fiksavus modelių klasę ir $\delta$,
mūsų klaidų dažnis vėl mažėja
įprasta sparta $\mathcal{O}(1/\sqrt{n})$.
Atrodo mažai tikėtina, kad galėtume pasiekti geriau pagal $n$.
Tačiau keičiant modelių klasę,
VC dimensija gali pateikti
pesimistinį apibendrinimo atotrūkio
vaizdą.





## Santrauka

Paprasčiausias būdas įvertinti modelį
yra pasinaudoti testavimo aibe, sudaryta iš anksčiau nematytų duomenų.
Testavimo aibės vertinimai suteikia nepaslinktą tikrosios paklaidos įvertį
ir konverguoja norima sparta $\mathcal{O}(1/\sqrt{n})$, didėjant testavimo aibei.
Galime pateikti apytikslius pasikliautinuosius intervalus,
pagrįstus tiksliais asimptotiniais skirstiniais,
arba galiojančius baigtinės imties pasikliautinuosius intervalus,
pagrįstus (konservatyvesnėmis) baigtinės imties garantijomis.
Iš tiesų testavimo aibės vertinimas yra
šiuolaikinių mašininio mokymosi tyrimų pagrindas.
Tačiau testavimo aibės retai kada būna tikros testavimo aibės
(kurias daug kartų naudoja keli tyrėjai).
Kai tik ta pati testavimo aibė naudojama
keliems modeliams įvertinti,
klaidingo atradimo kontrolė gali būti sudėtinga.
Teoriškai tai gali sukelti milžiniškų problemų.
Praktikoje problemos svarba
priklauso nuo nagrinėjamų atskirtų aibių dydžio
ir nuo to, ar jos tik naudojamos hiperparametrams rinktis,
ar iš jų tiesiogiau nuteka informacija.
Vis dėlto gera praktika yra kurti tikras testavimo aibes (ar kelias)
ir būti kuo konservatyvesniems dėl to, kaip dažnai jos naudojamos.


Tikėdamiesi pateikti patenkinamesnį sprendimą,
statistinio mokymosi teoretikai sukūrė metodus
tolygiajam konvergavimui modelių klasėje garantuoti.
Jei iš tiesų kiekvieno modelio empirinė paklaida vienu metu
konverguoja į jo tikrąją paklaidą,
tada galime laisvai rinktis modelį, kuris veikia
geriausiai, minimizuodamas apmokymo paklaidą,
žinodami, kad jis taip pat panašiai gerai veiks
atskirtuose duomenyse.
Esminis dalykas: bet kuris toks rezultatas turi priklausyti
nuo kokios nors modelių klasės savybės.
Vladimiras Vapnikas ir Aleksejus Chervonenkis
pristatė VC dimensiją,
pateikdami tolygiojo konvergavimo rezultatus,
galiojančius visiems VC klasės modeliams.
Visų klasės modelių apmokymo paklaidos
(vienu metu) garantuotai
yra artimos jų tikrosioms paklaidoms
ir garantuotai dar labiau artėja
sparta $\mathcal{O}(1/\sqrt{n})$.
Po revoliucinio VC dimensijos atradimo
pasiūlyta daugybė alternatyvių sudėtingumo matų,
kurių kiekvienas palengvina analogišką apibendrinimo garantiją.
Išsamų kelių pažangių funkcijų sudėtingumo matavimo būdų aptarimą žr. :citet:`boucheron2005theory`.
Deja, nors šie sudėtingumo matai
tapo plačiai naudingomis statistinės teorijos priemonėmis,
paaiškėja, kad jie yra bejėgiai
(tiesiogiai taikomi),
paaiškinti, kodėl giliesiems neuroniniams tinklams pavyksta apibendrinti.
Giliesiems neuroniniams tinklams dažnai tenka milijonai parametrų (ar daugiau),
ir jie lengvai gali priskirti atsitiktinius žymenis dideliems taškų rinkiniams.
Nepaisant to, jie gerai apibendrina praktiniuose uždaviniuose
ir, stebėtina, dažnai apibendrina geriau,
kai yra didesni ir gilesni,
nepaisant didesnių VC dimensijų.
Kitame skyriuje vėl grįšime prie apibendrinimo
giliojo mokymosi kontekste.

## Pratimai

1. Jei norime įvertinti fiksuoto modelio $f$ paklaidą
   $0.0001$ tikslumu su didesne nei 99,9 % tikimybe,
   kiek imčių mums reikia?
1. Tarkime, kad kažkas kitas turi pažymėtą testavimo aibę
   $\mathcal{D}$ ir pateikia tik nepažymėtas įvestis (požymius).
   Dabar tarkime, kad prie testavimo aibės žymenų galite prieiti tik
   paleisdami modelį $f$ (be apribojimų modelių klasei)
   kiekvienai nepažymėtai įvesčiai
   ir gaudami atitinkamą paklaidą $\epsilon_\mathcal{D}(f)$.
   Kiek modelių jums reikėtų įvertinti,
   kol nutekintumėte visą testavimo aibę
   ir taip galėtumėte atrodyti, kad turite paklaidą $0$,
   nepriklausomai nuo tikrosios paklaidos?
1. Kokia yra penktos eilės polinomų klasės VC dimensija?
1. Kokia yra ašims lygiagrečių stačiakampių VC dimensija dvimačiams duomenims?

[Diskusijos](https://discuss.d2l.ai/t/6829)
