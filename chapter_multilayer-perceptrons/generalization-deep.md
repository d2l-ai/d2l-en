# Apibendrinimas giliajame mokyme


:numref:`chap_regression` ir :numref:`chap_classification`
sprendėme regresijos ir klasifikavimo uždavinius,
pritaikydami tiesinius modelius apmokymo duomenims.
Abiem atvejais pateikėme praktinius algoritmus
parametrams rasti, kurie maksimizavo
stebimų apmokymo žymenų tikėtinumą.
O tada, kiekvieno skyriaus pabaigoje,
prisiminėme, kad apmokymo duomenų pritaikymas
buvo tik tarpinis tikslas.
Mūsų tikrasis siekis visą laiką buvo atrasti *bendrus dėsningumus*,
kuriais remiantis galime daryti tikslias prognozes
net naujiems pavyzdžiams, paimtiems iš tos pačios populiacijos.
Mašininio mokymosi tyrėjai yra optimizavimo algoritmų *vartotojai*.
Kartais tenka net kurti naujus optimizavimo algoritmus.
Bet galiausiai optimizavimas yra tik priemonė tikslui pasiekti.
Savo esme mašininis mokymasis yra statistikos disciplina,
ir apmokymo nuostolius norime optimizuoti tik tiek,
kiek kokio nors statistinio principo (žinomo ar nežinomo)
dėka gauti modeliai apibendrina už apmokymo aibės ribų.


Šviesioji pusė yra ta, kad giliuosius neuroninius tinklus,
apmokytus stochastiniu gradientiniu nusileidimu, apibendrina nepaprastai gerai
begalėje prognozavimo uždavinių, apimančių kompiuterinį matymą;
natūraliosios kalbos apdorojimą; laiko eilučių duomenis; rekomendavimo sistemas;
elektroninius sveikatos įrašus; baltymų lankstymąsi;
vertės funkcijos aproksimavimą vaizdo žaidimuose
ir stalo žaidimuose; ir daugelyje kitų sričių.
Tamsioji pusė – jei ieškojote
paprasto paaiškinimo
apie optimizavimą
(kodėl galime juos pritaikyti apmokymo duomenims)
ar apibendrinimą
(kodėl gauti modeliai apibendrina nematytiems pavyzdžiams),
tuomet galbūt norėsite įsipilti ko nors išgerti.
Nors mūsų tiesinių modelių optimizavimo procedūros
ir sprendinių statistinės savybės
yra gerai aprašytos išsamios teorijos,
mūsų giliojo mokymosi supratimas
abiem frontais vis dar primena laukinius vakarus.

Ir giliojo mokymosi teorija, ir praktika
sparčiai vystosi:
teoretikai pasitelkia naujas strategijas
paaiškinti, kas vyksta,
o praktikai ir toliau
diegia naujoves svaiginančiu tempu,
kurdami euristikų arsenalus giliesiems tinklams apmokyti
ir intuicijų bei liaudies žinių visumą,
padedančią nuspręsti,
kurias technikas taikyti kokiose situacijose.

Dabartinio momento santrauka yra tokia: giliojo mokymosi teorija
pateikė žadančių tyrimo krypčių ir pavienių žavių rezultatų,
bet vis dar atrodo toli nuo išsamaus paaiškinimo,
(i) kodėl galime optimizuoti neuroninius tinklus
ir (ii) kaip gradientiniu nusileidimu išmokyti modeliai
sugeba taip gerai apibendrinti, net didelės dimensijos uždaviniuose.
Tačiau praktikoje (i) retai kada yra problema
(visada galime rasti parametrus, kurie atitiks visus mūsų apmokymo duomenis),
todėl apibendrinimo supratimas yra gerokai didesnė problema.
Kita vertus, net neturėdami nuoseklios mokslinės teorijos paguodos,
praktikai sukūrė didelį technikų rinkinį,
kuris gali padėti sukurti modelius, gerai apibendrinančius praktikoje.
Nors joks glaustas apibendrinimas negali tinkamai atspindėti
didžiulės apibendrinimo giliajame mokyme temos,
o bendra tyrimų būklė toli gražu nėra išspręsta,
šiame skyrelyje tikimės pateikti platų tyrimų ir praktikos
būklės apžvalgą.


## Persimokymo ir reguliarizavimo peržiūra

Pagal „nemokamų pietų nėra“ teoremą, kurią pateikė :citet:`wolpert1995no`,
bet kuris mokymosi algoritmas geriau apibendrina duomenims su tam tikrais skirstiniais ir prasčiau – su kitais skirstiniais.
Taigi, turint baigtinę apmokymo aibę,
modelis remiasi tam tikromis prielaidomis:
norint pasiekti žmogaus lygio našumą,
gali būti naudinga nustatyti *indukcinius polinkius*,
atspindinčius, kaip žmonės mąsto apie pasaulį.
Tokie indukciniai polinkiai rodo pirmenybę
sprendiniams su tam tikromis savybėmis.
Pavyzdžiui,
gilusis MLP turi indukcinį polinkį
kurti sudėtingą funkciją sudedant paprastesnes funkcijas.

Kai mašininio mokymosi modeliai koduoja indukcinius polinkius,
jų apmokymo metodas
paprastai susideda iš dviejų fazių: (i) pritaikyti apmokymo duomenis;
ir (ii) įvertinti *apibendrinimo paklaidą*
(tikrąją paklaidą visoje populiacijoje),
įvertinant modelį su atidėtaisiais duomenimis.
Skirtumas tarp mūsų atitikimo apmokymo duomenims
ir atitikimo testavimo duomenims vadinamas *apibendrinimo atotrūkiu*, ir kai jis didelis,
sakome, kad mūsų modeliai *persimokė* apmokymo duomenims.
Kraštutiniais persimokymo atvejais
galime tiksliai pritaikyti apmokymo duomenis,
net kai testavimo paklaida išlieka didelė.
O klasikiniu požiūriu
aiškinimas yra toks, kad mūsų modeliai per sudėtingi,
todėl reikia sumažinti arba požymių skaičių,
arba išmoktų nenulinių parametrų skaičių,
arba parametrų dydį, kaip jis kiekybiškai įvertinamas.
Prisiminkite modelio sudėtingumo ir nuostolių grafiką
(:numref:`fig_capacity_vs_error`)
iš :numref:`sec_generalization_basics`.


Tačiau gilusis mokymasis šį vaizdą komplikuoja prieštaraujančiais intuicijai būdais.
Pirma, klasifikavimo uždaviniams
mūsų modeliai paprastai yra pakankamai raiškūs,
kad puikiai pritaikytų kiekvieną apmokymo pavyzdį,
net milijonus pavyzdžių turinčiuose duomenų rinkiniuose
:cite:`zhang2021understanding`.
Klasikiniame paveiksle galėtume manyti,
kad ši situacija yra kraštutiniame dešiniajame
modelio sudėtingumo ašies gale,
ir kad bet koks apibendrinimo paklaidos pagerėjimas
turi atsirasti per reguliarizavimą,
arba sumažinant modelių klasės sudėtingumą,
arba taikant baudą, stipriai apribojančią
reikšmių, kurias gali įgyti mūsų parametrai, aibę.
Bet čia viskas ir pradeda darytis keista.

Keista, bet daugeliui giliojo mokymosi užduočių
(pvz., vaizdų atpažinimui ir teksto klasifikavimui)
paprastai renkamės iš modelių architektūrų,
kurių visos gali pasiekti savavališkai mažus apmokymo nuostolius
(ir nulinę apmokymo paklaidą).
Kadangi visi nagrinėjami modeliai pasiekia nulinę apmokymo paklaidą,
*vienintelis tolesnių laimėjimų kelias yra sumažinti persimokymą*.
Dar keisčiau, dažnai nutinka taip, kad
nepaisant tobulo apmokymo duomenų pritaikymo,
galime dar labiau *sumažinti apibendrinimo paklaidą*,
padarydami modelį *dar raiškesnį*,
pvz., pridėdami sluoksnių, mazgų arba apmokydami
didesnį epochų skaičių.
Dar keisčiau, dėsningumas, susiejantis apibendrinimo atotrūkį
su modelio *sudėtingumu* (kaip, pavyzdžiui, atspindi tinklų gylis ar plotis),
gali būti nemonotoniškas:
didesnis sudėtingumas iš pradžių kenkia,
bet vėliau padeda, vadinamuoju „dvigubo nusileidimo“ dėsningumu
:cite:`nakkiran2021deep`.
Taigi giliojo mokymosi specialistas turi triukų krepšį,
kai kurie iš jų, atrodo, kažkokiu būdu riboja modelį,
o kiti, atrodo, daro jį dar raiškesnį,
ir visi jie tam tikra prasme taikomi persimokymui sumažinti.

Dar labiau viską komplikuoja tai, kad
nors klasikinės mokymosi teorijos suteikiamos garantijos
gali būti konservatyvios net klasikiniams modeliams,
jos atrodo bejėgės paaiškinti, kodėl
gilieji neuroniniai tinklai apskritai apibendrina.
Kadangi gilieji neuroniniai tinklai geba pritaikyti
savavališkus žymenis net ddideliems duomenų rinkiniams,
ir nepaisant pažįstamų metodų, tokių kaip $\ell_2$ reguliarizavimas, naudojimo,
tradiciniai sudėtingumu pagrįsti apibendrinimo rėžiai,
pvz., pagrįsti hipotezių klasės VC dimensija
arba Rademacherio sudėtingumu,
negali paaiškinti, kodėl neuroniniai tinklai apibendrina.

## Įkvėpimas iš neparametrinių metodų

Pirmą kartą artėjant prie giliojo mokymosi,
viliojanti mintis juos laikyti parametriniais modeliais.
Juk modeliai *turi* milijonus parametrų.
Kai atnaujiname modelius, atnaujiname jų parametrus.
Kai išsaugome modelius, rašome jų parametrus į diską.
Tačiau matematika ir informatika yra pilna
prieštaraujančių intuicijai požiūrio pokyčių
ir stebinančių izomorfizmų tarp iš pažiūros skirtingų problemų.
Nors neuroniniai tinklai akivaizdžiai *turi* parametrų,
tam tikrais atžvilgiais gali būti vaisingiau
galvoti apie juos kaip apie besielgiančius kaip neparametriniai modeliai.
Taigi kas tiksliai daro modelį neparametriniu?
Nors šis pavadinimas apima įvairius metodus,
viena bendra tema yra ta, kad neparametriniai metodai
paprastai turi sudėtingumo lygį, kuris auga
augant turimų duomenų kiekiui.

Galbūt paprasčiausias neparametrinio modelio pavyzdys
yra $k$ artimiausių kaimynų algoritmas (daugiau neparametrinių modelių aptarsime vėliau, pavyzdžiui, :numref:`sec_attention-pooling`).
Čia apmokymo metu
besimokantysis tiesiog įsimena duomenų rinkinį.
Tada prognozavimo metu,
susidūręs su nauju tašku $\mathbf{x}$,
besimokantysis suranda $k$ artimiausių kaimynų
($k$ taškų $\mathbf{x}_i'$, minimizuojančių
kokį nors atstumą $d(\mathbf{x}, \mathbf{x}_i')$).
Kai $k=1$, šis algoritmas vadinamas $1$ artimiausio kaimyno algoritmu,
ir algoritmas visada pasieks nulinę apmokymo paklaidą.
Tačiau tai nereiškia, kad algoritmas neapibendrins.
Iš tiesų paaiškėja, kad esant kai kurioms švelnioms sąlygoms,
1 artimiausio kaimyno algoritmas yra suderintas
(galiausiai konverguoja į optimalų prognozuotoją).


Atkreipkite dėmesį, kad $1$ artimiausio kaimyno algoritmui reikia nurodyti
kokią nors atstumo funkciją $d$, arba, lygiavertiškai,
nurodyti kokią nors vektorinių reikšmių bazinę funkciją $\phi(\mathbf{x})$
mūsų duomenų požymiams sudaryti.
Bet kuriam atstumo metrikos pasirinkimui
pasieksime nulinę apmokymo paklaidą
ir galiausiai pasieksime optimalų prognozuotoją,
tačiau skirtingos atstumo metrikos $d$
koduoja skirtingus indukcinius polinkius,
o esant baigtiniam turimų duomenų kiekiui
duos skirtingus prognozuotojus.
Skirtingi atstumo metrikos $d$ pasirinkimai
atspindi skirtingas prielaidas apie esamus dėsningumus,
o skirtingų prognozuotojų našumas
priklausys nuo to, kiek šios prielaidos
dera su stebimais duomenimis.

Tam tikra prasme, kadangi neuroniniai tinklai yra perparametrizuoti,
turintys daug daugiau parametrų nei reikia apmokymo duomenims pritaikyti,
jie linkę *interpoliuoti* apmokymo duomenis (puikiai juos pritaikyti)
ir todėl tam tikrais atžvilgiais elgiasi labiau kaip neparametriniai modeliai.
Naujesni teoriniai tyrimai nustatė
gilų ryšį tarp didelių neuroninių tinklų
ir neparametrinių metodų, ypač branduolių metodų.
Visų pirma :citet:`Jacot.Grabriel.Hongler.2018`
parodė, kad riboje, kai daugiasluoksniai perceptronai
su atsitiktinai inicializuotais svoriais tampa be galo platūs,
jie tampa ekvivalentūs (neparametriniams) branduolių metodams
su tam tikru branduolio funkcijos pasirinkimu
(iš esmės atstumo funkcija),
kurį jie vadina neuroniniu tangentiniu branduoliu.
Nors dabartiniai neuroninio tangentinio branduolio modeliai gali visiškai nepaaiškinti
šiuolaikinių giliųjų tinklų elgsenos,
jų sėkmė kaip analizės įrankio
pabrėžia neparametrinio modeliavimo naudingumą
siekiant suprasti perparametrizuotų giliųjų tinklų elgseną.


## Ankstyvasis sustabdymas

Nors gilieji neuroniniai tinklai geba pritaikyti savavališkus žymenis,
net kai žymenys priskirti neteisingai ar atsitiktinai
:cite:`zhang2021understanding`,
ši galimybė atsiranda tik po daugelio apmokymo iteracijų.
Nauja tyrimų kryptis :cite:`Rolnick.Veit.Belongie.Shavit.2017`
atskleidė, kad esant žymenų triukšmui,
neuroniniai tinklai linkę pirmiausia pritaikyti švariai pažymėtus duomenis
ir tik vėliau interpoliuoti neteisingai pažymėtus duomenis.
Be to, nustatyta, kad šis reiškinys
tiesiogiai virsta apibendrinimo garantija:
kai tik modelis pritaikė švariai pažymėtus duomenis,
bet ne atsitiktinai pažymėtus pavyzdžius, įtrauktus į apmokymo aibę,
jis iš tikrųjų apibendrino :cite:`Garg.Balakrishnan.Kolter.Lipton.2021`.

Kartu šie atradimai padeda motyvuoti *ankstyvąjį sustabdymą* –
klasikinę giliųjų neuroninių tinklų reguliarizavimo techniką.
Čia, užuot tiesiogiai ribojus svorių reikšmes,
ribojamas apmokymo epochų skaičius.
Dažniausias būdas nustatyti sustabdymo kriterijų
yra stebėti validavimo paklaidą viso apmokymo metu
(paprastai tikrinant kartą po kiekvienos epochos)
ir nutraukti apmokymą, kai validavimo paklaida
nesumažėjo daugiau nei kokiu nors mažu dydžiu $\epsilon$
per tam tikrą epochų skaičių.
Tai kartais vadinama *kantrybės kriterijumi*.
Be galimybės lemti geresnį apibendrinimą
esant triukšmingiems žymenims,
kita ankstyvojo sustabdymo nauda yra sutaupytas laikas.
Kai tik kantrybės kriterijus įvykdomas, apmokymą galima nutraukti.
Dideliems modeliams, kuriems gali prireikti dienų apmokymo
vienu metu aštuoniuose ar daugiau GPU,
gerai suderintas ankstyvasis sustabdymas gali sutaupyti tyrėjams dienų laiko
ir jų darbdaviams – daugybę tūkstančių dolerių.

Pažymėtina, kad kai nėra žymenų triukšmo ir duomenų rinkiniai yra *realizuojami*
(klasės iš tiesų atskiriamos, pvz., kačių ir šunų atskyrimas),
ankstyvasis sustabdymas paprastai neduoda reikšmingo apibendrinimo pagerėjimo.
Kita vertus, kai yra žymenų triukšmo
arba būdingo žymens kintamumo
(pvz., prognozuojant pacientų mirtingumą),
ankstyvasis sustabdymas yra esminis.
Modelių apmokymas, kol jie interpoliuoja triukšmingus duomenis, paprastai yra bloga idėja.


## Klasikiniai reguliarizavimo metodai giliesiems tinklams

:numref:`chap_regression` aprašėme
keletą klasikinių reguliarizavimo technikų
mūsų modelių sudėtingumui riboti.
Visų pirma :numref:`sec_weight_decay`
pristatė metodą, vadinamą svorių nykimu,
kurį sudaro reguliarizavimo nario pridėjimas prie nuostolių funkcijos,
siekiant nubausti dideles svorių reikšmes.
Priklausomai nuo to, kuri svorių norma baudžiama,
ši technika žinoma arba kaip kraigo reguliarizavimas ($\ell_2$ bauda),
arba kaip lasso reguliarizavimas ($\ell_1$ bauda).
Klasikinėje šių reguliarizatorių analizėje
jie laikomi pakankamai ribojančiais svorių įgyjamas reikšmes,
kad modelis negalėtų pritaikyti savavališkų žymenų.

Giliojo mokymosi įgyvendinimuose
svorių nykimas tebėra populiarus įrankis.
Tačiau tyrėjai pastebėjo,
kad tipiniai $\ell_2$ reguliarizavimo stiprumai
nepakankami, kad neleistų tinklams
interpoliuoti duomenų :cite:`zhang2021understanding`, todėl nauda, jei aiškinama
kaip reguliarizavimas, gali būti prasminga
tik kartu su ankstyvojo sustabdymo kriterijumi.
Nesant ankstyvojo sustabdymo, galima,
kad lygiai kaip sluoksnių skaičius
ar mazgų skaičius (giliajame mokyme)
arba atstumo metrika (1 artimiausio kaimyno algoritme),
šie metodai gali lemti geresnį apibendrinimą
ne todėl, kad prasmingai riboja
neuroninio tinklo galią,
o todėl, kad kažkokiu būdu koduoja indukcinius polinkius,
geriau dermančius su dėsningumais,
randamais dominančiuose duomenų rinkiniuose.
Todėl klasikiniai reguliarizatoriai tebėra populiarūs
giliojo mokymosi įgyvendinimuose,
net jei teorinis jų efektyvumo pagrindimas
gali būti radikaliai kitoks.

Pažymėtina, kad giliojo mokymosi tyrėjai taip pat rėmėsi
technikomis, pirmą kartą išpopuliarėjusiomis
klasikinio reguliarizavimo kontekstuose,
tokiomis kaip triukšmo pridėjimas prie modelio įvesčių.
Kitame skyrelyje pristatysime
garsųjį dropout metodą
(kurį išrado :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`),
kuris tapo giliojo mokymosi pagrindu,
nors teorinis jo efektyvumo pagrindas
tebėra panašiai paslaptingas.


## Santrauka

Skirtingai nei klasikiniai tiesiniai modeliai,
kurie paprastai turi mažiau parametrų nei pavyzdžių,
gilieji tinklai paprastai yra perparametrizuoti,
ir daugumai užduočių geba
puikiai pritaikyti apmokymo aibę.
Šis *interpoliacijos režimas* meta iššūkį
daugeliui tvirtai įsišaknijusių intuicijų.
Funkciniu požiūriu neuroniniai tinklai atrodo kaip parametriniai modeliai.
Tačiau galvojimas apie juos kaip apie neparametrinius modelius
kartais gali būti patikimesnis intuicijos šaltinis.
Kadangi dažnai visi nagrinėjami gilieji tinklai
geba pritaikyti visus apmokymo žymenis,
beveik visi laimėjimai turi atsirasti sumažinant persimokymą
(uždarant *apibendrinimo atotrūkį*).
Paradoksalu, bet intervencijos,
mažinančios apibendrinimo atotrūkį,
kartais atrodo didinančios modelio sudėtingumą,
o kartais – mažinančios jį.
Tačiau šie metodai retai sumažina sudėtingumą
pakankamai, kad klasikinė teorija
paaiškintų giliųjų tinklų apibendrinimą,
o *kodėl tam tikri pasirinkimai lemia geresnį apibendrinimą*,
daugiausia lieka milžinišku atviru klausimu,
nepaisant daugelio puikių tyrėjų suderintų pastangų.


## Pratimai

1. Kokia prasme tradiciniai sudėtingumu pagrįsti matai nepaaiškina giliųjų neuroninių tinklų apibendrinimo?
1. Kodėl *ankstyvasis sustabdymas* gali būti laikomas reguliarizavimo technika?
1. Kaip tyrėjai paprastai nustato sustabdymo kriterijų?
1. Koks svarbus veiksnys, atrodo, išskiria atvejus, kai ankstyvasis sustabdymas lemia didelį apibendrinimo pagerėjimą?
1. Be apibendrinimo, aprašykite kitą ankstyvojo sustabdymo naudą.

[Diskusijos](https://discuss.d2l.ai/t/7473)
