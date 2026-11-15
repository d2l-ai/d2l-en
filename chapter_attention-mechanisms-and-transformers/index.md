# Dėmesio sutelkimo mechanizmai ir transformeriai
:label:`chap_attention-and-transformers`


Ankstyvuosius giliojo mokymosi bumo metus daugiausia varė
rezultatai, gauti naudojant daugiasluoksnį perceptroną,
konvoliucinį tinklą ir rekurentinio tinklo architektūras.
Stebėtina, kad modelių architektūros, kuriomis rėmėsi
daugelis giliojo mokymosi proveržių 2010-aisiais,
palyginti su savo pirmtakais pasikeitė stebėtinai mažai,
nors praėjo beveik 30 metų.
Nors daug naujų metodologinių naujovių
pateko į daugumos praktikų įrankių rinkinį – prisiminkime ReLU
aktyvacijas, liekamuosius sluoksnius, rinkinio normalizaciją, dropout
ir adaptyvius mokymosi greičio tvarkaraščius – pagrindinės
architektūros buvo aiškiai atpažįstamos kaip
klasikinių idėjų išplėsti įgyvendinimai.
Nepaisant tūkstančių straipsnių, siūlančių alternatyvias idėjas,
modeliai, primenantys klasikinius konvoliucinius neuroninius tinklus (:numref:`chap_cnn`),
išlaikė *pažangiausio lygio* statusą kompiuterinėje regoje,
o modeliai, primenantys Sepo Hochreiterio originalų
LSTM rekurentinio neuroninio tinklo dizainą (:numref:`sec_lstm`),
dominavo daugumoje natūraliosios kalbos apdorojimo taikymų.
Galima teigti, kad iki tol sparti giliojo mokymosi iškilimas
atrodė daugiausia aiškinamas pokyčiais
turimuose skaičiavimo ištekliuose
(dėl lygiagrečiųjų skaičiavimų su GPU naujovių)
ir milžiniškų duomenų išteklių prieinamumu
(dėl pigios saugyklos ir interneto paslaugų).
Nors šie veiksniai iš tiesų gali likti pagrindinėmis varomosiomis jėgomis,
slypinčiomis už didėjančios šios technologijos galios,
mes taip pat stebime, pagaliau,
dominuojančių architektūrų kraštovaizdžio pasikeitimą.

Šiuo metu dominuojantys modeliai
beveik visoms natūraliosios kalbos apdorojimo užduotims
yra pagrįsti transformerių architektūra.
Turint bet kurią naują natūraliosios kalbos apdorojimo užduotį, numatytasis pirmasis metodas
yra paimti didelį transformeriais pagrįstą iš anksto apmokytą modelį
(pvz., BERT :cite:`Devlin.Chang.Lee.ea.2018`, ELECTRA :cite:`clark2019electra`, RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` ar Longformer :cite:`beltagy2020longformer`),
prireikus pritaikyti išvesties sluoksnius
ir adaptuoti modelį turimais
duomenimis tolesnei užduočiai.
Jei sekėte pastarųjų kelerių metų
sunkiai atsikvepiančią naujienų aprėptį, sutelktą ties „OpenAI“
dideliais kalbos modeliais, vadinasi, sekėte pokalbį,
sutelktą ties GPT-2 ir GPT-3 transformeriais pagrįstais modeliais :cite:`Radford.Wu.Child.ea.2019,brown2020language`.
Tuo tarpu regimasis transformeris tapo
numatytuoju modeliu įvairioms regos užduotims,
įskaitant vaizdų atpažinimą, objektų aptikimą,
semantinį segmentavimą ir superskiriamąją gebą :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`.
Transformeriai taip pat pasirodė kaip konkurencingi metodai
kalbos atpažinimui :cite:`gulati2020conformer`,
skatinamajam mokymuisi :cite:`chen2021decision`
ir grafų neuroniniams tinklams :cite:`dwivedi2020generalization`.

Pagrindinė transformerio modelio idėja yra *dėmesio sutelkimo mechanizmas* –
naujovė, iš pradžių sumanyta kaip patobulinimas
kodavimo ir dekodavimo RNN, taikomiems sekos į seką taikymams,
tokiems kaip mašininis vertimas :cite:`Bahdanau.Cho.Bengio.2014`.
Galbūt prisimenate, kad pirmuosiuose sekos į seką modeliuose
mašininiam vertimui :cite:`Sutskever.Vinyals.Le.2014`
visa įvestis buvo suspaudžiama kodavimo bloko
į vieną fiksuoto ilgio vektorių, paduodamą į dekodavimo bloką.
Dėmesio sutelkimo intuicija yra ta, kad užuot spaudus įvestį,
gali būti geriau, kad dekodavimo blokas kiekviename žingsnyje iš naujo peržvelgtų įvesties seką.
Be to, užuot visada matęs tą patį įvesties vaizdą,
galima įsivaizduoti, kad dekodavimo blokas turėtų selektyviai sutelkti dėmesį
į konkrečias įvesties sekos dalis konkrečiais dekodavimo žingsniais.
Bahdanau dėmesio sutelkimo mechanizmas pateikė paprastą būdą,
kuriuo dekodavimo blokas galėtų dinamiškai *kreipti dėmesį* į skirtingas
įvesties dalis kiekviename dekodavimo žingsnyje.
Aukšto lygio idėja yra ta, kad kodavimo blokas galėtų sukurti vaizdą,
kurio ilgis lygus originalios įvesties sekos ilgiui.
Tada dekodavimo metu dekodavimo blokas gali (per tam tikrą valdymo mechanizmą)
gauti kaip įvestį konteksto vektorių, sudarytą iš įvesties vaizdų
svertinės sumos kiekvienu laiko žingsniu.
Intuityviai svoriai nustato, kiek
kiekvieno žingsnio kontekstas „sutelkia dėmesį“ į kiekvieną įvesties teksto vienetą,
o svarbiausia yra padaryti šį svorių priskyrimo procesą
diferencijuojamą,
kad jį būtų galima išmokti kartu su
visais kitais neuroninio tinklo parametrais.

Iš pradžių idėja buvo nepaprastai sėkmingas
patobulinimas rekurentiniams neuroniniams tinklams,
kurie jau dominavo mašininio vertimo taikymuose.
Modeliai veikė geriau nei originalios
kodavimo ir dekodavimo sekos į seką architektūros.
Be to, tyrėjai pastebėjo, kad kai kurios malonios kokybinės įžvalgos
kartais atsirasdavo tikrinant dėmesio svorių šabloną.
Vertimo užduotyse dėmesio sutelkimo modeliai
dažnai priskirdavo didelius dėmesio svorius tarpkalbiniams sinonimams,
generuodami atitinkamus žodžius tikslo kalba.
Pavyzdžiui, verčiant sakinį „my feet hurt“
į „j'ai mal au pieds“, neuroninis tinklas galėtų priskirti
didelius dėmesio svorius žodžio „feet“ vaizdui,
generuodamas atitinkamą prancūzišką žodį „pieds“.
Šios įžvalgos paskatino teiginius, kad dėmesio sutelkimo modeliai suteikia „interpretuojamumo“,
nors ką tiksliai reiškia dėmesio svoriai – t. y.
kaip, jei apskritai, jie turėtų būti *interpretuojami* – tebėra neaiški tyrimų tema.

Tačiau dėmesio sutelkimo mechanizmai netrukus tapo reikšmingesniu rūpesčiu,
už jų naudingumo kaip kodavimo ir dekodavimo rekurentinių neuroninių tinklų patobulinimo
ir tariamo naudingumo išskiriant svarbias įvestis ribų.
:citet:`Vaswani.Shazeer.Parmar.ea.2017` pasiūlė
transformerių architektūrą mašininiam vertimui,
visiškai atsisakydami rekurentinių jungčių
ir vietoj to remdamiesi gudriai išdėstytais dėmesio sutelkimo mechanizmais
visiems ryšiams tarp įvesties ir išvesties teksto vienetų užfiksuoti.
Architektūra veikė nepaprastai gerai,
ir 2018 m. transformeris pradėjo pasirodyti
daugumoje pažangiausių natūraliosios kalbos apdorojimo sistemų.
Be to, tuo pačiu metu dominuojanti natūraliosios kalbos apdorojimo praktika
tapo didelio masto modelių išankstinis apmokymas
milžiniškuose bendrojo pobūdžio fono tekstynuose
tam tikram saviraiškiam išankstinio apmokymo tikslui optimizuoti,
o vėliau šių modelių adaptacija
naudojant turimus tolesnių užduočių duomenis.
Atotrūkis tarp transformerių ir tradicinių architektūrų
ypač išaugo taikant šią išankstinio apmokymo paradigmą,
todėl transformerių iškilimas sutapo
su tokių didelio masto iš anksto apmokytų modelių iškilimu,
dabar kartais vadinamų *pagrindiniais modeliais* (angl. *foundation models*) :cite:`bommasani2021opportunities`.


Šiame skyriuje pristatome dėmesio sutelkimo modelius,
pradėdami nuo pačių pagrindinių intuicijų
ir paprasčiausių idėjos įgyvendinimų.
Tada žingsnis po žingsnio pereiname prie transformerių architektūros,
regimojo transformerio ir šiuolaikinių transformeriais pagrįstų
iš anksto apmokytų modelių panoramos.

```toc
:maxdepth: 2

queries-keys-values
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```

