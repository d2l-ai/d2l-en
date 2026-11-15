# Prisidėjimo gairės

Dėkojame, kad domitės prisidėjimu prie šios atvirojo kodo knygos! Labai vertiname savo bendruomenės atsiliepimus ir indėlį.

Prieš pateikdami bet kokį pull request ar issue, perskaitykite šį dokumentą. Tai padės mums efektyviau dirbti kartu.

> **Lietuviško vertimo pastaba.** Šioje repozitorijoje yra lietuviškas knygos vertimas, kurį rengia ir už kurį atsako Linas Petkevičius ir [Dirbtinio intelekto metodų laboratorija](https://aiml.lt) (žr. [README](README.md)). Pastabas dėl vertimo (terminų, formuluočių, klaidų) teikite per šios repozitorijos *issues* ar *pull request*. Pastabas dėl originalaus anglų kalbos turinio teikite [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en). Toliau pateiktos gairės išverstos iš originalo, o nuorodos veda į originalią repozitoriją.

## Ko tikėtis, kai prisidedate

Kai pateikiate pull request, mūsų komanda gauna pranešimą ir atsakys kuo greičiau. Stengsimės dirbti kartu su jumis, kad jūsų pull request atitiktų mūsų stilių ir standartus. Jei sulieksime jūsų pull request, vėliau galime atlikti papildomų stiliaus ar aiškumo pataisymų.

GitHub šaltinio failai nėra publikuojami tiesiogiai oficialioje svetainėje. Jei sulieksime jūsų pull request, jūsų pakeitimus dokumentacijos svetainėje paskelbsime kuo greičiau, tačiau jie neatsiras iš karto ar automatiškai.

Lauksime jūsų pull request dėl:

* Naujo turinio, kurį norite pridėti (pavyzdžiui, naujų kodo pavyzdžių ar pamokų)
* Turinio netikslumų
* Informacijos spragų, kurias reikia užpildyti išsamesne informacija
* Rašybos ar gramatikos klaidų
* Siūlomų perrašymų, gerinančių aiškumą ir mažinančių painiavą

**Pastaba:** Mes visi rašome skirtingai, todėl jums gali nepatikti, kaip kažkas šiuo metu parašyta ar sutvarkyta. Mes norime to atsiliepimo. Tačiau įsitikinkite, kad jūsų prašymas perrašyti pagrįstas aukščiau išvardytais kriterijais. Jei ne, galime atsisakyti jį sulieti.

## Kaip prisidėti

Norėdami prisidėti, pirmiausia perskaitykite [prisidėjimo skyrių](https://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html) ir galiausiai
atsiųskite mums pull request. Nedideliems pakeitimams, pavyzdžiui, rašybos klaidos taisymui ar nuorodos pridėjimui, galite naudoti [GitHub redagavimo mygtuką](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files). Didesniems pakeitimams:

1. [Sukurkite repozitorijos šakutę (fork)](https://help.github.com/articles/fork-a-repo/).
2. Savo šakutėje atlikite pakeitimą naujoje šakoje (pvz., naudodami [`git branch`](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)), pagrįstoje šios repozitorijos **master** šaka.
3. Įrašykite pakeitimą (commit) į savo šakutę su aiškiu ir aprašomu commit pranešimu.
4. [Sukurkite pull request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/), atsakydami į visus pull request formos klausimus.

Prieš atsiųsdami mums pull request, įsitikinkite, kad:

1. Dirbate su naujausiu **master** šakos šaltiniu.
2. Patikrinote [esamus atvirus](https://github.com/d2l-ai/d2l-en/pulls) ir [neseniai uždarytus](https://github.com/d2l-ai/d2l-en/pulls?q=is%3Apr+is%3Aclosed) pull request, kad įsitikintumėte, jog kas nors kitas problemos dar neišsprendė.
3. [Sukūrėte issue](https://github.com/d2l-ai/d2l-en/issues/new) prieš pradėdami darbą, kuris užims daug jūsų laiko.

Jei prisidėjimas užims daug laiko, [atidarykite naują issue](https://github.com/d2l-ai/d2l-en/issues/new) ir pristatykite savo idėją prieš pradėdami. Paaiškinkite problemą ir aprašykite turinį, kurį norėtumėte matyti pridėtą prie dokumentacijos. Pasakykite, ar rašysite patys, ar norėtumėte mūsų pagalbos. Aptarsime jūsų pasiūlymą ir pasakysime, ar greičiausiai jį priimsime. Nenorime, kad praleistumėte daug laiko prisidėjimui, kuris gali būti už dokumentacijos aprėpties ribų arba kuris jau rengiamas.

## Kaip rasti prisidėjimo darbų

Jei norite prisidėti, bet neturite projekto galvoje, pažiūrėkite į šios repozitorijos [atvirus issue](https://github.com/d2l-ai/d2l-en/issues) idėjų. Issue su žymomis [help wanted](https://github.com/d2l-ai/d2l-en/labels/help%20wanted), [good first issue](https://github.com/d2l-ai/d2l-en/labels/good%20first%20issue) ar [enhancement](https://github.com/d2l-ai/d2l-en/labels/enhancement) yra puiki vieta pradėti.

Be rašytinio turinio, labai vertiname naujus pavyzdžius ir kodo fragmentus mūsų dokumentacijai, pavyzdžiui, pavyzdžius skirtingoms platformoms ar aplinkoms ir kodo pavyzdžius papildomomis kalbomis.


## Kaip pakeisti kodą viename iš karkasų?

Šiame skyriuje aprašoma kūrimo aplinkos paruošimas ir darbo eiga,
kurios reikia laikytis modifikuojant ar perkeliant Python kodą ir darant
pakeitimus viename iš knygoje naudojamų mašininio mokymo karkasų.
Laikomės iš anksto apibrėžtų [stiliaus gairių](https://github.com/d2l-ai/d2l-en/blob/master/STYLE_GUIDE.md),
kad kodo kokybė visoje knygoje būtų vienoda, ir tikimės to paties
iš bendruomenės dalyvių. Šiame žingsnyje gali tekti pažiūrėti ir kitus
skyrius, kuriuos parengė kiti dalyviai.

Visi skyrių poskyriai generuojami iš markdown (.md failų, ne .ipynb failų)
šaltinio failų. Keičiant kodą, kad būtų patogiau kurti
ir užtikrinti, jog nėra klaidų, markdown failų niekada nekeičiame tiesiogiai.
Vietoj to šiuos markdown failus galime nuskaityti/įkelti kaip jupyter užrašų knygutes
ir atlikti reikiamus pakeitimus užrašų knygutėje, kad markdown
failas būtų redaguojamas automatiškai (daugiau apie tai toliau). Šitaip, prieš sukuriant PR,
pakeitimus galima lengvai išbandyti vietoje jupyter užrašų knygutėje.

Pradėkite nuo repozitorijos klonavimo.

* Nuklonuokite savo d2l-en šakutę į vietinį kompiuterį.
```
git clone https://github.com/<UserName>/d2l-en.git
```

* Paruoškite vietinę aplinką: sukurkite tuščią conda aplinką
(galite remtis knygos skyreliu [Miniconda diegimas](https://d2l.ai/chapter_installation/index.html#installing-miniconda)).

* Aktyvavę aplinką, įdiekite reikalingus paketus.
Kokie paketai reikalingi? Tai priklauso nuo karkaso, kurį norite redaguoti. Atkreipkite dėmesį, kad master ir release šakos gali turėti skirtingas
karkaso versijas. Daugiau informacijos rasite [diegimo skyriuje](https://d2l.ai/chapter_installation/index.html).
Diegimo pavyzdys pateiktas toliau:

```bash
conda activate d2l

# PyTorch
pip install torch==<version> torchvision==<version>
# pip install torch==2.0.0 torchvision==0.15.0

# MXNet
pip install mxnet==<version>
# pip install mxnet==1.9.1
# arba GPU
# pip install mxnet-cu112==1.9.1

# Tensorflow
pip install tensorflow==<version> tensorflow-probability==<version>
# pip install tensorflow==2.12.0 tensorflow-probability==0.19.0
```

Knygos kompiliavimą užtikrina
[`d2lbook`](https://github.com/d2l-ai/d2l-book) paketas.
Tiesiog d2l conda aplinkoje paleiskite `pip install git+https://github.com/d2l-ai/d2l-book`,
kad įdiegtumėte paketą.
Toliau paaiškinsime keletą pagrindinių `d2lbook` galimybių. 

PASTABA: `d2l` ir `d2lbook` yra skirtingi paketai (kad nekiltų painiavos).

* Įdiekite `d2l` biblioteką kūrimo režimu (reikia paleisti tik vieną kartą)

```bash
# Vietinės repozitorijos šakutės šakniniame kataloge
cd d2l-en

# Įdiegti d2l paketą
python setup.py develop
```

Dabar aplinkoje galite naudoti `from d2l import <framework_name> as d2l`,
kad pasiektumėte išsaugotas funkcijas ir taip pat jas redaguoti iš karto.

Pridedant konkretaus karkaso kodo langelį, reikia nurodyti
karkasą užkomentuojant langelio viršuje, pavyzdžiui, `#@tab tensorflow`.
Jei kodo skirtukas visiems karkasams yra visiškai toks pats, naudokite
`#@tab all`. Šios informacijos reikia `d2lbook` paketui,
kad sukurtų svetainę, PDF ir pan. Rekomenduojame pažiūrėti keletą užrašų knygučių
kaip pavyzdžių.


### Kaip atidaryti / redaguoti markdown failus naudojant Jupyter Notebook?

Naudodami notedown įskiepį, md formato užrašų knygutes galime keisti tiesiogiai
jupyter. Pirmiausia įdiekite notedown įskiepį, paleiskite jupyter ir
įkelkite įskiepį, kaip parodyta toliau:

```bash
pip install mu-notedown  # Gali tekti pašalinti originalų notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Kad notedown įskiepis būtų įjungtas pagal numatytuosius nustatymus kaskart paleidus
`jupyter notebook`, atlikite šiuos veiksmus: pirmiausia sugeneruokite
Jupyter Notebook konfigūracijos failą
(jei jis jau sugeneruotas, šį žingsnį galite praleisti).

```bash
jupyter notebook --generate-config
```

Tada pridėkite šią eilutę Jupyter Notebook
konfigūracijos failo pabaigoje (Linux/macOS dažniausiai kelias `~/.jupyter/jupyter_notebook_config.py`):

```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

Po to pakanka paleisti jupyter notebook
komandą, ir notedown įskiepis bus įjungtas pagal numatytuosius nustatymus.

Daugiau informacijos rasite skyrelyje apie [markdown failus jupyter](https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter).


#### d2lbook activate

Norėdami pradėti dirbti su konkrečiu skyrelio karkasu,
aktyvuokite tik norimą naudoti karkaso skirtuką,
pavyzdžiui, -> `d2lbook activate <framework_name> chapter_preliminaries/ndarray.md`,
tada `<framework_name>` kodo blokai tampa python blokais, o
kiti karkasai ignoruojami vykdant užrašų knygutę.

Baigę redaguoti užrašų knygutę, ją išsaugokite ir
būtinai išvalykite visas išvestis bei aktyvuokite visus
skirtukus naudodami `d2lbook activate`.

```bash
# Pavyzdys
d2lbook activate all chapter_preliminaries/ndarray.md`
```

#### d2lbook build lib

Pastaba: funkciją, kuri bus naudojama vėliau, pažymėkite
`#save`, o kai atlikti visi aukščiau išvardyti žingsniai,
šakniniame kataloge tiesiog paleiskite toliau pateiktą komandą, kad visos
išsaugotos funkcijos / klasės būtų nukopijuotos į `d2l/<framework_name>.py`

```bash
d2lbook build lib
```

Jei išsaugotoms funkcijoms reikia importuoti kai kuriuos paketus, galite pridėti
juos į `chapter_preface/index.md` po atitinkamo karkaso skirtuku ir
paleisti `d2lbook build lib`. Tada importas atsispindės ir d2l
bibliotekoje, o išsaugotos funkcijos galės pasiekti importuotą biblioteką.

PASTABA: kelis kartus paleisdami užrašų knygutę vietoje, įsitikinkite, kad po pakeitimo išvestys / rezultatai visuose karkasuose yra nuoseklūs.


Galiausiai atsiųskite PR; jei visos patikros sėkmingos ir autoriai peržiūrės PR, jūsų indėlis bus sulietas. :)

Tikimės, kad tai pakankamai išsamu, jog galėtumėte pradėti. Nedvejodami klauskite autorių ir kitų dalyvių, jei kiltų abejonių. Visada džiaugiamės atsiliepimais.

## Elgesio kodeksas

Šiame projekte priimtas [Amazon atvirojo kodo elgesio kodeksas](https://aws.github.io/code-of-conduct). Daugiau informacijos rasite [Elgesio kodekso DUK](https://aws.github.io/code-of-conduct-faq) arba kreipkitės el. paštu [opensource-codeofconduct@amazon.com](mailto:opensource-codeofconduct@amazon.com) su papildomais klausimais ar pastabomis.

## Pranešimai apie saugumo problemas

Jei aptikote galimą saugumo problemą, praneškite AWS saugumo komandai per [pažeidžiamumų pranešimo puslapį](http://aws.amazon.com/security/vulnerability-reporting/). Prašome **nekurti** viešo issue GitHub.

## Licencijavimas

Šio projekto licencijavimo informacijos ieškokite [LICENSE](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE) faile. Paprašysime patvirtinti jūsų indėlio licencijavimą. Didesniems pakeitimams galime paprašyti pasirašyti [Dalyvio licencijos sutartį (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement).
