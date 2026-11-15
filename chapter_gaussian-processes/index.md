# Gauso procesai
:label:`chap_gp`

**Andrew Gordon Wilson** (*New York University ir Amazon*)


Gauso procesai (GP) yra visur. Jūs jau esate sutikę daug GP pavyzdžių to nesuvokdami. Bet koks modelis, tiesinis pagal savo parametrus, kurių skirstinys yra Gauso, yra Gauso procesas. Ši klasė apima diskrečiuosius modelius, įskaitant atsitiktinius klajojimus ir autoregresinius procesus, taip pat tolydžiuosius modelius, įskaitant Bajeso tiesinės regresijos modelius, polinomus, Furjė eilutes, radialiųjų bazinių funkcijų modelius ir net neuroninius tinklus su begaliniu paslėptųjų vienetų skaičiumi. Yra nuolatinis juokelis, kad „viskas yra ypatingas Gauso proceso atvejis“.

Gauso procesų mokytis svarbu dėl trijų priežasčių: (1) jie suteikia modeliavimo *funkcijų erdvės* perspektyvą, dėl kurios įvairias modelių klases, įskaitant giliuosius neuroninius tinklus, tampa daug lengviau suprasti; (2) jie turi nepaprastai platų taikymų spektrą, kuriame yra pažangiausi, įskaitant aktyvųjį mokymąsi, hiperparametrų mokymąsi, auto-ML ir erdvėlaikio regresiją; (3) pastaraisiais metais algoritminiai pasiekimai Gauso procesus padarė vis labiau mastelio didėjimui pritaikomus ir aktualius, jie derinami su giliuoju mokymusi per tokius karkasus kaip [GPyTorch](https://gpytorch.ai) :cite:`Gardner.Pleiss.Weinberger.Bindel.Wilson.2018`. Iš tiesų GP ir giliesiems neuroniniams tinklams yra ne konkuruojantys, o labai vienas kitą papildantys metodai, ir juos galima derinti labai efektyviai. Šie algoritminiai pasiekimai aktualūs ne tik Gauso procesams, bet ir suteikia skaitinių metodų pagrindą, kuris plačiai naudingas giliajame mokyme.

Šiame skyriuje pristatome Gauso procesus. Įvadinėje užrašų knygutėje pradedame nuo intuityvaus samprotavimo apie tai, kas yra Gauso procesai ir kaip jie tiesiogiai modeliuoja funkcijas. Apriorinių skirstinių užrašų knygutėje sutelkiame dėmesį į tai, kaip nurodyti Gauso procesų apriorinius skirstinius. Tiesiogiai susiejame tradicinį svorių erdvės modeliavimo požiūrį su funkcijų erdve, o tai padės mums samprotauti apie mašininio mokymosi modelių, įskaitant giliuosius neuroninius tinklus, kūrimą ir supratimą. Tada pristatome populiariąsias kovariacijos funkcijas, dar žinomas kaip _branduoliai_, kurios valdo Gauso proceso apibendrinimo savybes. GP su tam tikru branduoliu apibrėžia funkcijų apriorinį skirstinį. Inferencijos užrašų knygutėje parodysime, kaip naudoti duomenis _aposteriorinio_ skirstinio inferencijai, kad galėtume daryti prognozes. Šioje užrašų knygutėje yra nuo nulio parašytas kodas prognozėms daryti su Gauso procesu, taip pat GPyTorch įvadas. Būsimose užrašų knygutėse pristatysime skaitmeniką, slypinčią už Gauso procesų, kuri naudinga Gauso procesams didinant mastelį, bet taip pat yra galingas bendras giliojo mokymosi pagrindas, ir pažangesnius naudojimo atvejus, pavyzdžiui, hiperparametrų derinimą giliajame mokyme. Mūsų pavyzdžiuose bus naudojamas GPyTorch, kuris leidžia Gauso procesams didinti mastelį ir yra glaudžiai integruotas su giliojo mokymosi funkcionalumu ir PyTorch.

```toc
:maxdepth: 2

gp-intro
gp-priors
gp-inference
```
