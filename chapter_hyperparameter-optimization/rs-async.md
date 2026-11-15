```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# Asinchroninė atsitiktinė paieška
:label:`sec_rs_async`

Kaip matėme ankstesniame skyrelyje :numref:`sec_api_hpo`, gali tekti laukti
valandų ar net dienų, kol atsitiktinė paieška grąžins gerą hiperparametrų
konfigūraciją, nes hiperparametrų konfigūracijų įvertinimas yra brangus. Praktikoje dažnai turime prieigą prie išteklių telkinio, pavyzdžiui,
kelių GPU tame pačiame kompiuteryje arba kelių kompiuterių su vienu GPU. Tai
kelia klausimą: *kaip efektyviai paskirstyti atsitiktinę paiešką?*

Apskritai skiriame sinchroninį ir asinchroninį lygiagretųjį
hiperparametrų optimizavimą (žr. :numref:`distributed_scheduling`). Sinchroniniu
atveju laukiame, kol baigsis visi vienu metu vykdomi bandymai,
ir tik tada pradedame kitą partiją. Panagrinėkime konfigūracijų erdves, kuriose yra
tokių hiperparametrų kaip filtrų skaičius ar giliojo
neuroninio tinklo sluoksnių skaičius. Hiperparametrų konfigūracijos, turinčios daugiau
sluoksnių ar filtrų, natūraliai užtruks ilgiau, ir visi kiti tos pačios partijos bandymai
turės laukti sinchronizavimo taškuose (pilka sritis
:numref:`distributed_scheduling`), kol galėsime tęsti optimizavimo
procesą.

Asinchroniniu atveju naują bandymą suplanuojame iš karto, kai tik atsiranda
ištekliai. Tai optimaliai išnaudoja mūsų išteklius, nes išvengiame bet kokių
sinchronizavimo pridėtinių sąnaudų. Atsitiktinėje paieškoje kiekviena nauja hiperparametrų konfigūracija
parenkama nepriklausomai nuo visų kitų ir visų pirma neišnaudojant
jokio ankstesnio įvertinimo stebėjimų. Tai reiškia, kad atsitiktinę
paiešką galime lengvai asinchroniškai lygiagretinti. Su sudėtingesniais metodais,
kurie sprendimus priima pagal ankstesnius stebėjimus, tai padaryti nėra paprasta (žr. :numref:`sec_sh_async`).
Nors mums reikia prieigos prie daugiau išteklių nei nuosekliu atveju, asinchroninė
atsitiktinė paieška rodo tiesinį pagreitėjimą: tam tikras našumas pasiekiamas
$K$ kartų greičiau, jei galima lygiagrečiai vykdyti $K$ bandymų.


![Hiperparametrų optimizavimo proceso paskirstymas sinchroniškai arba asinchroniškai. Palyginti su nuosekliu atveju, galime sumažinti bendrą sienos laikrodžio laiką išlaikydami tą patį bendrą skaičiavimo kiekį. Sinchroninis planavimas gali lemti dykinančius darbininkus, kai yra atsiliekančių bandymų.](../img/distributed_scheduling.svg)
:label:`distributed_scheduling`

Šioje užrašų knygutėje nagrinėsime asinchroninę atsitiktinę paiešką, kai bandymai
vykdomi keliuose Python procesuose tame pačiame kompiuteryje. Paskirstytą užduočių planavimą
ir vykdymą sunku įgyvendinti nuo nulio. Naudosime *Syne Tune*
:cite:`salinas-automl22`, kuris suteikia paprastą sąsają asinchroniniam
HPO. Syne Tune sukurtas veikti su skirtingomis vykdymo vidinėmis dalimis, ir
besidominčius skaitytojus kviečiame išnagrinėti jo paprastas API, kad daugiau
sužinotų apie paskirstytąjį HPO.

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## Tikslo funkcija

Pirmiausia turime apibrėžti naują tikslo funkciją, kad ji dabar grąžintų
našumą į Syne Tune per atgalinio iškvietimo funkciją `report`.

```{.python .input  n=34}
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l    
    from syne_tune import Reporter

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter() 
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Initialize the state of Trainer
            trainer.fit(model=model, data=data) 
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

Atkreipkite dėmesį, kad Syne Tune `PythonBackend` reikalauja, kad priklausomybės būtų importuojamos
funkcijos apibrėžimo viduje.

## Asinchroninis planuoklis

Pirmiausia apibrėžiame darbininkų, kurie vienu metu vertina bandymus, skaičių. Taip pat
turime nurodyti, kiek laiko norime vykdyti atsitiktinę paiešką, apibrėždami
viršutinę bendro sienos laikrodžio laiko ribą.

```{.python .input  n=37}
n_workers = 2  # Needs to be <= the number of available GPUs

max_wallclock_time = 12 * 60  # 12 minutes
```

Toliau nurodome, kurią metriką norime optimizuoti ir ar ją norime minimizuoti, ar
maksimizuoti. Būtent `metric` turi atitikti argumento pavadinimą,
perduodamą atgaliniam iškvietimui `report`.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

Naudojame konfigūracijų erdvę iš ankstesnio pavyzdžio. Syne Tune šį
žodyną taip pat galima naudoti pastovioms savybėms perduoti apmokymo scenarijui.
Šia galimybe pasinaudojame, kad perduotume `max_epochs`. Be to, nurodome
pirmąją įvertinamą konfigūraciją per `initial_config`.

```{.python .input  n=39}
config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": 10,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Toliau turime nurodyti užduočių vykdymo vidinę dalį. Čia nagrinėjame tik
paskirstymą vietiniame kompiuteryje, kur lygiagrečios užduotys vykdomos kaip
posistemiai procesai. Tačiau didelio masto HPO galėtume tai vykdyti ir klasteryje
ar debesies aplinkoje, kur kiekvienas bandymas naudoja visą egzempliorių.

```{.python .input  n=40}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)
```

Dabar galime sukurti asinchroninės atsitiktinės paieškos planuoklį, kurio elgesys panašus
į mūsų `BasicScheduler` iš :numref:`sec_api_hpo`.

```{.python .input  n=41}
scheduler = RandomSearch(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
)
```

Syne Tune taip pat turi `Tuner`, kuriame centralizuoti pagrindinis eksperimento ciklas ir
apskaita, o sąveiką tarp planuoklio ir vidinės dalies
tarpininkauja.

```{.python .input  n=42}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler, 
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
```

Paleiskime savo paskirstytojo HPO eksperimentą. Pagal mūsų sustojimo kriterijų
jis vyks apie 12 minučių.

```{.python .input  n=43}
tuner.run()
```

Visų įvertintų hiperparametrų konfigūracijų žurnalai išsaugomi tolesnei
analizei. Bet kuriuo derinimo užduoties metu galime lengvai gauti
iki šiol gautus rezultatus ir nubraižyti geriausios esamos konfigūracijos trajektoriją.

```{.python .input  n=46}
d2l.set_figsize()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Asinchroninio optimizavimo proceso vizualizavimas

Toliau vizualizuojame, kaip kiekvieno bandymo mokymosi kreivės (kiekviena grafiko spalva atitinka bandymą) kinta
asinchroninio optimizavimo proceso metu. Bet kuriuo momentu vienu metu
vykdoma tiek bandymų, kiek turime darbininkų. Kai bandymas baigiasi, iš karto
pradedame kitą bandymą nelaukdami, kol baigsis kiti bandymai. Darbininkų
dykinimo laikas taikant asinchroninį planavimą sumažinamas iki minimumo.

```{.python .input  n=45}
d2l.set_figsize([6, 2.5])
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
    
d2l.plt.xlabel("wall-clock time")
d2l.plt.ylabel("objective function")
```

## Santrauka

Atsitiktinės paieškos laukimo laiką galime žymiai sumažinti paskirstydami
bandymus per lygiagrečius išteklius. Apskritai skiriame sinchroninį
planavimą ir asinchroninį planavimą. Sinchroninis planavimas reiškia, kad
atrenkame naują hiperparametrų konfigūracijų partiją, kai tik ankstesnė partija
baigiasi. Jei yra atsiliekančių bandymų, t. y. bandymų, kurie baigiasi ilgiau nei
kiti, mūsų darbininkai turi laukti sinchronizavimo taškuose. Asinchroninis
planavimas įvertina naują hiperparametrų konfigūraciją, kai tik
atsiranda ištekliai, todėl užtikrina, kad visi darbininkai bet kuriuo
momentu būtų užimti. Nors atsitiktinę paiešką lengva asinchroniškai paskirstyti ir jai
nereikia keisti paties algoritmo, kitiems metodams reikalingi tam tikri papildomi
pakeitimai.

## Pratimai

1. Panagrinėkite `DropoutMLP` modelį, įgyvendintą :numref:`sec_dropout` ir naudotą 1 pratime :numref:`sec_api_hpo`.
    1. Įgyvendinkite tikslo funkciją `hpo_objective_dropoutmlp_synetune`, skirtą naudoti su Syne Tune. Įsitikinkite, kad jūsų funkcija po kiekvienos epochos praneša patikros paklaidą.
    2. Naudodami 1 pratimo iš :numref:`sec_api_hpo` sąranką, palyginkite atsitiktinę paiešką su Bajeso optimizavimu. Jei naudojate SageMaker, galite naudoti Syne Tune lyginamosios analizės galimybes, kad eksperimentus vykdytumėte lygiagrečiai. Užuomina: Bajeso optimizavimas pateikiamas kaip `syne_tune.optimizer.baselines.BayesianOptimization`.
    3. Šiam pratimui reikia vykdyti egzemplioriuje, turinčiame bent 4 CPU branduolius. Vienam iš aukščiau naudotų metodų (atsitiktinė paieška, Bajeso optimizavimas) vykdykite eksperimentus su `n_workers=1`, `n_workers=2`, `n_workers=4` ir palyginkite rezultatus (geriausių esamų konfigūracijų trajektorijas). Bent jau atsitiktinei paieškai turėtumėte pastebėti tiesinį mastelio kitimą darbininkų skaičiaus atžvilgiu. Užuomina: patikimiems rezultatams gali tekti vidurkinti per kelis pakartojimus kiekvienu atveju.
2. *Pažengusiems*. Šio pratimo tikslas yra įgyvendinti naują planuoklį Syne Tune.
    1. Sukurkite virtualiąją aplinką, kurioje būtų ir [d2lbook](https://github.com/d2l-ai/d2l-en/blob/master/INFO.md#installation-for-developers), ir [syne-tune](https://syne-tune.readthedocs.io/en/latest/getting_started.html) šaltiniai.
    2. Įgyvendinkite `LocalSearcher` iš 2 pratimo :numref:`sec_api_hpo` kaip naują ieškiklį Syne Tune. Užuomina: perskaitykite [šią pamoką](https://syne-tune.readthedocs.io/en/latest/tutorials/developer/README.html). Arba galite sekti šiuo [pavyzdžiu](https://syne-tune.readthedocs.io/en/latest/examples.html#launch-hpo-experiment-with-home-made-scheduler).
    3. Palyginkite savo naująjį `LocalSearcher` su `RandomSearch` `DropoutMLP` lyginamojoje užduotyje.


:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/12093)
:end_tab:
