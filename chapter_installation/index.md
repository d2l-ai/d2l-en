# Diegimas
:label:`chap_installation`

Kad galėtume pradėti darbą,
mums reikės aplinkos, kurioje būtų galima paleisti Python,
„Jupyter Notebook", reikalingas bibliotekas
ir kodą, kurio reikia pačiai knygai vykdyti.

## Miniconda diegimas

Paprasčiausias variantas –
įdiegti [Miniconda](https://conda.io/en/latest/miniconda.html).
Atkreipkite dėmesį, kad reikalinga Python 3.x versija.
Šiuos veiksmus galite praleisti,
jei jūsų kompiuteryje jau įdiegta conda.

Apsilankykite Miniconda svetainėje ir nustatykite
tinkamą versiją savo sistemai
pagal savo Python 3.x versiją ir kompiuterio architektūrą.
Tarkime, kad naudojate Python 3.9
(tai mūsų išbandyta versija).
Jei naudojate macOS,
atsisiųskite bash scenarijų,
kurio pavadinime yra eilutė „MacOSX",
nueikite į atsisiuntimo vietą
ir paleiskite diegimą taip
(pavyzdys skirtas „Intel" procesorių „Mac" kompiuteriams):

```bash
# Failo pavadinimas gali keistis
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


„Linux" naudotojas
atsisiųstų failą,
kurio pavadinime yra eilutė „Linux",
ir atsisiuntimo vietoje paleistų:

```bash
# Failo pavadinimas gali keistis
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


„Windows" naudotojas Miniconda atsisiunčia ir įdiegia pagal [diegimo instrukcijas internete](https://conda.io/en/latest/miniconda.html).
„Windows" sistemoje komandoms vykdyti komandų eilutę (komandų interpretatorių) galite atidaryti ieškodami `cmd`.

Toliau inicializuokite komandų aplinką (angl. *shell*), kad galėtume tiesiogiai vykdyti `conda`.

```bash
~/miniconda3/bin/conda init
```


Tada uždarykite ir iš naujo atidarykite savo dabartinę komandų aplinką.
Dabar turėtumėte galėti sukurti
naują aplinką taip:

```bash
conda create --name d2l python=3.9 -y
```


Dabar galime aktyvuoti `d2l` aplinką:

```bash
conda activate d2l
```


## Giliojo mokymosi karkaso ir `d2l` paketo diegimas

Prieš diegdami bet kurį giliojo mokymosi karkasą (angl. *framework*),
pirmiausia patikrinkite, ar
jūsų kompiuteryje yra tinkamų grafinių procesorių (GPU)
(standartiniame nešiojamajame kompiuteryje
vaizdą rodantys grafiniai procesoriai mums netinka).
Pavyzdžiui,
jei jūsų kompiuteryje yra NVIDIA grafinių procesorių ir įdiegta [CUDA](https://developer.nvidia.com/cuda-downloads),
viskas paruošta.
Jei jūsų kompiuteryje grafinio procesoriaus nėra,
kol kas nėra dėl ko nerimauti.
Centrinio procesoriaus (CPU) galios
pakanka pirmiesiems skyriams.
Tik atminkite, kad didesniems modeliams
vykdyti grafinio procesoriaus prireiks.


:begin_tab:`mxnet`

Norėdami įdiegti MXNet versiją su GPU palaikymu,
turime išsiaiškinti, kokia CUDA versija įdiegta.
Tai galite patikrinti paleidę `nvcc --version`
arba `cat /usr/local/cuda/version.txt`.
Tarkime, kad įdiegta CUDA 11.2,
tada vykdykite šią komandą:

```bash
# macOS ir „Linux" naudotojams
pip install mxnet-cu112==1.9.1

# „Windows" naudotojams
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


Paskutinius skaitmenis galite keisti pagal savo CUDA versiją, pvz., `cu101`
CUDA 10.1 ir `cu90` CUDA 9.0.


Jei jūsų kompiuteryje nėra NVIDIA grafinių procesorių
ar CUDA,
galite įdiegti CPU versiją
taip:

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

PyTorch (nurodytos versijos išbandytos rašymo metu) su CPU arba GPU palaikymu galite įdiegti taip:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
TensorFlow su CPU arba GPU palaikymu galite įdiegti taip:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
JAX ir Flax su CPU arba GPU palaikymu galite įdiegti taip:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


Jei jūsų kompiuteryje nėra NVIDIA grafinių procesorių
ar CUDA,
galite įdiegti CPU versiją
taip:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


Kitas žingsnis – įdiegti
mūsų sukurtą `d2l` paketą,
kuriame sutelktos
visoje knygoje
dažnai naudojamos funkcijos ir klasės:

```bash
pip install d2l==1.0.3
```


## Kodo atsisiuntimas ir paleidimas

Toliau norėsite atsisiųsti sąsiuvinius (angl. *notebooks*),
kad galėtumėte paleisti kiekvieną knygos kodo bloką.
Tiesiog spustelėkite skirtuką „Notebooks" bet kurio HTML
puslapio viršuje [D2L.ai svetainėje](https://d2l.ai/),
kad atsisiųstumėte kodą, ir jį išarchyvuokite.
Arba sąsiuvinius galite gauti iš komandų eilutės taip:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

Jei `unzip` dar neįdiegta, pirmiausia paleiskite `sudo apt-get install unzip`.
Dabar „Jupyter Notebook" serverį galime paleisti vykdydami:

```bash
jupyter notebook
```


Šiuo metu naršyklėje galite atidaryti http://localhost:8888
(ji gali būti atsidariusi automatiškai).
Tada galime vykdyti kiekvieno knygos skyrelio kodą.
Kai tik atidarote naują komandų eilutės langą,
prieš paleisdami D2L sąsiuvinius
arba atnaujindami paketus
(giliojo mokymosi karkasą
ar `d2l` paketą),
turėsite paleisti `conda activate d2l`,
kad aktyvuotumėte vykdymo aplinką.
Norėdami išeiti iš aplinkos,
paleiskite `conda deactivate`.


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17964)
:end_tab:
