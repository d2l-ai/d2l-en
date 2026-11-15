```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# Gauso procesų inferencija

Šiame skyrelyje parodysime, kaip atlikti aposteriorinę inferenciją ir daryti prognozes naudojant GP apriorinius skirstinius, kuriuos pristatėme ankstesniame skyrelyje. Pradėsime nuo regresijos, kur inferenciją galime atlikti _uždaros formos_ išraiška. Tai „GP trumpai“ skyrelis, skirtas greitai pradėti praktiškai dirbti su Gauso procesais. Pradėsime nuo visų pagrindinių operacijų programavimo nuo nulio, o tada pristatysime [GPyTorch](https://gpytorch.ai/), kuris daug patogesnį padarys darbą su pažangiausiais Gauso procesais ir integravimą su giliaisiais neuroniniais tinklais. Šias pažangesnes temas išsamiai nagrinėsime kitame skyrelyje. Tame skyrelyje taip pat nagrinėsime aplinkybes, kai reikalinga apytikslė inferencija – klasifikavimą, taškinius procesus ar bet kokius ne Gauso tikėtinumus. 

## Aposteriorinė regresijos inferencija

_Stebėjimo_ modelis susieja funkciją, kurios norime išmokti, $f(x)$, su mūsų stebiniais $y(x)$, abiem indeksuotais tam tikra įvestimi $x$. Klasifikavime $x$ galėtų būti vaizdo pikseliai, o $y$ – susijusi klasės žymė. Regresijoje $y$ paprastai atitinka tolydžiąją išvestį, pavyzdžiui, žemės paviršiaus temperatūrą, jūros lygį, $CO_2$ koncentraciją ir pan.  

Regresijoje dažnai darome prielaidą, kad išvestis yra latentinė be triukšmo funkcija $f(x)$ plius nepriklausomas vienodai pasiskirstęs (i.i.d.) Gauso triukšmas $\epsilon(x)$: 

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

su $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$. Tegu $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$ yra mūsų apmokymo stebinių vektorius, o $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ yra latentinių be triukšmo funkcijos reikšmių vektorius, užklaustas apmokymo įvestyse $X = {x_1, \dots, x_n}$.

Darysime prielaidą, kad $f(x) \sim \mathcal{GP}(m,k)$, o tai reiškia, kad bet kuris funkcijos reikšmių rinkinys $\textbf{f}$ turi jungtinį daugiamatį Gauso skirstinį su vidurkio vektoriumi $\mu_i = m(x_i)$ ir kovariacijų matrica $K_{ij} = k(x_i,x_j)$. RBF branduolys $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$ būtų standartinis kovariacijos funkcijos pasirinkimas. Žymėjimo paprastumo dėlei darysime prielaidą, kad vidurkio funkcija $m(x)=0$; mūsų išvedimus vėliau galima lengvai apibendrinti.

Tarkime, norime daryti prognozes įvesčių aibėje $$X_* = x_{*1},x_{*2},\dots,x_{*m}.$$ Tada norime rasti $x^2$ ir $p(\mathbf{f}_* | \mathbf{y}, X)$. Regresijos atveju šį skirstinį galime patogiai rasti naudodami Gauso tapatybes, suradę jungtinį skirstinį $\mathbf{f}_* = f(X_*)$ ir $\mathbf{y}$. 

Jei lygtį :eqref:`eq_gp-regression` įvertiname apmokymo įvestyse $X$, turime $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$. Pagal Gauso proceso apibrėžimą (žr. ankstesnį skyrelį), $\mathbf{f} \sim \mathcal{N}(0,K(X,X))$, kur $K(X,X)$ yra $n \times n$ matrica, sudaryta įvertinus mūsų kovariacijos funkciją (dar vadinamą _branduoliu_) visose galimose įvesčių porose $x_i, x_j \in X$. $\mathbf{\epsilon}$ yra tiesiog vektorius, sudarytas iš i.i.d. imčių iš $\mathcal{N}(0,\sigma^2)$, todėl jo skirstinys yra $\mathcal{N}(0,\sigma^2I)$. Todėl $\mathbf{y}$ yra dviejų nepriklausomų daugiamačių Gauso kintamųjų suma, taigi jo skirstinys yra $\mathcal{N}(0, K(X,X) + \sigma^2I)$. Taip pat galima parodyti, kad $\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$, kur $K(X_*,X)$ yra $m \times n$ matrica, sudaryta įvertinus branduolį visose testavimo ir apmokymo įvesčių porose. 

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*) \\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}
\right)
$$

Tada galime naudoti standartines Gauso tapatybes, kad iš jungtinio skirstinio rastume sąlyginį skirstinį (žr., pvz., Bishop, 2 skyrių), 
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$, kur $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$, o $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$.

Paprastai mums nereikia naudoti pilnos prognozuojamosios kovariacijų matricos $S$, o vietoj to naudojame $S$ įstrižainę neapibrėžtumui apie kiekvieną prognozę. Dažnai dėl šios priežasties prognozuojamąjį skirstinį užrašome vienam testavimo taškui $x_*$, o ne testavimo taškų rinkiniui. 

Branduolio matrica turi parametrus $\theta$, kuriuos taip pat norime įvertinti, pavyzdžiui, aukščiau pateikto RBF branduolio amplitudę $a$ ir ilgio skalę $\ell$. Šiems tikslams naudojame _ribinį tikėtinumą_, $p(\textbf{y} | \theta, X)$, kurį jau išvedėme nustatydami ribinius skirstinius, kad rastume jungtinį skirstinį $\textbf{y},\textbf{f}_*$. Kaip pamatysime, ribinis tikėtinumas skyla į modelio pritaikymo ir modelio sudėtingumo narius ir automatiškai koduoja Okamo skustuvo (angl. *Occam's razor*) sąvoką hiperparametrams mokytis. Išsamų aptarimą žr. MacKay 28 sk. :cite:`mackay2003information` ir Rasmussen ir Williams 5 sk. :cite:`rasmussen2006gaussian`.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## Prognozavimo ir branduolio hiperparametrų mokymosi GP regresijoje lygtys

Čia pateikiame lygtis, kurias naudosite hiperparametrams mokytis ir prognozėms daryti Gauso procesų regresijoje. Vėlgi darome prielaidą, kad turime regresijos tikslų vektorių $\textbf{y}$, indeksuotą įvestimis $X = \{x_1,\dots,x_n\}$, ir norime padaryti prognozę testavimo įvestyje $x_*$. Darome prielaidą apie i.i.d. adityvų nulinio vidurkio Gauso triukšmą su dispersija $\sigma^2$. Latentinei be triukšmo funkcijai naudojame Gauso proceso apriorinį skirstinį $f(x) \sim \mathcal{GP}(m,k)$ su vidurkio funkcija $m$ ir branduolio funkcija $k$. Pats branduolys turi parametrus $\theta$, kuriuos norime išmokti. Pavyzdžiui, jei naudojame RBF branduolį, $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, norime išmokti $\theta = \{a^2, \ell^2\}$. Tegu $K(X,X)$ žymi $n \times n$ matricą, atitinkančią branduolio įvertinimą visose galimose $n$ apmokymo įvesčių porose. Tegu $K(x_*,X)$ žymi $1 \times n$ vektorių, sudarytą įvertinus $k(x_*, x_i)$, $i=1,\dots,n$. Tegu $\mu$ yra vidurkio vektorius, sudarytas įvertinus vidurkio funkciją $m(x)$ kiekviename apmokymo taške $x$.

Dirbant su Gauso procesais paprastai laikomasi dviejų žingsnių procedūros. 
1. Išmokstami branduolio hiperparametrai $\hat{\theta}$ maksimizuojant ribinį tikėtinumą šių hiperparametrų atžvilgiu.
2. Prognozuojamasis vidurkis naudojamas kaip taškinis prognozuotuvas, o 2 kartus prognozuojamasis standartinis nuokrypis – 95 % patikimajai aibei sudaryti, sąlygojant šiais išmoktais hiperparametrais $\hat{\theta}$.

Logaritminis ribinis tikėtinumas yra tiesiog logaritminis Gauso tankis, kurio forma yra:
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

Prognozuojamasis skirstinys turi formą:
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## Mokymosi ir prognozių lygčių interpretavimas

Yra keletas esminių dalykų, į kuriuos verta atkreipti dėmesį dėl Gauso procesų prognozuojamųjų skirstinių:

* Nepaisant modelių klasės lankstumo, GP regresijai galima atlikti _tikslią_ Bajeso inferenciją _uždaros formos_ išraiška. Be branduolio hiperparametrų mokymosi, nėra jokio _apmokymo_. Galime tiksliai užrašyti, kokias lygtis norime naudoti prognozėms daryti. Gauso procesai šiuo požiūriu yra gana išskirtiniai, ir tai labai prisidėjo prie jų patogumo, universalumo ir nuolatinio populiarumo. 

* Prognozuojamasis vidurkis $a_*$ yra apmokymo tikslų $\textbf{y}$ tiesinis darinys, pasvertas branduoliu $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$. Kaip pamatysime, branduolys (ir jo hiperparametrai) todėl atlieka lemiamą vaidmenį modelio apibendrinimo savybėse.

* Prognozuojamasis vidurkis aiškiai priklauso nuo tikslo reikšmių $\textbf{y}$, bet prognozuojamoji dispersija – ne. Vietoj to prognozuojamasis neapibrėžtumas didėja, kai testavimo įvestis $x_*$ tolsta nuo tikslų vietų $X$, kaip nulemia branduolio funkcija. Tačiau neapibrėžtumas netiesiogiai priklausys nuo tikslų reikšmių $\textbf{y}$ per branduolio hiperparametrus $\theta$, kurie išmokstami iš duomenų.

* Ribinis tikėtinumas skyla į modelio pritaikymo ir modelio sudėtingumo (logaritminio determinanto) narius. Ribinis tikėtinumas linkęs parinkti hiperparametrus, kurie suteikia paprasčiausius pritaikymus, vis dar suderinamus su duomenimis. 

* Pagrindinės skaičiavimo kliūtys kyla dėl tiesinės sistemos sprendimo ir logaritminio determinanto skaičiavimo $n \times n$ simetrinei teigiamai apibrėžtai matricai $K(X,X)$, turint $n$ apmokymo taškų. Naiviai kiekviena iš šių operacijų reikalauja $\mathcal{O}(n^3)$ skaičiavimų, taip pat $\mathcal{O}(n^2)$ saugyklos kiekvienam branduolio (kovariacijų) matricos elementui, dažnai pradedant Choleskio skaida. Istoriškai šios kliūtys apribojo GP uždaviniais, turinčiais mažiau nei maždaug 10 000 apmokymo taškų, ir suteikė GP reputaciją „lėtų“, kuri jau beveik dešimtmetį yra netiksli. Pažangiose temose aptarsime, kaip GP galima pritaikyti uždaviniams su milijonais taškų, didinant mastelį.

* Populiariems branduolio funkcijų pasirinkimams $K(X,X)$ dažnai yra artima singuliariajai, o tai gali sukelti skaitinių problemų atliekant Choleskio skaidas ar kitas operacijas, skirtas tiesinėms sistemoms spręsti. Laimei, regresijoje dažnai dirbame su $K_{\theta}(X,X)+\sigma^2I$, kad triukšmo dispersija $\sigma^2$ būtų pridedama prie $K(X,X)$ įstrižainės, žymiai pagerinant jos sąlygotumą. Jei triukšmo dispersija maža arba atliekame regresiją be triukšmo, įprasta praktika yra pridėti nedidelį „virpėjimo“ (angl. *jitter*) kiekį prie įstrižainės, apie $10^{-6}$, kad pagerintume sąlygotumą.


## Išdirbtas pavyzdys nuo nulio

Sukurkime kelis regresijos duomenis ir tada pritaikykime GP duomenims, įgyvendindami kiekvieną žingsnį nuo nulio. 
Duomenis atrinksime iš 
$$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon,$$ su $\epsilon \sim \mathcal{N}(0,\sigma^2)$. Be triukšmo funkcija, kurią norime rasti, yra $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$. Pradėsime nuo triukšmo standartinio nuokrypio $\sigma = 0.25$.

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

Čia matome triukšmingus stebinius kaip apskritimus ir be triukšmo funkciją mėlyna spalva, kurią norime rasti. 

Dabar nurodykime GP apriorinį skirstinį latentinei be triukšmo funkcijai, $f(x)\sim \mathcal{GP}(m,k)$. Naudosime vidurkio funkciją $m(x) = 0$ ir RBF kovariacijos funkciją (branduolį)
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

Pradėjome nuo 0.2 ilgio skalės. Prieš pritaikydami duomenims, svarbu apsvarstyti, ar nurodėme pagrįstą apriorinį skirstinį. Vizualizuokime kelias imties funkcijas iš šio apriorinio skirstinio, taip pat 95 % patikimąją aibę (manome, kad yra 95 % tikimybė, jog tikroji funkcija yra šioje srityje).

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

Ar šios imtys atrodo pagrįstos? Ar aukšto lygio funkcijų savybės atitinka duomenų tipą, kurį bandome modeliuoti?

Dabar sudarykime aposteriorinio prognozuojamojo skirstinio vidurkį ir dispersiją bet kuriame savavališkame testavimo taške $x_*$.

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

Prieš darydami prognozes turėtume išmokti savo branduolio hiperparametrus $\theta$ ir triukšmo dispersiją $\sigma^2$. Inicializuokime savo ilgio skalę ties 0.75, nes mūsų apriorinės funkcijos atrodė pernelyg sparčiai kintančios, palyginti su duomenimis, kuriuos pritaikome. Taip pat spėsime triukšmo standartinį nuokrypį $\sigma$ lygų 0.75. 

Kad išmoktume šiuos parametrus, maksimizuosime ribinį tikėtinumą šių parametrų atžvilgiu.

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$


Galbūt mūsų apriorinės funkcijos kito pernelyg sparčiai. Spėkime 0.4 ilgio skalę. Taip pat spėsime 0.75 triukšmo standartinį nuokrypį. Tai tik hiperparametrų inicializacijos – šiuos parametrus išmoksime iš ribinio tikėtinumo.

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

Šiuo atveju išmokstame 0.299 ilgio skalę ir 0.24 triukšmo standartinį nuokrypį. Atkreipkite dėmesį, kad išmoktas triukšmas yra labai artimas tikrajam triukšmui, o tai rodo, kad mūsų GP yra labai gerai nurodytas šiam uždaviniui. 

Apskritai labai svarbu atidžiai pagalvoti renkantis branduolį ir inicializuojant hiperparametrus. Nors ribinio tikėtinumo optimizavimas gali būti gana atsparus inicializacijai, jis nėra apsaugotas nuo prastų inicializacijų. Pabandykite paleisti aukščiau pateiktą scenarijų su įvairiomis inicializacijomis ir pažiūrėkite, kokius rezultatus gausite.

Dabar darykime prognozes su šiais išmoktais hiperparametrais.

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

Matome, kad aposteriorinis vidurkis oranžine spalva beveik tobulai atitinka tikrąją be triukšmo funkciją! Atkreipkite dėmesį, kad rodoma 95 % patikimoji aibė yra latentinei _be triukšmo_ (tikrajai) funkcijai, o ne duomenų taškams. Matome, kad ši patikimoji aibė visiškai apima tikrąją funkciją ir neatrodo pernelyg plati ar siaura. Nenorėtume ir nesitikėtume, kad ji apimtų duomenų taškus. Jei norime patikimosios aibės stebiniams, turėtume apskaičiuoti

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

Yra du neapibrėžtumo šaltiniai: _epistemilinis_ neapibrėžtumas, atitinkantis _sumažinamą_ neapibrėžtumą, ir _aleatorinis_ arba _nesumažinamas_ neapibrėžtumas. _Epistemilinis_ neapibrėžtumas čia atitinka neapibrėžtumą dėl tikrųjų be triukšmo funkcijos reikšmių. Šis neapibrėžtumas turėtų didėti tolstant nuo duomenų taškų, nes toli nuo duomenų yra didesnė funkcijos reikšmių, suderinamų su mūsų duomenimis, įvairovė. Stebint vis daugiau duomenų, mūsų įsitikinimai apie tikrąją funkciją tampa tikresni, ir epistemilinis neapibrėžtumas išnyksta. _Aleatorinis_ neapibrėžtumas šiuo atveju yra stebėjimo triukšmas, nes duomenys mums pateikiami su šiuo triukšmu, ir jo sumažinti negalima.

_Epistemilinį_ neapibrėžtumą duomenyse užfiksuoja latentinės be triukšmo funkcijos dispersija np.diag(post\_cov). _Aleatorinį_ neapibrėžtumą užfiksuoja triukšmo dispersija post_sig_est**2. 

Deja, žmonės dažnai nerūpestingai atvaizduoja neapibrėžtumą: daugelyje straipsnių rodomos visiškai neapibrėžtos paklaidos juostos, nėra aiškaus supratimo, ar vizualizuojame epistemilinį, ar aleatorinį neapibrėžtumą, ar abu, ir painiojamos triukšmo dispersijos su triukšmo standartiniais nuokrypiais, standartiniai nuokrypiai su standartinėmis paklaidomis, pasikliautinieji intervalai su patikimosiomis aibėmis ir pan. Be tikslumo, ką atvaizduoja neapibrėžtumas, jis iš esmės yra beprasmis. 

Atidžiai sekant, ką atvaizduoja mūsų neapibrėžtumas, labai svarbu pažymėti, kad imame _dvigubą_ be triukšmo funkcijos dispersijos įverčio _kvadratinę šaknį_. Kadangi mūsų prognozuojamasis skirstinys yra Gauso, šis dydis leidžia sudaryti 95 % patikimąją aibę, atvaizduojančią mūsų įsitikinimus apie intervalą, kuris su 95 % tikimybe turi etaloninę funkciją. Triukšmo _dispersija_ yra visiškai kitame mastelyje ir yra daug mažiau interpretuojama. 

Galiausiai pažiūrėkime į 20 aposteriorinių imčių. Šios imtys mums sako, kokių tipų funkcijos, mūsų manymu, galėtų atitikti mūsų duomenis aposteriori.

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

Pagrindiniuose regresijos taikymuose dažniausiai naudojamas aposteriorinis prognozuojamasis vidurkis ir standartinis nuokrypis atitinkamai kaip taškinis prognozuotuvas ir neapibrėžtumo metrika. Pažangesniuose taikymuose, pavyzdžiui, Bajeso optimizavime su Monte Karlo gavimo funkcijomis arba Gauso procesuose modeliu pagrįstam RL, dažnai būtina imti aposteriorines imtis. Tačiau net jei pagrindiniuose taikymuose jų ir nebūtina imti, šios imtys suteikia daugiau intuicijos apie pritaikymą duomenims ir dažnai naudingos vizualizacijose. 

## Gyvenimo palengvinimas su GPyTorch

Kaip matėme, iš tikrųjų yra gana lengva įgyvendinti pagrindinę Gauso procesų regresiją visiškai nuo nulio. Tačiau vos tik norime ištirti įvairius branduolių pasirinkimus, apsvarstyti apytikslę inferenciją (kuri reikalinga net klasifikavimui), derinti GP su neuroniniais tinklais ar net turime duomenų rinkinį, didesnį nei maždaug 10 000 taškų, įgyvendinimas nuo nulio tampa sunkiai valdomas ir varginantis. Kai kurie efektyviausi mastelio didėjimui pritaikytos GP inferencijos metodai, pavyzdžiui, SKI (dar žinomas kaip KISS-GP), gali reikalauti šimtų kodo eilučių, įgyvendinančių pažangias skaitines tiesinės algebros rutinas. 

Šiais atvejais _GPyTorch_ biblioteka mums gerokai palengvins gyvenimą. GPyTorch plačiau aptarsime būsimose užrašų knygutėse apie Gauso procesų skaitmeniką ir pažangius metodus. GPyTorch bibliotekoje yra [daug pavyzdžių](https://github.com/cornellius-gp/gpytorch/tree/master/examples). Kad pajaustume paketą, peržiūrėsime [paprastą regresijos pavyzdį](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb), parodydami, kaip jį galima pritaikyti, kad atkurtume aukščiau gautus rezultatus naudojant GPyTorch. Gali atrodyti, kad tai daug kodo vien pagrindinei regresijai atkurti, ir tam tikra prasme taip ir yra. Tačiau galime nedelsdami naudoti įvairius branduolius, mastelio didėjimui pritaikytus inferencijos metodus ir apytikslę inferenciją pakeitę tik kelias kodo eilutes žemiau, užuot rašę potencialiai tūkstančius naujo kodo eilučių.

```{.python .input}
# First let's convert our data into tensors for use with PyTorch
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# We are using exact GP inference with a zero mean and RBF kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

Šis kodo blokas pateikia duomenis GPyTorch tinkamu formatu ir nurodo, kad naudojame tikslią inferenciją, taip pat
vidurkio funkciją (nulinę) ir branduolio funkciją (RBF), kuriuos norime naudoti. Bet kurį kitą branduolį galime naudoti labai lengvai,
iškviesdami, pavyzdžiui, gpytorch.kernels.matern_kernel() arba gpyotrch.kernels.spectral_mixture_kernel(). Iki šiol aptarėme
tik tikslią inferenciją, kai prognozuojamąjį skirstinį įmanoma nustatyti nedarant jokių aproksimacijų.
Gauso procesams tikslią inferenciją galime atlikti tik tada, kai turime Gauso tikėtinumą; tiksliau, kai
darome prielaidą, kad mūsų stebiniai generuojami kaip be triukšmo funkcija, atvaizduojama Gauso procesu, plius Gauso triukšmas.
Būsimose užrašų knygutėse nagrinėsime kitas aplinkybes, pavyzdžiui, klasifikavimą, kai šių prielaidų daryti negalime.

```{.python .input}
# Initialize Gaussian likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use the adam optimizer, includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# Set our loss as the negative log GP marginal likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

Čia aiškiai nurodome tikėtinumą, kurį norime naudoti (Gauso), tikslą, kurį naudosime branduolio hiperparametrams apmokyti (čia – ribinį tikėtinumą), ir procedūrą, kurią norime naudoti tam tikslui optimizuoti (šiuo atveju – Adam). Pažymime, kad nors naudojame Adam, kuris yra „stochastinis“ optimizatorius, šiuo atveju tai viso rinkinio Adam. Kadangi ribinis tikėtinumas nefaktorizuojasi per duomenų egzempliorius, negalime naudoti optimizatoriaus per duomenų „minirinkinius“ ir tikėtis garantuotos konvergencijos. GPyTorch taip pat palaiko kitus optimizatorius, pavyzdžiui, L-BFGS. Skirtingai nei standartiniame giliajame mokyme, gerai optimizuotas ribinis tikėtinumas stipriai koreliuoja su geru apibendrinimu, todėl dažnai linkstame prie galingų optimizatorių, tokių kaip L-BFGS, jei jie nėra pernelyg brangūs.

```{.python .input}
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

Čia iš tikrųjų vykdome optimizavimo procedūrą, išvesdami nuostolio reikšmes kas 10 iteracijų.

```{.python .input}
# Get into evaluation (predictive posterior) mode
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

Aukščiau pateiktas kodo blokas leidžia mums daryti prognozes mūsų testavimo įvestyse.

```{.python .input}
with torch.no_grad():
    # Initialize plot
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower bounds for 95\% credible set (in this case, in
    # observation space)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

Galiausiai nubraižome pritaikymą.

Matome, kad pritaikymai yra praktiškai identiški. Keletas pastabų: GPyTorch dirba su ilgio skalių ir stebėjimo triukšmo _kvadratais_. Pavyzdžiui, mūsų išmoktas triukšmo standartinis nuokrypis kode nuo nulio yra apie 0.283. GPyTorch rasta triukšmo dispersija yra $0.81 \approx 0.283^2$. GPyTorch grafike taip pat rodome patikimąją aibę _stebėjimų erdvėje_, o ne latentinių funkcijų erdvėje, kad parodytume, jog ji iš tiesų apima stebėtus duomenų taškus.

## Santrauka

Gauso proceso apriorinį skirstinį galime derinti su duomenimis, kad sudarytume aposteriorinį skirstinį, kurį naudojame prognozėms daryti. Taip pat galime sudaryti ribinį tikėtinumą, kuris naudingas automatiniam branduolio hiperparametrų mokymuisi, valdančiam tokias savybes kaip Gauso proceso kitimo sparta. Aposteriorinio skirstinio sudarymo ir branduolio hiperparametrų mokymosi regresijai mechanika yra paprasta ir užima apie tuziną kodo eilučių. Ši užrašų knygutė yra gera nuoroda bet kuriam skaitytojui, norinčiam greitai „įsibėgėti“ su Gauso procesais. Taip pat pristatėme GPyTorch biblioteką. Nors GPyTorch kodas pagrindinei regresijai yra gana ilgas, jį galima nesunkiai modifikuoti kitoms branduolio funkcijoms ar pažangesnėms funkcijoms, kurias aptarsime būsimose užrašų knygutėse, pavyzdžiui, mastelio didėjimui pritaikytai inferencijai ar ne Gauso tikėtinumams klasifikavimui.


## Pratimai

1. Pabrėžėme _mokymosi_ branduolio hiperparametrų svarbą ir hiperparametrų bei branduolių poveikį Gauso procesų apibendrinimo savybėms. Pabandykite praleisti žingsnį, kuriame mokomės hiperparametrų, ir vietoj to spėkite įvairias ilgio skales ir triukšmo dispersijas bei patikrinkite jų poveikį prognozėms. Kas nutinka, kai naudojate didelę ilgio skalę? Mažą ilgio skalę? Didelę triukšmo dispersiją? Mažą triukšmo dispersiją?
2. Sakėme, kad ribinis tikėtinumas nėra iškilasis tikslas, bet kad hiperparametrai, pavyzdžiui, ilgio skalė ir triukšmo dispersija, GP regresijoje gali būti patikimai įvertinti. Tai paprastai tiesa – iš tikrųjų ribinis tikėtinumas _daug_ geriau išmoksta ilgio skalės hiperparametrus nei įprasti erdvinės statistikos metodai, apimantys empirinių autokoreliacijos funkcijų („kovariogramų“) pritaikymą. Galima teigti, kad didžiausias mašininio mokymosi indėlis į Gauso procesų tyrimus, bent jau iki pastarojo darbo apie mastelio didėjimui pritaikytą inferenciją, buvo ribinio tikėtinumo įvedimas hiperparametrų mokymuisi. 

*Tačiau* skirtingos net šių parametrų poros suteikia interpretuojamai skirtingus tikėtinus daugelio duomenų rinkinių paaiškinimus, o tai lemia lokaliuosius optimumus mūsų tiksle. Jei naudojame didelę ilgio skalę, tada darome prielaidą, kad tikroji pagrindinė funkcija kinta lėtai. Jei stebimi duomenys _kinta_ žymiai, vienintelis būdas, kaip galime tikėtinai turėti didelę ilgio skalę, yra didelė triukšmo dispersija. Kita vertus, jei naudojame mažą ilgio skalę, mūsų pritaikymas bus labai jautrus duomenų kitimui, paliekant mažai vietos kitimams paaiškinti triukšmu (aleatoriniu neapibrėžtumu). 

Pabandykite pažiūrėti, ar galite rasti šiuos lokaliuosius optimumus: inicializuokite su labai didele ilgio skale ir dideliu triukšmu bei su mažomis ilgio skalėmis ir mažu triukšmu. Ar konverguojate į skirtingus sprendinius?
  
3. Sakėme, kad esminis Bajeso metodų privalumas yra natūralus _epistemilinio_ neapibrėžtumo atvaizdavimas. Aukščiau pateiktame pavyzdyje negalime pilnai pamatyti epistemilinio neapibrėžtumo poveikio. Vietoj to pabandykite prognozuoti su `test_x = np.linspace(0, 10, 1000)`. Kas nutinka 95 % patikimajai aibei, kai jūsų prognozės išeina už duomenų ribų? Ar ji apima tikrąją funkciją tame intervale? Kas nutinka, jei tame regione vizualizuojate tik aleatorinį neapibrėžtumą? 

4. Pabandykite paleisti aukščiau pateiktą pavyzdį, bet vietoj to su 10 000, 20 000 ir 40 000 apmokymo taškų, ir išmatuokite vykdymo laikus. Kaip kinta apmokymo laikas didėjant mastui? Arba kaip vykdymo laikai kinta priklausomai nuo testavimo taškų skaičiaus? Ar tai skiriasi prognozuojamajam vidurkiui ir prognozuojamajai dispersijai? Atsakykite į šį klausimą tiek teoriškai išvesdami apmokymo ir testavimo laiko sudėtingumus, tiek paleisdami aukščiau pateiktą kodą su skirtingu taškų skaičiumi.

5. Pabandykite paleisti GPyTorch pavyzdį su skirtingomis kovariacijos funkcijomis, pavyzdžiui, Materno branduoliu. Kaip keičiasi rezultatai? O su spektrinių mišinių branduoliu, esančiu GPyTorch bibliotekoje? Ar kai kuriuos lengviau apmokyti pagal ribinį tikėtinumą nei kitus? Ar kai kurie vertingesni ilgojo, o ne trumpojo nuotolio prognozėms?

6. Mūsų GPyTorch pavyzdyje nubraižėme prognozuojamąjį skirstinį, įskaitant stebėjimo triukšmą, o savo pavyzdyje „nuo nulio“ įtraukėme tik epistemilinį neapibrėžtumą. Pakartokite GPyTorch pavyzdį, bet šį kartą braižydami tik epistemilinį neapibrėžtumą, ir palyginkite su rezultatais nuo nulio. Ar prognozuojamieji skirstiniai dabar atrodo vienodai? (Turėtų.)

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/12117)
:end_tab:
