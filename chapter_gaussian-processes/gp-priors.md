```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch'])
```

# Gauso procesų apriorinis skirstinys

Gauso procesų (GP) supratimas svarbus samprotaujant apie modelių konstravimą ir apibendrinimą bei siekiant pažangiausio našumo įvairiuose taikymuose, įskaitant aktyvųjį mokymąsi ir hiperparametrų derinimą giliajame mokyme. GP yra visur, ir mūsų interesas yra žinoti, kas jie yra ir kaip galime juos naudoti.

Šiame skyrelyje pristatome Gauso procesų _apriorinius_ skirstinius funkcijoms. Kitoje užrašų knygutėje parodome, kaip naudoti šiuos apriorinius skirstinius _aposteriorinei inferencijai_ atlikti ir prognozėms daryti. Kitą skyrelį galima laikyti „GP trumpai“, kuris greitai pateikia tai, ko reikia Gauso procesams praktiškai taikyti.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix

d2l.set_figsize()
```

## Apibrėžimas

Gauso procesas apibrėžiamas kaip _atsitiktinių dydžių rinkinys, kurio bet kuris baigtinis skaičius turi jungtinį Gauso skirstinį_. Jei funkcija $f(x)$ yra Gauso procesas su _vidurkio funkcija_ $m(x)$ ir _kovariacijos funkcija_ arba _branduoliu_ $k(x,x')$, $f(x) \sim \mathcal{GP}(m, k)$, tada bet kuris funkcijos reikšmių rinkinys, užklaustas bet kuriame įvesties taškų $x$ (laikų, erdvinių vietų, vaizdo pikselių ir kt.) rinkinyje, turi jungtinį daugiamatį Gauso skirstinį su vidurkio vektoriumi $\mu$ ir kovariacijų matrica $K$: $f(x_1),\dots,f(x_n) \sim \mathcal{N}(\mu, K)$, kur $\mu_i = E[f(x_i)] = m(x_i)$ ir $K_{ij} = \textrm{Cov}(f(x_i),f(x_j)) = k(x_i,x_j)$.

Šis apibrėžimas gali atrodyti abstraktus ir neprieinamas, bet Gauso procesai iš tikrųjų yra labai paprasti objektai. Bet kuri funkcija

$$f(x) = w^{\top} \phi(x) = \langle w, \phi(x) \rangle,$$ :eqlabel:`eq_gp-function`

kur $w$ atrinktas iš Gauso (normaliojo) skirstinio, o $\phi$ yra bet koks bazinių funkcijų vektorius, pavyzdžiui, $\phi(x) = (1, x, x^2, ..., x^d)^{\top}$,
yra Gauso procesas. Be to, bet kurį Gauso procesą f(x) galima išreikšti lygties :eqref:`eq_gp-function` forma. Panagrinėkime kelis konkrečius pavyzdžius, kad pradėtume susipažinti su Gauso procesais, ir tada galėsime įvertinti, kokie paprasti ir naudingi jie iš tikrųjų yra.

## Paprastas Gauso procesas

Tarkime, $f(x) = w_0 + w_1 x$, o $w_0, w_1 \sim \mathcal{N}(0,1)$, kur $w_0, w_1, x$ visi yra vienmačiai. Šią funkciją galime ekvivalenčiai užrašyti kaip skaliarinę sandaugą $f(x) = (w_0, w_1)(1, x)^{\top}$. Aukščiau pateiktoje :eqref:`eq_gp-function` $w = (w_0, w_1)^{\top}$ ir $\phi(x) = (1,x)^{\top}$.

Bet kuriam $x$ $f(x)$ yra dviejų Gauso atsitiktinių dydžių suma. Kadangi Gauso skirstiniai yra uždari sudėties atžvilgiu, $f(x)$ taip pat yra Gauso atsitiktinis dydis bet kuriam $x$. Iš tikrųjų galime apskaičiuoti, kad bet kuriam konkrečiam $x$ $f(x)$ yra $\mathcal{N}(0,1+x^2)$. Panašiai bet kurio funkcijos reikšmių rinkinio $(f(x_1),\dots,f(x_n))$ jungtinis skirstinys bet kuriam įvesčių rinkiniui $x_1,\dots,x_n$ yra daugiamatis Gauso skirstinys. Todėl $f(x)$ yra Gauso procesas.

Trumpai tariant, $f(x)$ yra _atsitiktinė funkcija_ arba _funkcijų skirstinys_. Galime šiek tiek geriau suprasti šį skirstinį pakartotinai atrinkdami $w_0, w_1$ reikšmes ir vizualizuodami atitinkamas funkcijas $f(x)$, kurios yra tiesės su skirtingais nuolydžiais ir skirtingais laisvaisiais nariais, taip:

```{.python .input}
def lin_func(x, n_sample):
    preds = np.zeros((n_sample, x.shape[0]))
    for ii in range(n_sample):
        w = np.random.normal(0, 1, 2)
        y = w[0] + w[1] * x
        preds[ii, :] = y
    return preds

x_points = np.linspace(-5, 5, 50)
outs = lin_func(x_points, 10)
lw_bd = -2 * np.sqrt((1 + x_points ** 2))
up_bd = 2 * np.sqrt((1 + x_points ** 2))

d2l.plt.fill_between(x_points, lw_bd, up_bd, alpha=0.25)
d2l.plt.plot(x_points, np.zeros(len(x_points)), linewidth=4, color='black')
d2l.plt.plot(x_points, outs.T)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("f(x)", fontsize=20)
d2l.plt.show()
```

Jei $w_0$ ir $w_1$ vietoj to atrenkami iš $\mathcal{N}(0,\alpha^2)$, kaip, jūsų manymu, $\alpha$ keitimas veikia funkcijų skirstinį?

## Iš svorių erdvės į funkcijų erdvę

Aukščiau pateiktame grafike matėme, kaip modelio parametrų skirstinys indukuoja funkcijų skirstinį. Nors dažnai turime nuojautų apie funkcijas, kurias norime modeliuoti – ar jos glotnios, periodinės, sparčiai kintančios ir pan., – samprotauti apie parametrus, kurie iš esmės neinterpretuojami, yra gana nuobodu. Laimei, Gauso procesai suteikia paprastą mechanizmą samprotauti _tiesiogiai_ apie funkcijas. Kadangi Gauso skirstinį visiškai apibrėžia jo pirmieji du momentai – vidurkis ir kovariacijų matrica, Gauso procesą, platesne prasme, apibrėžia jo vidurkio funkcija ir kovariacijos funkcija.

Aukščiau pateiktame pavyzdyje vidurkio funkcija

$$m(x) = E[f(x)] = E[w_0 + w_1x] = E[w_0] + E[w_1]x = 0+0 = 0.$$

Panašiai kovariacijos funkcija yra

$$k(x,x') = \textrm{Cov}(f(x),f(x')) = E[f(x)f(x')]-E[f(x)]E[f(x')] = E[w_0^2 + w_0w_1x' + w_1w_0x + w_1^2xx'] = 1 + xx'.$$

Dabar mūsų funkcijų skirstinį galima tiesiogiai nurodyti ir iš jo atrinkti, nereikalaujant atrinkti iš parametrų skirstinio. Pavyzdžiui, norėdami atrinkti iš $f(x)$, galime tiesiog sudaryti daugiamatį Gauso skirstinį, susijusį su bet kuriuo $x$ rinkiniu, kurį norime užklausti, ir atrinkti iš jo tiesiogiai. Pradėsime matyti, kokia naudinga yra ši formuluotė.

Pirmiausia pastebime, kad iš esmės tą patį išvedimą, kurį atlikome paprastam tiesės modeliui, galima pritaikyti vidurkio ir kovariacijos funkcijai rasti _bet kuriam_ modeliui $f(x) = w^{\top} \phi(x)$ su $w \sim \mathcal{N}(u,S)$. Šiuo atveju vidurkio funkcija $m(x) = u^{\top}\phi(x)$, o kovariacijos funkcija $k(x,x') = \phi(x)^{\top}S\phi(x')$. Kadangi $\phi(x)$ gali atvaizduoti bet kokių netiesinių bazinių funkcijų vektorių, nagrinėjame labai bendrą modelių klasę, įskaitant modelius net su _begaliniu_ parametrų skaičiumi.

## Radialiosios bazinės funkcijos (RBF) branduolys

_Radialiosios bazinės funkcijos_ (RBF) branduolys yra populiariausia Gauso procesų, ir apskritai branduolinių mašinų, kovariacijos funkcija.
Šio branduolio forma yra $k_{\textrm{RBF}}(x,x') = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, kur $a$ yra amplitudės parametras, o $\ell$ yra _ilgio skalės_ hiperparametras.

Išveskime šį branduolį pradėdami nuo svorių erdvės. Panagrinėkime funkciją

$$f(x) = \sum_{i=1}^J w_i \phi_i(x), w_i  \sim \mathcal{N}\left(0,\frac{\sigma^2}{J}\right), \phi_i(x) = \exp\left(-\frac{(x-c_i)^2}{2\ell^2 }\right).$$

$f(x)$ yra radialiųjų bazinių funkcijų, kurių plotis $\ell$, o centrai taškuose $c_i$, suma, kaip parodyta šiame paveiksle.


Galime atpažinti, kad $f(x)$ turi formą $w^{\top} \phi(x)$, kur $w = (w_1,\dots,w_J)^{\top}$, o $\phi(x)$ yra vektorius, kuriame yra kiekviena iš radialiųjų bazinių funkcijų. Šio Gauso proceso kovariacijos funkcija tada yra

$$k(x,x') = \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x').$$

Dabar panagrinėkime, kas nutinka, kai parametrų (ir bazinių funkcijų) skaičių nukreipiame į begalybę. Tegu $c_J = \log J$, $c_1 = -\log J$, o $c_{i+1}-c_{i} = \Delta c = 2\frac{\log J}{J}$, ir $J \to \infty$. Kovariacijos funkcija tampa Rymano suma:

$$k(x,x') = \lim_{J \to \infty} \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x') = \int_{c_0}^{c_\infty} \phi_c(x)\phi_c(x') dc.$$

Nustatę $c_0 = -\infty$ ir $c_\infty = \infty$, begalinį skaičių bazinių funkcijų paskleidžiame per visą realiųjų skaičių tiesę, kiekviena
vienas nuo kitos $\Delta c \to 0$ atstumu:

$$k(x,x') = \int_{-\infty}^{\infty} \exp(-\frac{(x-c)^2}{2\ell^2}) \exp(-\frac{(x'-c)^2}{2\ell^2 }) dc = \sqrt{\pi}\ell \sigma^2 \exp(-\frac{(x-x')^2}{2(\sqrt{2} \ell)^2}) \propto k_{\textrm{RBF}}(x,x').$$

Verta akimirką pagalvoti apie tai, ką čia padarėme. Pereidami prie funkcijų erdvės atvaizdavimo, išvedėme, kaip atvaizduoti modelį su _begaliniu_ parametrų skaičiumi naudojant baigtinį skaičiavimų kiekį. Gauso procesas su RBF branduoliu yra _universalusis aproksimatorius_, gebantis atvaizduoti bet kurią tolydžiąją funkciją savavališku tikslumu. Intuityviai matome, kodėl, iš aukščiau pateikto išvedimo. Kiekvieną radialiąją bazinę funkciją galime sutraukti į taškinę masę imdami $\ell \to 0$ ir suteikti kiekvienai taškinei masei bet kokį norimą aukštį.

Taigi Gauso procesas su RBF branduoliu yra modelis su begaliniu parametrų skaičiumi ir daug didesniu lankstumu nei bet kuris baigtinis neuroninis tinklas. Galbūt visas šurmulys dėl _perparametrizuotų_ neuroninių tinklų yra nevietoje. Kaip pamatysime, GP su RBF branduoliais nepersimoko, ir iš tikrųjų užtikrina ypač įtikinamą apibendrinimo našumą mažuose duomenų rinkiniuose. Be to, :cite:`zhang2021understanding` pavyzdžius, pavyzdžiui, gebėjimą tobulai pritaikyti vaizdus su atsitiktinėmis žymėmis, bet vis tiek gerai apibendrinti struktūrizuotuose uždaviniuose, (galima tobulai atkurti naudojant Gauso procesus) :cite:`wilson2020bayesian`. Neuroniniai tinklai nėra tokie išskirtiniai, kaip juos vaizduojame.

Daugiau intuicijos apie Gauso procesus su RBF branduoliais ir hiperparametrus, pavyzdžiui, _ilgio skalę_, galime įgyti atrinkdami tiesiogiai iš funkcijų skirstinio. Kaip ir anksčiau, tai apima paprastą procedūrą:

1. Pasirinkite įvesties $x$ taškus, kuriuose norime užklausti GP: $x_1,\dots,x_n$.
2. Įvertinkite $m(x_i)$, $i = 1,\dots,n$, ir $k(x_i,x_j)$, kai $i,j = 1,\dots,n$, kad atitinkamai sudarytumėte vidurkio vektorių ir kovariacijų matricą $\mu$ ir $K$, kur $(f(x_1),\dots,f(x_n)) \sim \mathcal{N}(\mu, K)$.
3. Atrinkite iš šio daugiamačio Gauso skirstinio, kad gautumėte imties funkcijos reikšmes.
4. Atrinkite daugiau kartų, kad vizualizuotumėte daugiau imties funkcijų, užklaustų tuose taškuose.

Šį procesą iliustruojame paveiksle žemiau.

```{.python .input}
def rbfkernel(x1, x2, ls=4.):  #@save
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls / 2) * (dist ** 2))

x_points = np.linspace(0, 5, 50)
meanvec = np.zeros(len(x_points))
covmat = rbfkernel(x_points,x_points, 1)

prior_samples= np.random.multivariate_normal(meanvec, covmat, size=5);
d2l.plt.plot(x_points, prior_samples.T, alpha=0.5)
d2l.plt.show()
```

## Neuroninio tinklo branduolys

Gauso procesų tyrimus mašininiame mokymesi paskatino neuroninių tinklų tyrimai. Radfordas Nealas tyrinėjo vis didesnius Bajeso neuroninius tinklus ir galiausiai 1994 m. parodė (vėliau paskelbta 1996 m., nes tai buvo vienas liūdniausiai pagarsėjusių NeurIPS atmetimų), kad tokie tinklai su begaliniu paslėptųjų vienetų skaičiumi tampa Gauso procesais su tam tikromis branduolio funkcijomis :cite:`neal1996bayesian`. Susidomėjimas šiuo išvedimu atgijo, o tokios idėjos kaip neuroninis tangentinis branduolys naudojamos neuroninių tinklų apibendrinimo savybėms tirti :cite:`matthews2018gaussian` :cite:`novak2018bayesian`. Neuroninio tinklo branduolį galime išvesti taip.

Panagrinėkime neuroninio tinklo funkciją $f(x)$ su vienu paslėptuoju sluoksniu:

$$f(x) = b + \sum_{i=1}^{J} v_i h(x; u_i).$$

$b$ yra poslinkis, $v_i$ yra paslėptųjų į išvesties svoriai, $h$ yra bet kuri aprėžta paslėptųjų vienetų perdavimo funkcija, $u_i$ yra įvesties į paslėptuosius svoriai, o $J$ yra paslėptųjų vienetų skaičius. Tegu $b$ ir $v_i$ yra nepriklausomi su nuliniu vidurkiu ir atitinkamai dispersijomis $\sigma_b^2$ ir $\sigma_v^2/J$, o $u_i$ turi nepriklausomus vienodus skirstinius. Tada galime naudoti centrinę ribinę teoremą, kad parodytume, jog bet kuris funkcijos reikšmių rinkinys $f(x_1),\dots,f(x_n)$ turi jungtinį daugiamatį Gauso skirstinį.

Atitinkamo Gauso proceso vidurkio ir kovariacijos funkcijos yra:

$$m(x) = E[f(x)] = 0$$

$$k(x,x') = \textrm{cov}[f(x),f(x')] = E[f(x)f(x')] = \sigma_b^2 + \frac{1}{J} \sum_{i=1}^{J} \sigma_v^2 E[h_i(x; u_i)h_i(x'; u_i)]$$

Kai kuriais atvejais šią kovariacijos funkciją galime iš esmės įvertinti uždara forma. Tegu $h(x; u) = \textrm{erf}(u_0 + \sum_{j=1}^{P} u_j x_j)$, kur $\textrm{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt$, o $u \sim \mathcal{N}(0,\Sigma)$. Tada $k(x,x') = \frac{2}{\pi} \textrm{sin}(\frac{2 \tilde{x}^{\top} \Sigma \tilde{x}'}{\sqrt{(1 + 2 \tilde{x}^{\top} \Sigma \tilde{x})(1 + 2 \tilde{x}'^{\top} \Sigma \tilde{x}')}})$.

RBF branduolys yra _stacionarus_, tai reiškia, kad jis yra _invariantiškas postūmiui_, todėl jį galima užrašyti kaip funkciją nuo $\tau = x-x'$. Intuityviai stacionarumas reiškia, kad aukšto lygio funkcijos savybės, pavyzdžiui, kitimo sparta, nesikeičia judant įvesčių erdvėje. Tačiau neuroninio tinklo branduolys yra _nestacionarus_. Žemiau parodome imties funkcijas iš Gauso proceso su šiuo branduoliu. Matome, kad funkcija kokybiškai atrodo kitaip prie pradžios taško.

## Santrauka


Pirmasis Bajeso inferencijos atlikimo žingsnis yra apriorinio skirstinio nurodymas. Gauso procesus galima naudoti visą apriorinį funkcijų skirstinį nurodyti. Pradėdami nuo tradicinio modeliavimo „svorių erdvės“ požiūrio, galime indukuoti funkcijų apriorinį skirstinį pradėdami nuo modelio funkcinės formos ir įvesdami jo parametrų skirstinį. Taip pat galime apriorinį skirstinį nurodyti tiesiogiai funkcijų erdvėje su savybėmis, kurias valdo branduolys. Funkcijų erdvės požiūris turi daug privalumų. Galime kurti modelius, kurie iš tikrųjų atitinka begalinį parametrų skaičių, bet naudoja baigtinį skaičiavimų kiekį! Be to, nors šie modeliai turi daug lankstumo, jie taip pat daro stiprias prielaidas apie tai, kokio tipo funkcijos yra apriori tikėtinos, o tai lemia gana gerą apibendrinimą mažuose duomenų rinkiniuose.

Funkcijų erdvės modelių prielaidas intuityviai valdo branduoliai, kurie dažnai koduoja aukštesnio lygio funkcijų savybes, pavyzdžiui, glotnumą ir periodiškumą. Daugelis branduolių yra stacionarūs, tai reiškia, kad jie yra invariantiški postūmiui. Funkcijos, atrinktos iš Gauso proceso su stacionariu branduoliu, turi maždaug tas pačias aukšto lygio savybes (pavyzdžiui, kitimo spartą) nepriklausomai nuo to, kur žiūrime įvesčių erdvėje.

Gauso procesai yra gana bendra modelių klasė, apimanti daug modelių, su kuriais jau esame susipažinę, pavyzdžių, įskaitant polinomus, Furjė eilutes ir kt., jei tik turime Gauso apriorinį parametrų skirstinį. Jie taip pat apima neuroninius tinklus su begaliniu parametrų skaičiumi, net be Gauso parametrų skirstinių. Šis ryšys, kurį atrado Radfordas Nealas, paskatino mašininio mokymosi tyrėjus nusigręžti nuo neuroninių tinklų ir pereiti prie Gauso procesų.


## Pratimai

1. Atrinkite apriorines imties funkcijas iš GP su Ornšteino–Ulenbeko (OU) branduoliu, $k_{\textrm{OU}}(x,x') = \exp\left(-\frac{1}{2\ell}||x - x'|\right)$. Jei ilgio skalę $\ell$ fiksuosite vienodą, kuo šios funkcijos atrodo kitaip nei imties funkcijos iš GP su RBF branduoliu?

2. Kaip RBF branduolio _amplitudės_ $a^2$ keitimas veikia funkcijų skirstinį?

3. Tarkime, sudarome $u(x) = f(x) + 2g(x)$, kur $f(x) \sim \mathcal{GP}(m_1,k_1)$ ir $g(x) \sim \mathcal{GP}(m_2,k_2)$. Ar $u(x)$ yra Gauso procesas, ir jei taip, kokia yra jo vidurkio ir kovariacijos funkcija?

4. Tarkime, sudarome $g(x) = a(x)f(x)$, kur $f(x) \sim \mathcal{GP}(0,k)$ ir $a(x) = x^2$. Ar $g(x)$ yra Gauso procesas, ir jei taip, kokia yra jo vidurkio ir kovariacijos funkcija? Koks yra $a(x)$ poveikis? Kaip atrodo imties funkcijos, atrinktos iš $g(x)$?

5. Tarkime, sudarome $u(x) = f(x)g(x)$, kur $f(x) \sim \mathcal{GP}(m_1,k_1)$ ir $g(x) \sim \mathcal{GP}(m_2,k_2)$. Ar $u(x)$ yra Gauso procesas, ir jei taip, kokia yra jo vidurkio ir kovariacijos funkcija?

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/12116)
:end_tab:
