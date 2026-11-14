```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Skaitinis stabilumas ir inicializavimas
:label:`sec_numerical_stability`


Iki šiol kiekvienam mūsų įgyvendintam modeliui
reikėjo inicializuoti jo parametrus
pagal tam tikrą iš anksto nurodytą skirstinį.
Iki šiol inicializavimo schemą laikėme savaime suprantama,
praleisdami detales, kaip šie pasirinkimai daromi.
Galbūt net susidarėte įspūdį,
kad šie pasirinkimai nėra ypač svarbūs.
Priešingai, inicializavimo schemos pasirinkimas
vaidina svarbų vaidmenį neuroninių tinklų mokymesi,
ir jis gali būti lemiamas išlaikant skaitinį stabilumą.
Be to, šie pasirinkimai gali būti įdomiais būdais susiję
su netiesinės aktyvacijos funkcijos pasirinkimu.
Kokią funkciją pasirenkame ir kaip inicializuojame parametrus,
gali nulemti, kaip greitai konverguoja mūsų optimizavimo algoritmas.
Prasti pasirinkimai čia gali lemti,
kad apmokymo metu susidursime su
sprogstančiais arba nykstančiais gradientais.
Šiame skyrelyje išsamiau nagrinėjame šias temas
ir aptariame keletą naudingų euristikų,
kurios jums praverstų
visoje jūsų giliojo mokymosi karjeroje.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Nykstantys ir sprogstantys gradientai

Panagrinėkime gilųjį tinklą su $L$ sluoksnių,
įvestimi $\mathbf{x}$ ir išvestimi $\mathbf{o}$.
Kiekvieną sluoksnį $l$ apibrėžia transformacija $f_l$,
parametrizuota svoriais $\mathbf{W}^{(l)}$,
kurio paslėptojo sluoksnio išvestis yra $\mathbf{h}^{(l)}$ (tegul $\mathbf{h}^{(0)} = \mathbf{x}$),
tada mūsų tinklą galima išreikšti taip:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

Jei visos paslėptojo sluoksnio išvestys ir įvestis yra vektoriai,
$\mathbf{o}$ gradientą bet kurios parametrų aibės $\mathbf{W}^{(l)}$ atžvilgiu
galime užrašyti taip:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

Kitaip tariant, šis gradientas yra
$L-l$ matricų
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
ir gradiento vektoriaus $\mathbf{v}^{(l)}$ sandauga.
Todėl esame jautrūs toms pačioms
skaitinio nuopildymo (angl. *underflow*) problemoms, kurios dažnai iškyla
dauginant per daug tikimybių.
Dirbant su tikimybėmis, įprastas triukas yra
pereiti į logaritminę erdvę, t. y. perkelti
apkrovą nuo mantisės prie skaitinio vaizdavimo
eksponentės.
Deja, mūsų aukščiau minėta problema rimtesnė:
iš pradžių matricos $\mathbf{M}^{(l)}$ gali turėti labai įvairias tikrines reikšmes.
Jos gali būti mažos arba didelės, ir
jų sandauga gali būti *labai didelė* arba *labai maža*.

Nestabilių gradientų keliama rizika
neapsiriboja skaitiniu vaizdavimu.
Nenuspėjamo dydžio gradientai
taip pat kelia grėsmę mūsų optimizavimo algoritmų stabilumui.
Galime susidurti su parametrų atnaujinimais, kurie yra arba
(i) pernelyg dideli, sugadinantys mūsų modelį
(*sprogstančių gradientų* problema);
arba (ii) pernelyg maži
(*nykstančių gradientų* problema),
dėl ko mokymasis tampa neįmanomas, nes parametrai
vos juda per kiekvieną atnaujinimą.


### (**Nykstantys gradientai**)

Vienas dažnas nykstančių gradientų problemos kaltininkas
yra aktyvacijos funkcijos $\sigma$ pasirinkimas,
kuri pridedama po kiekvieno sluoksnio tiesinių operacijų.
Istoriškai sigmoidinė funkcija
$1/(1 + \exp(-x))$ (pristatyta :numref:`sec_mlp`)
buvo populiari, nes primena slenkstinę funkciją.
Kadangi ankstyvieji dirbtiniai neuroniniai tinklai buvo įkvėpti
biologinių neuroninių tinklų,
idėja apie neuronus, kurie šauna arba *visiškai*, arba *visai nešauna*
(kaip biologiniai neuronai), atrodė patraukli.
Pažvelkime į sigmoidinę funkciją atidžiau,
kad pamatytume, kodėl ji gali sukelti nykstančius gradientus.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Kaip matote, (**sigmoidinės funkcijos gradientas išnyksta,
kai jos įvestys yra ir didelės, ir mažos**).
Be to, atgalinio sklidimo per daugelį sluoksnių metu,
nebent būtume „Auksaplaukės zonoje“, kurioje
daugelio sigmoidinių funkcijų įvestys artimos nuliui,
bendros sandaugos gradientai gali išnykti.
Kai mūsų tinklas turi daug sluoksnių,
jei nebūsime atsargūs, gradientas
greičiausiai bus nutrauktas kuriame nors sluoksnyje.
Iš tiesų ši problema anksčiau kamavo giliųjų tinklų apmokymą.
Todėl ReLU, kurios stabilesnės
(bet mažiau neuroniškai įtikimos),
tapo numatytuoju specialistų pasirinkimu.


### [**Sprogstantys gradientai**]

Priešinga problema, kai gradientai sprogsta,
gali būti lygiai taip pat erzinanti.
Kad tai šiek tiek geriau iliustruotume,
generuojame 100 Gauso atsitiktinių matricų
ir dauginame jas iš kokios nors pradinės matricos.
Mūsų pasirinktam mastelio dydžiui
(dispersijos pasirinkimas $\sigma^2=1$)
matricų sandauga sprogsta.
Kai taip nutinka dėl gilaus tinklo
inicializavimo, neturime jokių šansų priversti
gradientinio nusileidimo optimizatoriaus konverguoti.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # Generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### Simetrijos laužymas

Kita neuroninių tinklų projektavimo problema
yra jų parametrizavimui būdinga simetrija.
Tarkime, kad turime paprastą MLP
su vienu paslėptuoju sluoksniu ir dviem vienetais.
Šiuo atveju galėtume sukeisti vietomis pirmojo sluoksnio svorius $\mathbf{W}^{(1)}$
ir taip pat sukeisti
išvesties sluoksnio svorius,
kad gautume tą pačią funkciją.
Nėra nieko ypatingo, kas skirtų
pirmąjį ir antrąjį paslėptuosius vienetus.
Kitaip tariant, turime keitinių simetriją
tarp kiekvieno sluoksnio paslėptųjų vienetų.

Tai daugiau nei tik teorinis nepatogumas.
Panagrinėkime minėtą vieno paslėptojo sluoksnio MLP
su dviem paslėptaisiais vienetais.
Iliustracijai
tarkime, kad išvesties sluoksnis du paslėptuosius vienetus transformuoja tik į vieną išvesties vienetą.
Įsivaizduokite, kas nutiktų, jei inicializuotume
visus paslėptojo sluoksnio parametrus
kaip $\mathbf{W}^{(1)} = c$ su kokia nors konstanta $c$.
Šiuo atveju tiesioginio sklidimo metu
abu paslėptieji vienetai gauna tas pačias įvestis ir parametrus,
sukurdami tą pačią aktyvaciją,
kuri paduodama išvesties vienetui.
Atgalinio sklidimo metu
diferencijavus išvesties vienetą parametrų $\mathbf{W}^{(1)}$ atžvilgiu, gaunamas gradientas, kurio visi elementai įgyja tą pačią reikšmę.
Taigi po gradientinės iteracijos (pvz., minirinkinio stochastinio gradientinio nusileidimo)
visi $\mathbf{W}^{(1)}$ elementai vis dar įgyja tą pačią reikšmę.
Tokios iteracijos
niekada savaime nesulaužytų *simetrijos*
ir galbūt niekada nesugebėtume realizuoti
tinklo raiškos galios.
Paslėptasis sluoksnis elgtųsi
taip, tarsi turėtų tik vieną vienetą.
Atkreipkite dėmesį, kad nors minirinkinio stochastinis gradientinis nusileidimas šios simetrijos nesulaužytų,
dropout reguliarizavimas (pristatytas vėliau) ją sulaužytų!


## Parametrų inicializavimas

Vienas būdas spręsti – ar bent sušvelninti –
aukščiau iškeltas problemas yra atsargus inicializavimas.
Kaip pamatysime vėliau,
papildomas atsargumas optimizavimo metu
ir tinkamas reguliarizavimas gali dar labiau pagerinti stabilumą.


### Numatytasis inicializavimas

Ankstesniuose skyreliuose, pvz., :numref:`sec_linear_concise`,
naudojome normalųjį skirstinį
savo svorių reikšmėms inicializuoti.
Jei nenurodome inicializavimo metodo, karkasas
naudos numatytąjį atsitiktinio inicializavimo metodą, kuris praktikoje dažnai gerai veikia
vidutinio dydžio problemoms.






### Xavier inicializavimas
:label:`subsec_xavier`

Pažvelkime į išvesties $o_{i}$ mastelio skirstinį
kokiame nors pilnai sujungtame sluoksnyje
*be netiesiškumų*.
Turint $n_\textrm{in}$ įvesčių $x_j$
ir su jomis susijusius svorius $w_{ij}$ šiam sluoksniui,
išvestis duodama

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

Svoriai $w_{ij}$ visi traukiami
nepriklausomai iš to paties skirstinio.
Be to, tarkime, kad šis skirstinys
turi nulinį vidurkį ir dispersiją $\sigma^2$.
Atkreipkite dėmesį, kad tai nereiškia, jog skirstinys turi būti Gauso,
tik kad vidurkis ir dispersija turi egzistuoti.
Kol kas tarkime, kad sluoksnio įvestys $x_j$
taip pat turi nulinį vidurkį ir dispersiją $\gamma^2$
ir kad jos yra nepriklausomos nuo $w_{ij}$ bei viena nuo kitos.
Šiuo atveju galime apskaičiuoti $o_i$ vidurkį:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

ir dispersiją:

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Vienas būdas išlaikyti dispersiją fiksuotą
yra nustatyti $n_\textrm{in} \sigma^2 = 1$.
Dabar panagrinėkime atgalinį sklidimą.
Ten susiduriame su panašia problema,
nors gradientai sklinda iš sluoksnių, esančių arčiau išvesties.
Naudodami tą patį samprotavimą kaip tiesioginiam sklidimui,
matome, kad gradientų dispersija gali sprogti,
nebent $n_\textrm{out} \sigma^2 = 1$,
kur $n_\textrm{out}$ yra šio sluoksnio išvesčių skaičius.
Tai palieka mus dilemoje:
negalime vienu metu patenkinti abiejų sąlygų.
Vietoj to tiesiog stengiamės patenkinti:

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

Tai yra samprotavimas, kuriuo grindžiamas dabar jau standartinis
ir praktiškai naudingas *Xavier inicializavimas*,
pavadintas pirmojo jo kūrėjų autoriaus vardu :cite:`Glorot.Bengio.2010`.
Paprastai Xavier inicializavimas
semia svorius iš Gauso skirstinio
su nuliniu vidurkiu ir dispersija
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$.
Taip pat galime tai pritaikyti
dispersijai pasirinkti, kai svoriai semiami
iš tolygiojo skirstinio.
Atkreipkite dėmesį, kad tolygusis skirstinys $U(-a, a)$ turi dispersiją $\frac{a^2}{3}$.
Įstatę $\frac{a^2}{3}$ į savo sąlygą $\sigma^2$,
gauname, kad turėtume inicializuoti pagal

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Nors prielaida apie netiesiškumų nebuvimą
aukščiau pateiktame matematiniame samprotavime
neuroniniuose tinkluose gali būti lengvai pažeista,
Xavier inicializavimo metodas
praktikoje veikia gerai.


### Daugiau

Aukščiau pateiktas samprotavimas vos paliečia
šiuolaikinių parametrų inicializavimo metodų paviršių.
Giliojo mokymosi karkasas dažnai įgyvendina daugiau nei tuziną skirtingų euristikų.
Be to, parametrų inicializavimas ir toliau yra
karšta fundamentinių giliojo mokymosi tyrimų sritis.
Tarp jų yra euristikos, specializuotos
susietiems (bendriems) parametrams, superrezoliucijai,
sekų modeliams ir kitoms situacijoms.
Pavyzdžiui,
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` parodė galimybę apmokyti
10 000 sluoksnių neuroninius tinklus be architektūrinių gudrybių,
naudojant kruopščiai suprojektuotą inicializavimo metodą.

Jei ši tema jus domina, siūlome
pasigilinti į šio modulio siūlomas galimybes,
perskaityti straipsnius, pasiūliusius ir išanalizavusius kiekvieną euristiką,
o tada panagrinėti naujausias publikacijas šia tema.
Galbūt užtiksite ar net išrasite
gudrią idėją ir prisidėsite įgyvendinimu prie giliojo mokymosi karkasų.


## Santrauka

Nykstantys ir sprogstantys gradientai yra dažnos problemos giliuosiuose tinkluose. Reikia didelio atidumo inicializuojant parametrus, kad gradientai ir parametrai išliktų gerai kontroliuojami.
Reikalingos inicializavimo euristikos, kad pradiniai gradientai nebūtų nei per dideli, nei per maži.
Atsitiktinis inicializavimas yra esminis užtikrinant, kad simetrija būtų sulaužyta prieš optimizavimą.
Xavier inicializavimas siūlo, kad kiekvieno sluoksnio bet kurios išvesties dispersijos neveiktų įvesčių skaičius, o bet kurio gradiento dispersijos neveiktų išvesčių skaičius.
ReLU aktyvacijos funkcijos sušvelnina nykstančių gradientų problemą. Tai gali paspartinti konvergavimą.

## Pratimai

1. Ar galite sugalvoti kitų atvejų, kai neuroninis tinklas gali pasižymėti simetrija, kurią reikia laužyti, be keitinių simetrijos MLP sluoksniuose?
1. Ar galime inicializuoti visus svorių parametrus tiesinėje regresijoje arba softmax regresijoje tokia pačia reikšme?
1. Susiraskite analizines dviejų matricų sandaugos tikrinių reikšmių ribas. Ką tai pasako apie tai, kaip užtikrinti, kad gradientai būtų gerai sąlygoti?
1. Jei žinome, kad kai kurie nariai diverguoja, ar galime tai ištaisyti vėliau? Įkvėpimo ieškokite straipsnyje apie sluoksnių adaptyvųjį sparto mastelio keitimą :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17986)
:end_tab:
