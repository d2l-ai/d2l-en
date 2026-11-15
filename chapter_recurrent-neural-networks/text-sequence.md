# Neapdoroto teksto konvertavimas į sekų duomenis
:label:`sec_text-sequence`

Visoje šioje knygoje
dažnai dirbsime su tekstiniais duomenimis,
pavaizduotais kaip žodžių, simbolių
ar žodžių dalių sekos.
Kad galėtume pradėti, mums reikės keleto pagrindinių
įrankių neapdorotam tekstui paversti
tinkamo pavidalo sekomis.
Tipiniai išankstinio apdorojimo konvejeriai
atlieka šiuos žingsnius:

1. Įkelti tekstą kaip eilutes į atmintį.
1. Suskaidyti eilutes į teksto vienetus (pvz., žodžius ar simbolius).
1. Sudaryti žodyną, susiejantį kiekvieną žodyno elementą su skaitiniu indeksu.
1. Konvertuoti tekstą į skaitinių indeksų sekas.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## Duomenų rinkinio skaitymas

Čia dirbsime su H. G. Wellso
[Laiko mašina](http://www.gutenberg.org/ebooks/35),
knyga, kurioje yra kiek daugiau nei 30 000 žodžių.
Nors realūs taikymai paprastai
apima gerokai didesnius duomenų rinkinius,
to pakanka išankstinio apdorojimo konvejeriui pademonstruoti.
Toliau pateiktas metodas `_download`
(**nuskaito neapdorotą tekstą į eilutę**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

Paprastumo dėlei, apdorodami neapdorotą tekstą, ignoruojame skyrybos ženklus ir didžiąsias raides.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Teksto skaidymas į teksto vienetus

*Teksto vienetai* (angl. *tokens*) yra atominiai (nedalomi) teksto vienetai.
Kiekvienas laiko žingsnis atitinka 1 teksto vienetą,
tačiau tai, kas tiksliai sudaro teksto vienetą, yra dizaino pasirinkimas.
Pavyzdžiui, galėtume pavaizduoti sakinį
„Baby needs a new pair of shoes“
kaip 7 žodžių seką,
kur visų žodžių aibė sudaro
didelį žodyną (paprastai dešimtis
ar šimtus tūkstančių žodžių).
Arba tą patį sakinį pavaizduotume
kaip daug ilgesnę 30 simbolių seką,
naudodami daug mažesnį žodyną
(yra tik 256 skirtingi ASCII simboliai).
Toliau suskaidome savo iš anksto apdorotą tekstą
į simbolių seką.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Žodynas

Šie teksto vienetai vis dar yra eilutės.
Tačiau mūsų modelių įvestis
galiausiai turi būti sudaryta
iš skaitinių įvesčių.
[**Toliau pristatome klasę
*žodynams* sudaryti,
t. y. objektams, kurie susieja
kiekvieną skirtingą teksto vieneto reikšmę
su unikaliu indeksu.**]
Pirmiausia nustatome unikalių teksto vienetų aibę mūsų mokymo *tekstyne*.
Tada kiekvienam unikaliam teksto vienetui priskiriame skaitinį indeksą.
Reti žodyno elementai patogumo dėlei dažnai atmetami.
Kai apmokymo ar testavimo metu sutinkame teksto vienetą,
kurio anksčiau nebuvome matę arba kuris buvo pašalintas iš žodyno,
jį pavaizduojame specialiu „&lt;unk&gt;“ teksto vienetu,
rodančiu, kad tai *nežinoma* reikšmė.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

Dabar [**sudarome žodyną**] savo duomenų rinkiniui,
konvertuodami eilučių seką
į skaitinių indeksų sąrašą.
Atkreipkite dėmesį, kad neprarandame jokios informacijos
ir galime lengvai konvertuoti savo duomenų rinkinį
atgal į originalų (eilučių) atvaizdą.

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Viskas kartu

Naudodami aukščiau pateiktas klases ir metodus,
[**viską supakuojame į šį
`TimeMachine` klasės metodą `build`**],
kuris grąžina `corpus` – teksto vienetų indeksų sąrašą, ir `vocab` –
*Laiko mašinos* tekstyno žodyną.
Čia atlikti pakeitimai:
(i) tekstą skaidome į simbolius, o ne į žodžius,
kad supaprastintume apmokymą vėlesniuose skyreliuose;
(ii) `corpus` yra vienas sąrašas, o ne teksto vienetų sąrašų sąrašas,
nes kiekviena *Laiko mašinos* duomenų rinkinio teksto eilutė
nebūtinai yra sakinys ar pastraipa.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Tiriamoji kalbos statistika
:label:`subsec_natural-lang-stat`

Naudodami realų tekstyną ir žodžiams apibrėžtą klasę `Vocab`,
galime patikrinti pagrindinę statistiką apie žodžių vartojimą mūsų tekstyne.
Toliau sudarome žodyną iš *Laiko mašinoje* vartojamų žodžių
ir išspausdiname dešimt dažniausiai pasitaikančių.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Atkreipkite dėmesį, kad (**dešimt dažniausių žodžių**)
nėra labai aprašomieji.
Galite net įsivaizduoti,
kad matytume labai panašų sąrašą,
jei būtume pasirinkę bet kurią knygą atsitiktinai.
Artikeliai, tokie kaip „the“ ir „a“,
įvardžiai, tokie kaip „i“ ir „my“,
ir prielinksniai, tokie kaip „of“, „to“ ir „in“,
pasitaiko dažnai, nes atlieka dažnus sintaksinius vaidmenis.
Tokie žodžiai, kurie yra dažni, bet nelabai aprašomieji,
dažnai vadinami (***stop žodžiais***, angl. *stop words*) ir
ankstesnės kartos teksto klasifikatoriuose,
pagrįstuose vadinamosiomis žodžių maišo reprezentacijomis,
jie dažniausiai būdavo filtruojami.
Tačiau jie turi prasmę, todėl
dirbant su šiuolaikiniais RNN ir
transformeriais pagrįstais neuroniniais modeliais
jų filtruoti nebūtina.
Jei pažvelgsite žemiau sąraše,
pastebėsite,
kad žodžių dažnis greitai mažėja.
$10$-as dažniausias žodis
yra mažiau nei $1/5$ tokio paplitimo kaip populiariausias.
Žodžių dažnis, einant per rangus žemyn, linkęs sekti laipsninį skirstinį
(konkrečiai Zipfo).
Kad susidarytume geresnį vaizdą, [**nubraižome žodžių dažnio paveikslą**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Susitvarkius su pirmaisiais keliais žodžiais kaip išimtimis,
visi likę žodžiai maždaug seka tiesią liniją logaritminėje–logaritminėje diagramoje.
Šį reiškinį užfiksuoja *Zipfo dėsnis*,
teigiantis, kad $i$-ojo dažniausio žodžio dažnis $n_i$
yra:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

o tai ekvivalentu

$$\log n_i = -\alpha \log i + c,$$

kur $\alpha$ yra rodiklis, apibūdinantis
skirstinį, o $c$ – konstanta.
Tai jau turėtų priversti mus susimąstyti, jei norime
modeliuoti žodžius pagal skaičiavimo statistikas.
Juk gerokai pervertinsime uodegos, dar vadinamos retais žodžiais, dažnį. Bet [**kaip dėl kitų žodžių derinių, tokių kaip du iš eilės einantys žodžiai (bigramos), trys iš eilės einantys žodžiai (trigramos)**] ir daugiau?
Pažiūrėkime, ar bigramų dažnis elgiasi taip pat kaip pavienio žodžio (unigramos) dažnis.

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Čia verta pažymėti vieną dalyką. Iš dešimties dažniausių žodžių porų devynios sudarytos iš stop žodžių, ir tik viena yra aktuali pačiai knygai – „the time“. Be to, pažiūrėkime, ar trigramų dažnis elgiasi taip pat.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Dabar [**vizualizuokime teksto vienetų dažnį**] tarp šių trijų modelių: unigramų, bigramų ir trigramų.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Šis paveikslas yra gana įdomus.
Pirma, be pavienių žodžių (unigramų), žodžių sekos
taip pat, atrodo, seka Zipfo dėsnį,
nors su mažesniu rodikliu
$\alpha$ :eqref:`eq_zipf_law`,
priklausomai nuo sekos ilgio.
Antra, skirtingų $n$-gramų skaičius nėra toks didelis.
Tai teikia vilčių, kad kalboje yra nemažai struktūros.
Trečia, daug $n$-gramų pasitaiko labai retai.
Dėl to tam tikri metodai netinka kalbos modeliavimui
ir tai motyvuoja naudoti giliojo mokymosi modelius.
Tai aptarsime kitame skyrelyje.


## Santrauka

Tekstas yra viena dažniausių sekų duomenų formų, sutinkamų giliajame mokyme.
Dažni pasirinkimai, kas sudaro teksto vienetą, yra simboliai, žodžiai ir žodžių dalys.
Tekstui iš anksto apdoroti paprastai (i) skaidome tekstą į teksto vienetus; (ii) sudarome žodyną, susiejantį teksto vienetų eilutes su skaitiniais indeksais; ir (iii) konvertuojame tekstinius duomenis į teksto vienetų indeksus, kuriuos modeliai galėtų apdoroti.
Praktikoje žodžių dažnis linkęs sekti Zipfo dėsnį. Tai teisinga ne tik pavieniams žodžiams (unigramoms), bet ir $n$-gramoms.


## Pratimai

1. Šio skyrelio eksperimente suskaidykite tekstą į žodžius ir keiskite `Vocab` egzemplioriaus argumento `min_freq` reikšmę. Kokybiškai apibūdinkite, kaip `min_freq` pokyčiai veikia gauto žodyno dydį.
1. Įvertinkite Zipfo skirstinio rodiklį unigramoms, bigramoms ir trigramoms šiame tekstyne.
1. Raskite kitų duomenų šaltinių (atsisiųskite standartinį mašininio mokymosi duomenų rinkinį, pasirinkite kitą viešojo domeno knygą,
   nuskaitykite svetainę ir pan.). Kiekvieną iš jų suskaidykite į teksto vienetus ir žodžių, ir simbolių lygiu. Kaip žodynų dydžiai palyginami su *Laiko mašinos* tekstynu esant lygiavertėms `min_freq` reikšmėms? Įvertinkite Zipfo skirstinio rodiklį, atitinkantį unigramų ir bigramų skirstinius šiems tekstynams. Kaip jie palyginami su reikšmėmis, kurias stebėjote *Laiko mašinos* tekstynui?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18011)
:end_tab:
