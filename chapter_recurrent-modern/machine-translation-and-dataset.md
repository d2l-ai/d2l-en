```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Mašininis vertimas ir duomenų rinkinys
:label:`sec_machine_translation`

Tarp pagrindinių proveržių, sukėlusių
platų susidomėjimą šiuolaikiniais RNN,
buvo didelė pažanga taikomojoje statistinio
*mašininio vertimo* srityje.
Čia modeliui pateikiamas sakinys viena kalba
ir jis turi prognozuoti atitinkamą sakinį kita kalba.
Atkreipkite dėmesį, kad čia sakiniai gali būti skirtingo ilgio,
o atitinkami žodžiai dviejuose sakiniuose
gali eiti ne ta pačia tvarka
dėl dviejų kalbų
gramatinės struktūros skirtumų.


Daug problemų turi tokį atvaizdavimo
tarp dviejų „nesulygiuotų“ sekų pobūdį.
Pavyzdžiai: atvaizdavimas
iš dialogo užklausų į atsakymus
arba iš klausimų į atsakymus.
Plačiąja prasme tokios problemos vadinamos
*sekos į seką* (angl. *seq2seq*) problemomis,
ir jos yra mūsų dėmesio centre
tiek likusioje šio skyriaus dalyje,
tiek didelėje :numref:`chap_attention-and-transformers` dalyje.

Šiame skyrelyje pristatome mašininio vertimo problemą
ir pavyzdinį duomenų rinkinį, kurį naudosime tolesniuose pavyzdžiuose.
Dešimtmečius populiarios buvo statistinės vertimo tarp kalbų formuluotės
:cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`,
dar prieš tyrėjams priversdami veikti neuroninių tinklų metodus
(metodai dažnai buvo suvedami po terminu *neuroninis mašininis vertimas*).


Pirmiausia mums reikės naujo kodo duomenims apdoroti.
Skirtingai nuo kalbos modeliavimo, kurį matėme :numref:`sec_language-model`,
čia kiekvieną pavyzdį sudaro dvi atskiros teksto sekos:
viena šaltinio kalba, o kita (vertimas) – tikslo kalba.
Toliau pateikti kodo fragmentai parodys,
kaip įkelti iš anksto apdorotus duomenis į minirinkinius apmokymui.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**Duomenų rinkinio atsisiuntimas ir išankstinis apdorojimas**]

Pradžioje atsisiunčiame anglų–prancūzų duomenų rinkinį,
sudarytą iš [dvikalbių sakinių porų iš Tatoeba projekto](http://www.manythings.org/anki/).
Kiekviena duomenų rinkinio eilutė yra tabuliacijos ženklais atskirta pora,
sudaryta iš angliškos teksto sekos (*šaltinio*)
ir išverstos prancūziškos teksto sekos (*tikslo*).
Atkreipkite dėmesį, kad kiekviena teksto seka
gali būti tik vienas sakinys
arba kelių sakinių pastraipa.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Atsisiuntę duomenų rinkinį,
neapdorotiems tekstiniams duomenims
[**atliekame kelis išankstinio apdorojimo žingsnius**].
Pavyzdžiui, nekeičiamą tarpą (angl. *non-breaking space*) pakeičiame tarpu,
didžiąsias raides paverčiame mažosiomis
ir tarp žodžių bei skyrybos ženklų įterpiame tarpą.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**Skaidymas į teksto vienetus**]

Skirtingai nei simbolių lygmens skaidymas į teksto vienetus
:numref:`sec_language-model`,
mašininiam vertimui
čia pirmenybę teikiame žodžių lygmens skaidymui
(šiandieniniai pažangiausi modeliai naudoja
sudėtingesnes skaidymo technikas).
Toliau pateiktas metodas `_tokenize`
suskaido į teksto vienetus pirmąsias `max_examples` teksto sekų poras,
kur kiekvienas teksto vienetas yra žodis arba skyrybos ženklas.
Prie kiekvienos sekos pabaigos pridedame specialų „&lt;eos&gt;“ teksto vienetą,
nurodantį sekos pabaigą.
Kai modelis prognozuoja
generuodamas seką teksto vienetas po teksto vieneto,
„&lt;eos&gt;“ teksto vieneto sugeneravimas
gali rodyti, kad išvesties seka baigta.
Galiausiai toliau pateiktas metodas grąžina
du teksto vienetų sąrašų sąrašus: `src` ir `tgt`.
Konkrečiai, `src[i]` yra teksto vienetų sąrašas iš
$i$-osios teksto sekos šaltinio kalba (čia anglų),
o `tgt[i]` – tikslo kalba (čia prancūzų).

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

[**Nubraižykime teksto vienetų skaičiaus kiekvienoje teksto sekoje histogramą.**]
Šiame paprastame anglų–prancūzų duomenų rinkinyje
dauguma teksto sekų turi mažiau nei 20 teksto vienetų.

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Fiksuoto ilgio sekų įkėlimas
:label:`subsec_loading-seq-fixed-len`

Prisiminkite, kad kalbos modeliavime
[**kiekviena pavyzdinė seka**],
tiek vieno sakinio segmentas,
tiek kelių sakinių atkarpa,
(**turėjo fiksuotą ilgį.**)
Tai buvo nurodoma argumentu `num_steps`
(laiko žingsnių arba teksto vienetų skaičius) iš :numref:`sec_language-model`.
Mašininiame vertime kiekvienas pavyzdys yra
šaltinio ir tikslo teksto sekų pora,
kurioje dvi teksto sekos gali būti skirtingo ilgio.

Skaičiavimo efektyvumui
vis tiek galime apdoroti teksto sekų minirinkinį
vienu metu naudodami *nukirtimą* ir *užpildymą*.
Tarkime, kad kiekviena to paties minirinkinio seka
turėtų turėti tą patį ilgį `num_steps`.
Jei teksto seka turi mažiau nei `num_steps` teksto vienetų,
prie jos pabaigos tiesiog pridėsime specialų „&lt;pad&gt;“ teksto vienetą,
kol jos ilgis pasieks `num_steps`.
Priešingu atveju teksto seką nukirsime,
paimdami tik pirmuosius jos `num_steps` teksto vienetų
ir atmesdami likusius.
Tokiu būdu kiekviena teksto seka
turės tą patį ilgį
ir bus įkeliama į tos pačios formos minirinkinius.
Be to, taip pat įrašome šaltinio sekos ilgį neįskaitant užpildymo teksto vienetų.
Šios informacijos prireiks kai kuriems vėliau nagrinėjamiems modeliams.


Kadangi mašininio vertimo duomenų rinkinį
sudaro kalbų poros,
galime sudaryti du žodynus –
atskirai šaltinio kalbai ir
tikslo kalbai.
Naudojant žodžių lygmens skaidymą,
žodyno dydis bus gerokai didesnis
nei naudojant simbolių lygmens skaidymą.
Kad tai sušvelnintume,
čia retus teksto vienetus,
pasirodančius mažiau nei du kartus,
traktuojame kaip tą patį nežinomą („&lt;unk&gt;“) teksto vienetą.
Kaip paaiškinsime vėliau (:numref:`fig_seq2seq`),
apmokant su tikslo sekomis,
dekodavimo bloko išvestis (žymenų teksto vienetai)
gali būti ta pati dekodavimo bloko įvestis (tikslo teksto vienetai),
pastumta vienu teksto vienetu;
o specialus sekos pradžios „&lt;bos&gt;“ teksto vienetas
bus naudojamas kaip pirmasis įvesties teksto vienetas
tikslo sekai prognozuoti (:numref:`fig_seq2seq_predict`).

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**Duomenų rinkinio skaitymas**]

Galiausiai apibrėžiame metodą `get_dataloader`,
kuris grąžina duomenų iteratorių.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

[**Perskaitykime pirmąjį minirinkinį iš anglų–prancūzų duomenų rinkinio.**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Parodome šaltinio ir tikslo sekų porą,
apdorotą aukščiau pateikto metodo `_build_arrays`
(eilutės formatu).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Santrauka

Natūraliosios kalbos apdorojime *mašininis vertimas* reiškia užduotį automatiškai atvaizduoti seką, vaizduojančią teksto eilutę *šaltinio* kalba, į eilutę, vaizduojančią tikėtiną vertimą *tikslo* kalba. Naudojant žodžių lygmens skaidymą, žodyno dydis bus gerokai didesnis nei naudojant simbolių lygmens skaidymą, bet sekų ilgiai bus daug trumpesni. Kad sumažintume didelį žodyno dydį, retus teksto vienetus galime traktuoti kaip kokį nors „nežinomą“ teksto vienetą. Teksto sekas galime nukirpti ir užpildyti, kad visos jos turėtų tą patį ilgį ir galėtų būti įkeliamos į minirinkinius. Šiuolaikiniai įgyvendinimai dažnai sugrupuoja panašaus ilgio sekas, kad neeikvotų pernelyg daug skaičiavimų užpildymui.


## Pratimai

1. Išbandykite skirtingas argumento `max_examples` reikšmes metode `_tokenize`. Kaip tai veikia šaltinio ir tikslo kalbos žodynų dydžius?
1. Kai kurių kalbų, pavyzdžiui, kinų ir japonų, tekste nėra žodžių ribų žymeklių (pvz., tarpų). Ar žodžių lygmens skaidymas tokiais atvejais vis dar yra gera idėja? Kodėl taip arba kodėl ne?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/18020)
:end_tab:
