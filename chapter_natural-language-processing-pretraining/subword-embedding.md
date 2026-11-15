# Žodžių dalių įterpiniai
:label:`sec_fasttext`

Anglų kalboje
žodžiai, tokie kaip
„helps“, „helped“ ir „helping“, yra
to paties žodžio „help“ (padėti)
linksniuotės ar asmenuotės formos.
Sąryšis
tarp „dog“ ir „dogs“
yra toks pat kaip
tarp „cat“ ir „cats“,
o
sąryšis
tarp „boy“ ir „boyfriend“
yra toks pat kaip
tarp „girl“ ir „girlfriend“.
Kitose kalbose,
pavyzdžiui, prancūzų ir ispanų,
daugelis veiksmažodžių turi daugiau nei 40 kaitomų formų,
o suomių kalboje
daiktavardis gali turėti iki 15 linksnių.
Kalbotyroje
morfologija tiria žodžių darybą ir žodžių sąryšius.
Tačiau
vidinė žodžių struktūra
nebuvo nagrinėjama nei word2vec,
nei GloVe.

## fastText modelis

Prisiminkite, kaip žodžiai vaizduojami word2vec.
Ir skip-gram modelyje,
ir nuolatinio žodžių krepšio modelyje
skirtingos to paties žodžio kaitomos formos
tiesiogiai vaizduojamos skirtingais vektoriais
be bendrų parametrų.
Kad būtų panaudota morfologinė informacija,
*fastText* modelis
pasiūlė *žodžių dalių įterpinių* (angl. *subword embedding*) metodą,
kur žodžio dalis yra simbolių $n$-grama :cite:`Bojanowski.Grave.Joulin.ea.2017`.
Užuot mokęsis žodžių lygmens vektorinių vaizdavimų,
fastText galima laikyti
žodžių dalių lygmens skip-gram,
kur kiekvienas *centrinis žodis* vaizduojamas
jo žodžių dalių vektorių suma.

Iliustruokime, kaip gauti
kiekvieno centrinio žodžio žodžių dalis fastText
naudojant žodį „where“.
Pirmiausia pridėkite specialius simbolius „&lt;“ ir „&gt;“
žodžio pradžioje ir pabaigoje, kad atskirtumėte priešdėlius ir priesagas nuo kitų žodžių dalių.
Tada iš žodžio išgaukite simbolių $n$-gramas.
Pavyzdžiui, kai $n=3$,
gauname visas 3 ilgio žodžio dalis: „&lt;wh“, „whe“, „her“, „ere“, „re&gt;“ ir specialią žodžio dalį „&lt;where&gt;“.


fastText bet kuriam žodžiui $w$
pažymėkime $\mathcal{G}_w$
visų jo žodžių dalių, kurių ilgis tarp 3 ir 6,
ir jo specialios žodžio dalies sąjungą.
Žodynas
yra visų žodžių žodžių dalių sąjunga.
Tegul $\mathbf{z}_g$
yra žodžio dalies $g$ žodyne vektorius;
tuomet žodžio $w$ kaip centrinio žodžio
skip-gram modelyje vektorius $\mathbf{v}_w$
yra jo žodžių dalių vektorių suma:

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

Likusi fastText dalis yra tokia pati kaip skip-gram modelio. Palyginti su skip-gram modeliu,
fastText žodynas yra didesnis,
todėl yra daugiau modelio parametrų.
Be to,
kad apskaičiuotume žodžio vaizdavimą,
reikia susumuoti
visus jo žodžių dalių vektorius,
o tai lemia didesnį skaičiavimo sudėtingumą.
Tačiau
dėl bendrų parametrų iš žodžių dalių tarp panašios struktūros žodžių
reti žodžiai ir net žodžiai, kurių nėra žodyne,
fastText gali gauti geresnius vektorinius vaizdavimus.



## Baitų porų kodavimas
:label:`subsec_Byte_Pair_Encoding`

fastText visos išgautos žodžių dalys turi būti nurodytų ilgių, pavyzdžiui, nuo $3$ iki $6$, todėl žodyno dydžio iš anksto nustatyti negalima.
Kad fiksuoto dydžio žodyne būtų galima turėti kintamo ilgio žodžių dalis,
galime taikyti glaudinimo algoritmą,
vadinamą *baitų porų kodavimu* (angl. *byte pair encoding*, BPE), žodžių dalims išgauti :cite:`Sennrich.Haddow.Birch.2015`.

Baitų porų kodavimas atlieka apmokymo duomenų rinkinio statistinę analizę, kad atrastų dažnus simbolius žodyje,
pavyzdžiui, bet kokio ilgio iš eilės einančių simbolių sekas.
Pradedant nuo 1 ilgio simbolių,
baitų porų kodavimas iteratyviai sujungia dažniausią iš eilės einančių simbolių porą, kad sukurtų naujus ilgesnius simbolius.
Atkreipkite dėmesį, kad dėl efektyvumo poros, kertančios žodžių ribas, neįtraukiamos.
Galiausiai tokius simbolius galime naudoti kaip žodžių dalis žodžiams segmentuoti.
Baitų porų kodavimas ir jo variantai buvo naudojami įvesties vaizdavimams populiariuose natūraliosios kalbos apdorojimo išankstinio apmokymo modeliuose, tokiuose kaip GPT-2 :cite:`Radford.Wu.Child.ea.2019` ir RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
Toliau iliustruosime, kaip veikia baitų porų kodavimas.

Pirmiausia simbolių žodyną inicializuojame visais angliškais mažosiomis raidėmis, specialiu žodžio pabaigos simboliu `'_'` ir specialiu nežinomo simboliu `'[UNK]'`.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Kadangi nenagrinėjame simbolių porų, kertančių žodžių ribas,
mums reikia tik žodyno `raw_token_freqs`, kuris atvaizduoja žodžius į jų dažnius (pasirodymų skaičių)
duomenų rinkinyje.
Atkreipkite dėmesį, kad specialus simbolis `'_'` pridedamas prie kiekvieno žodžio, kad
galėtume lengvai atkurti žodžių seką (pvz., „a taller man“)
iš išvesties simbolių sekos (pvz., „a_ tall er_ man“).
Kadangi sujungimo procesą pradedame nuo žodyno, sudaryto tik iš pavienių simbolių ir specialių simbolių, tarpas įterpiamas tarp kiekvienos iš eilės einančių simbolių poros kiekviename žodyje (žodyno `token_freqs` raktai).
Kitaip tariant, tarpas yra skirtukas tarp simbolių žodyje.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Apibrėžiame šią funkciją `get_max_freq_pair`, kuri
grąžina dažniausią iš eilės einančių simbolių porą žodyje,
kur žodžiai gaunami iš įvesties žodyno `token_freqs` raktų.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

Kaip godusis metodas, pagrįstas iš eilės einančių simbolių dažniu,
baitų porų kodavimas naudos šią funkciją `merge_symbols`, kad sujungtų dažniausią iš eilės einančių simbolių porą ir sukurtų naujus simbolius.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Dabar iteratyviai vykdome baitų porų kodavimo algoritmą žodyno `token_freqs` raktams. Pirmoje iteracijoje dažniausia iš eilės einančių simbolių pora yra `'t'` ir `'a'`, todėl baitų porų kodavimas juos sujungia ir sukuria naują simbolį `'ta'`. Antroje iteracijoje baitų porų kodavimas toliau sujungia `'ta'` ir `'l'`, kad gautų dar vieną naują simbolį `'tal'`.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Po 10 baitų porų kodavimo iteracijų matome, kad sąraše `symbols` dabar yra 10 daugiau simbolių, iteratyviai sujungtų iš kitų simbolių.

```{.python .input}
#@tab all
print(symbols)
```

Tam pačiam duomenų rinkiniui, nurodytam žodyno `raw_token_freqs` raktuose,
kiekvienas duomenų rinkinio žodis dabar segmentuotas žodžių dalimis „fast_“, „fast“, „er_“, „tall_“ ir „tall“
baitų porų kodavimo algoritmo dėka.
Pavyzdžiui, žodžiai „faster_“ ir „taller_“ segmentuojami atitinkamai kaip „fast er_“ ir „tall er_“.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Atkreipkite dėmesį, kad baitų porų kodavimo rezultatas priklauso nuo naudojamo duomenų rinkinio.
Taip pat galime naudoti žodžių dalis, išmoktas iš vieno duomenų rinkinio,
kito duomenų rinkinio žodžiams segmentuoti.
Kaip godusis metodas, ši funkcija `segment_BPE` bando suskaidyti žodžius į ilgiausias galimas žodžių dalis iš įvesties argumento `symbols`.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

Toliau naudojame sąraše `symbols` esančias žodžių dalis, išmoktas iš aukščiau minėto duomenų rinkinio,
kad segmentuotume `tokens`, vaizduojančius kitą duomenų rinkinį.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Santrauka

* fastText modelis siūlo žodžių dalių įterpinių metodą. Remdamasis word2vec skip-gram modeliu, jis vaizduoja centrinį žodį kaip jo žodžių dalių vektorių sumą.
* Baitų porų kodavimas atlieka apmokymo duomenų rinkinio statistinę analizę, kad atrastų dažnus simbolius žodyje. Kaip godusis metodas, baitų porų kodavimas iteratyviai sujungia dažniausią iš eilės einančių simbolių porą.
* Žodžių dalių įterpiniai gali pagerinti retų žodžių ir žodžių, kurių nėra žodyne, vaizdavimų kokybę.

## Pratimai

1. Pavyzdžiui, anglų kalboje yra apie $3\times 10^8$ galimų $6$-gramų. Kokia problema kyla, kai žodžių dalių per daug? Kaip ją išspręsti? Užuomina: žr. fastText straipsnio 3.2 skyriaus pabaigą :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. Kaip suprojektuoti žodžių dalių įterpinių modelį, pagrįstą nuolatinio žodžių krepšio modeliu?
1. Kad gautume $m$ dydžio žodyną, kiek sujungimo operacijų reikia, kai pradinio simbolių žodyno dydis yra $n$?
1. Kaip išplėsti baitų porų kodavimo idėją frazėms išgauti?



:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/4587)
:end_tab:
