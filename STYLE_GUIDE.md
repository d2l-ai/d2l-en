# STILIAUS GAIRĖS

> Pastaba: šios gairės išverstos iš originalo. Kodo identifikatoriai, funkcijų ir karkasų pavadinimai paliekami angliški.

## Bendrai

* Būkite tikslūs, aiškūs, įtraukiantys, pragmatiški ir nuoseklūs

## Tekstas

* Skyriai ir poskyriai
    * Kiekvieno skyriaus pradžioje pateikite apžvalgą
    * Laikykitės nuoseklios kiekvieno poskyrio struktūros
        * Santrauka
        * Pratimai
* Citatos
    * Naudokite dvigubas kabutes
* Simbolių aprašymai
    * laiko žingsnis t（ne t laiko žingsnis）
* Įrankiai, klasės ir funkcijos
    * Gluon, MXNet, NumPy, spaCy, NDArray, Symbol, Block, HybridBlock, ResNet-18, Fashion-MNIST, matplotlib
        * Laikykite juos žodžiais be kabučių (``)
    * Sequential klasė / objektas, HybridSequential klasė / objektas
        * Be kabučių (``)
    * `backward` funkcija
        * ne `backward()` funkcija
    * „for-loop“, ne „for loop“
* Terminai
    * Nuosekliai naudokite
        * funkcija (ne metodas)
        * objektas (angl. *instance*, ne *object*)
        * svoris, poslinkis, žymė
        * modelio mokymas, modelio prognozavimas (modelio inferencija)
        * mokymo / testavimo / validavimo duomenų rinkinys
        * vietoj „duomenų objektas“ ar „duomenų taškas“ pirmenybę teikite „duomenų / mokymo / testavimo pavyzdys“
    * Skirkite：
        * hiperparametras ir parametras
        * mažos imties stochastinis gradientinis nusileidimas ir stochastinis gradientinis nusileidimas
* Skaitmenis naudokite, kai jie paaiškina kodą arba yra kodo ar matematikos dalis.
* Priimtinos santrumpos
    * DI, DBP (MLP), KNT (CNN), RNT (RNN), GRU, LSTM, modelių pavadinimai (pvz., ELMo, GPT, BERT)
    * Daugeliu atvejų, kad būtų aiškiau, rašome pilnus pavadinimus (pvz., NLP -> natūraliosios kalbos apdorojimas)

## Matematika

* Laikykitės nuoseklaus [matematinio žymėjimo](chapter_notation/index.md)
* Jei reikia, skyrybos ženklus dėkite lygčių viduje
    * pvz., kablelį ir tašką
* Priskyrimo simbolis
    * \leftarrow
* Matematinius skaitmenis naudokite tik kai jie yra matematikos dalis: „$x$ yra arba $1$, arba $-1$“, „skaičių $12$ ir $18$ didžiausias bendrasis daliklis yra $6$“.
* Netaikome „tūkstančių skyriklio“ (nes skirtingos leidyklos laikosi skirtingų stilių). Pvz., 10,000 šaltiniuose markdown failuose rašoma kaip 10000.

## Paveikslai

* Programinė įranga
    * Paveikslams kurti naudokite OmniGraffle.
      * Eksportuokite pdf (begalinė drobė) 100 %, tada pdf2svg konvertuokite į svg
        * `ls *.pdf | while read f; do pdf2svg $f ${f%.pdf}.svg; done`
      * Neeksportuokite svg tiesiai iš Omnigraffle (šrifto dydis gali šiek tiek pasikeisti)
* Stilius
    * Dydis：
        * Horizontaliai：<= 400 pikselių  (riboja puslapio plotis)
        * Vertikaliai：<= 200 pikselių (galimos išimtys)
    * Storis：
        * StickArrow
        * 1pt
        * rodyklės galvutės dydis: 50 %
    * Šriftas：
        * Arial (tekstui), STIXGeneral (matematikai), 9pt（apatiniai / viršutiniai indeksai：6pt）
        * Nekursyvinkite skaičių ar skliaustų apatiniuose ar viršutiniuose indeksuose
    * Spalva：
        * Mėlyna kaip fonas (tekstas juodas)
            * (Stenkitės vengti) Ypač tamsi：3FA3FD
            * Tamsi：66BFFF
            * Šviesi：B2D9FF
            * (Stenkitės vengti) Ypač šviesi: CFF4FF
* Būkite atidūs dėl autorių teisių


## Kodas

* Kiekviena eilutė turi turėti <=78 simbolius (riboja puslapio plotis). [Cambridge stiliui](https://github.com/d2l-ai/d2l-en/pull/2187) kiekviena eilutė turi turėti <=79 simbolius.
* Python
    * PEP8
        * pvz., (https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator)
* Kad sutaupytumėte vietos, kelis priskyrimus dėkite toje pačioje eilutėje
  * pvz., `num_epochs, lr = 5, 0.1`
* Būkite nuoseklūs kintamųjų pavadinimuose
    * `num_epochs`
        * epochų skaičius
    * `num_hiddens`
        * paslėptųjų vienetų skaičius
    * `num_inputs`
        * įvesčių skaičius
    * `num_outputs`
        * išvesčių skaičius
    * `net`
        * modelis
    * `lr`
        * mokymosi sparta
    * `acc`
        * tikslumas
    * Iteracijų metu
        * požymiai：`X`
        * žymės：`y`, `y_hat` arba `Y`, `Y_hat`
        * `for X, y in data_iter`
    * Duomenų rinkiniai：
        * požymiai：`features` arba `images`
        * žymės：`labels`
        * DataLoader objektas：`train_iter`, `test_iter`, `data_iter`
* Komentarai
    * Komentarų gale taško nededame.
    * Kad būtų aiškiau, kintamųjų pavadinimus apskliauskite kabutėmis, pvz.,  # `X` forma
* Importai
    * importuokite abėcėlės tvarka
* Kintamųjų spausdinimas
    * jei įmanoma, kodo bloko pabaigoje naudokite `x, y` vietoj `print('x:', x, 'y:', y)`
* Eilutės
    * Naudokite viengubas kabutes
    * Naudokite f-eilutes. Norėdami ilgą f-eilutę suskaidyti į kelias eilutes, naudokite po vieną f-eilutę eilutėje.
* Kita
    * `nd.f(x)` → `x.nd`
    * `.1` → `1.0`
    * 1. → `1.0`


## Nuorodos

* Kaip pridėti nuorodas į paveikslus, lenteles ir lygtis, žr. [d2lbook](https://book.d2l.ai/user/markdown.html#cross-references).


## URL

Nustačius `style = cambridge`, URL bus paversti QR kodais, todėl specialiuosius simbolius reikia pakeisti [URL kodavimu](https://www.urlencoder.io/learn/). Pavyzdžiui:

`Stanford's [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)`
->
`Stanford's [large movie review dataset](https://ai.stanford.edu/%7Eamaas/data/sentiment/)`


## Citavimas

1. Paleiskite `pip install git+https://github.com/d2l-ai/d2l-book`
1. Naudokite bibtool nuosekliems bibtex įrašų raktams generuoti. Įdiekite: `brew install bib-tool`
1. Pridėkite bibtex įrašą į `d2l.bib` šakniniame kataloge. Tarkime, originalus įrašas yra
```
@article{wood2011sequence,
  title={The sequence memoizer},
  author={Wood, Frank and Gasthaus, Jan and Archambeau, C{\'e}dric and James, Lancelot and Teh, Yee Whye},
  journal={Communications of the ACM},
  volume={54},
  number={2},
  pages={91--98},
  year={2011},
  publisher={ACM}
}
```
4. Paleiskite `bibtool -s -f "%3n(author).%d(year)" d2l.bib -o d2l.bib`. Dabar pridėtas įrašas turės nuoseklius raktus, o kaip šalutinis efektas, jis atsiras abėcėlės tvarka kitų failo darbų atžvilgiu:
```
@Article{	  Wood.Gasthaus.Archambeau.ea.2011,
  title		= {The sequence memoizer},
  author	= {Wood, Frank and Gasthaus, Jan and Archambeau, C{\'e}dric
		  and James, Lancelot and Teh, Yee Whye},
  journal	= {Communications of the ACM},
  volume	= {54},
  number	= {2},
  pages		= {91--98},
  year		= {2011},
  publisher	= {ACM}
}
```
5. Tekste pridėtam darbui cituoti naudokite:
```
:cite:`Wood.Gasthaus.Archambeau.ea.2011`
```


## Kodo redagavimas ir testavimas viename karkase

1. Tarkime, norime redaguoti ir testuoti MXNet kodą xx.md: paleiskite `d2lbook activate default xx.md`. Tada kitų karkasų kodas xx.md faile deaktyvuojamas.
2. Atidarykite xx.md naudodami Jupyter notebook, redaguokite kodą ir kodui testuoti naudokite „Kernel -> Restart & Run All“.
3. Paleiskite `d2lbook activate all xx.md`, kad iš naujo aktyvuotumėte visų karkasų kodą. Tada git push.

Panašiai `d2lbook activate pytorch/tensorflow xx.md` aktyvuos tik PyTorch/TensorFlow kodą xx.md faile.
