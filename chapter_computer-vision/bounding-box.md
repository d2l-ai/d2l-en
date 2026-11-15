# Objektų aptikimas ir apibrėžiamieji stačiakampiai
:label:`sec_bbox`


Ankstesniuose skyreliuose (pvz., :numref:`sec_alexnet`–:numref:`sec_googlenet`)
pristatėme įvairius vaizdų klasifikavimo modelius.
Vaizdų klasifikavimo užduotyse
darome prielaidą, kad vaizde yra tik *vienas*
pagrindinis objektas
ir mus domina tik tai, kaip
atpažinti jo kategoriją.
Tačiau dominančiame vaizde dažnai būna *keli* objektai.
Norime žinoti ne tik jų kategorijas, bet ir jų konkrečias padėtis vaizde.
Kompiuterinėje regoje tokias užduotis vadiname *objektų aptikimu* (arba *objektų atpažinimu*).

Objektų aptikimas
plačiai taikomas daugelyje sričių.
Pavyzdžiui, savaeigėms transporto priemonėms reikia planuoti
kelionės maršrutus
aptinkant
transporto priemonių, pėsčiųjų, kelių ir kliūčių padėtis
užfiksuotuose vaizdo įrašų kadruose.
Be to,
robotai gali naudoti šią techniką
dominantiems objektams aptikti ir lokalizuoti
naršydami aplinkoje.
Be to,
saugumo sistemoms
gali prireikti aptikti nenormalius objektus, tokius kaip įsilaužėliai ar bombos.

Kituose keliuose skyreliuose pristatysime
kelis giliojo mokymosi metodus objektams aptikti.
Pradėsime nuo
objektų *padėčių* (arba *vietų*) pristatymo.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Įkelsime pavyzdinį vaizdą, kurį naudosime šiame skyrelyje. Matome, kad vaizdo kairėje yra šuo, o dešinėje – katė.
Tai du pagrindiniai šio vaizdo objektai.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Apibrėžiamieji stačiakampiai


Objektų aptikime
paprastai naudojame *apibrėžiamąjį stačiakampį* (angl. *bounding box*) objekto erdvinei vietai apibūdinti.
Apibrėžiamasis stačiakampis yra stačiakampis, kurį nustato stačiakampio viršutinio kairiojo kampo $x$ ir $y$ koordinatės bei tokios pat apatinio dešiniojo kampo koordinatės.
Kitas dažnai naudojamas apibrėžiamojo stačiakampio vaizdavimas yra
apibrėžiamojo stačiakampio centro $(x, y)$ ašių
koordinatės ir stačiakampio plotis bei aukštis.

[**Čia apibrėžiame funkcijas, skirtas konvertuoti tarp**] šių (**dviejų
vaizdavimų**):
`box_corner_to_center` konvertuoja iš dviejų kampų
vaizdavimo į centro, pločio ir aukščio vaizdavimą,
o `box_center_to_corner` – atvirkščiai.
Įvesties argumentas `boxes` turėtų būti dvimatis tenzorius,
kurio forma ($n$, 4), kur $n$ yra apibrėžiamųjų stačiakampių skaičius.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Remdamiesi koordinačių informacija [**apibrėšime vaizdo šuns ir katės apibrėžiamuosius stačiakampius**].
Koordinačių pradžia vaizde
yra viršutinis kairysis vaizdo kampas, o į dešinę ir žemyn yra
atitinkamai teigiamos $x$ ir $y$ ašių kryptys.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Dviejų apibrėžiamųjų stačiakampių konvertavimo funkcijų teisingumą galime patikrinti konvertuodami du kartus.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

[**Nubraižykime apibrėžiamuosius stačiakampius vaizde**], kad patikrintume, ar jie tikslūs.
Prieš braižydami apibrėšime pagalbinę funkciją `bbox_to_rect`. Ji vaizduoja apibrėžiamąjį stačiakampį `matplotlib` paketo apibrėžiamojo stačiakampio formatu.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Pridėję apibrėžiamuosius stačiakampius prie vaizdo,
matome, kad pagrindiniai dviejų objektų kontūrai iš esmės yra dviejų stačiakampių viduje.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Santrauka

* Objektų aptikimas ne tik atpažįsta visus dominančius vaizdo objektus, bet ir jų padėtis. Padėtis paprastai vaizduojama apibrėžiamuoju stačiakampiu.
* Galime konvertuoti tarp dviejų dažnai naudojamų apibrėžiamojo stačiakampio vaizdavimų.

## Pratimai

1. Raskite kitą vaizdą ir pabandykite pažymėti apibrėžiamąjį stačiakampį, kuriame yra objektas. Palyginkite apibrėžiamųjų stačiakampių ir kategorijų žymėjimą: kas paprastai trunka ilgiau?
1. Kodėl giliausia įvesties argumento `boxes` dimensija funkcijose `box_corner_to_center` ir `box_center_to_corner` visada yra 4?


:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1527)
:end_tab:
