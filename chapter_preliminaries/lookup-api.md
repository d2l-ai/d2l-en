```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dokumentacija
:begin_tab:`mxnet`
Nors negalime pristatyti kiekvienos MXNet funkcijos ir klasės
(o informacija gali greitai pasenti),
[API dokumentacija](https://mxnet.apache.org/versions/1.8.0/api)
ir papildomi [vadovai](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) bei pavyzdžiai
suteikia tokią dokumentaciją.
Šiame skyrelyje pateikiame šiek tiek patarimų, kaip tyrinėti MXNet API.
:end_tab:

:begin_tab:`pytorch`
Nors negalime pristatyti kiekvienos PyTorch funkcijos ir klasės
(o informacija gali greitai pasenti),
[API dokumentacija](https://pytorch.org/docs/stable/index.html) ir papildomi [vadovai](https://pytorch.org/tutorials/beginner/basics/intro.html) bei pavyzdžiai
suteikia tokią dokumentaciją.
Šiame skyrelyje pateikiame šiek tiek patarimų, kaip tyrinėti PyTorch API.
:end_tab:

:begin_tab:`tensorflow`
Nors negalime pristatyti kiekvienos TensorFlow funkcijos ir klasės
(o informacija gali greitai pasenti),
[API dokumentacija](https://www.tensorflow.org/api_docs) ir papildomi [vadovai](https://www.tensorflow.org/tutorials) bei pavyzdžiai
suteikia tokią dokumentaciją.
Šiame skyrelyje pateikiame šiek tiek patarimų, kaip tyrinėti TensorFlow API.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## Modulio funkcijos ir klasės

Norėdami sužinoti, kurias funkcijas ir klases galima kviesti modulyje,
iškviečiame funkciją `dir`. Pavyzdžiui, galime
(**užklausti visų atsitiktinių skaičių generavimo modulio savybių**):

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

Paprastai galime nepaisyti funkcijų, prasidedančių ir pasibaigiančių `__` (specialūs Python objektai),
arba funkcijų, prasidedančių vienu `_` (paprastai vidinės funkcijos).
Pagal likusius funkcijų ar atributų pavadinimus
galime drįsti spėti, kad šis modulis siūlo
įvairius atsitiktinių skaičių generavimo metodus,
įskaitant atranką iš tolygiojo skirstinio (`uniform`),
normaliojo skirstinio (`normal`) ir daugianario skirstinio (`multinomial`).

## Konkrečios funkcijos ir klasės

Konkrečių nurodymų, kaip naudoti tam tikrą funkciją ar klasę,
galime gauti iškviesdami funkciją `help`. Pavyzdžiui,
[**panagrinėkime tenzorių funkcijos `ones` naudojimo instrukcijas**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

Iš dokumentacijos matome, kad funkcija `ones`
sukuria naują tenzorių su nurodyta forma
ir visus elementus nustato lygius 1.
Kai tik įmanoma, turėtumėte (**atlikti greitą bandymą**),
kad patvirtintumėte savo interpretaciją:

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

„Jupyter" sąsiuvinyje galime naudoti `?` dokumentui parodyti kitame
lange. Pavyzdžiui, `list?` sukurs turinį,
beveik identišką `help(list)`,
ir parodys jį naujame naršyklės lange.
Be to, jei naudojame du klaustukus, pvz., `list??`,
bus parodytas ir funkciją įgyvendinantis Python kodas.

Oficiali dokumentacija pateikia daug aprašymų ir pavyzdžių, kurie yra už šios knygos ribų.
Pabrėžiame svarbius naudojimo atvejus,
kurie leis jums greitai pradėti spręsti praktinius uždavinius,
o ne aprėpties išsamumą.
Taip pat skatiname studijuoti bibliotekų šaltinio kodą,
kad pamatytumėte aukštos kokybės gamybinio kodo įgyvendinimo pavyzdžių.
Tai darydami taptumėte geresniu inžinieriumi
ir kartu geresniu mokslininku.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Diskusijos](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[Diskusijos](https://discuss.d2l.ai/t/17972)
:end_tab:
