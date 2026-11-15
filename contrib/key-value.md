# Paskirstytoji raktų ir reikšmių saugykla
:label:`sec_key_value`

KVStore yra vieta duomenims dalytis. Įsivaizduokite ją kaip vieną objektą, bendrą skirtingiems įrenginiams (GPU ir kompiuteriams), kai kiekvienas įrenginys gali įrašyti duomenis (angl. *push*) ir juos nuskaityti (angl. *pull*).

## Inicializavimas
Panagrinėkime paprastą pavyzdį: į saugyklą inicializuojame (int, NDArray) porą, o tada reikšmę nuskaitome:

```{.python .input  n=1}
#@tab mxnet
from mxnet import np, npx, kv
npx.set_np()
```

```{.python .input  n=2}
#@tab mxnet
np.ones((2,3))
```

```{.python .input  n=11}
#@tab mxnet
help(kv)
```

```{.python .input  n=3}
#@tab mxnet
kv = kv.create('local')  # Create a local kv store.
shape = (2,3)
kv.init(3, np.ones(shape) * 2)
a = np.zeros(shape)
kv.pull(3, out = a)
print(a)
```

## Įrašymas, agregavimas ir atnaujinimas

Bet kuriam inicializuotam raktui galite įrašyti naują tokios pat formos reikšmę:

```{.python .input  n=4}
#@tab mxnet
kv.push(3, np.ones(shape)*8)
kv.pull(3, out = a)  # Pull out the value
print(a.asnumpy())
```

Įrašomi duomenys gali būti saugomi bet kuriame įrenginyje. Be to, į tą patį raktą galite įrašyti kelias reikšmes: tada KVStore pirmiausia susumuos visas šias reikšmes ir įrašys agreguotą reikšmę. Čia tik pademonstruosime reikšmių sąrašo įrašymą CPU. Atkreipkite dėmesį, kad sumavimas vyksta tik tada, kai reikšmių sąrašas ilgesnis nei vienas elementas

```{.python .input  n=5}
#@tab mxnet
devices = [npx.cpu(i) for i in range(4)]
b = [np.ones(shape, ctx=device) for device in devices]
kv.push(3, b)
kv.pull(3, out = a)
print(a)
```

Kiekvieno įrašymo metu KVStore sujungia įrašomą reikšmę su saugoma reikšme naudodama atnaujintuvą (angl. *updater*). Numatytasis atnaujintuvas yra ASSIGN. Galite pakeisti numatytąjį, kad valdytumėte, kaip duomenys sujungiami:

```{.python .input  n=6}
#@tab mxnet
def update(key, input, stored):
    print(f'update on key: {key}')
    stored += input * 2
kv._set_updater(update)
kv.pull(3, out=a)
print(a)
```

```{.python .input  n=7}
#@tab mxnet
kv.push(3, np.ones(shape))
kv.pull(3, out=a)
print(a)
```

## Nuskaitymas

Jau matėte, kaip nuskaityti vieną rakto ir reikšmės porą. Panašiai kaip įrašant, reikšmę į kelis įrenginius galite nuskaityti vienu kreipiniu:

```{.python .input  n=8}
#@tab mxnet
b = [np.ones(shape, ctx=device) for device in devices]
kv.pull(3, out = b)
print(b[1])
```

## Raktų ir reikšmių porų sąrašo apdorojimas

Visos iki šiol pristatytos operacijos naudojo vieną raktą. KVStore taip pat turi sąsają raktų ir reikšmių porų sąrašui.

Vienam įrenginiui:

```{.python .input  n=9}
#@tab mxnet
keys = [5, 7, 9]
kv.init(keys, [np.ones(shape)]*len(keys))
kv.push(keys, [np.ones(shape)]*len(keys))
b = [np.zeros(shape)]*len(keys)
kv.pull(keys, out = b)
print(b[1])
```

Keliems įrenginiams:

```{.python .input  n=10}
#@tab mxnet
b = [[np.ones(shape, ctx=device) for device in devices]] * len(keys)
kv.push(keys, b)
kv.pull(keys, out = b)
print(b[1][1])
```
