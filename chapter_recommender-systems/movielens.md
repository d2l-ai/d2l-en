#  MovieLens duomenų rinkinys

Rekomendacijų tyrimams yra prieinama daug duomenų rinkinių. Vienas populiariausių iš jų yra [MovieLens](https://movielens.org/) duomenų rinkinys. MovieLens yra nekomercinė internetinė filmų rekomendacinė sistema. Ją 1997 m. sukūrė ir valdo GroupLens – Minesotos universiteto tyrimų laboratorija, siekdama rinkti filmų įvertinimų duomenis tyrimų tikslais. MovieLens duomenys buvo labai svarbūs keliems moksliniams tyrimams, įskaitant personalizuotas rekomendacijas ir socialinę psichologiją.


## Duomenų gavimas


MovieLens duomenų rinkinį talpina svetainė [GroupLens](https://grouplens.org/datasets/movielens/). Yra kelios versijos. Naudosime MovieLens 100K duomenų rinkinį :cite:`Herlocker.Konstan.Borchers.ea.1999`. Šį duomenų rinkinį sudaro $100\,000$ įvertinimų nuo 1 iki 5 žvaigždučių, kuriuos 1682 filmams skyrė 943 naudotojai. Jis išvalytas taip, kad kiekvienas naudotojas būtų įvertinęs bent 20 filmų. Taip pat prieinama kai kuri paprasta demografinė informacija, pavyzdžiui, naudotojų amžius, lytis ir elementų žanrai. Galime atsisiųsti [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) ir išskleisti failą `u.data`, kuriame yra visi $100\,000$ įvertinimų csv formatu. Aplanke yra daug kitų failų; išsamų kiekvieno failo aprašymą galima rasti duomenų rinkinio [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) faile.

Pirmiausia importuokime paketus, reikalingus šio skyrelio eksperimentams vykdyti.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

Tada atsisiunčiame MovieLens 100k duomenų rinkinį ir įkeliame sąveikas kaip `DataFrame`.

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## Duomenų rinkinio statistika

Įkelkime duomenis ir patikrinkime pirmuosius penkis įrašus rankiniu būdu. Tai efektyvus būdas sužinoti duomenų struktūrą ir įsitikinti, kad jie įkelti teisingai.

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

Matome, kad kiekvieną eilutę sudaro keturi stulpeliai: „user id“ 1–943, „item id“ 1–1682, „rating“ 1–5 ir „timestamp“. Galime sukonstruoti $n \times m$ dydžio sąveikų matricą, kur $n$ ir $m$ yra atitinkamai naudotojų ir elementų skaičius. Šis duomenų rinkinys fiksuoja tik esamus įvertinimus, todėl ją galime vadinti ir įvertinimų matrica; sąveikų matricą ir įvertinimų matricą vartosime pakaitomis, kai šios matricos reikšmės yra tikslūs įvertinimai. Dauguma įvertinimų matricos reikšmių nežinomos, nes naudotojai neįvertino daugumos filmų. Taip pat parodome šio duomenų rinkinio retumą. Retumas apibrėžiamas kaip `1 - nenulinių elementų skaičius / ( naudotojų skaičius * elementų skaičius)`. Akivaizdu, kad sąveikų matrica yra labai reta (t. y. retumas = 93,695 %). Realaus pasaulio duomenų rinkiniai gali būti dar retesni, o tai jau seniai yra iššūkis kuriant rekomendacines sistemas. Perspektyvus sprendimas yra naudoti papildomą šalutinę informaciją, pavyzdžiui, naudotojų ir elementų požymius, retumui sušvelninti.

Tada nubraižome skirtingų įvertinimų skaičiaus pasiskirstymą. Kaip ir tikėtasi, jis atrodo kaip normalusis pasiskirstymas, kurio dauguma įvertinimų susitelkę ties 3–4.

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## Duomenų rinkinio skaidymas

Duomenų rinkinį padalijame į apmokymo ir testavimo aibes. Toliau pateikta funkcija siūlo du skaidymo režimus: `random` ir `seq-aware`. Režimu `random` funkcija padalija 100 tūkst. sąveikų atsitiktinai neatsižvelgdama į laiko žymas ir pagal numatytuosius nustatymus 90 % duomenų naudoja kaip apmokymo pavyzdžius, o likusius 10 % – kaip testavimo pavyzdžius. Režimu `seq-aware` testavimui paliekame elementą, kurį naudotojas įvertino paskutinį, o naudotojų istorines sąveikas naudojame kaip apmokymo aibę. Naudotojų istorinės sąveikos surikiuojamos nuo seniausios iki naujausios pagal laiko žymą. Šis režimas bus naudojamas sekai jautraus rekomendavimo skyrelyje.

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

Atkreipkite dėmesį, kad praktikoje gera praktika be testavimo aibės naudoti ir patikros aibę. Tačiau trumpumo dėlei to nedarome. Šiuo atveju testavimo aibę galima laikyti atidėta patikros aibe.

## Duomenų įkėlimas

Padalijus duomenų rinkinį, patogumo dėlei apmokymo ir testavimo aibes paversime sąrašais ir žodynais ar matricomis. Toliau pateikta funkcija skaito duomenų lentelę eilutė po eilutės ir sunumeruoja naudotojų ir elementų indeksus pradėdama nuo nulio. Tada funkcija grąžina naudotojų, elementų, įvertinimų sąrašus ir žodyną ar matricą, kurioje saugomos sąveikos. Grįžtamosios informacijos tipą galime nurodyti kaip `explicit` (aiškioji) arba `implicit` (numanomoji).

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Po to sujungiame aukščiau pateiktus žingsnius, ir tai bus naudojama kitame skyrelyje. Rezultatai apgaubiami `Dataset` ir `DataLoader`. Atkreipkite dėmesį, kad apmokymo duomenų `DataLoader` parametras `last_batch` nustatytas į režimą `rollover` (likę pavyzdžiai perkeliami į kitą epochą), o tvarka sumaišoma.

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## Santrauka

* MovieLens duomenų rinkiniai plačiai naudojami rekomendacijų tyrimams. Jie viešai prieinami ir nemokami.
* Apibrėžiame funkcijas MovieLens 100k duomenų rinkiniui atsisiųsti ir iš anksto apdoroti, kad galėtume jį naudoti tolesniuose skyreliuose.


## Pratimai

* Kokius kitus panašius rekomendacijų duomenų rinkinius galite rasti?
* Peržiūrėkite svetainę [https://movielens.org/](https://movielens.org/) ir sužinokite daugiau apie MovieLens.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/399)
:end_tab:
