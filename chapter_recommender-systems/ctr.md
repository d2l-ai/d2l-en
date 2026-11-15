# Požymiais turtingos rekomendacinės sistemos

Sąveikų duomenys yra pats pagrindinis naudotojų pageidavimų ir interesų rodiklis. Jie atlieka lemiamą vaidmenį anksčiau pristatytuose modeliuose. Tačiau sąveikų duomenys paprastai būna itin reti ir kartais triukšmingi. Šiai problemai spręsti į rekomendacinį modelį galime integruoti šalutinę informaciją, pavyzdžiui, elementų požymius, naudotojų profilius ir net kontekstą, kuriame įvyko sąveika. Šių požymių naudojimas padeda rekomenduoti, nes jie gali būti veiksmingi naudotojų interesų prognozuotojai, ypač kai sąveikų duomenų trūksta. Todėl būtina, kad rekomendaciniai modeliai taip pat gebėtų tvarkyti tokius požymius ir turėtų tam tikrą turinio ar konteksto jautrumą. Kad parodytume šio tipo rekomendacinius modelius, pristatome kitą užduotį – internetinės reklamos rekomendacijų paspaudimų dažnio (angl. *click-through rate*, CTR) prognozavimą :cite:`McMahan.Holt.Sculley.ea.2013` – ir pateikiame anonimizuotą reklamos duomenų rinkinį. Tikslinės reklamos paslaugos sulaukė didelio dėmesio ir dažnai laikomos rekomendavimo varikliais. Rekomenduoti reklamas, atitinkančias asmeninį naudotojų skonį ir interesus, svarbu siekiant padidinti paspaudimų dažnį.


Skaitmeninės rinkodaros specialistai naudoja internetinę reklamą, kad rodytų reklamas klientams. Paspaudimų dažnis yra metrika, matuojanti reklamuotojų gaunamų paspaudimų ant reklamų skaičių, tenkantį peržiūrų skaičiui, ir išreiškiama procentais, apskaičiuojamais pagal formulę:

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

Paspaudimų dažnis yra svarbus signalas, rodantis prognozavimo algoritmų veiksmingumą. Paspaudimų dažnio prognozavimas yra užduotis prognozuoti tikimybę, kad kažkas svetainėje bus paspausta. CTR prognozavimo modeliai gali būti naudojami ne tik tikslinės reklamos sistemose, bet ir bendrose elementų (pvz., filmų, naujienų, produktų) rekomendacinėse sistemose, el. pašto kampanijose ir net paieškos sistemose. Jis taip pat glaudžiai susijęs su naudotojų pasitenkinimu, konversijos rodikliu ir gali padėti nustatant kampanijos tikslus, nes padeda reklamuotojams kelti realistiškus lūkesčius.

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Internetinės reklamos duomenų rinkinys

Dėl didelės interneto ir mobiliųjų technologijų pažangos internetinė reklama tapo svarbiu pajamų šaltiniu ir sukuria didžiąją dalį interneto pramonės pajamų. Svarbu rodyti aktualias reklamas arba reklamas, sužadinančias naudotojų susidomėjimą, kad atsitiktiniai lankytojai taptų mokančiais klientais. Mūsų pristatomas duomenų rinkinys yra internetinės reklamos duomenų rinkinys. Jį sudaro 34 laukai, kurių pirmasis stulpelis yra tikslinis kintamasis, nurodantis, ar reklama buvo paspausta (1), ar ne (0). Visi kiti stulpeliai yra kategoriniai požymiai. Stulpeliai gali reikšti reklamos ID, svetainės ar programos ID, įrenginio ID, laiką, naudotojų profilius ir t. t. Tikrosios požymių reikšmės neatskleidžiamos dėl anonimizavimo ir privatumo sumetimų.

Šis kodas atsisiunčia duomenų rinkinį iš mūsų serverio ir išsaugo jį vietiniame duomenų aplanke.

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Yra apmokymo ir testavimo aibės, kurias sudaro atitinkamai 15000 ir 3000 pavyzdžių (eilučių).

## Duomenų rinkinio apvalkalas

Duomenų įkėlimo patogumui įgyvendiname `CTRDataset`, kuris įkelia reklamos duomenų rinkinį iš CSV failo ir gali būti naudojamas su `DataLoader`.

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

Šis pavyzdys įkelia apmokymo duomenis ir išspausdina pirmąjį įrašą.

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Kaip matyti, visi 34 laukai yra kategoriniai požymiai. Kiekviena reikšmė žymi atitinkamo įrašo fiktyvaus kintamojo indeksą. Žymuo $0$ reiškia, kad nebuvo paspausta. Šis `CTRDataset` taip pat gali būti naudojamas kitiems duomenų rinkiniams įkelti, pavyzdžiui, Criteo rodomosios reklamos iššūkio [duomenų rinkiniui](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) ir Avazu paspaudimų dažnio prognozavimo [duomenų rinkiniui](https://www.kaggle.com/c/avazu-ctr-prediction).  

## Santrauka 
* Paspaudimų dažnis yra svarbi metrika, naudojama reklamos sistemų ir rekomendacinių sistemų veiksmingumui matuoti.
* Paspaudimų dažnio prognozavimas paprastai paverčiamas dvejetainio klasifikavimo problema. Tikslas – prognozuoti, ar reklama (elementas) bus paspausta, remiantis duotais požymiais.

## Pratimai

* Ar galite įkelti Criteo ir Avazu duomenų rinkinius su pateiktu `CTRDataset`? Verta pažymėti, kad Criteo duomenų rinkinį sudaro realiosios reikšmės požymiai, todėl gali tekti šiek tiek pakeisti kodą.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/405)
:end_tab:
