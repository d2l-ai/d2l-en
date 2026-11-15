# AutoRec: įvertinimų prognozavimas su autokoduotuvais

Nors matricos faktorizavimo modelis pasiekia gerą našumą įvertinimų prognozavimo užduotyje, iš esmės jis yra tiesinis modelis. Todėl tokie modeliai negali užfiksuoti sudėtingų netiesinių ir painių sąryšių, kurie gali padėti prognozuoti naudotojų pageidavimus. Šiame skyrelyje pristatome netiesinį neuroninio tinklo kolaboratyviojo filtravimo modelį AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`. Jis kolaboratyvųjį filtravimą (CF) sieja su autokoduotuvo architektūra ir siekia integruoti netiesines transformacijas į CF aiškiosios grįžtamosios informacijos pagrindu. Įrodyta, kad neuroniniai tinklai gali aproksimuoti bet kurią tolydžiąją funkciją, todėl jie tinka matricos faktorizavimo apribojimui įveikti ir jo išraiškingumui praturtinti.

Viena vertus, AutoRec turi tokią pačią struktūrą kaip autokoduotuvas, kurį sudaro įvesties sluoksnis, paslėptasis sluoksnis ir rekonstrukcijos (išvesties) sluoksnis. Autokoduotuvas yra neuroninis tinklas, kuris mokosi nukopijuoti savo įvestį į išvestį, kad užkoduotų įvestis į paslėptuosius (ir paprastai mažesnės dimensijos) vaizdavimus. AutoRec vietoj to, kad aiškiai įterptų naudotojus ar elementus į mažesnės dimensijos erdvę, kaip įvestį naudoja sąveikų matricos stulpelį ar eilutę, o tada išvesties sluoksnyje rekonstruoja sąveikų matricą.

Kita vertus, AutoRec skiriasi nuo tradicinio autokoduotuvo: užuot mokęsis paslėptųjų vaizdavimų, AutoRec daugiausia dėmesio skiria išvesties sluoksnio mokymuisi ir rekonstravimui. Jis kaip įvestį naudoja iš dalies stebėtą sąveikų matricą, siekdamas rekonstruoti užbaigtą įvertinimų matricą. Tuo pat metu trūkstami įvesties elementai užpildomi išvesties sluoksnyje rekonstrukcijos būdu rekomendavimo tikslais.

Yra du AutoRec variantai: naudotojais pagrįstas ir elementais pagrįstas. Trumpumo dėlei čia pristatome tik elementais pagrįstą AutoRec. Naudotojais pagrįstą AutoRec galima išvesti analogiškai.


## Modelis

Tegul $\mathbf{R}_{*i}$ žymi $i$-ąjį įvertinimų matricos stulpelį, kuriame nežinomi įvertinimai pagal numatytuosius nustatymus yra nuliai. Neuroninė architektūra apibrėžiama taip:

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

kur $f(\cdot)$ ir $g(\cdot)$ yra aktyvinimo funkcijos, $\mathbf{W}$ ir $\mathbf{V}$ yra svorių matricos, o $\mu$ ir $b$ – poslinkiai. Tegul $h( \cdot )$ žymi visą AutoRec tinklą. Išvestis $h(\mathbf{R}_{*i})$ yra $i$-ojo įvertinimų matricos stulpelio rekonstrukcija.

Šia tikslo funkcija siekiama minimizuoti rekonstrukcijos paklaidą:

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

kur $\| \cdot \|_{\mathcal{O}}$ reiškia, kad atsižvelgiama tik į stebėtų įvertinimų indėlį, t. y. atgalinio sklidimo metu atnaujinami tik svoriai, susiję su stebėtomis įvestimis.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Modelio įgyvendinimas

Tipinį autokoduotuvą sudaro koduotuvas ir dekoduotuvas. Koduotuvas projektuoja įvestį į paslėptuosius vaizdavimus, o dekoduotuvas atvaizduoja paslėptąjį sluoksnį į rekonstrukcijos sluoksnį. Laikomės šios praktikos ir koduotuvą bei dekoduotuvą kuriame iš pilnai sujungtų sluoksnių. Koduotuvo aktyvinimas pagal numatytuosius nustatymus yra `sigmoid`, o dekoduotuvui aktyvinimas netaikomas. Po kodavimo transformacijos įtrauktas dropout, kad sumažėtų persimokymas. Nestebėtų įvesčių gradientai užmaskuojami, kad tik stebėti įvertinimai prisidėtų prie modelio mokymosi proceso.

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## Vertintojo įgyvendinimas iš naujo

Kadangi įvestis ir išvestis pasikeitė, reikia iš naujo įgyvendinti vertinimo funkciją, o kaip tikslumo matą ir toliau naudojame RMSE.

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Modelio apmokymas ir vertinimas

Dabar apmokykime ir įvertinkime AutoRec MovieLens duomenų rinkinyje. Aiškiai matome, kad testavimo RMSE yra mažesnis nei matricos faktorizavimo modelio, o tai patvirtina neuroninių tinklų veiksmingumą įvertinimų prognozavimo užduotyje.

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Santrauka

* Matricos faktorizavimo algoritmą galime perkelti į autokoduotuvų rėmus, kartu integruodami netiesinius sluoksnius ir dropout reguliarizavimą.
* Eksperimentai su MovieLens 100K duomenų rinkiniu rodo, kad AutoRec pasiekia geresnį našumą nei matricos faktorizavimas.



## Pratimai

* Keiskite AutoRec paslėptąją dimensiją, kad pamatytumėte jos poveikį modelio našumui.
* Pabandykite pridėti daugiau paslėptųjų sluoksnių. Ar tai padeda pagerinti modelio našumą?
* Ar galite rasti geresnį dekoduotuvo ir koduotuvo aktyvinimo funkcijų derinį?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/401)
:end_tab:
