# Kompiuterinė rega
:label:`chap_cv`

Nesvarbu, ar tai medicininė diagnostika, savaeigės transporto priemonės, kamerų stebėjimas ar išmanieji filtrai, daugelis kompiuterinės regos taikymų yra glaudžiai susiję su dabartiniu ir būsimu mūsų gyvenimu.
Pastaraisiais metais gilusis mokymasis buvo
transformuojanti jėga, tobulinanti kompiuterinės regos sistemų našumą.
Galima sakyti, kad pažangiausi kompiuterinės regos taikymai beveik neatskiriami nuo giliojo mokymosi.
Atsižvelgiant į tai, šiame skyriuje susitelksime į kompiuterinės regos sritį ir panagrinėsime metodus bei taikymus, kurie pastaruoju metu buvo įtakingi akademinėje bendruomenėje ir pramonėje.


:numref:`chap_cnn` ir :numref:`chap_modern_cnn` nagrinėjome įvairius konvoliucinius neuroninius tinklus,
dažnai naudojamus kompiuterinėje regoje, ir pritaikėme juos
paprastoms vaizdų klasifikavimo užduotims.
Šio skyriaus pradžioje aprašysime
du metodus,
kurie gali pagerinti modelio apibendrinimą, būtent *vaizdų augmentaciją* ir *patikslinimą*,
ir pritaikysime juos vaizdų klasifikavimui.
Kadangi gilieji neuroniniai tinklai gali veiksmingai vaizduoti vaizdus keliais lygiais,
tokie sluoksniniai vaizdai sėkmingai
naudojami įvairiose kompiuterinės regos užduotyse, tokiose kaip *objektų aptikimas*, *semantinis segmentavimas* ir *stiliaus perkėlimas*.
Laikydamiesi pagrindinės idėjos – kompiuterinėje regoje išnaudoti sluoksninius vaizdus,
pradėsime nuo pagrindinių objektų aptikimo komponentų ir technikų. Toliau parodysime, kaip naudoti *pilnai konvoliucinius tinklus* vaizdų semantiniam segmentavimui. Tada paaiškinsime, kaip naudoti stiliaus perkėlimo technikas vaizdams, tokiems kaip šios knygos viršelis, generuoti.
Galiausiai skyrių užbaigsime
pritaikydami šio ir kelių ankstesnių skyrių medžiagą dviem populiariems kompiuterinės regos lyginamiesiems duomenų rinkiniams.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```

