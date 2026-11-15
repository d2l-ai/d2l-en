```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q mokymasis
:label:`sec_qlearning`

Ankstesniame skyrelyje aptarėme reikšmių iteracijos algoritmą, kuriam reikia prieigos prie viso Markovo sprendimų proceso (MDP), pavyzdžiui, perėjimo ir atlygio funkcijų. Šiame skyrelyje nagrinėsime Q mokymąsi :cite:`Watkins.Dayan.1992` – algoritmą, skirtą išmokti reikšmės funkciją nebūtinai žinant MDP. Šis algoritmas įkūnija pagrindinę skatinamojo mokymosi idėją: jis leis robotui pačiam gauti savo duomenis.
<!-- , instead of relying upon the expert. -->

## Q mokymosi algoritmas

Prisiminkite, kad veiksmo reikšmės funkcijos reikšmių iteracija :ref:`sec_valueiter` atitinka atnaujinimą

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

Kaip aptarėme, šiam algoritmui įgyvendinti reikia žinoti MDP, būtent perėjimo funkciją $P(s' \mid s, a)$. Pagrindinė Q mokymosi idėja yra aukščiau pateiktoje išraiškoje pakeisti sumą per visus $s' \in \mathcal{S}$ suma per būsenas, kuriose lankėsi robotas. Tai leidžia apeiti poreikį žinoti perėjimo funkciją.

## Optimizavimo uždavinys, slypintis už Q mokymosi

Įsivaizduokime, kad robotas veiksmams atlikti naudoja strategiją $\pi_e(a \mid s)$. Kaip ir ankstesniame skyriuje, jis surenka $n$ trajektorijų, kurių kiekviena po $T$ laiko žingsnių, duomenų rinkinį $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$. Prisiminkite, kad reikšmių iteracija iš tiesų yra ribojimų aibė, siejanti skirtingų būsenų ir veiksmų veiksmo reikšmes $Q^*(s, a)$ tarpusavyje. Apytikslę reikšmių iteracijos versiją galime įgyvendinti naudodami duomenis, kuriuos robotas surinko naudodamas $\pi_e$, taip

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

Pirmiausia pastebėkime šios išraiškos ir aukščiau pateiktos reikšmių iteracijos panašumus bei skirtumus. Jei roboto strategija $\pi_e$ būtų lygi optimaliai strategijai $\pi^*$ ir jei jis surinktų begalinį duomenų kiekį, šis optimizavimo uždavinys būtų identiškas optimizavimo uždaviniui, slypinčiam už reikšmių iteracijos. Tačiau nors reikšmių iteracijai reikia žinoti $P(s' \mid s, a)$, optimizavimo tikslo funkcijoje šio nario nėra. Mes nesukčiavome: kai robotas naudoja strategiją $\pi_e$, kad atliktų veiksmą $a_t^i$ būsenoje $s_t^i$, kita būsena $s_{t+1}^i$ yra imtis, atrinkta pagal perėjimo funkciją. Taigi optimizavimo tikslas taip pat turi prieigą prie perėjimo funkcijos, bet netiesiogiai – per roboto surinktus duomenis.

Mūsų optimizavimo uždavinio kintamieji yra $Q(s, a)$ visiems $s \in \mathcal{S}$ ir $a \in \mathcal{A}$. Tikslo funkciją galime minimizuoti gradientiniu nusileidimu. Kiekvienai porai $(s_t^i, a_t^i)$ mūsų duomenų rinkinyje galime užrašyti

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$
:eqlabel:`q_learning`

kur $\alpha$ yra mokymosi greitis. Paprastai realiuose uždaviniuose, kai robotas pasiekia tikslo vietą, trajektorijos baigiasi. Tokios galutinės būsenos reikšmė yra nulis, nes robotas už šios būsenos nebeatlieka jokių tolesnių veiksmų. Savo atnaujinimą turėtume modifikuoti, kad jis tvarkytų tokias būsenas, taip

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

kur $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ yra indikatorinis kintamasis, lygus vienam, jei $s_{t+1}^i$ yra galutinė būsena, ir nuliui kitu atveju. Būsenos ir veiksmo kortežų $(s, a)$, kurie nėra duomenų rinkinio dalis, reikšmė nustatoma į $-\infty$. Šis algoritmas žinomas kaip Q mokymasis.

Turėdami šių atnaujinimų sprendinį $\hat{Q}$, kuris yra optimalios reikšmės funkcijos $Q^*$ aproksimacija, galime lengvai gauti šią reikšmės funkciją atitinkančią optimalią deterministinę strategiją naudodami

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

Gali pasitaikyti situacijų, kai yra kelios deterministinės strategijos, atitinkančios tą pačią optimalią reikšmės funkciją; tokias lygybes galima nutraukti savavališkai, nes jos turi tą pačią reikšmės funkciją.

## Tyrinėjimas Q mokymesi

Strategija, kurią robotas naudoja duomenims rinkti, $\pi_e$, yra esminė, kad Q mokymasis veiktų gerai. Juk vidurkį per $s'$, naudojantį perėjimo funkciją $P(s' \mid s, a)$, pakeitėme roboto surinktais duomenimis. Jei strategija $\pi_e$ nepasiekia įvairių būsenų ir veiksmų erdvės dalių, nesunku įsivaizduoti, kad mūsų įvertis $\hat{Q}$ bus prasta optimalios $Q^*$ aproksimacija. Taip pat svarbu pažymėti, kad tokioje situacijoje $Q^*$ įvertis *visose būsenose* $s \in \mathcal{S}$ bus blogas, ne tik tose, kuriose lankėsi $\pi_e$. Taip yra todėl, kad Q mokymosi tikslas (arba reikšmių iteracija) yra ribojimas, siejantis visų būsenos ir veiksmo porų reikšmes. Todėl labai svarbu duomenims rinkti pasirinkti teisingą strategiją $\pi_e$.

Šią problemą galime sušvelninti pasirinkdami visiškai atsitiktinę strategiją $\pi_e$, kuri veiksmus iš $\mathcal{A}$ atrenka tolygiai atsitiktinai. Tokia strategija aplankytų visas būsenas, bet tam prireiktų didelio trajektorijų skaičiaus.

Taip prieiname prie antrosios pagrindinės Q mokymosi idėjos – tyrinėjimo. Tipiniai Q mokymosi įgyvendinimai susieja dabartinį $Q$ įvertį ir strategiją $\pi_e$ nustatydami

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

kur $\epsilon$ vadinamas „tyrinėjimo parametru“ ir yra pasirenkamas naudotojo. Strategija $\pi_e$ vadinama tyrinėjimo strategija. Ši konkreti $\pi_e$ vadinama $\epsilon$-godžiąja tyrinėjimo strategija, nes ji pasirenka optimalų veiksmą (pagal dabartinį įvertį $\hat{Q}$) su tikimybe $1-\epsilon$, o su likusia tikimybe $\epsilon$ tyrinėja atsitiktinai. Taip pat galime naudoti vadinamąją softmax tyrinėjimo strategiją

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

kur hiperparametras $T$ vadinamas temperatūra. Didelė $\epsilon$ reikšmė $\epsilon$-godžiojoje strategijoje veikia panašiai kaip didelė temperatūros $T$ reikšmė softmax strategijoje.

Svarbu pažymėti, kad kai pasirenkame tyrinėjimą, priklausantį nuo dabartinio veiksmo reikšmės funkcijos įverčio $\hat{Q}$, optimizavimo uždavinį turime periodiškai spręsti iš naujo. Tipiniai Q mokymosi įgyvendinimai po kiekvieno veiksmo, atlikto naudojant $\pi_e$, atlieka vieną minirinkinio atnaujinimą naudodami kelias surinkto duomenų rinkinio būsenos ir veiksmo poras (paprastai surinktas ankstesniame roboto laiko žingsnyje).

## „Savaime besitaisanti“ Q mokymosi savybė

Duomenų rinkinys, kurį robotas surenka Q mokymosi metu, laikui bėgant auga. Tiek tyrinėjimo strategija $\pi_e$, tiek įvertis $\hat{Q}$ kinta robotui renkant vis daugiau duomenų. Tai suteikia pagrindinį supratimą, kodėl Q mokymasis veikia gerai. Panagrinėkime būseną $s$: jei tam tikras veiksmas $a$ pagal dabartinį įvertį $\hat{Q}(s,a)$ turi didelę reikšmę, tada ir $\epsilon$-godžioji, ir softmax tyrinėjimo strategijos turi didesnę tikimybę pasirinkti šį veiksmą. Jei šis veiksmas iš tikrųjų *nėra* idealus veiksmas, tada vėlesnės būsenos, atsirandančios dėl šio veiksmo, turės prastus atlygius. Todėl kitas Q mokymosi tikslo atnaujinimas sumažins reikšmę $\hat{Q}(s,a)$, o tai sumažins tikimybę pasirinkti šį veiksmą kitą kartą, kai robotas aplankys būseną $s$. Blogus veiksmus, pvz., tuos, kurių reikšmė pervertinta $\hat{Q}(s,a)$, robotas ištyrinėja, bet jų reikšmė pataisoma kitame Q mokymosi tikslo atnaujinime. Gerus veiksmus, pvz., tuos, kurių reikšmė $\hat{Q}(s, a)$ didelė, robotas tyrinėja dažniau ir taip juos sustiprina. Pasinaudojus šia savybe galima įrodyti, kad Q mokymasis gali konverguoti į optimalią strategiją, net jei pradedama nuo atsitiktinės strategijos $\pi_e$ :cite:`Watkins.Dayan.1992`.

Šis gebėjimas ne tik rinkti naujus duomenis, bet ir rinkti tinkamo tipo duomenis yra pagrindinė skatinamojo mokymosi algoritmų savybė, ir būtent ji juos skiria nuo prižiūrimojo mokymosi. Q mokymasis, naudojantis giliuosius neuroninius tinklus (kuriuos pamatysime vėlesniame DQN skyriuje), yra atsakingas už skatinamojo mokymosi atgimimą :cite:`mnih2013playing`.

## Q mokymosi įgyvendinimas

Dabar parodome, kaip įgyvendinti Q mokymąsi FrozenLake aplinkoje iš [Open AI Gym](https://gym.openai.com). Atkreipkite dėmesį, kad tai ta pati sąranka, kurią nagrinėjome :ref:`sec_valueiter` eksperimente.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

FrozenLake aplinkoje robotas juda $4 \times 4$ tinkleliu (tai yra būsenos) su veiksmais „aukštyn“ ($\uparrow$), „žemyn“ ($\rightarrow$), „į kairę“ ($\leftarrow$) ir „į dešinę“ ($\rightarrow$). Aplinkoje yra keletas duobių (H) langelių ir užšalusių (F) langelių, taip pat tikslo langelis (G), ir visi jie robotui nežinomi. Kad uždavinys būtų paprastas, darome prielaidą, kad robotas turi patikimus veiksmus, t. y. $P(s' \mid s, a) = 1$ visiems $s \in \mathcal{S}, a \in \mathcal{A}$. Jei robotas pasiekia tikslą, bandymas baigiasi ir robotas gauna $1$ atlygį nepriklausomai nuo veiksmo; atlygis bet kurioje kitoje būsenoje yra $0$ visiems veiksmams. Roboto tikslas yra išmokti strategiją, kuri iš nurodytos pradinės vietos (S) (tai yra $s_0$) pasiekia tikslo vietą (G), kad būtų maksimizuota *grąža*.

Pirmiausia įgyvendiname $\epsilon$-godųjį metodą taip:

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

Dabar esame pasirengę įgyvendinti Q mokymąsi:

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Reset environment
        state, done = env.reset(), False
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

Šis rezultatas rodo, kad Q mokymasis gali rasti optimalų šio uždavinio sprendinį maždaug po 250 iteracijų. Tačiau palyginus šį rezultatą su reikšmių iteracijos algoritmo rezultatu (žr. :ref:`subsec_valueitercode`), matome, kad reikšmių iteracijos algoritmui šio uždavinio optimaliam sprendiniui rasti reikia daug mažiau iteracijų. Taip nutinka todėl, kad reikšmių iteracijos algoritmas turi prieigą prie viso MDP, o Q mokymasis – ne.


## Santrauka
Q mokymasis yra vienas iš pačių pagrindinių skatinamojo mokymosi algoritmų. Jis buvo naujausios skatinamojo mokymosi sėkmės epicentre, ypač mokantis žaisti vaizdo žaidimus :cite:`mnih2013playing`. Q mokymosi įgyvendinimui nereikia, kad visiškai žinotume Markovo sprendimų procesą (MDP), pavyzdžiui, perėjimo ir atlygio funkcijas.

## Pratimai

1. Pabandykite padidinti tinklelio dydį iki $8 \times 8$. Kiek iteracijų prireikia optimaliai reikšmės funkcijai rasti, palyginti su $4 \times 4$ tinkleliu?
1. Dar kartą paleiskite Q mokymosi algoritmą su $\gamma$ (t. y. „gamma“ aukščiau pateiktame kode), kai jis lygus $0$, $0.5$ ir $1$, ir išanalizuokite rezultatus.
1. Dar kartą paleiskite Q mokymosi algoritmą su $\epsilon$ (t. y. „epsilon“ aukščiau pateiktame kode), kai jis lygus $0$, $0.5$ ir $1$, ir išanalizuokite rezultatus.

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/12103)
:end_tab:
