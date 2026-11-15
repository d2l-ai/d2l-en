```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Reikšmių iteracija
:label:`sec_valueiter`

Šiame skyrelyje aptarsime, kaip kiekvienoje būsenoje parinkti geriausią roboto veiksmą, kad būtų maksimizuota trajektorijos *grąža*. Aprašysime algoritmą, vadinamą reikšmių iteracija, ir įgyvendinsime jį imituojamam robotui, keliaujančiam per užšalusį ežerą.

## Stochastinė strategija

Stochastinė strategija, žymima $\pi(a \mid s)$ (trumpiau – strategija), yra sąlyginis veiksmų $a \in \mathcal{A}$ skirstinys, esant būsenai $s \in \mathcal{S}$, $\pi(a \mid s) \equiv P(a \mid s)$. Pavyzdžiui, jei robotas turi keturis veiksmus $\mathcal{A}=$ {eiti į kairę, eiti žemyn, eiti į dešinę, eiti aukštyn}. Strategija būsenoje $s \in \mathcal{S}$ tokiai veiksmų aibei $\mathcal{A}$ yra kategorinis skirstinys, kuriame keturių veiksmų tikimybės galėtų būti $[0.4, 0.2, 0.1, 0.3]$; kitoje būsenoje $s' \in \mathcal{S}$ tų pačių keturių veiksmų tikimybės $\pi(a \mid s')$ galėtų būti $[0.1, 0.1, 0.2, 0.6]$. Atkreipkite dėmesį, kad bet kuriai būsenai $s$ turime $\sum_a \pi(a \mid s) = 1$. Deterministinė strategija yra ypatingas stochastinės strategijos atvejis, kai skirstinys $\pi(a \mid s)$ nenulinę tikimybę suteikia tik vienam konkrečiam veiksmui, pvz., $[1, 0, 0, 0]$ mūsų pavyzdyje su keturiais veiksmais.

Kad žymėjimas būtų mažiau sudėtingas, dažnai rašysime $\pi(s)$ kaip sąlyginį skirstinį vietoj $\pi(a \mid s)$.

## Reikšmės funkcija

Įsivaizduokite, kad robotas pradeda nuo būsenos $s_0$ ir kiekvienu laiko momentu pirmiausia atrenka veiksmą iš strategijos $a_t \sim \pi(s_t)$ ir jį atlieka, taip patekdamas į kitą būseną $s_{t+1}$. Trajektorija $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ gali skirtis priklausomai nuo to, kuris konkretus veiksmas $a_t$ atrenkamas tarpiniais momentais. Apibrėžiame visų tokių trajektorijų vidutinę *grąžą* $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

kur $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ yra kita roboto būsena, o $r(s_t, a_t)$ yra betarpiškas atlygis, gaunamas atlikus veiksmą $a_t$ būsenoje $s_t$ laiko momentu $t$. Tai vadinama strategijos $\pi$ „reikšmės funkcija“. Paprastais žodžiais, būsenos $s_0$ reikšmė strategijai $\pi$, žymima $V^\pi(s_0)$, yra tikėtina $\gamma$ diskontuota *grąža*, kurią robotas gauna, jei pradeda būsenoje $s_0$ ir kiekvienu laiko momentu atlieka veiksmus pagal strategiją $\pi$.

Toliau suskaidome trajektoriją į du etapus: (i) pirmąjį etapą, atitinkantį $s_0 \to s_1$ atlikus veiksmą $a_0$, ir (ii) antrąjį etapą, kuris yra vėlesnė trajektorija $\tau' = (s_1, a_1, r_1, \ldots)$. Pagrindinė visų skatinamojo mokymosi algoritmų idėja yra ta, kad būsenos $s_0$ reikšmę galima užrašyti kaip vidutinį atlygį, gautą pirmajame etape, ir reikšmės funkciją, suvidurkintą per visas galimas kitas būsenas $s_1$. Tai gana intuityvu ir kyla iš mūsų Markovo prielaidos: vidutinė grąža iš dabartinės būsenos yra vidutinės grąžos iš kitos būsenos ir vidutinio atlygio už perėjimą į kitą būseną suma. Matematiškai du etapus užrašome kaip

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$
:eqlabel:`eq_dynamic_programming`

Šis skaidymas yra labai galingas: tai dinaminio programavimo principo pagrindas, kuriuo remiasi visi skatinamojo mokymosi algoritmai. Atkreipkite dėmesį, kad antrasis etapas turi du vidurkius: vieną pagal pirmajame etape atlikto veiksmo $a_0$ pasirinkimus naudojant stochastinę strategiją, o kitą pagal galimas būsenas $s_1$, gautas atlikus pasirinktą veiksmą. Lygtį :eqref:`eq_dynamic_programming` galime užrašyti naudodami Markovo sprendimų proceso (MDP) perėjimo tikimybes kaip

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$
:eqlabel:`eq_dynamic_programming_val`

Svarbu pastebėti, kad aukščiau pateikta tapatybė galioja visoms būsenoms $s \in \mathcal{S}$, nes galime apsvarstyti bet kurią trajektoriją, prasidedančią toje būsenoje, ir suskaidyti ją į du etapus.

## Veiksmo ir reikšmės funkcija

Įgyvendinimuose dažnai naudinga palaikyti dydį, vadinamą „veiksmo reikšmės“ funkcija, kuri yra glaudžiai susijęs su reikšmės funkcija dydis. Ji apibrėžiama kaip vidutinė trajektorijos, prasidedančios $s_0$, *grąža*, kai pirmojo etapo veiksmas fiksuojamas kaip $a_0$

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

atkreipkite dėmesį, kad suma po vidurkio ženklu eina nuo $t=1,\ldots, \infty$, nes šiuo atveju pirmojo etapo atlygis yra fiksuotas. Galime vėl suskaidyti trajektoriją į dvi dalis ir užrašyti

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$
:eqlabel:`eq_dynamic_programming_q`

Ši versija yra :eqref:`eq_dynamic_programming_val` analogas veiksmo reikšmės funkcijai.

## Optimali stochastinė strategija

Tiek reikšmės funkcija, tiek veiksmo reikšmės funkcija priklauso nuo strategijos, kurią pasirenka robotas. Toliau pagalvosime apie „optimalią strategiją“, kuri pasiekia didžiausią vidutinę *grąžą*
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

Iš visų galimų stochastinių strategijų, kurias galėjo pasirinkti robotas, optimali strategija $\pi^*$ pasiekia didžiausią vidutinę diskontuotą *grąžą* trajektorijoms, prasidedančioms būsenoje $s_0$. Optimalios strategijos reikšmės funkciją ir veiksmo reikšmės funkciją žymėkime $V^* \equiv V^{\pi^*}$ ir $Q^* \equiv Q^{\pi^*}$.

Panagrinėkime deterministinę strategiją, kai bet kurioje būsenoje pagal strategiją galimas tik vienas veiksmas. Tai mums duoda

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

Gera mnemonika šiam prisiminti yra tokia: optimalus veiksmas būsenoje $s$ (deterministinei strategijai) yra tas, kuris maksimizuoja pirmojo etapo atlygio $r(s, a)$ ir trajektorijų, prasidedančių kita būsena $s'$, vidutinės *grąžos* sumą, suvidurkintą per visas galimas kitas būsenas $s'$ iš antrojo etapo.

## Dinaminio programavimo principas

Mūsų ankstesnio skyrelio išvedžiojimą :eqref:`eq_dynamic_programming` arba :eqref:`eq_dynamic_programming_q` galima paversti algoritmu, skirtu apskaičiuoti atitinkamai optimalią reikšmės funkciją $V^*$ arba veiksmo reikšmės funkciją $Q^*$. Pastebėkite, kad
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Deterministinei optimaliai strategijai $\pi^*$, kadangi būsenoje $s$ galima atlikti tik vieną veiksmą, taip pat galime rašyti 

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

visoms būsenoms $s \in \mathcal{S}$. Ši tapatybė vadinama „dinaminio programavimo principu“ :cite:`BellmanDPPaper,BellmanDPBook`. Jį 1950-aisiais suformulavo Richardas Bellmanas, ir galime jį prisiminti kaip „likusi optimalios trajektorijos dalis taip pat yra optimali“.

## Reikšmių iteracija

Dinaminio programavimo principą galime paversti algoritmu, skirtu optimaliai reikšmės funkcijai rasti, vadinamu reikšmių iteracija. Pagrindinė reikšmių iteracijos idėja yra šią tapatybę laikyti ribojimų aibe, siejančia $V^*(s)$ skirtingose būsenose $s \in \mathcal{S}$. Reikšmės funkciją inicializuojame tam tikromis savavališkomis reikšmėmis $V_0(s)$ visoms būsenoms $s \in \mathcal{S}$. $k$-ojoje iteracijoje reikšmių iteracijos algoritmas atnaujina reikšmės funkciją taip

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

Pasirodo, kad kai $k \to \infty$, reikšmių iteracijos algoritmo įvertinta reikšmės funkcija konverguoja į optimalią reikšmės funkciją nepriklausomai nuo inicializacijos $V_0$,
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Tą patį reikšmių iteracijos algoritmą galima ekvivalenčiai užrašyti naudojant veiksmo reikšmės funkciją
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

Šiuo atveju inicializuojame $Q_0(s, a)$ tam tikromis savavališkomis reikšmėmis visiems $s \in \mathcal{S}$ ir $a \in \mathcal{A}$. Vėl turime $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ visiems $s \in \mathcal{S}$ ir $a \in \mathcal{A}$.

## Strategijos įvertinimas

Reikšmių iteracija leidžia apskaičiuoti optimalią reikšmės funkciją, t. y. optimalios deterministinės strategijos $\pi^*$ funkciją $V^{\pi^*}$. Panašius iteracinius atnaujinimus taip pat galime naudoti reikšmės funkcijai, susijusiai su bet kuria kita, galbūt stochastine, strategija $\pi$, apskaičiuoti. Vėl inicializuojame $V^\pi_0(s)$ tam tikromis savavališkomis reikšmėmis visoms būsenoms $s \in \mathcal{S}$, o $k$-ojoje iteracijoje atliekame atnaujinimus

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Šis algoritmas žinomas kaip strategijos įvertinimas ir yra naudingas reikšmės funkcijai pagal strategiją apskaičiuoti. Vėl pasirodo, kad kai $k \to \infty$, šie atnaujinimai konverguoja į teisingą reikšmės funkciją nepriklausomai nuo inicializacijos $V_0$,

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Strategijos $\pi$ veiksmo reikšmės funkcijos $Q^\pi(s, a)$ skaičiavimo algoritmas yra analogiškas.

## Reikšmių iteracijos įgyvendinimas
:label:`subsec_valueitercode`
Toliau parodome, kaip įgyvendinti reikšmių iteraciją navigacijos uždaviniui, vadinamam FrozenLake iš [Open AI Gym](https://gym.openai.com). Pirmiausia turime nustatyti aplinką, kaip parodyta šiame kode.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations
random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

FrozenLake aplinkoje robotas juda $4 \times 4$ tinkleliu (tai yra būsenos) su veiksmais „aukštyn“ ($\uparrow$), „žemyn“ ($\rightarrow$), „į kairę“ ($\leftarrow$) ir „į dešinę“ ($\rightarrow$). Aplinkoje yra keletas duobių (H) langelių ir užšalusių (F) langelių, taip pat tikslo langelis (G), ir visi jie robotui nežinomi. Kad uždavinys būtų paprastas, darome prielaidą, kad robotas turi patikimus veiksmus, t. y. $P(s' \mid s, a) = 1$ visiems $s \in \mathcal{S}, a \in \mathcal{A}$. Jei robotas pasiekia tikslą, bandymas baigiasi ir robotas gauna $1$ atlygį nepriklausomai nuo veiksmo; atlygis bet kurioje kitoje būsenoje yra $0$ visiems veiksmams. Roboto tikslas yra išmokti strategiją, kuri iš nurodytos pradinės vietos (S) (tai yra $s_0$) pasiekia tikslo vietą (G), kad būtų maksimizuota *grąža*.

Ši funkcija įgyvendina reikšmių iteraciją, kur `env_info` turi su MDP ir aplinka susijusią informaciją, o `gamma` yra diskonto koeficientas:

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

Aukščiau pateiktuose paveiksluose parodyta strategija (rodyklė nurodo veiksmą) ir reikšmės funkcija (spalvos kitimas rodo, kaip reikšmės funkcija kinta laikui bėgant nuo pradinės reikšmės, rodomos tamsia spalva, iki optimalios reikšmės, rodomos šviesiomis spalvomis). Kaip matome, reikšmių iteracija optimalią reikšmės funkciją randa po 10 iteracijų, o tikslo būsenos (G) galima pasiekti iš bet kurios būsenos, jei tai nėra H langelis. Kitas įdomus įgyvendinimo aspektas yra tas, kad, be optimalios reikšmės funkcijos radimo, mes taip pat automatiškai radome ir šią reikšmės funkciją atitinkančią optimalią strategiją $\pi^*$.


## Santrauka
Pagrindinė reikšmių iteracijos algoritmo idėja yra naudoti dinaminio programavimo principą optimaliai vidutinei grąžai, gaunamai iš tam tikros būsenos, rasti. Atkreipkite dėmesį, kad reikšmių iteracijos algoritmui įgyvendinti reikia, kad mes visiškai žinotume Markovo sprendimų procesą (MDP), pavyzdžiui, perėjimo ir atlygio funkcijas.


## Pratimai

1. Pabandykite padidinti tinklelio dydį iki $8 \times 8$. Kiek iteracijų prireikia optimaliai reikšmės funkcijai rasti, palyginti su $4 \times 4$ tinkleliu?
1. Koks yra reikšmių iteracijos algoritmo skaičiavimo sudėtingumas?
1. Dar kartą paleiskite reikšmių iteracijos algoritmą su $\gamma$ (t. y. „gamma“ aukščiau pateiktame kode), kai jis lygus $0$, $0.5$ ir $1$, ir išanalizuokite rezultatus. 
1. Kaip $\gamma$ reikšmė veikia iteracijų, per kurias reikšmių iteracija konverguoja, skaičių? Kas atsitinka, kai $\gamma=1$?

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/12005)
:end_tab:
