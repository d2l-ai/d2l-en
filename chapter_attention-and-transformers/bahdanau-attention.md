```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Bahdanau Attention

:label:`sec_seq2seq_attention` 

Nous avons étudié le problème de la traduction automatique
dans :numref:`sec_seq2seq`,
où nous avons conçu
une architecture encodeur-décodeur basée sur deux RNN
pour l'apprentissage de séquence à séquence.
Plus précisément,
l'encodeur RNN
transforme
une séquence de longueur variable
en une variable contextuelle de forme fixe,
puis
le décodeur RNN
génère la séquence de sortie (cible) jeton par jeton
sur la base des jetons générés et de la variable contextuelle.
Cependant,
même si tous les jetons d'entrée (source)
ne sont pas utiles pour décoder un certain jeton,
la *même* variable contextuelle
qui code la séquence d'entrée entière
est toujours utilisée à chaque étape du décodage.

Dans un défi distinct mais connexe
de la génération d'écriture manuscrite pour une séquence de texte donnée,
Graves a conçu un modèle d'attention différentiable
pour aligner les caractères du texte avec le tracé beaucoup plus long du stylo,
où l'alignement ne se déplace que dans une seule direction :cite:`Graves.2013`.
Inspirés par l'idée d'apprendre à aligner,
Bahdanau et al. ont proposé un modèle d'attention différenciable
sans la sévère limitation de l'alignement unidirectionnel :cite:`Bahdanau.Cho.Bengio.2014`.
Lors de la prédiction d'un token,
si tous les tokens d'entrée ne sont pas pertinents,
le modèle aligne (ou assiste)
uniquement sur les parties de la séquence d'entrée qui sont pertinentes pour la prédiction actuelle.
Ceci est réalisé
en traitant la variable de contexte comme une sortie de la mise en commun de l'attention.

## Modèle

Lorsque nous décrirons
l'attention Bahdanau
pour l'encodeur-décodeur RNN ci-dessous,
nous suivrons la même notation dans
:numref:`sec_seq2seq`.
Le nouveau modèle basé sur l'attention
est le même que celui de
dans :numref:`sec_seq2seq` 
sauf que
la variable de contexte
$\mathbf{c}$ 
dans
:eqref:`eq_seq2seq_s_t` 
est remplacée par
$\mathbf{c}_{t'}$ 
 à tout pas de temps de décodage $t'$.
Supposons que
il y ait $T$ tokens dans la séquence d'entrée,
la variable de contexte au pas de temps de décodage $t'$
est la sortie du pooling d'attention :

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

où l'état caché du décodeur
$\mathbf{s}_{t' - 1}$ au pas de temps $t' - 1$
est la requête,
et les états cachés du codeur $\mathbf{h}_t$
sont à la fois les clés et les valeurs,
et le poids de l'attention $\alpha$
est calculé comme dans
:eqref:`eq_attn-scoring-alpha` 
en utilisant la fonction de notation additive de l'attention
définie par
:eqref:`eq_additive-attn`.

Légèrement différente de,
l'architecture d'encodeur-décodeur RNN vanille
dans :numref:`fig_seq2seq_details`,
la même architecture
avec l'attention de Bahdanau est représentée dans
:numref:`fig_s2s_attention_details`.

![Layers in an RNN encoder-decoder model with Bahdanau attention.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Définition du décodeur avec attention

Pour mettre en œuvre l'encodeur-décodeur RNN
avec l'attention de Bahdanau,
il suffit de redéfinir le décodeur.
Pour visualiser les poids d'attention appris de manière plus pratique,
la classe suivante `AttentionDecoder`
définit [**l'interface de base pour
décodeurs avec mécanismes d'attention**].

```{.python .input}
%%tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

Implémentons maintenant [**le décodeur RNN avec attention Bahdanau**]
dans la classe `Seq2SeqAttentionDecoder` suivante.
L'état du décodeur
est initialisé avec
(i) les états cachés de la couche finale du codeur à tous les pas de temps (en tant que clés et valeurs de l'attention) ;
(ii) l'état caché de toutes les couches du codeur au dernier pas de temps (pour initialiser l'état caché du décodeur) ;
et (iii) la longueur valide du codeur (pour exclure les jetons de remplissage dans le regroupement de l'attention).
À chaque étape temporelle de décodage,
l'état caché de la couche finale du décodeur à l'étape temporelle précédente est utilisé comme requête de l'attention.
Par conséquent, la sortie de l'attention
et l'intégration d'entrée sont concaténées
comme entrée du décodeur RNN.

```{.python .input}
%%tab mxnet
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = np.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            hidden_state = hidden_state[0]
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (batch_size, num_steps, num_hiddens).
        # Length of list hidden_state is num_layers, where the shape of its
        # element is (batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (tf.transpose(outputs, (1, 0, 2)), hidden_state,
                enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Shape of output enc_outputs: # (batch_size, num_steps, num_hiddens)
        # Length of list hidden_state is num_layers, where the shape of its
        # element is (batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X)  # Input X has shape: (batch_size, num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Concatenate on the feature dimension
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (batch_size, num_steps, vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

Dans ce qui suit, nous [**testons le décodeur implémenté**]
avec l'attention Bahdanau
en utilisant un minibatch de 4 entrées de séquence
de 7 pas de temps.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
if tab.selected('mxnet'):
    X = d2l.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('pytorch'):
    X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('tensorflow'):
    X = tf.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X, training=False), None)
    output, state = decoder(X, state, training=False)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

## [**Training**]

Similaire à :numref:`sec_seq2seq_training`,
ici nous spécifions les hyperparamètres,
instancions
un encodeur et un décodeur avec l'attention de Bahdanau,
et entraînons ce modèle pour la traduction automatique.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128) 
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch'):
    encoder = d2l.Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = d2l.Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.005)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

Une fois le modèle entraîné,
nous l'utilisons pour [**traduire quelques phrases anglaises**]
en français et calculer leurs scores BLEU.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)        
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')  
```

En [**visualisant les poids d'attention**]
lors de la traduction de la dernière phrase anglaise,
nous pouvons voir que chaque requête attribue des poids non uniformes
sur les paires clé-valeur.
Cela montre qu'à chaque étape du décodage,
différentes parties des séquences d'entrée
sont sélectivement agrégées dans le regroupement d'attention.

```{.python .input}
%%tab all
_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weights], 0),
    (1, 1, -1, data.num_steps))
```

```{.python .input}
%%tab mxnet
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab tensorflow
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

## Résumé

* Lors de la prédiction d'un token, si tous les tokens d'entrée ne sont pas pertinents, le codeur-décodeur RNN avec attention Bahdanau agrège sélectivement différentes parties de la séquence d'entrée. Ceci est réalisé en traitant la variable de contexte comme une sortie de la mise en commun de l'attention additive.
* Dans le codeur-décodeur RNN, l'attention de Bahdanau traite l'état caché du décodeur au pas de temps précédent comme la requête, et les états cachés du codeur à tous les pas de temps comme les clés et les valeurs.

## Exercices

1. Remplacez GRU par LSTM dans l'expérience.
1. Modifiez l'expérience pour remplacer la fonction de notation additive de l'attention par le produit scalaire du point. Comment cela influence-t-il l'efficacité de l'entrainement ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3868)
:end_tab:
