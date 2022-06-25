# Large-Scale Pretraining with Transformers
:label:`sec_large-pretraining-transformers`

So far in our image classification and machine translation experiments, models were trained on datasets with input-output examples *from scratch* to perform specific tasks. For example, a transformer was trained with English-French pairs (:numref:`sec_transformer`) so that this model can translate input English text into French. As a result, each model becomes a *specific expert* that is sensitive to even slight shift in data distribution (:numref:`sec_environment-and-distribution-shift`). 
For better generalized models, or even more competent *generalists* that can perform multiple tasks with or without adaptation, *pretraining* models on large data has been pervasive. 

Given larger data for pretraining, the transformer architecture performs better with an increased model size and training compute, demonstrating superior *scaling* behavior. Specifically, performance of transformer-based language models scales as a power-law with the amount of model parameters, training tokens, and training compute :cite:`kaplan2020scaling`. The scalability of transformers is also evidenced by the significantly boosted performance from larger vision transformers trained on larger data (discussed in :numref:`sec_vision-transformer`). More recent success stories include Gato, a *generalist* model that can play Atari, caption images, chat, and act as a robot :cite:`reed2022generalist`. Gato is a single  transformer that scales well when pretrained on diverse modalities including text, images, joint torques, and button presses. Notably, all such multi-modal data is serialized into a flat sequence of tokens, which can be processed akin to text tokens (:numref:`sec_transformer`) or image patches (:numref:`sec_vision-transformer`) by transformers.

Before compelling success of pretraining transformers for multi-modal data, transformers were extensively pretrained  with a wealth of text. Originally proposed for machine translation, the transformer architecture in :numref:`fig_transformer` consists of an encoder for representing input sequences and a decoder for generating target sequences. Primarily, transformers can be used in three different modes: *encoder-only*, *encoder-decoder*, and *decoder-only*. To conclude this chapter, we will review these three modes and explain the scalability in pretraining transformers.

```{.python .input}
# TOREMOVE
```

## Encoder-Only

In encoder-only mode, only the transformer encoder is used, converting a sequence of input tokens into the same number of representations that can be further projected to output (e.g., classification). A transformer encoder consists of multiple self-attention layers, so any output token representation depends on all input tokens.
For example, vision transformers depicted in :numref:`fig_vit` are encoder-only, converting a sequence of input image patches into 
a global representation of a special “&lt;cls&gt;” token, and further projecting it into classification labels. This design was inspired by an earlier encoder-only  transformer pretrained on text: BERT (Bidirectional Encoder Representations from Transformers) :cite:`Devlin.Chang.Lee.ea.2018`.

![Left: Pretraining encoder-only BERT with masked language modeling. Prediction of the masked "love" token depends on all input tokens before and after "love". Right: Attention pattern in the encoder. Each token along the vertical axis attends to all input tokens along the horizontal axis.](../img/bert-encoder-only.svg)
:label:`fig_bert-encoder-only`

BERT is pretrained on text sequences using *masked language modeling*: input text with randomly masked tokens is fed into a transformer encoder to predict the masked tokens. As illustrated in :numref:`fig_bert-encoder-only`, an original text sequence "I", "love", "this", "red", "car" is prepended with the “&lt;cls&gt;” token and the “&lt;mask&gt;” token randomly replaces "love"; then the cross-entropy loss between the ground truth "love" and the prediction at the position of “&lt;mask&gt;” is to be minimized during pretraining. Note that there is no constraint in the attention pattern of transformer encoders (right of :numref:`fig_bert-encoder-only`) so any output depends on all input tokens. Thus, prediction of "love" depends on input tokens before and after it in the sequence. This is why BERT is a "bidirectional encoder". 

Without need for manual labeling, pretraining of BERT can leverage large-scale text data from books and Wikipedia. Pretrained BERT can be *fine tuned* to downstream encoding tasks involving single text or text pairs. During fine tuning, additional layers are appended to BERT with randomized parameters, and these parameters and BERT parameters will be updated on downstream task training data. 


![Fine tuning encoder-only BERT for sentiment analysis.](../img/bert-finetune-classification.svg)
:label:`fig_bert-finetune-classification`

:numref:`fig_bert-finetune-classification` illustrates fine tuning of BERT for sentiment analysis. The transformer encoder is a pretrained BERT, which takes a text sequence as input and feeds representation of “&lt;cls&gt;” (global representation of the input sequence) into an additional MLP to output the sentiment prediction. During fine tuning on a sentiment analysis dataset, MLP is trained from scratch while BERT parameters are updated. 

The largest version of BERT has 350 million parameters and is pretrained on 250 billion tokens. The general language representations learned from BERT advanced the state of the art for natural language tasks such as single text classification, text pair classification or regression, text tagging, and question answering.


The original BERT pretraining includes another loss for predicting whether one sentence is next to the other. However, it was 


BERT variants:

<!--
XLNET :cite:`yang2019xlnet`
RoBERTa :cite:`liu2019roberta`
ALBERT :cite:`lan2019albert`
SpanBERT :cite:`joshi2020spanbert`
DistilBERT :cite:`sanh2019distilbert`
ELECTRA :cite:`clark2019electra`
-->




## Encoder-Decoder

* Although BERT covers many tasks, it's not universal, especially for generation.
* As another pretrain-finetune, so T5 casts every text problem as a "text-to-text" problem. Explain how T5 can be used once pretrained.
* To enable sequence generation, it adopts encoder-decoder

![Encoder-decoder T5 pretraining (left) and attention pattern in the encoder-decoder (right).](../img/t5-encoder-decoder.svg)
:label:`fig_t5-encoder-decoder`

* To pretrain, use span corruption objective to reconstruct masked span.
* Same as BERT, self-supervised learning. Different: on C4
* When using for downstream, fine tune. Explain how with news summarization.

![Encoder-only T5 fine tuning.](../img/t5-finetune-summarization.svg)
:label:`fig_t5-finetune-summarization`

* T5 achieves SOTA.

* Concurrently work BART encodes then decodes. However when fine-tuning, it needs additional encoder. T5 is used in Switch transformer, LaMDA, Imagen.


T5

<!--
BART :cite:`lewis2019bart`
T5 :cite:`raffel2020exploring`
Switch Transformer :cite:`fedus2022switch`
-->



## Decoder-Only 

* Fine-tuning is bad: need downstream data, computationally expensive for gradient update.
* GPT-2 demonstrates zero-shot possibility, GPT-3 shows few-shot. No gradient update.
* Let's start with GPT.


### GPT

* GPT is a transformer decoder LM. 
* Recall LM
* How to use transformer decoder for LM:

![Decoder-only GPT pretraining (left) and attention pattern in the decoder (right).](../img/gpt-decoder-only.svg)
:label:`fig_gpt-decoder-only`
* To pretrain, use LM, same as before: self-supervised learning.
* When using for downstream, fine tune. Explain how with classification.

![Decoder-only GPT fine tuning.](../img/gpt-finetune-classification.svg)
:label:`fig_gpt-finetune-classification`

* Note that GPT inspired BERT. To demonstrate more benefits of decoder-only scheme, they train on larger data, leading to GPT-2.


### GPT-2

* GPT, BERT, T5 use fine-tuning. It can be bad
* GPT-2 considers multitask learning and zero-shot
* So more data is needed
* GPT-2's architectural difference
* GPT-2's pretraining and finetuning
* GPT-2's SOTA on zero-shot LM. Explain zero-shot with summarization TL;DR, but non-LM zero-shot performance is not so good.


### GPT-3


* GPT-2 also evalutes few-shot on machine translation, poor perf.
* GPT-3 explores zero-shot, few-shot, one-shot

![Zero-shot, one-shot, few-shot learning with language models.](../img/gpt-3-xshot.svg)
:label:`fig_gpt-3-xshot`

* Kaplan's scaling law suggests bigger model with more data
* GPT-3 is bigger, trains on more data
* GPT-3's architectural difference, sparse transformer
* GPT-3's SOTA perf.
* Cool demos of GPT-3.
* summarize training compute, param size of BERT, T5, GPT. Similar to Table D.1 of GPT-3 paper.





:Pretraining BERT, T5, and GPT-3 at multiple 

|Model|Parameters| Data (training tokens)|
|:--|:-|:-|
|GPT|100M | | 
|BERT-Base|109M |250B | 
|BERT-Large|355M  |250B |
|GPT-2|1.5B | |
|T5-Base|220M  |1000B |
|T5-Large| 770M  |1000B | 
|T5-11B|11B  |1000B | 
|GPT-3|175B  |300B |
:label:`tab_bert-t5-gpt-scale`


<!--
GPT-1 :cite:`Radford.Narasimhan.Salimans.ea.2018`
GPT-2 :cite:`Radford.Wu.Child.ea.2019`
GPT-3 :cite:`brown2020language`
-->

<!--
Sparse transformer :cite:`child2019generating`
-->




## Scaling Up

### Scaling Laws

<!--
Scaling laws for neural LM :cite:`kaplan2020scaling`
Scaling laws for transfer :cite:`hernandez2021scaling`
Scale efficiently :cite:`tay2021scale`
-->


### Larger Models 


<!--
GLaM :cite:`du2021glam`
Gopher :cite:`rae2021scaling`
Megatron-Turing NLG 530B :cite:`smith2022using`

LaMDA :cite:`thoppilan2022lamda`
Chinchilla :cite:`hoffmann2022training`
Gopher :cite:`zhang2022opt`
PaLM :cite:`chowdhery2022palm`
-->



Emergent Abilities

<!--
Emergent Abilities :cite:`wei2022emergent`
-->




## Discussions

Swin Transformer

<!--
Swin Transformer :cite:`liu2021swin`
-->

MAE

<!--
MAE :cite:`he2022masked`
-->

iGPT

<!--
iGPT :cite:`chen2020generative`
-->


More Modalities

<!--
CLIP :cite:`radford2021learning`
DALL-E :cite:`ramesh2021zero`
DALL-E 2 :cite:`ramesh2022hierarchical`
Flamingo :cite:`alayrac2022flamingo`
Imagen :cite:`saharia2022photorealistic`
Generalist Agent :cite:`reed2022generalist`
-->

Scaling law by Gato: Figure 8 in Gato paper


## Exercises


[Discussions](https://discuss.d2l.ai/t/)



## References
