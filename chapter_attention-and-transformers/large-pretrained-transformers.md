# Large Pretrained Models with Transformers
:label:`sec_large-pretrained-transformers`




* Lacking inductive biases from CNN/RNN, Transformers scale well.
* There's more data and compute.
* Large pretrained models are successful, SOTA are virtually all with Transformers, such as in LM.
* Before delving into scaling law, it's important to note different transformer mode. For historical reasons, the following discussion leans slightly towards language, which also fits sequence processing for other modalities, or even multimodal applications.


Large pretrained models generalize from labeled and unlabeled data.

In practice, we use large pretrained models as a backbone for downstream tasks, rather than training pretrained models from scratch.




## Encoder-Only

* ViT is encoder-only, a sequence of image patches, a cls token is inspired by BERT.
* BERT encodes a text sequence.
* Encoder predicts mask.


`fig`: encoder-only, attention mask, masked LM prediction

* Mask from original text. Advantage: self-supervised
* Self-supervised enables large-scale training
* Large-scale training generalizes well before downstream.
* When using for downstream, fine tune. Explain how with classification example:

`fig`: bert fine tune classification

* Fine-tuning produces SOTA
* BERT has other objective like next sentence prediction, but found not useful in followups. There are other followups.



BERT

<!--
BERT :cite:`Devlin.Chang.Lee.ea.2018`
-->

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

`fig`: encoder-decoder, attention mask, reconstruct masked span

* To pretrain, use span corruption objective to reconstruct masked span.
* Same as BERT, self-supervised learning. Different: on C4
* When using for downstream, fine tune. Explain how with news summarization.

`fig`: T5 fine tune news summarization

* T5 achieves SOTA.

* Concurrently work BART encodes then decodes. However when fine-tuning, it needs additional encoder. T5 is used in Switch transformer, LaMDA, Imagen.




T5

<!--
BART :cite:`lewis2019bart`
T5 :cite:`raffel2020exploring`
Switch Transformer :cite:`fedus2022switch`
-->



## Decoder-Only 


### GPT


### GPT-2


### GPT-3




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



### Emergent Abilities

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



## Exercises


[Discussions](https://discuss.d2l.ai/t/)



## References
