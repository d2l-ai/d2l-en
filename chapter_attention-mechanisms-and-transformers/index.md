# Attention Mechanisms and Transformers
:label:`chap_attention-and-transformers`



The earliest years of the deep learning boom were driven primarily
by results produced using the multilayer perceptron, 
convolutional network, and recurrent network architectures. 
Remarkably, the model architectures that underpinned 
many of deep learning's breakthroughs in the 2010s
had changed remarkably little relative to their
antecedents despite the lapse of nearly 30 years. 
While plenty of new methodological innovations 
made their way into most practitioner's toolkits---ReLU 
activations, residual layers, batch normalization, dropout, 
and adaptive learning rate schedules come to mind---the core
underlying architectures were clearly recognizable as 
scaled-up implementations of classic ideas.
Despite thousands of papers proposing alternative ideas,
models resembling classical convolutional neural networks (:numref:`chap_cnn`) 
retained *state of the art* status in computer vision
and models resembling Sepp Hochreiters original design
for the LSTM recurrent neural netowrk (:numref:`sec_lstm`),
dominated most applications in NLP. 
Arguably, to that point, the rapid emergence of deep learning
appeared to be primarily attributable to shifts 
in the available computational resources 
(due to innovations in parralel computing with GPUs)
and the availability of massive data resources
(due to cheap storage and internet services).
While these factors may indeed remain the primary drivers
behind this technology's increasing power
we are also witnessing, at long last, 
a sea change in the landscape of dominant architectures. 


At the present moment, the dominant models 
for nearly all natural language processing tasks
are based on the Transformer architecture.
Given any new task in NLP, the default first-pass approach
is to grab a large Transformer-based pretrained model,
(e.g., BERT, ELECTRA, ROBERTA, or Longformer)
adapting the output layers as necessary, 
and fine-tuning the model on the available 
data for the downstream task. 
If you've been paying attention to the last few years
of breathless news coverage centered on OpenAI's
large language models, then you've been tracking a conversation 
centered on the GPT-2 and GPT-3 Transformer-based models.
Meanwhile, the Visual Transformer has emerged 
as a default model for diverse vision tasks,
including image recognition, object detection,
semantic segmentation, and superresolution. 
Transformers are also showing up as competitive methods 
in research papers on graph neural networks,
electronic health records, and drug discovery.


The core idea behind the Transformer model is the *attention mechanism*,
an innovation that was originally envisioned as an enhancement 
for encoder-decoder RNNs applied to sequence-to-sequence applications,
like machine translations :cite:`Bahdanau.Cho.Bengio.2014`.
You might recall that in the first sequence-to-sequence models
for machine translation :cite:`Sutskever.Vinyals.Le.2014`,
the entire input was compressed by the encoder 
into a single fixed-length vector to be fed into the decoder. 
The intuition behind attention is that the rather than compressing the input,
it might be better for the decoder to revisit the input sequence at every step.
Moreover, rather than always seeing the same representation of the input,
one might imagine that the decoder should selectively focus 
on particular parts of the input sequence at particular decoding steps. 
Bahdanau's attention mechanism provided a simple means 
by which the decoder could dynamically *attend* to different 
parts of the input at each decoding step. 
The high level idea is that the encoder could produce a representation
of length equal to the original input seuqence. 
Then, at decoding time, the decoder can (via some control mechanism)
receive as input a context vector consisting of a weighted sum 
of the representations on the input at each sequence step. 
Intuitively, the weights determine the extend 
to which each step's context "focuses" on each input token,
and the key is to make this process 
for assigning the weights differentiable
so that it can be learned along with 
all of the other neural network parameters. 


Intitially, the idea was a remarkably successful 
enhancement to the recurrent neural networks 
that already dominated speech translation applications.
The models performed better than the original
encoder-decoder sequence-to-sequence architectures.
Moreover, researchers noted that some nice qualitative insights
sometimes emerged form inspecting the pattern of attention weights.
In translation tasks, attention models 
often assigned high attention weights to cross-lingual synonyms
when generating the corresponding words in the target language. 
For example, when translating the sentence "my feet hurt"
to "j'ai mal au pieds", the neural network might assign
high attention weights to the representation of "feet"
when generating the corresponding French word "pieds".
These insights spurred claims that attention modesl confer "interpretability"
although what precisely the attention weights mean---i.e.,
how, if at all, they should be *interpreted* remains a hazy research topic.

However attention mechanisms soon emerged as more significant concerns,
beyond their usefulness as an enhancements for encoder-decoder RNNs
and their putative usefulness for picking out salient inputs. 
In 2017, citet:`Vaswani.Shazeer.Parmar.ea.2017` proposed 
the Transformer architecture for machine translation, 
dispensing with recurrent connections together,
and instead relying on cleverly arranged attention mechanisms
to capture all relationships among input and output tokens. 
The architecture performed remarkably well, 
and by 2018 the Transformer began showing up
in the majority of state-of-the-art NLP systems. 
Moreover, at the same time, the dominant practice in NLP
became to pre-train large-scale models 
on enormous generic background corpora
to optimize some self-supervised pretraining objective,
and then to fine-tune these models 
using the available downstream data. 
Thew gap between Transformers and tradition architectures
grew especially wide when applied in this pretraining paradigm,
and thus the ascendance of Transformers coincided 
with the ascendence of such large-scale pretrained models,
now sometimes called "Foundation models" :cite:`bommasani2021opportunities`.


In this chapter, we introduce attention models, 
starting with the most basic intuitions 
and the simplest instantiationes of the idea.
We then work our way up to the Transformer architecture, 
the Visual Transformer, and the landscape 
of modern Transformer-based pretrained models. 


```toc
:maxdepth: 2

attention-basics
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```

