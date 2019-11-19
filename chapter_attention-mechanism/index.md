# Attention Mechanism
:label:`chap_attention`

Recurrent neural network (RNN) in :numref:`chap_rnn` has been trying to massage away the aches and pains caused by complex sequential dependencies in data. However, even though RNN is able to capture the long-range, variable-length sequential information, it suffers from inability to parallelize within a sequence.

Luckily, 30 years after the invention of RNN, a brilliant paper :cite:`Vaswani.Shazeer.Parmar.ea.2017` breathed new life into modeling sequential data. In this paper, a novel architecture called *attention* can improve on the results in machine translation task and other tasks which need to process sequential data. The attention architecture, which is the backbone of the *transformer*, achieves parallelization by capturing recurrence sequence with attention but at the same time encoding each item's position in the sequence. As a result, the transformer leads to a compatible model with significantly shorter training time.

As a bit of a historical digression, attention research is an enormous field with a long history in cognitive neuroscience. Focalization, concentration of consciousness are of the essence of attention, which enable the human to prioritize the perception in order to deal effectively with others. As a result, we do not process all the information that is available in the sensory input. At any time, we are aware of only a small fraction of the information in the environment. In cognitive neuroscience, there are several types of attentions such as selective attention, covert attention, and spatial attention. The theory ignites the spark in recent deep learning is the *feature integration theory* of the selective attention, which was developed by Anne Treisman and Garry Gelade through the paper :cite:`Treisman.Gelade.1980` in 1980. This paper declares that when perceiving a stimulus, features are registered early, automatically, and in parallel, while objects are identified separately and at a later stage in processing. The theory has been one of the most influential psychological models of human visual attention. 

However, we will not indulge in too much theory of attention in neuroscience, but rather focusing on applying the attention idea in deep learning. In this chapter, we will provide you with some intuition about how to transform the attention idea to the concrete mathematics models. By studying several essential attention models such as dot product attention, multi-head attention, and transformer, you will be master throughout the attention mechanism and be able to apply it to implement specific model such as Bidirectional Encoder Representations from Transformers (BERT).

```toc
:maxdepth: 2

attention
seq2seq-attention
transformer
```

