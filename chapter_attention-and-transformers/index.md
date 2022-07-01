# Attention and Transformers
:label:`chap_attention`

The optic nerve of a primate's visual system
receives massive sensory input,
far exceeding what the brain can fully process.
Fortunately,
not all stimuli are created equal.
Focalization and concentration of consciousness
have enabled primates to direct attention
to objects of interest,
such as preys and predators,
in the complex visual environment.
The ability of paying attention to
only a small fraction of the information
has evolutionary significance,
allowing human beings
to live and succeed.

Scientists have been studying attention
in the cognitive neuroscience field
since the 19th century.
In this chapter,
we will begin by reviewing a popular framework
explaining how attention is deployed in a visual scene.
Inspired by the attention cues in this framework,
we will design models
that leverage such attention cues.
Notably, the Nadaraya-Watson kernel regression
in 1964 is a simple demonstration of machine learning with *attention mechanisms*.
Next, we will introduce attention functions
that have been extensively used in
the design of attention models in deep learning.
Specifically,
we will show how to use these functions
to design the *Bahdanau attention*,
a groundbreaking attention model in deep learning
that can align bidirectionally and is differentiable.

Equipped with
the more recent
*multi-head attention*
and *self-attention* designs,
the *transformer* architecture is solely
based on attention mechanisms.
We will go on to describe its original encoder-decoder design for machine translation.
Then we will show how its encoder can 
represent images, leading to the development of vision transformers.
When training very large models on very large datasets (e.g., 300 million images),
vision transformers outperform ResNets significantly in image classification, demonstrating superior scalability of transformers.
Thus, transformers have been extensively used in large-scale *pretraining*, which can be adapted to perform different tasks with model update (e.g., *fine tuning*) or not (e.g., *few shot*).
In the end, we will review how to pretrain transformers as encoder-only (e.g., BERT), encoder-decoder (e.g., T5), and decoder-only (e.g., GPT series).
Compelling success of large-scale pretraining with transformers in areas as diverse as
language,
vision, speech,
and reinforcement learning
suggests that better performance benefits from larger models, more training data, and more training compute.

```toc
:maxdepth: 2

attention-cues
nadaraya-watson
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```

