# Attention
:label:`sec_attention`

Graves proposed the first attention model (though it is not called "attention" in the paper) in a differentiable fashion to address the challenge of handwriting generation for a given text sequence :cite:`Graves.2013`.
Since pen trace is usually much longer than text,
Graves's attention model aligns text characters with the pen trace, however, only in one direction.

Mnih et al. proposed a non-differentiable attention model to selectively process regions or locations from images or videos :cite:`Mnih.Heess.Graves.ea.2014`.



