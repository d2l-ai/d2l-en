
# Problem Set 

"For the things we have to learn before we can do them, we learn by doing them." - Aristotle

There's nothing quite like working with a new tool to really understand it, so we have put together some exercises through this book to give you a chance to put into practice what you learned in the previous lesson(s). 

## Problems using NDarray [(Official Documentation)](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html) 


Problem 1: Initialize an ndarray of dimension 1x256 on the GPU without overwriting its memory. Then, find the index corresponding to the maximum value in the array (argmax)


```python
# Problem 1 Work Area
```

## Problems from Linear Algebra

Problem 2: Create a 4x4 matrix of random values (where values are uniformly random on the iterval [0,1]. Then create an 4x4 identity matrix (an identity of size n is the n × n square matrix with ones on the main diagonal and zeros elsewhere). Multiply the two together and verify that you get the original matrix back.


```python
# Problem 2 Work Area
```

Problem 3: Create a 3x3x20 tensor such that at every x,y coordinate, moving through the z coordinate lists the [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_number). So, at a z position of 0, the 3x3 matrix will be all 1s. At z-position 1, the 3x3 matrix will be all 1s. At z-position 2, the 3x3 matrix will be all 2s, at z-position 3, the 3x3 matrix will be all 3s and so forth.

Hint: Create the first 2 matrices by hand and then use element-wise operations in a loop to construct the rest of the tensor. 


```python
# Problem 3 Work Area
```

Problem 4: What is the sum of the vector you created? What is the mean?


```python
# Problem 4 Work Area
```

Problem 5: Create a vector [0,1], and another vector [1,0], and use mxnet to calculate the angle between them. Remember that the dot product of two vectors is equal to the cossine of the angle between the vectors, and that the arccos function is the inverse of cosine.


```python
# Problem 5 Work Area
```

## Problems from Probability

Problem 6: In the classic game of Risk, the attacker can roll a maximum of three dice, while the defender can roll a maximum of two dice. Simulate the attacking and defending dice using `sample_multinomial` to try to estimate the odds that an attacker will win against a defender when both are rolling the maximum number of dice.


```python
# Problem 6 Work Area
```

## Problems from Automatic differentiation with ``autograd`` 

Problem 7: The formula for a parabola is y=ax^2+bx+c. If a=5 and b = 13, what is the slope of y when x=0.  How about when x=7? 


```python
# Problem 7 Work Area
```

Problem 8: Graph the parabola described in Problem 6 and inspect the slope of y when x = 0 and x = 7. Does it match up with your answer from Problem 6?



```python
# Problem 8 Work Area
```

## Next
[Chapter 2: Linear regression from scratch](../chapter02_supervised-learning/linear-regression-scratch.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)


```python

```
