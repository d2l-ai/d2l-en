# Intermediate Linear Algebra
:label:`appendix_linear_algebra`

In :numref:`chapter_linear_algebra`, we saw the fundamentals of linear algebra and how to use it to store and transform data.  Indeed, it is safe to say that linear algebra is one of the key mathematical pillars of modern deep learning.  However the theory goes much deeper than we have seen, and in this section we will dive into the some of the key geometric notions associated with linear algebra, and see another interpretation of many of the operations we have met before.

## Geometry of Vectors
Before diving in to the geometric interpretation of the many operations from :numref:`chapter_linear_algebra`, we first need to discuss the two common geometric interpretations of vectors, as either points or directions in space. Fundamentally, a vector is a list of numbers like

```{.python .input}
[1,7,0,1]
```

Mathematicians most often write this as either a *column* or *row* vector, which is to say either as
$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix} \text{ or } \mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

These often have different interpretations where data points are column vectors and weights used to form weighted sums are row vectors.  However, as we will see in this text, it can sometimes pay to be flexable.  As a general rule of thumb, we have adopted the convention of writing single data points and theoretical investigations in column vectors, however we switch to row vectors representing data points when working with actual data as this is the format of the majority of data we encounter.  This dichotomy in notation is ubiquitous in machine learning literature, so we match it while recognizing that it can be a bit combersome to the beginner.

Given a vector, the first interpretation that we should give it is as a point in space.  In two or three dimensions, we can visualize these points by using the components of the vectors to define the location of the points in space compared to a fixed reference called the *origin*.

![An illustration of visualizing vectors as points in the plane.  The first component of the vector gives the $x$-coordinate, the second component gives the $y$-coordinate.  Higher dimensions are analogous, although much harder to visualize.](../img/GridPoints.svg)

This way of visualizing the data can be freeing, and allows us to consider the problem on a more abstract level.  No longer faced with some insurmountable seeming problem like classifying pictures as either cats or dogs, we can start considering tasks abstractly as collections of points in space and picturing the task as discovering how to separate two distinct clusters of points.


In parallel, there is a second point of view that people often take of vectors: as directions in space.  Equally as well as thinking of the vector $\mathbf{v} = (2,3)^\top$ as the location $2$ units to the right and $3$ units up from the origin, we can think of it as the direction itself to take $2$ steps to the right and $3$ steps up.  In this way, we consider all the vectors below should the same.

![Any vector can be visualized as an arrow in the plane.  In this case, every vector drawn is a representation of the vector $(2,3)$.](../img/ParVec.svg)

One of the big benefits of this shift is that we can make visual sense of the act of vector addition: we follow the directions given by one vector, and then follow the directions given by the other.

![We can visualize vector addition by fiirst following one vector, and then another.](../img/VecAdd.svg)

Vector subtraction has a similar interpretation.  By considering the identity that $$\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v}),$$ we see that the vector $\mathbf{u}-\mathbf{v}$ is the directions that takes us from the point $\mathbf{u}$ to the point $\mathbf{v}$.


## Dot Products and Angles
As we saw in :numref:`chapter_linear_algebra`, if we take two column vectors say $\mathbf{u}$ and $\mathbf{v}$ we can form the dot product by computing:
$$
\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.
$$

While discussion geometric interpretations of vectors, it is best to try and view this relationship more symmetrically, so we'll use the notation 
$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$
to highlight the fact that we can interchange the order of the vectors and get the same answer, as is the case in usual multiplication.

We already have one interpretation of vector dot products in terms of data transformation.  However, it turns out there is a nice geometric story we can tell.  In particular, let us see the relationship between the dot product and the angle between two vectors.

![Between any two vectors in the plane there is a well defined angle $\theta$.  We will see this angle is intimately tied to the dot product.](../img/VecAngle.svg)

To start with, let us consider two specific vectors:
$$
\mathbf{v} = (r,0) \text{ and } \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$
The vector $\mathbf{v}$ is length $r$ and runs parallel to the $x$-axis, and the vector $\mathbf{w}$ is of length $s$ and at angle $\theta$ with the $x$-axis.  If we compute the dot product of these two, we see that
$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$
Rearranging a bit, this becomes
$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$
In other words, for these two specific vectors, the dot product tells us the angle between the two vectors.  This same fact is true in general.  Indeed we have the following: for any two vectors $\mathbf{v}$ and $\mathbf{w}$, the angle between the two vectors is

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

This is a nice result since nothing in the computation references two-dimensions.  Indeed, we can use this in three or even three million dimensions without issue.

As a simple example, let's see how to compute the angle between a pair of vectors:

```{.python .input}
import numpy as np

def angle(v, w) :
    return np.arccos(v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w)))

print(angle(np.array([0,1,2]),np.array([2,3,4])))
```

It is reasonable to ask: why is this useful?  And the answer comes in the kind of invariances we expect data to have.  Consider an image, and an image where every pixel value is $10\%$ the brightness in the original image.  The values of the individual pixels are in general far from the original values.  Thus, if one computed the distance between the original image and the darker one, the distance can be enormous.  However, for most ML applications, the *content* is the same (still an image of a cat for instance in a cat/dog classifier).  If we now consider the angle however, it is not hard to see that for any vector $\mathbf{v}$, the angle between $\mathbf{v}$ and $0.1\cdot\mathbf{v}$ is zero.  This corresponds to the fact that scaling vectors keeps the same direction, just changing the length.  The angle will consider the darker image identical.  

Examples like this are everywhere.  In text, we might want the topic being discussed to not change if we write twice as long of document that says the same thing.  For some encoding (such as counting the number of occurrences of words in some vocabulary), this corresponds to a doubling of the vector encoding the document, so again we can use the angle.

### Cosine Similarity
Indeed the use of this as a measure of closeness, people have adopted the term *cosine similarity* to refer to the portion
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$
This takes a maximum value of $1$ when things are perfectly similar, a minimum value of $-1$ when they are perfectly dissimilar, and generally a value near zero when we consider the two unrelated objects.

## Hyperplanes

One of the key concepts we should understand is the notion of a *hyperplane*.  This is the high dimensional analogue of a line in two dimensions or a plan in three dimensions.

Let us start with an example.  Suppose we have a column vector $\mathbf{w}=(2,1)^\top$.  We want to know, "what are the points $\mathbf{v}$ with $\mathbf{w}\cdot\mathbf{v} = 1$?"  By recalling the connection between dot products and angles above, we can see that this is equivalent to 
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}
$$

![Recalling trigonometry, we see the formula $\|\mathbf{v}\|\cos(\theta)$ is the length of the projection of the vector $\mathbf{v}$ onto the direction of $\mathbf{w}$](../img/ProjVec.svg)

If we consider the geometric meaning of this expression (recalling our days in trigonometry) we see that this is equivalent to saying that the length of the projection of $\mathbf{v}$ onto the direction of $\mathbf{w}$ is exactly $1/\|\mathbf{w}\|$.  The set of all points where this is true is a line at right angles to the vector $\mathbf{w}$.  If we wanted, we could find the equation for this line and see that it is $2x+y = 1$ or equivalently $y=1-2x$.

If we now look at what happens when we ask about the set of points with
$$
\mathbf{w}\cdot\mathbf{v} > 1,
$$
or
$$
\mathbf{w}\cdot\mathbf{v} < 1,
$$
we can see that these cases correspond to those where the projections are longer or shorter than $1/\|\mathbf{w}\|$ respectively, thus those two inequalities define either side of the line.  In this way, we have found a way to cut our space into two halves, where all the points on one side have dot product below a threshold, and the other side above.

![If we now consider the inequality version of the expression, we see that our hyperplane (in this case: just a line) separates the space into two halves.](../img/SpaceDivision.svg)

The story in higer dimension is much the same.  If we now take $\mathbf{w} = (1,2,3)^\top$, and ask about the points in three dimensions with $\mathbf{w}\cdot\mathbf{v} = 1$, we obtain a plane at right angles to the given vector $\mathbf{w}$.  The two inequalities again define the two sides of the plane.

![Hyperplanes in any dimension separate the space into two halves.](../img/SpaceDivision3D.svg)

While our ability to picture runs out at this point, nothing stops us from doing this in tens, hundreds, or billions of dimensions.

This occurs often when thinking about machine learned models.  For instance, we can understand linear classificaiton models like those from :numref:`chapter_softmax` as methods to find hyperplanes that separate the different target classes.  In this context, such hyperplanes are often referred to as *decision planes*.  Indeed, the majority of deep learned classificaiton models end with a linear layer fed into a softmax, so one can imagine the role of the deep neural network as finding a non-linear embedding so that the target classes can be separated cleanly by hyperplanes.

To give a hand-built example, notice that we can produce a reasonable model to classify hand drawn zeros from hand drawn ones, by just taking the vector between their means to define the decision plane.

```{.python .input} 
### Load in the dataset and split it ###
 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
 
X, y = load_digits(n_class = 2, return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 8675309, test_size = 0.5)

### Compute and visualize the means
    
ave_0 = np.mean(X_train[y_train == 0],axis=0)
ave_1 = np.mean(X_train[y_train == 1],axis=0)
 
plt.imshow(ave_0.reshape(8,8),cmap='Greys')
plt.show()
plt.imshow(ave_1.reshape(8,8),cmap='Greys')
plt.show()

### Compute the weight and use as a decision plane ###
w = (ave_1 - ave_0).T
predictions = 1*(X_test.dot(w) > 0) 
 
print("Accuracy: {}".format(np.mean(predictions==y_test)))
```

## Geometry of linear transformations
Through :numref:`chapter_linear_algebra` and the above discussions, we have a solid understanding of the geometry of vectors, lengths, and angles, however there is one important object we have omitted discussing, and that is a geometric understanding of linear transformations represented by matrices.  Fully internalizing what matrices can do to transform data between two potentially different high dimensional spaces takes significant practice, and is beyond the scope of this appendix.  However, we can work to build a strong intuition in two dimensions.

Suppose we have some matrix, say
$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$
If we want to apply this to an arbitrary vector $\mathbf{v} = (x,y)$, we multiply and see that
$$
\begin{align*}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{align*}
$$
This may seem like an odd computation where something clear became somewhat impenetrable.  However, what it tells us is that we can write the way that a $2\times 2$ matrix transforms *any* vector in terms of how $\mathbf{A}$ transforms *two specific vectors*: $(1,0)^\top$ and $(0,1)^\top$.  This is worth considering for a moment, since we have essentially reduced an infinite problem (what happens to any pair of real numbers) to a finite one (what happens to these specific vectors).  These vectors are an example a *basis*, where we can write any vector in our space as a weighted sum of these *basis vectors*.

Let us draw what happens when we use the specific matrix
$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

If we look at the specific vector $\mathbf{v} = (2,-1)^\top$, we see this is $2\cdot(1,0)^\top + -1\cdot(0,1)^\top$, and thus we know that the matrix $A$ will send this to $2(\mathbf{A}(1,0)^\top) + -1(\mathbf{A}(0,1))^\top = 2(1,-1)^\top - (2,3)^\top = (0,-5)^\top$.  If we follow this logic through carefully, say by considering the grid of all integer pairs of points, we see that what happens is that the matrix multiplication distorts and rearranges the original coordinates of our space while keeping the overall structure of the grid.

![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.  This is a result of our computation above.](../img/GridTransform.svg)

This is the most important intuitive point to internalize about linear transformations represented by matrices.  They are incapable of distorting some parts of space differently than others.  All they can do is take the original coordinates on our space and distort them.

Some distortions can be severe.  For instance the matrix
$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$
Compresses the entire two-dimensional plane down to a single line.  Identifying and working with such transformations are the topic of a later section, but geometrically we can see that this is fundamentally different from the types of transformations we worked with before.  For instance, the result from matrix $\mathbf{A}$ can be "bent back" to the original grid.  The results from matrix $\mathbf{B}$ cannot since we will never know where the vector $(1,2)^\top$ came from---was it $(1,1)^\top$ or $(0,-1)^\top$?

While this picture was for a $2\times2$ matrix, nothing prevents us from taking the lessons learned into higher dimensions.  If we take similar basis vectors like $(1,0,\ldots,0)$ and see where our matrix sends them, we can start to get a feeling for how the matrix multiplication distorts the entire space.

## Linear Dependence
Consider again the matrix
$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$
This compresses the entire plane down to live on the single line $y = 2x$.  The question now arrises: is there some way we can detect this just looking at the matrix itself?  The answer is that indeed we can.  Lets take $\mathbf{b}_1 = (2,4)^\top$ and $\mathbf{b}_2 = (-1,-2)^\top$ be the two columns of $\mathbf{B}$.  Remember that we can write everything tranformed by the matrix $\mathbf{B}$ as a weighted sum of the columns of the matrix: like $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.  We call this a *linear combination*.  The fact that $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ means that we can write any linear combination of those two columns entirely in terms of say $\mathbf{b}_2$ since
$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$
This means that one of the columns is in a sense redundant in that it does not define a unique direction in space.  This should not suprise us too much since we already saw in the last section that this matrix collapses the entire plane down into a single line.  We can capture this collapse with our linear dependence: $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$.  To make this more symmetrical between the two vectors, we will write this as
$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

In general, we will say that a collection of vectors $\mathbf{v}_1, \ldots \mathbf{v}_k$ are *linearly dependent* if there exist coefficients $a_1, \ldots, a_k$ *not all equal to zero* so that
$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$
In this case, we could solve for one of the vectors in terms of some combination of the others, and effectively render it redundant.  Thus a linear dependence in the columns of a matrix is a witness to the fact that our matrix is compressing the space down to something lower dimension than it potentially could be.  If there is no linear dependence, then there is no compression into a lower dimensional space, and we say the vectors are *linearly independent*. 

## Rank

If we have a general $n\times m$ matrix, it is reasonable to ask what dimension space the matrix maps into.  A concept known as the *rank* will be our answer.  In the previous section, we noted that a linear dependence bears witness to compression of space into a lower dimension than it might originally appear, and so we will be able to use this to define the notion of rank.  In particular, the rank of a matrix $\mathbf{A}$ is the largest number of linearly independent columns that we can find.  For example, the matrix
$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$
has $\mathrm{rank}(B)=1$, since the two columns are linearly dependent, but either column by itself is not linearly dependent.  For a more challenging example, we can consider
$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$
and show that $\mathbf{C}$ has rank two since, for instance, the first two columns are linearly independent, however any of the four collections of three columns are dependent.  

This procedure, as described, is very inefficient.  It requires looking at every subset of the columns of our given matrix, and thus is potentially exponential in the number of columns.  Later we will see a more computationally efficient way to compute the rank of a matrix, but for now, this is sufficient to see that the concept is well defined and understand what it intuitively means.

## Invertibility
We have seen intuitively above that matrices which compress down to lower dimensions cannot be undone.  However, if the matrix has full rank (that is if $\mathbf{A}$ is an $n \times n$ matrix with rank $n$), we should always be able to undo it.  Consider the matrix
$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1 
\end{bmatrix}.
$$
which is the matrix with ones along the diagonal, and zeros elsewhere.  We call this the *identity* matrix.  It is the matrix which leaves our data alone when applied.  To find a matrix which undoes what our matrix $\mathbf{A}$ has done, we want to find a matrix $\mathbf{A}^{-1}$ such that
$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}.
$$

If we look at this as a system, we have $n \times n$ unknowns (the entries of $\mathbf{A}^{-1}$) and $n \times n$ equations (the equality that needs to hold between every entry of the product $\mathbf{A}^{-1}\mathbf{A}$ and every entry of $\mathbf{I}$) so we should generically expect a solution to exist.  Indeed, in the next section we will see a quantity called the *determinant* which has the property that as long as the determinant is not zero, we can find a solution.  We call this the *inverse* matrix.  As an example, if $\mathbf{A}$ is the general $2 \times 2$ matrix 
$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d 
\end{bmatrix},
$$
then we can see that the inverse is
$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a 
\end{bmatrix}.
$$

We can test to see this matches by examining the inversion method in numpy:
```{.python .input}
M = np.array([[1,2],[1,4]])
print(np.linalg.inv(M))
```

### Numerical Issues
While the inverse of a matrix is useful in theory, we must say that most of the time we do not wish to *use* the matrix inverse to solve a problem in practice.  In general, there are far more numericaly stable algorithms for solving linear equations like
$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$
than computing the inverse and multiplying to get
$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$
Indeed, just by a small number division can lead to numerical instability, so can inversion of a matrix which is nearly 

In addition, it is common that the matrix $\mathbf{A}$ is *sparse*, which is to say that it contains only a small number of non-zero values.  If we were to explore examples, we would see that this does not mean the inverse is sparse, so even if $\mathbf{A}$ was a $1$ million by $1$ million matrix with only $5$ milion non-zero entries (and thus we need only store those $5$ million), the inverse will generically have almost every entry non-negative meaning we need to store all $1$ million squared entries---$1$ trillion entries!

A deep dive into the numerical issues encountered when working with linear algebra is beyond the scope of our text, however it is worth knowing that we should be caution, and generally avoid inversion if we can.

## Determinant
Our geometric deep-dive gives the perfect opportunity to get an intuitive understanding of a quantity known as the *determinant*.  Consider the grid image from before.

![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/GridTransformFilled.svg)

Look at the highlighted square.  This is a square with edges given by $(0,1)$ and $(1,0)$ and thus it has area one.  After $\mathbf{A}$ transforms this square, we see that it becomes a parallelogram.  There is no reason this parallelogram should have the same area that we started with, and indeed in the specific case shown here of
$$
\mathbf{A} = \begin{bmatrix}
1 & -1 \\
2 & 3
\end{bmatrix},
$$
it is an exercise in coordinate geometry to compute the area of this parallelogram with edges $(1,-1)$ and $(2,3)$, and what we obtain is that the area is $5$.

In general, if we have a matrix
$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$
we can see with some computation that the area of the resulting parallelogram is $ad-bc$.  This area is referred to as the *determinant*.

Let's check this quickly with some example code.
```
import numpy as np
print(np.linalg.det(np.array[[1,-1],[2,3]]))
```

The eagle-eyed amongst us will notice that this expression can be zero or even negative.  For the negative term, this is a matter of convention taken generally in mathematics: if the matrix flips the figure, we say the area is negated.  Let us see now that when the determinant is zero, we learn more.

Let us consider
$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$
If we compute the determinant of this matrix we get $2\cdot(-2 ) - 4\cdot(-1) = 0$.  Given our understanding above, I claim this makes perfect sense.  $\mathbf{B}$ compresses the square from the original image down to a line segment, which has zero area.  And indeed, being compressed into a lower dimensional space is the only way to have zero area after the transformation.  Thus we see the following result is true: a matrix $A$ is invertible if and only if the determinant is not equal to zero.

As a final comment, imagine that we have any figure drawn on the plane.  Thinking like computer scientists, we can decompose that figure into a collection of little squares so that the area of the figure is in essence just the number of squares in the decomposition.  If we now transform that figure by a matrix, we send each of these squares to parallelograms, each one of which has area given by the determinant.  We see that for any figure, the determinant gives the (signed) number that a matrix scales the area of any figure.

We will return to understand more on how to compute determinants in a later section for larger matrices, but the basic intuition is the same.  The determinant is the factor that $n\times n$ matricies scale $n$-dimensional volumes.

## Eigendecompositions
Eigenvalues are often one of the most useful notions we will encounter when studying linear algebra, however in the beginning it is easy to overlook their importance.  Let us explore what these objects are!

Suppose we have a matrix $A$ where
$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$
If we apply $A$ to any vector $\mathbf{v} = (x,y)$, we obtain a vector $\mathbf{v}A = (2x,-y)$.  This has an intuitive interpretation: stretch the vector to be twice as wide in the $x$-direction, and then flip it in the $y$-direction.  The matrxi sends almost every vector to one completely geometrically unrelated.  

However, there are *some* vectors for which something remains unchanged.  Namely $(1,0)$ gets sent to $(2,0)$ and $(0,1$ get sent to $(0,-1)$.  These vectors are still in the same line, and the only modification is that the matrix stretches them by a factor of $2$ and $-1$ respectively.  We call such vectors *eigenvectors* and the factor they are stretched by *eigenvalues*.

In general, if we can find a number $\lambda$ and a vector $\mathbf{v}$ such that 
$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$
We say that $\mathbf{v}$ is an eigenvector for $A$ and $\lambda$ is an eigenvalue.

### Finding Eigenvalues
Let us figure out how to find them.  By subtracting off the $\lambda \vec v$ from both sides, and then factoring out the vector, we see the above is equivalent to:
$$
(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.
$$
For this to happen, we see that $(\mathbf{A} - \lambda \mathbf{I})$ must compress some direction down to zero, hence it is not invertible, and thus the determinant is zero. Thus, we can find the *eigenvalues* by finding for what $\lambda$ is $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$. Once we find the eigenvalues, we can solve $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ to find the associated *eigenvector(s)*.

### An Example
Let us see this with a more challenging matrix
$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$
If we consider $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$, we see this is equivalent to the polynomial equation $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$. Thus, two eigenvalues are $4$ and $1$.  To find the associated vectors, we then need to solve
$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \text{ and }
\begin{bmatrix}
2 & 2\\
1 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$
We can solve this with the vectors $(1,-1)^\top$ and $(1,2)^\top$ respectively.

We can check this in code using the built-in numpy `numpy.linalg.eig` routine.
```{.python .input}
import numpy as np

w, v = np.linalg.eig(np.array([[2,1],[2,3]]))
print(w)
print(v)
```
*Note*: this normalizes the eigenvectors to be of length one, wheras we took ours to be of arbitrary length.  Additionally, the choice of sign is arbitrary.

### Decomposing Matrices
Let us continue the previous example one step further.  Let
$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$
be the matrix where the columns are the eigenvectors of the matrix $\mathbf{A}$.  Let
$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$
be the matrix with the associated eigenvalues on the diagonal.  Then the definition of eigenvalues and eigenvectors tells us that
$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

The matrix $W$ is invertible, so we may multiply both sides by $W^{-1}$ on the right, we see that we may write
$$
\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.
$$

In the next section we will see some nice consequences of this, but for now we need only know that such a decomposition will exist as long as we can find a full collection of linearly independent eigenvectors (so that $W$ is invertible).
### Operations on Eigendecompositions
One nice thing about eigendecompositions is that we can write many operations we usually encounter cleanly in terms of the eigendecomposition.  As a first example, consider:
$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

This tells us that for any positive power of a matrix, the eigendecomposition is obtained by just raising the eigenvalues to the same power.  The same can be shown for negative powers, so if we want to invert a matrix we need only consider
$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$
or in other words, just invert each eigenvalue.  This will work as long as each eigenvalue is non-zero, so we see that invertible is the same as having no zero eigenvalues.  

Indeed, additional work can show that if $\lambda_1, \ldots, \lambda_n$ are the eigenvalues of a matrix, then the determinant of that matrix is
$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$
or the product of all the eigenvalues.  This makes good intuitive sense because whatever stretching $\mathbf{W}$ does, $W^{-1}$ undoes it, so in net the only stretching that happens is by multiplication by the diagonal matrix $\boldsymbol{\Sigma}$ which stretched volumes by the product of the diagonal elements.

The examples could continue, but I think the point is clear: eigendecompositions can simplify many linear algebraic computations and form a core component of a deep understanding of how matrices act. 
### Eigendecompositions of Symmetric Matrices
It is not always possible to find enough linearly independent eigenvectors for the above process to work.  For instance the matrix
$$\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$
has only a single eigenvector, namely $(0,1)$.  To handle such matrices requires a more advanced techniques than we can cover (such as the Jordan Normal Form, or Singular Value Decomposition).  We will often need to restrict our attention to those matrices where we can guarantee the existence of a full set of eigenvectors.

The most commonly encountered family are the *symmetric matrices* which are those matrices where $\mathbf{A} = \mathbf{A}^\top$.  In this case, we may take $W$ to be an *orthogonal matrix* (a matrix whose columns are all length one vectors which are at right angles to one-another where $\mathbf{W}^\top = \mathbf{W}^{-1}$) and all the eigenvalues will be real.  Thus, in this special case we can write the decomposition as
$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

### Gershgorin Circle Theorem
Eigenvalues are often difficult to reason with intuitively.  If presented an arbitrary matrix, it is there is little that can be said about what the eigenvalues are without computing them.  There is, however, one theorem that can make it easy to approximate if the largest values are on the diagonal.

Let $\mathbf{A} = (a_{ij})$ be any square matrix ($n\times n$).  We will define $r_i = \sum_{j \neq i} |a_{ij}|$.  Define $\mathcal{D}_i$ as the disk in the complex plane with center $a_{ii}$ radius $r_i$.  Then, every eigenvalue of $\mathbf{A}$ is contained in one of the $\mathcal{D}_i$.

This can be a bit to unpack, so let us look at an example.  Consider the matrix:
$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

We have $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ and $r_4 = 0.9$.  The matrix is symmetric, so all eigenvalues are real.  This means that all of our eigenvalues will be in one of the ranges of $[0.7,1.3]$, $[2.4,3.6]$, $[4.2,5.8]$, and $[8.1,9.9]$.  Indeed performing the numerical computation shows that the eigenvalues are approximately $0.99, 2.97, 4.95, 9.08$, all comfortably inside the ranges provided.  

```{.python .input}
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
print(v)
```

In this way, eigenvalues can be approximated, and the approximations will be pretty strong as long in the case that the diagonal is significantly larger than all the other elements.  

It is a small thing, but with a complex and subtle topic like eigendecomposition, it is good to get any intuitive grasp we can!

### A useful application: The growth of iterated maps

Now that we understand what eigenvectors are in principle, let us see how they can be used to provide a deep understanding of a problem central to neural network behavior: proper weight initialization. 

### Eigenvectors as Long Term Behavior

The full mathematical investigation of the initialization of deep neural networks is beyond the scope of the text, but we can see a toy version here to understand how eigenvalues can help us see how these models work.  As we know, neural networks operate by interspersing layers of linear transformations with non-linear operations.  For simplicity here, we will assume that there is no non-linearity, and that the transformation is a single repeated matrix operation $A$, so that the output of our model is
$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

When these models are initialized, $A$ is taken to be a random matrix with Gaussian entries, so lets make one of those.  We will start mean zero, variance one Gaussians.  Just to be concrete, lets make it a five by five matrix.

```{.python .input}
import numpy as np
np.random.seed(8675309)

k = 5
A = np.random.randn(k,k)
A
```

### Behavior on random data
For simplicity in our toy model, we will assume that the data vector we feed in $\mathbf{v}_{in}$ is a random five dimensional Gaussian vector.

Let us think about what we want to have happen.  For context, lets think of a generic ML problem where we are trying to turn input data, like an image, into a prediction, like the probability the image is a picture of a cat.  If repeated application of $\mathbf{A}$ stretches a random vector out to be very long, then small changes in input will be amplified into large changes in output---tiny modifications of the input image would lead to vastly different predictions.  This does not seem right!

On the flip side, if $\mathbf{A}$ shrinks random vectors to be shorter, then after running through many layers, the vector will essentially shrink to nothing, and the output will not depend on the input. This is also clearly not right either!

We need to walk the narrow line between growth and decay to make sure that our output changes depending on our input---but not much!

Let us see what happens when we repeatedly multiply our matrix $\mathbf{A}$ against a random input vector, and keep track of the norm.

```{.python .input}
# Calculate the sequence of norms after repeatedly applying A
v_in = np.random.randn(k,1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1,100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))
    
print("Sequence of norms: {}".format(norm_list))

```

The norm is growing uncontrollably!  Indeed if we take the list of quotients, we will see a pattern.

```{.python .input}
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1,100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])
    
print("Sequence of ratios: {}".format(norm_ratio_list))
```

If we look at the last portion of the above computation, we see that the random vector is stretched by a factor of ```1.974459321485[...]``` where the portion at the end shifts a little, but the stretching factor is stable.  

### Relating back to eigenvectors

We have seen that there are these eigenvectors and eigenvalues which correspond to the amount something is stretched, but that was for specific vectors, and specific stretches.  Let us take a look at what they are for $\mathbf{A}$.  A bit of a caveat here: it turns out that to see them all, we will need to go to complex numbers.  You can think of these as stretches and rotations.  By taking the norm of the complex number (square root of the sums of squares of real and imaginary parts) we can measure that stretching factor. Let us also sort them.

```{.python .input}
# Compute the Eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print("Norms of eigenvalues: {}".format(norm_eigs))
```

### An observation

We see something a bit unexpected happening here: that number we identified before for the long term stretching of our matrix $\mathbf{A}$ applied to a random vector is *exactly* (accurate to thirteen decimal places!) the largest eigenvalue of $\mathbf{A}$.  This is clearly not a coincidence!

But, if we now think about what is happening geometrically, this starts to make sense.  Consider a random vector.  This random vector points a little in every direction, so in particular, it points at least a little bit in the same direction as the eigenvector of $\mathbf{A}$ associated with the largest eigenvalue.  This is so important that it is called the *principle eigenvalue* and *principle eigenvector*.  After applying $\mathbf{A}$, our random vector gets stretched in every possible direction, as is associated with every possible eigenvector, but it is stretched most of all in the direction associated with this principle eigenvector.  What this means is that after apply in $A$, our random vector is longer, and points in a direction closer to being aligned with the principle eigenvector.  After applying the matrix many times, the alignment with the principle eigenvector becomes closer and closer until, for all practical purposes, our random vector has been transformed into the principle eigenvector!  Indeed this algorithm is the basis for what is known as the [power iteration method](https://en.wikipedia.org/wiki/Power_iteration) for finding the largest eigenvalue and eigenvector of a matrix.

### Fixing the normalization
Now, from above discussions, we concluded that we do not want a random vector to be stretched or squished at all, we would like random vectors to stay about the same size through the entire process.  To do so, we now rescale our matrix by this principle eigenvalue so that the largest eigenvalue is instead now just one.  Let us see what happens in this case.

```{.python .input}
# Rescale the matrix A
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k,1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1,100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))
    
print("Sequence of norms: {}".format(norm_list))

# Also the ratio
norm_ratio_list = []
for i in range(1,100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])
    
print("Sequence of ratios: {}".format(norm_ratio_list))
```

### Conclusions

We now see exactly what we hoped for!  After normalizing the matrices by the principle eigenvalue, we see that the random data does not explode as before, but rather eventually equilibrates to a specific value.  It would be nice to be able to do these things from first principles, and it turns out that if we look deeply at the mathematics of it, we can see that the largest eigenvalue of a large random matrix with independent mean zero, variance one Gaussians is on average about $\sqrt{n}$, or in our case $\sqrt{5} \approx 2.2$, due to a fascinating fact known as the [circular law](https://en.wikipedia.org/wiki/Circular_law).  This is why almost all initialization strategies are of the format of [Gaussians divided by the square root of the number of inputs](https://arxiv.org/abs/1704.08863).  In this way, eigenvalues form a key component of understanding the behavior of many ML systems.

## Tensors and Common Linear Algebra Operations
In :numref:`chapter_linear_algebra` the concept of tensors was introduced.  In this section we will dive more deeply into tensor contractions (the tensor equivalent of matrix multiplication), and see how it can provide a unified view on a number of matrix and vector operations.  

With matrices and vectors we knew how to multiply them to transform data.  We need to have a similar definition for tensors if they are to be useful to us.  Think about matrix multiplication:
$$
\mathbf{C} = \mathbf{A}\mathbf{B} \;\text{ or equivalently }\; c_{i,j} = \sum_{k} a_{i,k}b_{k,j}.
$$
This pattern is one we can repeat for tensors.  For tensors, there is no one case of what to sum over that can be universally chosen, so we need specify exactly which indices we want to sum over.  For instance we could consider
$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$
Such a transformation is called a *tensor contraction*.  It can represent a far more flexible family of transformations that matrix multiplication alone. 

As a often-used notational simplification, we can notice that the sum is over exactly those indices that occur more than once in the expression, thus people often work with *Einstein notation* where the summation is implicitly taken over all repeated indices.  This gives the compact expression:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

## Common Examples from Linear Algebra
Let us see how many of the linear algebraic definitions we have seen before can be expressed in this compressed tensor notation:
* $\\mathbf{v} \cdot \mathbf{w} = v_iw_i$ 
* $\|\mathbf{v}\|_2^{2} = v_iv_i$
* $\mathbf{A}\mathbf{v} = a_{ij}v_j$
* $\mathbf{A}\mathbf{B} = a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = a_{ii}$
In this way, we can replace a myriad of specialized notations with short tensor expressions.

## Expressing in numpy
Tensors may flexibly be operated on in code as well.  As seen in :numref:`chapter_linear_algebra`, we can create tensors using numpy arrays.

```{.python .input}
import numpy as np

# Define Tensors
B = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
A = np.array([[1,2],[3,4]])
v = np.array([1,2])

# Print out the shapes
print(A.shape)
print(B.shape)
print(v.shape)
```

Einstein summation has been implemented directly in numpy via ```np.einsum```.  The indices that occurs in the Einstein summation can be passed as a string, followed by the tensors that are being acted upon.  For instance, to implement matrix multiplication, we can consider the Einstein summation seen above ($\mathbf{A}\mathbf{v} = a_{ij}v_j$) and strip out the indices themselves to get the implementation in numpy:

```{.python .input}
# Reimplement matrix multiplication
print(np.einsum("ij,j -> i",A,v))
print(A.dot(v))
```

This is a highly flexible notation.  For instance if we want to compute what would be traditionally written as
$$
c_{kl} = \sum_{ij} \mathbf{B}_{ijk}\mathbf{A}_{il}v_j.
$$
it can be implemented via Einstein summation as:

```{.python .input}
print(np.einsum("ijk,il,j -> kl",B,A,v))
```

This notation is readable and efficient for humans, however bulky if for whatever reason we need generate a tensor contraction programatically.  For this reason, numpy provides an alternative notation by providing integer indices for each tensor.  For example, the same tensor contraction can also be written as:

```{.python .input}
print(np.einsum(B,[0,1,2],A,[0,3],v,[1],[2,3]))
```

Either notation allows for concise and efficient representation of tensor contractions in code.
