"""
Linear classifier:
f(x) = s = Wx + b
or
f(x) = W'b, where W' = W append b

image preprocessing:
- Center your data bt subtracting the mean from every feature
- Zero mean centering
"""

"""
Loss function: Multiclass SVM

Li = ∑ max(0,s_j−s_yi+Δ)
    j≠yi

Li = ∑ max(0, w_j.x_i - w_yi.x_i + Δ)
    j≠yi

The threshold at zero max(0,_) function is called the HINGE loss
- squared hinge loss SVM (or L2 SVM) is max(0,_)^2

The Multiclass Support Vector Machine "wants" the score of the correct class to be higher
than all other scores by at least a margin of Δ.

Regularization:
Most common is L2 Norm -> R(W) = sum(sum(W^2))
                                  k   l
L = 1/N * ∑ L_i  +  λ R(W)
    data loss          regularization loss

 It turns out that this hyperparameter can safely be set to Δ=1.0 in all cases. The hyperparameters Δ and λ seem like
 two different hyperparameters, but in fact they both control the same tradeoff between the data loss and the
 regularization loss. But λ has more effect on the magnitude of W than Δ and hence setting it to 1 is the best option.
"""
import numpy as np


def L_i(x, y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
    """
    delta = 1.0
    scores = W.dot(x)

    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes
    loss_i = 0.0

    for j in range(D):
        if j != y:
            loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0

    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)

    return loss_i


def L(X, y, W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    delta = 1.0
    scores = W.dot(X)  # 10x50000
    Y = y.reshape(1, -1)
    correct_class_scores = scores[Y]  # 50000

    margins = np.maximum(0, scores - correct_class_scores + delta)  # 10, 50000
    margins[Y] = 0

    loss = np.mean(np.sum(margins, axis=0))  # sum the loss of each example and then take the mean of all the examples
    # axis = 0 is summing along the column and axis=1 is summing along the row

    return loss
