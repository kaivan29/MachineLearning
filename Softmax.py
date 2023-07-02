import numpy as np
import math

"""
Interpret the scores as the un-normalized log probabilities for each class and replace the HINGE loss with a
CROSS-ENTROPY loss

Li= −log( e^f_yi  )
          --------
       ( ∑ j * e^fj )

Li = −f_yi + log ∑ j * e^fj

The cross-entropy is a measure of the difference between two probability distributions. In this case, it measures the
difference between the true distribution of classes (represented by p) and the estimated distribution of classes (
represented by q).

The Softmax classifier aims to minimize the cross-entropy between the estimated class probabilities and the true
distribution. It does this by adjusting its parameters (W) to make the predicted probabilities (q) as close as
possible to the true probabilities (p).

From a probabilistic standpoint, the Softmax classifier can be interpreted as assigning probabilities to each class
given an input image. The softmax function transforms the scores assigned to each class into probabilities. By
minimizing the negative log likelihood (which is equivalent to minimizing the cross-entropy), the Softmax classifier
aims to find the parameters (W) that maximize the likelihood of the correct class given the input.

Maximum Likelihood Estimation (MLE) is a common approach to estimate the parameters of a model by maximizing the
likelihood of the observed data given those parameters. In the case of the Softmax classifier, MLE aims to find the
parameters (W) that maximize the likelihood of the correct class given the input.

On the other hand, Maximum a posteriori (MAP) estimation goes beyond MLE by incorporating a prior belief about the
parameters. In MAP estimation, not only is the likelihood considered, but also a prior distribution over the
parameters. The prior distribution reflects any existing knowledge or assumptions about the parameters before
observing the data.

By including a regularization term in the loss function, the Softmax classifier is performing MAP estimation. The
regularization term, often represented as R(W), penalizes certain configurations of the weight matrix that do not
align with the prior belief. It helps to prevent overfitting and encourages solutions that are consistent with the
prior knowledge or assumptions about the parameters.

In summary, by incorporating the regularization term, the Softmax classifier combines the likelihood (data-driven)
and the prior (prior belief) to find the best parameters that balance the fit to the data and the prior knowledge.
This is known as MAP estimation.

Numeric Stability:

e^fyi / ∑ e^fj = C * e^fyi / C * ∑ e^fj = e^(fyi+logC) / ∑ e^(fj+logC)
Set, logC = − max fj this simply states that we should shift the values inside the vector f so that highest value is 0
"""

f = np.array([123, 456, 789])  # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f))  # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f)  # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer

w = [2, -3, -3]  # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]  # -2 +6 -3 = 1
f = 1.0 / (1 + math.exp(-dot))  # sigmoid function e/e+1

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f  # gradient on dot variable, using the sigmoid gradient derivation; e/(e+1)^2
dx = [w[0] * ddot, w[1] * ddot]  # backprop into x; 
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]  # backprop into w
# we're done! we have the gradients on the inputs to the circuit
