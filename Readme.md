# MNIST NUMPY

<p align="center">
    <img src="./assets/predictions.png" alt="predictions" width="50%">
</p>

## SUMMARY

This repository contains a Perceptron implementation using NumPy. There are no hidden details in the implementation, meaning that the work focuses more on the learning process rather than on how "clean" or "abstract" the implementation is. The task is to classify handwritten digits using the Perceptron.

## ARCHITECTURE

The Perceptron has **3 layers**: the input layer with **784 neurons** (28x28), a hidden layer with **n** neurons (n is a hyperparameter), and the output layer with **10 neurons** (10 classes).
Below is a simple representation of the architecture.

<p align="center">
    <img src="./assets/architecture.png" alt="architecture" width="75%">
</p>

## TRAINING PROCESS

The learning process can be divided into **4 parts** that we repeat **e** epochs.

### Forward Propagation

The forward propagation consists of 3 steps in our case:

1. Compute $\hat{y} = Wx + b$ where $x$ is the flattened input **(mx784)**, $W$ the weights **(784xn)**, and $b$ the bias **(1xn)**. The output of this operation will have a shape of **(mxn)**.

2. The linear operation is followed by an **"activation"** function, in our case ReLU, that can be seen as if a neuron is **"fired or not"**. ReLU equation is:
   $$
   f(x) = \begin{cases} x & \text{if } x > 0,\\ 0 & \text{otherwise} \end{cases}
   $$

3. We apply the linear operation again to the output of ReLU, this time $W$ has the shape **(nx10)**, where 10 is the number of classes in MNIST.

4. The last linear operation is followed by a Softmax function that will turn the output into class probabilities. The formula for Softmax is:
   $$
   \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
   $$
   for $i = 1, \ldots, K$.

### Metrics Computing

After forward propagation, the loss and accuracy are computed. The loss is quantified using Negative Log-Likelihood (NLL):
$$
L_i = -\sum_j y_{i,j} \log(\hat{y}_{i,j})
$$
The accuracy is also computed as:
$$
\text{Accuracy} = \frac{1}{k} \sum_{i=1}^{k} (\hat{y}_i == y_i)
$$

### Backward Propagation

The backward propagation involves computing the gradient of the loss with respect to the weights and biases. The steps are:

1. Gradient of the loss with respect to Softmax's output (S):
   $$ 
   \frac{\delta L}{\delta Z_2} = S - Y
   $$
2. Gradient of the loss with respect to $W_2$:
   $$ 
   \frac{\delta L}{\delta W_{2}} = A_1^T \cdot \frac{\delta L}{\delta Z_2} 
   $$
3. Gradient of the loss with respect to $b_2$:
   $$ 
   \frac{\delta L}{\delta b_{2}} = \sum(\frac{\delta L}{\delta Z_2}, \text{axis} = 0) 
   $$
4. Gradient of the loss with respect to $A_1$:
   $$ 
   \frac{\delta L}{\delta A_1} = \frac{\delta L}{\delta Z_2} \cdot W_2^T 
   $$
5. Gradient of the loss with respect to $Z_1$:
   $$ 
   \frac{\delta L}{\delta Z_1} = \frac{\delta L}{\delta A_1} \odot \frac{\delta A_1}{\delta Z_1}
   $$
6. Gradient of the loss with respect to $W_1$:
   $$ 
   \frac{\delta L}{\delta W_1} =  X^T \cdot \frac{\delta L}{\delta Z_1}
   $$
7. Gradient of the loss with respect to $b_1$:
   $$ 
   \frac{\delta L}{\delta b_{1}} =  \sum(\frac{\delta L}{\delta Z_1}, \text{axis} = 0)
