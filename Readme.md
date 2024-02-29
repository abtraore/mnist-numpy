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

1. Compute $\hat{y}$ by following the formula: $$\hat{y} = Wx + b$$ where $x$ is the flattened input **(mx784)**, $W$ the weights **(784xn)**, and $b$ the bias **(1xn)**. The output of this operation will have a shape of **(mxn)**.

2. The linear operation is followed by an **"activation"** function, in our case [ReLU](<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>), that can be seen as if a neuron is **"fired or not"**. ReLU equation is:

    <p align="center">
        <img src="./assets/relu.png">
    </p>

   
   
   Note that the shape of the input remains unchanged after this operation.

3. We apply the linear operation again to the output of ReLU, this time $W$ has the shape **(nx10)**, where 10 is the number of classes in MNIST.

4. The last linear operation is followed by a [Softmax](https://en.wikipedia.org/wiki/Softmax_function) function that will turn the output of the linear operation (logits) into class probabilities. Below is the formula to compute Softmax:

<p align="center">
    <img src="./assets/softmax.png">
</p>

   for $i = 1, \ldots, K$.

### Metrics Computing

After the forward propagation, the loss and accuracy are computed. The loss is a way to quantify "how much error" the Perceptron makes. To compute the error, we use the Negative Log-Likelihood (NLL):

<p align="center">
    <img src="./assets/nll.png">
</p>

The accuracy is also computed as it is an easier metric to understand:

<p align="center">
    <img src="./assets/accuracy.png">
</p>

Here, $\hat{y}_i$ and $y_i$ are binary.

### Backward Propagation

<p align="center">
    <img src="./assets/loss.png">
</p>

$Z_1 = XW_1+b_1$<br>
$A_1 = ReLU(Z_1)$<br>
$Z_2 = H_1W_2+b_2$<br>
$S = Softmax(A_2)$<br>

The goal is to compute the gradent of L w.r.t to W and b the parameters. Bellow are the steps to follow

1. Compute the gradient loss gradient w.r.t Softmax's output (S) (the simple expression of the gradient is the result of a more elaborate calculus simplication): 

<p align="center">
    <img src="./assets/dl.png">
</p>

2. Compute the loss gradient w.r.t $W_2$:

<p align="center">
    <img src="./assets/dw2.png">
</p>

3. Compute the loss gradient w.r.t $b_2$:

<p align="center">
    <img src="./assets/b2.png">
</p>

4. Compute the loss gradient w.r.t $A1$ (intermediate step to compute $\delta{W_1}$):

<p align="center">
    <img src="./assets/da1.png">
</p>

5. Compute the loss gradient w.r.t $Z_1$:

<p align="center">
    <img src="./assets/dz1.png">
</p>

6. Compute the loss gradient w.r.t $W_1$:

<p align="center">
    <img src="./assets/dw1.png">
</p>

7. Compute the loss gradient w.r.t $b_1$:

<p align="center">
    <img src="./assets/db1.png">
</p>