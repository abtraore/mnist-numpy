# MNINST NUMPY

<p align="center" style="width:100%">
    <img src="./assets/predictions.png" alt="predictions" style="width:50%">
</p>

## SUMMARY

This repo contains a perceptron implementation using numpy. There is no hidden details implementation meaning that the work focus more on the learning proccess rather than "how clean" or "abstract" I can implement it. The task is to classify handwriting digit using the perceptron.

## ARCHITECTURE

The perceptron has <B>3 layers</B>, the input layers with <B>784 neurons</B> (28x28), a hidden layer with <B>n</B> neurons (n is an hyper-parameter) and the output layer with <B>10 neurons</B> (10 classes).
Below is a simple representation of the architecture.

<p align="center" style="width:100%">
<img src="./assets/architecture.png" alt="architecture" style="width:75%">
</p>

## Training proccess

The learning proccess can be divided into <B>4 parts</B> that we repeat <B>"e"</B> epochs time.

### Forward propagation

The forward propagation consist for 3 steps in our case:

1. Compute $\hat{y}$ by following the formula: $$\hat{y}=Wx+b$$ where x is the flattened input <B>(mx784)</B>, W the weights <B>(784xn)</B> and b the bias <B>(1xn)</B>. The output of this operation will have a shape of <B>(mxn)</B>.

2. The linear operation is followed by an <B>"activation"</B> function in our case [ReLU](<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>) that can be seen as if a neuron is <B>"fired or not"</B>. ReLU equation is:

   $$
   f(x) =
   \begin{cases}
       x \text{ if } x > 0,\\
       0 \text{ otherwhise}
   \end{cases}
   $$

   Note that the shape of the input remain unchanged after this operation.

3. We apply again the linear operation to the output of ReLU this time W has the shape (n, 10). 10 is the number of classes in MNIST.

4. The last linear operation is followed by a [Softmax](https://en.wikipedia.org/wiki/Softmax_function) that will turn the output of the linear operation (logits) to class propabilities. Below is the formula to compute softmax:$$\sigma(z)_i=\frac{e^{\beta{_{z_{i}}}}}{\sum_{j=1}^{K}e^{\beta{_{z_{j}}}}}$$ for i = 1,...,K.

### Metrics computing

After the forward propagation, the loss and the accuracy is computed. The loss is a way to quantify "how much error" the perceptron make. To compute the error we use the negative-log-likelyhood (NLL):

$$
L_i=-\sum_j{y_{i,j}log(\hat{y}_{i,j})}
$$

The accuracy also is computed as is the a more easy metric to understand:

$$
\frac{1}{k}\sum^k_{i=1}{(\hat{y_i}-y_i)}
$$

$\hat{y_i}$ and $y$ are binary.

### Backward propagation
