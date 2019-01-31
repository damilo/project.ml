
# Mathematical Representation of a Perceptron Layer
## (with example in TensorFlow)
---

## Motivation
When I started to learn about artificial neural networks, I quickly became excited about the insides of those networks to solve real-world tasks. I am a person who understands because of his imagination and thus being in the need of graphics, animations as well as mathematical formulas if applicable.

The first question I asked myself was: 'How does the transformation from inputs to outputs work in an ANN?', and is exactly what this article covers.

Sure there are many articles about this topic already, but none is as good as your own written one. Furthermore, the contents of articles are slightly different and there may be persons who are in the need of this article to understand the 'How' and 'Why' better, which is another reason for making it public.

## Contents / Structure
The article is divided in two parts:
- Mathematics of a Perceptron
- Application in TensorFlow

The math starts with a Linear Threshold Unit (LTU), the first artificial neuron proposed by Waren McCulloch and Walter Pitts, then combines those units to a Perceptron, and finally clarifies the insides of a Multi-Layer Perceptron ready to be fed by multiple instances at once for batch training.

The second part uses the explained math to build ANN layers for solving the task to classify handwritten digits.

## Prerequisites
To understand the math, you need to know how vector and matrix calculus works - dot product and addition are of particular importance.

For running the TensorFlow application, you need to have a Python environment with all required packages installed, especially NumPy, TensorFlow as well as its dependencies.

## Mathematics of a Perceptron

### Linear Threshold Unit (LTU)
The linear threshold unit (LTU) consists of one input $x$ with n values, one single-value output $y$, and in-between mathematical operations to calculate the linear combination of the inputs and their weights (i.e. the weighted sum) and to apply an activation function on that calculation.

<img src='.\src\ltu.png' width='25%'/>

The weighted sum is the dot product of $w$ and $x$, written as a matrix product:

$z = w^T \cdot x = \sum\limits_{i=1}^n {w_i x_i}$

and the used activation function is the Heaviside step function:

$step (z) = \begin{cases} 0 & \text {if } z < 0 \\ 1 & \text {if } z \geq 0 \end{cases}$

The resulting output value $y$ is binary, since the step function outputs 0 and 1 only:

$y = step(z) = step (w^T \cdot x)$

By that, a single LTU can be used for binary classification, just like Logistic Regression.

### Perceptron

A Perceptron is a simple artificial neural network (ANN) based on a single layer of LTUs, where each LTU is connected to all inputs of vector $x$ as well as a bias vector $b$.

<img src='.\src\perceptron.png' width='15%'/>

The above shown example takes one input vector $x$ and a bias vector $b = (1, 1, 1)^T$ (consists of ones only). It outputs 3 binary values $y = (y_1, y_2, y_3)^T$.

Important to note is, that the weight vector $w_i$ exists per LTU:

$y_1 = step (z_1) = step (w_1^T \cdot x + b_1)$

$y_2 = step (z_2) = step (w_2^T \cdot x + b_2)$

$y_3 = step (z_3) = step (w_3^T \cdot x + b_3)$

The equations can be combined to:

$y = step (W \cdot x + b)$ with

$W = \begin{pmatrix} w_1^T \\ ... \\ w_u^T \end{pmatrix} = \begin{pmatrix} w_{1,1} \text{ ... } w_{1,n} \\ \text{ ... } \\ w_{u,1} \text{ ... } w_{u,n} \end{pmatrix}$

$W$ is a matrix of shape (u, n), where u = number of LTUs and  n = number of input values.

The input vector $x$ is of shape (n, 1), the bias vector $b$ is of shape (u, 1) and the output vector $y$ is of shape (u, 1).

By that, the Perceptron can be used for multi-class classification.

### Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) is a composition of an input layer, at least one hidden layer of LTUs and an output layer of LTUs. If an MLP has two or more hidden layer, it is called a deep neural network (DNN).

<img src='.\src\mlp.png' width='40%'/>

The calculations are the same as for a Perceptron, but now there exist more layers of LTUs to combine until you reach the output $y$:

$h^1 = step (z^1) = step (W^1 \cdot x + b^1)$

$y = step (z^2) = step (W^2 \cdot h^1 + b^2)$

### Input in Batches of k Instances

ANNs are usually trained in batches of instances (an instance is one input vector x). Hereby k instances are drawn from the m instances available:

$x_1 = \begin{pmatrix} x_{1,1} \\ ... \\ x_{1,n} \end{pmatrix} \text{ , ... ,  } x_k = \begin{pmatrix} x_{k,1} \\ ... \\ x_{k,n} \end{pmatrix}$

The k instances can be combined to:

$X = \begin{pmatrix} x_1^T \\ ... \\ x_k^T \end{pmatrix} = \begin{pmatrix} x_{1,1} \text{ ... } x_{1,n} \\ \text{ ... } \\ x_{k,1} \text{ ... } x_{k,n} \end{pmatrix}$

By that, the equation to calculate $y$ changes to:

$y = step (Z) = step (X \cdot W + b)$ with

The input $X$ is now a matrix of shape (k, n), where k = number of instances per batch and n = number of input values.

$W$ is a matrix, but the shape changed to (n, u) ($W$ is just the transpose of itself $= W^T$).

The bias vector $b$ is of shape (u, 1) and the output $y$ changed to a matrix of shape (k, u).

### Modern MLPs

Todays MLPs make use of other activation functions in order to work better with Gradient Descent.

Hereby the heaviside step function is replaced by one of the following activations:

Logistic Function (Sigmoid) $\sigma (z) = \frac{1}{1+e^{-z}}$

Rectifying Linear Unit $ReLU (z) = max (0, z)$

Hyperbolic Tangent $tanh (z) = 2\sigma (2z) - 1$

The activation function at the output layer depends on the task to be solved by the ANN:
- classification tasks: Softmax activation function
- regression tasks: no activation function

## Application in Tensorflow

With the math in the back, we can build our own neural network layers.

The task to solve is to classify handwritten digits from 0 to 9. I take the digits dataset provided by scikit-learn which itself is a subset of the <a href = 'http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits' target = '_blank'>'Optical Recognition of Handwritten Digits Data Set'</a> from the UCI ML Repository.

The data exploration, data preprocessing and model evaluation are not part of this article to focus on the implementation of Perceptron layers in Tensorflow only. If you want to take a look at the whole machine learning project, <a href = 'https://github.com/damilo/udacity.mlnd/blob/master/project.ml.tfmathmlp/project.ml.tfmathmlp.imp.ipynb' target = 'blank'>visit my GitHub ML repository with a Jupyter Notebook</a>.

### Overview
The digits dataset consists of m = 1797 instances with grayscale images showing numbers. One image is 8 by 8 pixels and each pixel value is within the range [0.0, 16.0]. The targets, i.e. the classes y to output, are within the range [0, 9], where each class is represented by roughly 10% of all images.

<img src='.\src\digits.png' width='50%' />

Prior implementation, the dataset has been scaled and split into a training, validation and testing set. Since the input vector $x_i$ must be one dimensional, the images have been flattened beforehand.

### Implementation
The implementation of a layer is a one-to-one mapping of the math:


```python
# layer
def nn_layer (inputs, units, activation = None):
        
    # X, shape (k, n)
    X = inputs
    
    # W, shape (n, u)
    # The initialization of the weights is very important in order to make the ANN learn something
    r = 2 / np.sqrt (units)
    W_init = tf.random_uniform (shape = [int (X.get_shape ()[1]), units], minval = -1.*r, maxval = 1.*r)
    W = tf.Variable (
        initial_value = W_init,
        name = 'weights'
    )
    
    # b, shape (u, 1)
    b = tf.Variable (
        initial_value = tf.zeros (shape = [units]),
        name = 'bias'
    )
    
    # Z = X * W + b with shape (k, u)
    Z = tf.matmul (X, W) + b
    
    # H = activation (Z)
    if activation:
        return activation (Z)
    else:
        return Z
```

Well, that's it.

Remark: Be aware of the weights initialization - wrong values lead to untrainable ANNs.

Now you can construct the ANN adding layer by layer:


```python
# construction phase
n_inputs = 8*8
n_hidden1 = 256
n_hidden2 = 128
n_outputs = 10

# ...

fc1 = nn_layer (
    inputs = X,
    units = n_hidden1,
    activation = tf.nn.relu
)
fc2 = nn_layer (
    inputs = fc1,
    units = n_hidden2,
    activation = tf.nn.relu
)
logits = nn_layer (
    inputs = fc2,
    units = n_outputs,
    activation = None
)

# ...
```

This article presented the math of the transformation from inputs to outputs in an ANN and proves it by implementing an example in Tensorflow.

The contents was strongly influenced by the book <a href = 'http://shop.oreilly.com/product/0636920052289.do'>'Hands-On Machine Learning with Scikit-Learn and TensorFlow'</a> by Aurélien Géron. It is a great book for everyone who wants to get a solid foundation in Machine Learning algorithms and their utilization for real-world tasks.

By the way: You don't need to implement your own layer in Tensorflow everytime. The API contains a functional interface for such a <a href = 'https://www.tensorflow.org/api_docs/python/tf/layers/dense'>dense layer</a>.


```python

```
