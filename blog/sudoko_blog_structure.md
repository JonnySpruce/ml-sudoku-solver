# Sudoku Blog

## Introduction

We often see neural networks used with sudoku in the realm of computer vision (link to Colin blog) to recognise a sudoku puzzle from a photo or video but rarely do we see machine learning used as a tool to solve sudoku. In this post, we aim to take you through our journey to cracking sudoku with artificial intelligence.

## Model Architectures

We explored two different model architectures in our quest to solve sudoku with neural networks.

### Multilayer Perceptron (MLP)

The most recognisable neural network. The MLP consists of at least three layers: input, hidden, and output. All neurons are connected to all the rest in the layer ahead of them. The weights of the neurons' connections are altered throughout the learning process via backpropagation in order to allow the network to learn based on input data.

Our first model consisted of an MLP with three hidden linear layers, interspersed with non-linear activation functions in the form of rectified linear units (ReLUs). This model performed...

<div align="center">
    <img src="mlpsvg.svg" height=200px title="Example MLP." alt="Example MLP."/>
</div>

### Convolutional Neural Network (CNN)

A CNN uses convolutions over tensors to facilitate machine learning. The kernel moves over...

Explanation of kernels and incorporating rules.

<div align="center">
    <img src="./threeThree.gif" height=200px />
    <img src="./oneNine.gif" height=200px />
    <img src="./nineOne.gif" height=200px />
</div>

## Loss Function

<div align="center">
    <img src="./colouredSudoku.png" width=200px />
    <img src="./lossVisualIdea.png" width=400px />
</div>

## Batch Normalisation

What is normalisation - what is batch version? Reduces effect of internal covariate shift in model parameters which is changes in their distributions. Makes training quicker and less all over the place (technical term).

## Optimiser

The optimiser is involved in updating the parameters/weights of the model during the training process. It aims to minimise the loss function and does so by adjusting model parameters depending on the loss function output. The optimiser is a crucial facet of training a neural network as without it, there would be no learning. One should note that the optimiser aims to minimise the loss function - not to improve the accuracy (or any other metric) of a model. Whilst a reduced loss may lead to an increased accuracy, the accuracy and its maximisation have no bearing on the work of the optimiser.

Throughout this work, we investigated a handful of different optimisers ranging from basic to advanced in a bid to improve our model's performance.

### Stochastic Gradient Descent (SGD)

SGD is likely the first optimiser you will hear about when learning about neural networks. It introduces randomness to the base gradient descent algorithm to improve efficiency and reduce computation massively. Instead of looking at every data point to determine the next step towards the minimum, SGD takes a shuffled group of data points. Due to this, the path SGD takes to the minimum will appear much more unstable and erratic than usual SGD, but it will reach the lowest point in a much faster time.

### Adaptive Moment Estimation (Adam)

Adam is an improvement on SGD. With SGD, the learning rate is set at the start and remains unchanged throughout the learning process. This can make it hard for the model to escape local minima and continue improving. Adam remedies this through the use of an adaptive learning rate. The learning rate is changed depending on (among other things) momentum. Momentum in the case of neural network training is calculated as the moving average of loss function gradients. In general, a higher momentum causes the adaptive learning rate to look at past gradients (loss function gradients) as well as the current one whereas a lower momentum leads the focus to be more on the current gradient.

### AdamW

In our training, we saw improvements with Adam over SGD. In order to better incorporate weight decay (a method of reducing overfitting), we opted to use the AdamW optimiser.

GRAPH to show differences between optimisers

## More Data

Does 4 million data set bring any advantages? Are our architectures large enough to make use of more data?

## Results

How accurate? Compare to different implementations both AI and programmatic ones potentially.

## Conclusion

Is using neural networks to solve sudoku a good idea? What did we learn from this task?
