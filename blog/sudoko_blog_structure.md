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

The optimiser is involved in updating the parameters of the model during the training process. It aims to minimise the loss function...

### SGD vs Adam

Adam uses momentum (what is?), cares less about hyper parameters, changes learning rate. SGD is the most basic. Adam converges faster, SGD may generalise better

## More Data

Does 4 million data set bring any advantages? Are our architectures large enough to make use of more data?

## Results

How accurate? Compare to different implementations both AI and programmatic ones potentially.

## Conclusion

Is using neural networks to solve sudoku a good idea? What did we learn from this task?
