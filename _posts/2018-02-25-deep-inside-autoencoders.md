---
toc: true
layout: post
description: A brief overview of autoencoders
categories: ["Deep Learning"]
title: Deep Inside Autoencoders
image: images/AE.png
comments: true
---

## Introduction

Autoencoders (AE) are neural networks that aims to copy their inputs to their outputs. They work by compressing the input into a latent-space representation, and then reconstructing the output from this representation. This kind of network is composed of two parts :

1. **Encoder**: This is the part of the network that compresses the input into a latent-space representation. It can be represented by an encoding function $$ h=f(x) $$.
2. **Decoder**: This part aims to reconstruct the input from the latent space representation. It can be represented by a decoding function $r=g(h)$.

<img src="{{ site.baseurl }}/images/autoencoders/AE.svg">

---

## Architecture of an Autoencoder
The autoencoder as a whole can thus be described by the function $$ g(f(x)) = r $$ where you want $$ r $$ as close as the original input $$ x $$.

> Why copying the input to the output ?

If the only purpose of autoencoders was to copy the input to the output, they would be useless. Indeed, we hope that, by training the autoencoder to copy the input to the output, the latent representation $$ h $$ will take on useful properties.

This can be achieved by creating constraints on the copying task. One way to obtain useful features from the autoencoder is to constrain $$ h $$ to have smaller dimensions than $$ x $$, in this case the autoencoder is called **undercomplete**. By training an undercomplete representation, we force the autoencoder to learn the most salient features of the training data. If the autoencoder is given too much capacity, it can learn to perform the copying task without extracting any useful information about the distribution of the data. This can also occur if the dimension of the latent representation is the same as the input, and in the **overcomplete** case, where the dimension of the latent representation is greater than the input. In these cases, even a linear encoder and linear decoder can learn to copy the input to the output without learning anything useful about the data distribution. Ideally, one could train any architecture of autoencoder successfully, choosing the code dimension and the capacity of the encoder and decoder based on the complexity of distribution to be modeled.

---

## What are autoencoders used for ?

Today data denoising and dimensionality reduction for data visualization are considered as two main interesting practical applications of autoencoders. With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.
Autoencoders are learned automatically from data examples. It means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input and that it does not require any new engineering, only the appropriate training data.

However, autoencoders will do a poor job for image compression. As the autoencoder is trained on a given set of data, it will achieve reasonable compression results on data similar to the training set used but will be poor general-purpose image compressors. Compression techniques like JPEG will do vastly better.
Autoencoders are trained to preserve as much information as possible when an input is run through the encoder and then the decoder, but are also trained to make the new representation have various nice properties. Different kinds of autoencoders aim to achieve different kinds of properties. We will focus on four types on autoencoders.

---

## Types of autoencoder :
In this article, the four following types of autoencoders will be described:
* Vanilla autoencoder
* Multilayer autoencoder
* Convolutional autoencoder
* Regularized autoencoder

In order to illustrate the different types of autoencoder, an example of each has been created, using the Keras framework and the MNIST dataset. The code for each type of autoencoder is available on my GitHub.

### Vanilla autoencoder
In its simplest form, the autoencoder is a three layers net, i.e. a neural net with one hidden layer. The input and output are the same, and we learn how to reconstruct the input, for example using the adam optimizer and the mean squared error loss function.


```python
input_size = 784
hidden_size = 64
output_size = 784

x = Input(shape=(input_size,))

# Encoder
h = Dense(hidden_size, activation='relu')(x)

# Decoder
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```


Here, we see that we have an undercomplete autoencoder as the hidden layer dimension (64) is smaller than the input (784). This constraint will impose our neural net to learn a compressed representation of data.


### Multilayer autoencoder
If one hidden layer is not enough, we can obviously extend the autoencoder to more hidden layers.

```python
input_size = 784
hidden_size = 128
code_size = 64

x = Input(shape=(input_size,))

# Encoder
hidden_1 = Dense(hidden_size, activation='relu')(x)
h = Dense(code_size, activation='relu')(hidden_1)

# Decoder
hidden_2 = Dense(hidden_size, activation='relu')(h)
r = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```



Now our implementation uses 3 hidden layers instead of just one. Any of the hidden layers can be picked as the feature representation but we will make the network symmetrical and use the middle-most layer.


### Convolutional autoencoder
We may also ask ourselves: can autoencoders be used with Convolutions instead of Fully-connected layers ?

The answer is yes and the principle is the same, but using images (3D vectors) instead of flattened 1D vectors. The input image is downsampled to give a latent representation of smaller dimensions and force the autoencoder to learn a compressed version of the images.


```python
x = Input(shape=(28, 28,1)) 

# Encoder
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
h = MaxPooling2D((2, 2), padding='same')(conv1_3)


# Decoder
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```



### Regularized autoencoder
There are other ways we can constraint the reconstruction of an autoencoder than to impose a hidden layer of smaller dimension than the input. Rather than limiting the model capacity by keeping the encoder and decoder shallow and the code size small, regularized autoencoders use a loss function that encourages the model to have other properties besides the ability to copy its input to its output. In practice, we usually find two types of regularized autoencoder: the sparse autoencoder and the denoising autoencoder.

**Sparse autoencoder**:  Sparse autoencoders are typically used to learn features for another task such as classification. An autoencoder that has been regularized to be sparse must respond to unique statistical features of the dataset it has been trained on, rather than simply acting as an identity function. In this way, training to perform the copying task with a sparsity penalty can yield a model that has learned useful features as a byproduct.
Another way we can constraint the reconstruction of autoencoder is to impose a constraint in its loss. We could, for example, add a reguralization term in the loss function. Doing this will make our autoencoder learn sparse representation of data.

```python
input_size = 784
hidden_size = 64
output_size = 784

x = Input(shape=(input_size,))

# Encoder
h = Dense(hidden_size, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)

# Decoder
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```



Notice in our hidden layer, we added an l1 activity regularizer, that will apply a penalty to the loss function during the optimization phase. As a result, the representation is now sparser compared to the vanilla autoencoder.



**Denoising autoencoder** : Rather than adding a penalty to the loss function, we can obtain an autoencoder that learns something useful by changing the reconstruction error term of the loss function. This can be done by adding some noise of the input image and make the autoencoder learn to remove it. By this means, the encoder will extract the most important features and learn a robuster representation of the data.

```python
x = Input(shape=(28, 28, 1))

# Encoder
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
h = MaxPooling2D((2, 2), padding='same')(conv1_2)


# Decoder
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')
```
---

## Summary
In this article, we went through the basic architecture of autoencoders. We also looked at many different types of autoencoders: vanilla, multilayer, convolutional and regularized. Each has different properties depending on the imposed constraints : either the reduced dimension of the hidden layers or another kind of penalty.

<br>

**I hope this article was clear and useful for new Deep Learning practitioners and that it gave you a good insight on what autoencoders are ! Feel free to give me feed back or ask me questions is something is not clear enough. The whole code is available at [this address!](https://github.com/nathanhubens/Autoencoders)**

