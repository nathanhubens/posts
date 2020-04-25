---
toc: true
layout: post
description: Image Retrieval with autoencoders
categories: ["Deep Learning"]
title: Build a simple Image Retrieval System with an Autoencoder
comments: true
---


Image retrieval is a very active and fast-advancing field of research area in the past decade. The most well-known systems being the Google Image Search and Pinterest Visual Pin Search. In this article, we will learn to build a very simple image retrieval system using a special type of Neural Network, called an **autoencoder**. The way we are going to proceed is in an unsupervised way, i.e without looking at the image labels. Indeed, we will retrieve images only by using their visual contents (textures, shapes,…). This type of image retrieval is called **content-based image retrieval (CBIR)**, opposed to keywords or text-based image retrieval.
For this article, we will use images of handwritten digits, the MNIST dataset and the Keras deep-learning framework.


![]({{ site.baseurl }}/images/autoencoders/mnist.png "The MNIST dataset")


## Autoencoders

Briefly, autoencoders are neural networks that aims to copy their inputs to their outputs. They work by compressing the input into a latent-space representation, and then reconstructing the output from this representation.This kind of network is composed of two parts :

1. **Encoder**: This is the part of the network that compresses the input into a latent-space representation. It can be represented by an encoding function $h=f(x)$.

2. **Decoder**: This part aims to reconstruct the input from the latent space representation. It can be represented by a decoding function $r=g(h)$.

![]({{ site.baseurl }}/images/autoencoders/AE.png "The architecture of an autoencoder")

<br>

> *If you want to learn more about autoencoders, I suggest you to read [my previous blog post](http://localhost:1313/2018/deepinsideautoencoders/).*

<br>

This latent representation, or code, is what will interest us here as it is the way the neural network as found to compress the visual content about each image. It means that all of similar images will be encoded (hopefully) in a similar way.
There are several types of autoencoders but since we are dealing with images, the most efficient is to use a **convolutional autoencoder**, that uses convolution layers to encode and decode images.

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


The first step is thus to train our autoencoder with our training set, to make it learn the way to encode our images into a latent-space representation.
Once the training as been performed, we only need the encoding part of the network.

```python
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
```
This encoder can now be used to encode our query image.

![]({{ site.baseurl }}/images/autoencoders/query.png "The query image")

The same encoding must be done on our searching database, where we want to find similar images to the query image. We can then compare the query code to the database code and try to find the closest ones. To perform this comparison, we will use the nearest-neighbors technique.

## Nearest-neighbors
The way we are going to retrieve the closest codes is by performing the nearest-neighbors algorithm. The principle behind nearest neighbor methods is to find a predefined number of samples closest in distance to the new point. The distance can be any metric measure but the most common choice is the Euclidean distance. For a query image $q$ and a sample $s$, both of dimension $n$, this distance can be computed by the following formula.

\begin{equation} 
d(q,s) = \sqrt{(q_1-s_1)^2 + (q_2-s_2)^2 + ... + (q_n-s_n)^2}
\end{equation}



In this example, we will retrieve the 5 closest images to the query image.

```python
# Fit the NN algorithm to the encoded test set
nbrs = NearestNeighbors(n_neighbors=5).fit(codes)

# Find the closest images to the encoded query image
distances, indices = nbrs.kneighbors(np.array(query_code))
```


## Results
These are the images we retrieved, it looks great ! All the retrieved images are pretty similar to our query image and they also all correspond to the same digit. This shows that the autoencoder, even without being shown the corresponding labels of the images, has found a way to encode similar images in a very similar way.

![]({{ site.baseurl }}/images/autoencoders/retrieved.png "The 5 retrieved images")

## Summary
In this article, we learned to create a very simple image retrieval system by using an autoencoder and the nearest-neighbors algorithm. We proceeded by training our autoencoder on a big dataset, to make it learn the way to encode efficiently the visual content of each image. We then compared the code of our query image to the codes of our searching dataset and retrieve the 5 closest. We saw that our system was giving pretty good results as the visual content of our 5 retrieved images was close to our query image and also that they all represented the same digit, even without using any label in the process.

<br>

**I hope this article was clear and useful for new Deep Learning practitioners and that it gave you a good insight on what image retrieval with autoencoders looks like ! Feel free to give me feedback or ask me questions is something is not clear enough. The whole code is available at [this address!](https://github.com/nathanhubens/Unsupervised-Image-Retrieval)**

