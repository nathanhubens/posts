---
toc: true
layout: post
description: How to remove the batch normalization layer to make your neural networks faster.
categories: ["Deep Learning"]
title: Speed-up inference with Batch Normalization Folding
comments: true
---

## Introduction
Batch Normalization is a technique which takes care of normalizing the input of each layer to make the training process faster and more stable. In practice, it is an extra layer that we generally add after the computation layer and before the non-linearity.

It consists of 2 steps:
1. Normalize the batch by first subtracting its mean $\mu$, then dividing it by its standard deviation $\sigma$.
2. Further scale by a factor $\gamma$ and shift by a factor $\beta$. Those are the parameters of the batch normalization layer, required in case of the network not needing the data to have a mean of 0 and a standard deviation of 1.

$$
\Large
\begin{aligned}
&\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i}\\
&\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}\\
&\widehat{x}_{i} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}\\
&y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{BN}_{\gamma, \beta}\left(x_{i}\right)
\end{aligned}
$$

Due to its efficiency for training neural networks, batch normalization is now widely used. But how useful is it at inference time?

Once the training has ended, each batch normalization layer possesses a specific set of $\gamma$ and $\beta$, but also $\mu$ and $\sigma$, the latter being computed using an exponentially weighted average during training. It means that during inference, the batch normalization acts as a simple linear transformation of what comes out of the previous layer, often a convolution.

As a convolution is also a linear transformation, it also means that both operations can be merged into a single linear transformation!

This would remove some unnecessary parameters but also reduce the number of operations to be performed at inference time.

---

## How to do that in practice?


With a little bit of math, we can easily rearrange the terms of the convolution to take the batch normalization into account.

As a little reminder, the convolution operation followed by the batch normalization operation can be expressed, for an input $x$, as:

$$
\Large
\begin{aligned}
z &=W * x+b \\
\text { out } &=\gamma \cdot \frac{z-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta
\end{aligned}
$$

So, if we re-arrange the $W$ and $b$ of the convolution to take the parameters of the batch normalization into account, as such:

$$
\Large
\begin{aligned}
w_{\text {fold }} &=\gamma \cdot \frac{W}{\sqrt{\sigma^{2}+\epsilon}} \\
b_{\text {fold }} &=\gamma \cdot \frac{b-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta
\end{aligned}
$$

We can remove the batch normalization layer and still have the same results!

> Note: Usually, you don’t have a bias in a layer preceding a batch normalization layer. It is useless and a waste of parameters as any constant will be canceled out by the batch normalization.

---


## How efficient is it?

We will try for 2 common architectures:
1. VGG16 with batch norm
2. ResNet50

Just for the demonstration, we will use ImageNette dataset and PyTorch. Both networks will be trained for 5 epochs and what changes in terms of parameter number and inference time.


### VGG16

Let’s start by training VGG16 for 5 epochs (the final accuracy doesn’t matter):

![]({{ site.baseurl }}/images/bn_folding/vgg_train.png "Training VGG16 for 5epochs")


Then show its number of parameters:

![]({{ site.baseurl }}/images/bn_folding/vgg_param.png "Number of parameters of VGG16")


The initial inference time for a single image is:

![]({{ site.baseurl }}/images/bn_folding/vgg_inf.png "Inference time of VGG16")


So now if we apply batch normalization folding, we have:

![]({{ site.baseurl }}/images/bn_folding/folded_vgg_param.png "Remaining number of parameters after BN folding")


And:

![]({{ site.baseurl }}/images/bn_folding/folded_vgg_inf.png "Inference time after BN folding")


So **8448** parameters removed and even better, almost **0.4 ms** faster inference! Most importantly, this is completely lossless, there is absolutely no change in terms of performance:

Let’s see how it behaves in the case of Resnet50!


### Resnet50

Same, we start by training it for 5 epochs:

![]({{ site.baseurl }}/images/bn_folding/resnet_train.png "Training Resnet50 for 5epochs")


The initial amount of parameters is:

![]({{ site.baseurl }}/images/bn_folding/resnet_param.png "Number of parameters of Resnet50")


And inference time is:

![]({{ site.baseurl }}/images/bn_folding/resnet_inf.png "Inference time of Resnet50")


After using batch normalization folding, we have:

![]({{ site.baseurl }}/images/bn_folding/folded_resnet_param.png "Remaining number of parameters after BN folding")

And:

![]({{ site.baseurl }}/images/bn_folding/folded_resnet_inf.png "Inference time after BN folding")

So now, we have **26,560** parameters removed and even more impressive, an inference time reduce by **1.5ms**! And still without any drop in performance.

So if we can reduce the inference time and the number of parameters of our models without enduring any drop in performance, why shouldn’t we always do it?

**I hope that this blog post helped you! Feel free to give me feedback or ask me questions is something is not clear enough.**

Code available at [this address!](https://github.com/nathanhubens/fasterai)

---

### References and further readings:

- [The Batch Normalization paper](https://arxiv.org/pdf/1502.03167.pdf)
- [DeepLearning.ai Batch Normalization Lesson](https://www.youtube.com/watch?v=tNIpEZLv_eg&t=1s)