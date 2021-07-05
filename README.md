<p align="center"><img src="images/logo.png" width="480"\></p>
#　transferLearningAlgorithms

Work are welcome to visit my space, I'm Yiyang Li, at least in the next three years (2021-2024), I will be here to record what I studied in graduate student stage about transfer learning, such as literature introduction and code implementation, etc. I look forward to working with you scholars and experts in communication and begging your comments.

# Contents

- [transferLearningAlgorithms](#transferlearningalgorithms)
- [Contents](#contents)
- [Installation](#installation)
- [Implementations](#implementations)
  - [GAN——Generative Adversarial Nets](#gangenerative-adversarial-nets)

# Installation

```
$ cd yourfolder
$ git clone https://github.com/CtrlZ1/transferLearningAlgorithms.git
```

# Implementations

## GAN——Generative Adversarial Nets

**Times**

2014 NIPS

**Authors**

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron Courville, Y oshua Bengio

**Abstract**

We propose a new framework for estimating generative models via an adversarial 
process, in which we simultaneously train two models: a generative model G that 
captures the data distribution, and a discriminative model D that estimates the 
probability that a sample came from the training data rather than G. The 
training procedure for G is to maximize the probability of D making a mistake. 
This framework corresponds to a minimax two-player game. In the space of 
arbitrary functions G and D, a unique solution exists, with G recovering the 
training data distribution and D equal to1 2everywhere. In the case where G and 
D are defined by multilayer perceptrons, the entire system can be trained with 
backpropagation. There is no need for any Markov chains or unrolled approximate 
inference networks during either training or generation of samples. Experiments 
demonstrate the potential of the framework through qualitative and quantitative 
evaluation of the generated samples.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/118483802

**Paper address**

https://arxiv.org/abs/1406.2661