<p align="center"><img src="images/logo.png" width="480"\></p>
# transferLearningAlgorithms

Work are welcome to visit my space, I'm Yiyang Li, at least in the next three years (2021-2024), I will be here to record what I studied in graduate student stage about transfer learning, such as literature introduction and code implementation, etc. I look forward to working with you scholars and experts in communication and begging your comments.



# Contents

- [transferLearningAlgorithms](#transferlearningalgorithms)
- [Contents](#contents)
- [Installation](#installation)
- [Implementations](#implementations)
  - [GAN](#gan)
  - [WGAN](#wgan)
  - [WGAN-GP](#wgan-gp)
  - [LargeScaleOT](#LargeScaleOT)
  - [JCPOT](#JCPOT)

# Installation

```
$ cd yourfolder
$ git clone https://github.com/CtrlZ1/transferLearningAlgorithms.git
```

# Implementations

## GAN

**title**

Generative Adversarial Nets

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

## WGAN

**title**

Wasserstein GAN

**Times**

2017

**Authors**

Martin Arjovsky, Soumith Chintala, Léon Bottou

**Abstract**

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/116898649

**Paper address**

https://arxiv.org/abs/1701.07875

## WGAN-GP

**title**

Improved Training of Wasserstein GANs

**Times**

2017

**Authors**

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron 
Courville

**Abstract**

Generative Adversarial Networks (GANs) are powerful generative models, but 
suffer from training instability. The recently proposed Wasserstein GAN (WGAN) 
makes progress toward stable training of GANs, but sometimes can still generate 
only poor samples or fail to converge. We find that these problems are often due 
to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the 
critic, which can lead to undesired behavior. We propose an alternative to 
clipping weights: penalize the norm of gradient of the critic with respect to 
its input. Our proposed method performs better than standard WGAN and enables 
stable training of a wide variety of GAN architectures with almost no 
hyperparameter tuning, including 101-layer ResNets and language models with 
continuous generators. We also achieve high quality generations on CIFAR-10 and 
LSUN bedrooms.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/118458028

**Paper address**

https://arxiv.org/abs/1704.00028



## LargeScaleOT

**title**

Large scale optimal transport and mapping estimation

**Times**

2018

**Authors**

Vivien Seguy、Bharath Bhushan Damodaran、Rémi Flamary、Nicolas Courty、Antoine Rolet、Mathieu Blondel

**Abstract**

This paper presents a novel two-step approach for the fundamental problem of 
learning an optimal map from one distribution to another. First, we learn an 
optimal transport (OT) plan, which can be thought as a one-to-many map between 
the two distributions. To that end, we propose a stochastic dual approach of 
regularized OT, and show empirically that it scales better than a recent related 
approach when the amount of samples is very large. Second, we estimate a Monge 
map as a deep neural network learned by approximating the barycentric projection 
of the previously-obtained OT plan. This parameterization allows generalization 
of the mapping outside the support of the input measure. We prove two 
theoretical stability results of regularized OT which show that our estimations 
converge to the OT plan and Monge map between the underlying continuous 
measures. We showcase our proposed approach on two applications: domain 
adaptation and generative modeling.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/118878524

**Paper address**

https://arxiv.org/abs/1711.02283

## JCPOT

**title**

Optimal Transport for Multi-source Domain Adaptation under Target
Shift

**Times**

2019

**Authors**

Ievgen Redko 、Nicolas Courty 、Rémi Flamary 、Devis Tuia

**Abstract**

In this paper, we tackle the problem of reducing discrepancies between multiple 
domains, i.e. multi-source domain adaptation, and consider it under the target 
shift assumption: in all domains we aim to solve a classification problem with 
the same output classes, but with different labels proportions. This problem, 
generally ignored in the vast majority of domain adaptation papers, is 
nevertheless critical in real-world applications, and we theoretically show its 
impact on the success of the adaptation. Our proposed method is based on optimal 
transport, a theory that has been successfully used to tackle adaptation 
problems in machine learning. The introduced approach, Joint Class Proportion 
and Optimal Transport (JCPOT), performs multi-source adaptation and target shift 
correction simultaneously by learning the class probabilities of the unlabeled 
target sample and the coupling allowing to align two (or more) probability 
distributions. Experiments on both synthetic and real-world data (satellite 
image pixel classification) task show the superiority of the proposed method 
over the state-of-the-art.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/117151400

**Paper address**

http://proceedings.mlr.press/v89/redko19a/redko19a.pdf