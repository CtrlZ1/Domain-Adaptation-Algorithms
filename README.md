<p align="center"><img src="images/logo.png" width="480"\></p>
# TransferLearningAlgorithms

Work are welcome to visit my space, I'm Yiyang Li, at least in the next three years (2021-2024), I will be here to record what I studied in graduate student stage about transfer learning, such as literature introduction and code implementation, etc. I look forward to working with you scholars and experts in communication and begging your comments.



# Contents

- [TransferLearningAlgorithms](#Transferlearningalgorithms)
- [Contents](#contents)
- [Installation](#installation)
- [Implementations](#implementations)
  - [GAN](#gan)
  - [WGAN](#wgan)
  - [WGAN-GP](#wgan-gp)
  - [LargeScaleOT](#LargeScaleOT)
  - [JCPOT](#JCPOT)
  - [JDOT](#JDOT)
  - [DCWD](#DCWD)
  - [WDGRL](#WDGRL)

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

## JDOT

**title**

Joint distribution optimal transportation for domain adaptation

**Times**

2017

**Authors**

Nicolas Courty 、Rémi Flamary 、Amaury Habrard 、Alain Rakotomamonjy

**Abstract**

This paper deals with the unsupervised domain adaptation problem, where one 
wants to estimate a prediction function f in a given target domain without any 
labeled sample by exploiting the knowledge available from a source domain where 
labels are known. Our work makes the following assumption: there exists a 
non-linear transformation between the joint feature/label space distributions of 
the two domain Ps and Pt. We propose a solution of this problem with optimal 
transport, that allows to recover an estimated target P^f_t= (X, f(X)) by 
optimizing simultaneously the optimal coupling and f. We show that our method 
corresponds to the minimization of a bound on the target error, and provide an 
efficient algorithmic solution, for which convergence is proved. The versatility 
of our approach, both in terms of class of hypothesis or loss functions is 
demonstrated with real world classification and regression problems, for which 
we reach or surpass state-of-the-art results.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/116608774

**Paper address**

https://proceedings.neurips.cc/paper/2017/file/0070d23b06b1486a538c0eaa45dd167a-Paper.pdf

## DCWD

**title**

Domain-attention Conditional Wasserstein Distance
for Multi-source Domain Adaptation

**Times**

2020

**Authors**

HANRUI WU 、YUGUANG YAN  、 MICHAEL K. NG 、QINGYAO WU

**Abstract**

Multi-source domain adaptation has received considerable attention due to its 
effectiveness of leveraging the knowledge from multiple related sources with 
different distributions to enhance the learning performance. One of the 
fundamental challenges in multi-source domain adaptation is how to determine the 
amount of knowledge transferred from each source domain to the target domain. To 
address this issue, we propose a new algorithm, called Domain-attention 
Conditional Wasserstein Distance (DCWD), to learn transferred weights for 
evaluating the relatedness across the source and target domains. In DCWD, we 
design a new conditional Wasserstein distance objective function by taking the 
label information into consideration to measure the distance between a given 
source domain and the target domain. We also develop an attention scheme to 
compute the transferred weights of different source domains based on their 
conditional Wasserstein distances to the target domain. After that, the 
transferred weights can be used to reweight the source data to determine their 
importance in knowledge transfer. We conduct comprehensive experiments on 
several real-world data sets, and the results demonstrate the effectiveness and 
efficiency of the proposed method.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/118358520

**Paper address**

https://dl.acm.org/doi/10.1145/3391229

## WDGRL

**title**

Wasserstein Distance Guided Representation Learning
for Domain Adaptation

**Times**

2018

**Authors**

Jian Shen, Yanru Qu, Weinan Zhang∗, Y ong Yu

**Abstract**

Domain adaptation aims at generalizing a high-performance learner on a target 
domain via utilizing the knowledge distilled from a source domain which has a 
different but related data distribution. One solution to domain adaptation is to 
learn domain invariant feature representations while the learned representations 
should also be discriminative in prediction. To learn such representations, 
domain adaptation frameworks usually include a domain invariant representation 
learning approach to measure and reduce the domain discrepancy, as well as a 
discriminator for classification. Inspired by Wasserstein GAN, in this paper we 
propose a novel approach to learn domain invariant feature representations, 
namely Wasserstein Distance Guided Representation Learning (WDGRL). WDGRL 
utilizes a neural network, denoted by the domain critic, to estimate empirical 
Wasserstein distance between the source and target samples and optimizes the 
feature extractor network to minimize the estimated Wasserstein distance in an 
adversarial manner. The theoretical advantages of Wasserstein distance for 
domain adaptation lie in its gradient property and promising generalization 
bound. Empirical studies on common sentiment and image classification adaptation 
datasets demonstrate that our proposed WDGRL outperforms the state-of-the-art 
domain invariant representation learning approaches.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/116942752

**Paper address**

https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17155

## WDGRL

**title**

Deep Domain Confusion: Maximizing for Domain Invariance

**Times**

2014

**Authors**

Eric Tzeng, Judy Hoffman, Ning Zhang, Kate Saenko, Trevor Darrell

**Abstract**

Recent reports suggest that a generic supervised deep CNN model trained on a 
large-scale dataset reduces, but does not remove, dataset bias on a standard 
benchmark. Fine-tuning deep models in a new domain can require a significant 
amount of data, which for many applications is simply not available. We propose 
a new CNN architecture which introduces an adaptation layer and an additional 
domain confusion loss, to learn a representation that is both semantically 
meaningful and domain invariant. We additionally show that a domain confusion 
metric can be used for model selection to determine the dimension of an 
adaptation layer and the best position for the layer in the CNN architecture. 
Our proposed adaptation method offers empirical performance which exceeds 
previously published results on a standard benchmark visual domain adaptation 
task.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/119698726

**Paper address**

https://arxiv.org/abs/1412.3474

## JAN

**title**

Deep Transfer Learning with Joint Adaptation Networks

**Times**

2017

**Authors**

Mingsheng Long  Han Zhu  Jianmin Wang  Michael I. Jordan

**Abstract**

Deep networks have been successfully applied to learn transferable features for 
adapting models from a source domain to a different target domain. In this 
paper, we present joint adaptation networks (JAN), which learn a transfer 
network by aligning the joint distributions of multiple domain-specific layers 
across domains based on a joint maximum mean discrepancy (JMMD) criterion. 
Adversarial training strategy is adopted to maximize JMMD such that the 
distributions of the source and target domains are made more distinguishable. 
Learning can be performed by stochastic gradient descent with the gradients 
computed by back-propagation in linear-time. Experiments testify that our model 
yields state of the art results on standard datasets.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/119850543

**Paper address**

http://proceedings.mlr.press/v70/long17a.html