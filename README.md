<p align="center"><img src="images/logo.png" width="480"\></p>



# Domain-Adaptation-Algorithms

Welcome to visit my work space, I'm Yiyang Li, at least in the next three years (2021-2024), I will be here to record what I studied in graduate student stage about Domain Adaptation, such as literature introduction and code implementation, etc. I look forward to working with you scholars and experts in communication and begging your comments. 

**PS.** This code base is based on **models**, which is more convenient for learning a single model. If you want to avoid cumbersome conventional code (such as data reading, etc.), you can visit the following link:https://github.com/CtrlZ1/Domain-Adaptive-CodeBase. It presents various domain adaptive codes in the form of projects.

# Contents

- [Domain-Adaptation-Algorithms](#Domain-Adaptation-Algorithms)
- [Contents](#contents)
- [Installation](#installation)
- [Implementations](#implementations)
  - [GAN](#gan)
  - [WGAN](#wgan)
  - [WGAN-GP](#wgan-gp)
  - [LargeScaleOT](#LargeScaleOT)
  - [JCPOT](#JCPOT)
  - [JDOT](#JDOT)
  - [Deep-JDOT](#Deep-JDOT)
  - [DCWD](#DCWD)
  - [DAN](#DAN)
  - [WDGRL](#WDGRL)
  - [DDC](#DDC)
  - [JAN](#JAN)
  - [MCD](#MCD)
  - [SWD](#SWD)
  - [JPOT](#JPOT)
  - [NW](#NW)
  - [WDAN](#WDAN)
  - [ADDA](#ADDA)
  - [CoGAN](#CoGAN)
  - [CDAN](#CDAN)
  - [M3SDA](#M3SDA)
  - [CMSS](#CMSS)
  - [LtC-MSDA](#LtC-MSDA)
  - [Dirt-T](#Dirt-T)

# Installation

```
$ cd yourfolder
$ git clone https://github.com/CtrlZ1/Domain-Adaptation-Algorithms.git
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

## Deep-JDOT

**title**

DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation

**Times**

2018

**Authors**

Bharath Bhushan Damodaran, Benjamin Kellenberger, Remi Flamary, Devis Tuia, Nicolas Courty

**Abstract**

In computer vision, one is often confronted with problems of domain shifts, 
which occur when one applies a classifier trained on a source dataset to target 
data sharing similar characteristics (e.g. same classes), but also different 
latent data structures (e.g. different acquisition conditions). In such a 
situation, the model will perform poorly on the new data, since the classifier 
is specialized to recognize visual cues specific to the source domain. In this 
work we explore a solution, named DeepJDOT, to tackle this problem: through a 
measure of discrepancy on joint deep representations/labels based on optimal 
transport, we not only learn new data representations aligned between the source 
and target domain, but also simultaneously preserve the discriminative 
information used by the classifier. We applied DeepJDOT to a series of visual 
recognition tasks, where it compares favorably against state-of-the-art deep 
domain adaptation methods.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/116698770

**Paper address**

https://arxiv.org/abs/1803.10081

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

## DDC

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

## DAN

**title**

Learning Transferable Features with Deep Adaptation Networks

**Times**

2015

**Authors**

Mingsheng Long  Yue Cao  Jianmin Wang  Michael I. Jordan

**Abstract**

Recent studies reveal that a deep neural network can learn transferable features 
which generalize well to novel tasks for domain adaptation. However, as deep 
features eventually transition from general to specific along the network, the 
feature transferability drops significantly in higher layers with increasing 
domain discrepancy. Hence, it is important to formally reduce the dataset bias 
and enhance the transferability in task-specific layers. In this paper, we 
propose a new Deep Adaptation Network (DAN) architecture, which generalizes deep 
convolution alneural network to the domain adaptation scenario. In DAN, hidden 
representations of all task-specific layers are embeddedin a reproducing kernel 
Hilbert space where the mean embeddingsof different domain distributions can be 
explicitly matched. The domain discrepancy is further reduced using an optimal 
multi-kernel selection method for mean embedding matching. DAN can learn 
transferable features with statistic alguarantees,and can scale linearly by 
unbiased estimate of kernel embedding. Extensive empirical evidence shows that 
the proposed architecture yields state-of-the-art image classification error 
rates on standard domain adaptation benchmarks.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/119829512

**Paper address**

http://proceedings.mlr.press/v37/long15.html

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

## MCD

**title**

Maximum Classifier Discrepancy for Unsupervised Domain Adaptation

**Times**

2018

**Authors**

Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, and Tatsuya Harada

**Abstract**

In this work, we present a method for unsupervised domain adaptation. Many 
adversarial learning methods train domain classifier networks to distinguish the 
features as either a source or target and train a feature generator network to 
mimic the discriminator. Two problems exist with these methods. First, the 
domain classifier only tries to distinguish the features as a source or target 
and thus does not consider task-specific decision boundaries between classes. 
Therefore, a trained generator can generate ambiguous features near class 
boundaries. Second, these methods aim to completely match the feature 
distributions between different domains, which is difficult because of each 
domain’s characteristics. To solve these problems, we introduce a new approach 
that attempts to align distributions of source and target by utilizing the 
task-specific decision boundaries. We propose to maximize the discrepancy 
between two classifiers’ outputs to detect target samples that are far from the 
support of the source. A feature generator learns to generate target features 
near the support to minimize the discrepancy. Our method outperforms other 
methods on several datasets of image classification and semantic segmentation.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/119991815

**Paper address**

https://openaccess.thecvf.com/content_cvpr_2018/html/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.html

## SWD

**title**

Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation

**Times**

2019

**Authors**

Chen-Yu Lee, Tanmay Batra, Mohammad Haris Baig, Daniel Ulbricht

**Abstract**

In this work, we connect two distinct concepts for unsupervised domain 
adaptation: feature distribution alignment between domains by utilizing the 
task-specificdecision boundary [57] and the Wasserstein metric [72]. Our 
proposed sliced Wasserstein discrepancy (SWD) is designed to capture the natural 
notion of dissimilarity between the outputs of task-specific classifiers. It 
provides a geometrically meaningful guidance to detect target samples that are 
far from the support of the source and enables efficient distribution alignment 
in an end-to-end trainable fashion. In the experiments, we validate the 
effectiveness and genericness of our method on digit and sign recognition, image 
classification, semantic segmentation, and object detection.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/119979243

**Paper address**

https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html

## JPOT

**title**

Joint Partial Optimal Transport for Open Set Domain Adaptation

**Times**

2020

**Authors**

Renjun Xu, Pelen Liu, Yin Zhang, Fang Cai, Jindong Wang, Shuoying Liang, Heting

**Abstract**

Domain adaptation (DA) has achieved a resounding success to learn a good 
classifier by leveraging labeled data from a source domain to adapt to an 
unlabeled target domain. However, in a general setting when the target domain 
contains classes that are never observed in the source domain, namely in Open 
Set Domain Adaptation (OSDA), existing DA methods failed to work because of the 
interference of the extra unknown classes. This is a much more challenging 
problem, since it can easily result in negative transfer due to the mismatch 
between the unknown and known classes. Existing researches are susceptible to 
misclassification when target domain unknown samples in the feature space 
distributed near the decision boundary learned from the labeled source domain. 
To overcome this, we propose Joint Partial Optimal Transport (JPOT), fully 
utilizing information of not only the labeled source domain but also the 
discriminative representation of unknown class in the target domain. The 
proposed joint discriminative prototypical compactness loss can not only achieve 
intra-class compactness and inter-class separability, but also estimate the mean 
and variance of the unknown class through backpropagation, which remains 
intractable for previous methods due to the blindness about the structure of the 
unknown classes. To our best knowledge, this is the first optimal transport 
model for OSDA. Extensive experiments demonstrate that our proposed model can 
significantly boost the performance of open set domain adaptation on standard DA 
datasets.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120133702

**Paper address**

https://www.ijcai.org/proceedings/2020/352

## NW

**title**

Normalized Wasserstein for Mixture Distributions with Applications in 
Adversarial Learning and Domain Adaptation

**Times**

2019

**Authors**

Yogesh Balaji, Rama Chellappa, Soheil Feizi

**Abstract**

Understanding proper distance measures between distributions is at the core of 
several learning tasks such as generative models, domain adaptation, clustering, 
etc. In this work, we focus on mixture distributions that arise naturally in 
several application domains where the data contains different sub-populations. F 
or mixture distributions, established distance measures such as the Wasserstein 
distance do not take into account imbalanced mixture proportions. Thus, even if 
two mixture distributions have identical mixture components but different 
mixture proportions, the Wasserstein distance between them will be large. This 
often leads to undesired results in distance-based learning methods for mixture 
distributions. In this paper , we resolve this issue by introducing the 
Normalized Wasserstein measure. The key idea is to introduce mixture proportions 
as optimization variables, effectively normalizing mixture proportions in the 
Wasserstein formulation. Using the proposed normalized Wasserstein measure leads 
to significant performance gains for mixture distributions with imbalanced 
mixture proportions compared to the vanilla Wasserstein distance. We demonstrate 
the effectiveness of the proposed measure in GANs, domain adaptation and 
adversarial clustering in several benchmark datasets.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120086168

**Paper address**

https://arxiv.org/abs/1902.00415

## WDAN

**title**

Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy for Unsupervised 
Domain Adaptation

**Times**

2017

**Authors**

Hongliang Yan, Y ukang Ding, Peihua Li, Qilong Wang, Y ong Xu, Wangmeng Zuo

**Abstract**

In domain adaptation, maximum mean discrepancy (MMD) has been widely adopted as 
a discrepancy metric between the distributions of source and target domains. 
However , existing MMD-based domain adaptation methods generally ignore the 
changes of class prior distributions, i.e., class weight bias across domains. 
This remains an open problem but ubiquitous for domain adaptation, which can be 
caused by changes in sample selection criteria and application scenarios. We 
show that MMD cannot account for class weight bias and results in degraded 
domain adaptation performance. To address this issue, a weighted MMD model is 
proposed in this paper . Specifically, we introduce class-specific auxiliary 
weights into the original MMD for exploiting the class prior probability on 
source and target domains,whose challengelies inthe factthattheclass label in 
target domain is unavailable. To account for it, our proposed weighted MMD model 
is defined by introducing an auxiliary weight for each class in the source 
domain, and a classification EM algorithm is suggested by alternating between 
assigning the pseudo-labels, estimating auxiliary weights and updating model 
parameters. Extensive experiments demonstrate the superiority of our weighted 
MMD over conventional MMD for domain adaptation.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120054974

**Paper address**

https://arxiv.org/abs/1705.00609

## MCDA

**title**

Deep multi-Wasserstein unsupervised domain adaptation

**Times**

2019

**Authors**

Tien-Nam Le , Amaury Habrard , Marc Sebban

**Abstract**

In unsupervised domain adaptation (DA), 1 aims at learning from labeled source 
data and fully unlabeled target examples a model with a low error on the target 
domain. In this setting, standard generalization bounds prompt us to minimize 
the sum of three terms: (a) the source true risk, (b) the divergence be- tween 
the source and target domains, and (c) the combined error of the ideal joint 
hypothesis over the two domains. Many DA methods – e s p e c i a l l y those 
using deep neural networks – h a v e focused on the first two terms by using 
different divergence measures to align the source and target distributions on a 
shared latent feature space, while ignoring the third term, assuming it is 
negligible to perform the adaptation. However, it has been shown that purely 
aligning the two distributions while minimizing the source error may lead to 
so-called negative transfer . In this paper, we address this issue with a new 
deep unsupervised DA method – called MCDA – minimizing the first two terms while 
controlling the third one. MCDA benefits from highly-confident target samples 
(using softmax predictions) to minimize class- wise Wasserstein distances and 
efficiently approximate the ideal joint hypothesis. Empirical results show that 
our approach outperforms state of the art methods.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120110987

**Paper address**

https://linkinghub.elsevier.com/retrieve/pii/S0167865519301400

## ADDA

**title**

Adversarial Discriminative Domain Adaptation

**Times**

2017

**Authors**

Eric Tzeng , Judy Hoffman , Kate Saenko , Trevor Darrell

**Abstract**

Adversarial learning methods are a promising approach to training robust deep 
networks, and can generate complex samples across diverse domains. They can also 
improve recognition despite the presence of domain shift or dataset bias: recent 
adversarial approaches to unsupervised domain adaptation reduce the difference 
between the training and test domain distributions and thus improve 
generalization performance. However , while generative adversarial networks 
(GANs) show compelling visualizations, they are not optimal on discriminative 
tasks and can be limited to smaller shifts. On the other hand, discriminative 
approaches can handle larger domain shifts, but impose tied weights on the model 
and do not exploit a GAN-based loss. In this work, we first outline a novel 
generalized framework for adversarial adaptation, which subsumes recent 
state-of-the-art approaches as special cases, and use this generalized view to 
better relate prior approaches. We then propose a previously unexplored instance 
of our general framework which combines discriminative modeling, untied weight 
sharing, and a GAN loss, which we call Adversarial Discriminative Domain 
Adaptation (ADDA). We show that ADDA is more effective yet considerably simpler 
than competing domainadversarial methods, and demonstrate the promise of our 
approach by exceeding state-of-the-art unsupervised adaptation results on 
standard domain adaptation tasks as well as a difficult cross-modality object 
classification task.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120273707

**Paper address**

https://openreview.net/forum?id=B1Vjl1Stl

## CoGAN

**title**

Coupled generative adversarial networks

**Times**

2016

**Authors**

Ming-Yu Liu , Oncel Tuzel

**Abstract**

We propose coupled generative adversarial network (CoGAN) for learning a joint 
distribution of multi-domain images. In contrast to the existing approaches, 
which require tuples of corresponding images in different domains in the 
training set, CoGAN can learn a joint distribution without any tuple of 
corresponding images. It can learn a joint distribution with just samples drawn 
from the marginal distributions. This is achieved by enforcing a weight-sharing 
constraint that limits the network capacity and favors a joint distribution 
solution over a product of marginal distributions one. We apply CoGAN to several 
joint distribution learning tasks, including learning a joint distribution of 
color and depth images, and learning a joint distribution of face images with 
different attributes. For each task it successfully learns the joint 
distribution without any tuple of corresponding images. We also demonstrate its 
applications to domain adaptation and image transformation.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120347149

**Paper address**

https://proceedings.neurips.cc/paper/2016/hash/502e4a16930e414107ee22b6198c578f-Abstract.html

## CDAN

**title**

Conditional Adversarial Domain Adaptation

**Times**

2018

**Authors**

Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I. Jordan

**Abstract**

Adversarial learning has been embedded into deep networks to learn disentangled 
and transferable representations for domain adaptation. Existing adversarial 
domain adaptation methods may not effectively align different domains of 
multimodal distributions native in classification problems. In this paper, we 
present conditional adversarial domain adaptation, a principled framework that 
conditions the adversarial adaptation models on discriminative information 
conveyed in the classifier predictions. Conditional domain adversarial networks 
(CDANs) are designed with two novel conditioning strategies: multilinear 
conditioning that captures the crosscovariance between feature representations 
and classifier predictions to improve the discriminability, and entropy 
conditioning that controls the uncertainty of classifier predictions to 
guarantee the transferability. With theoretical guarantees and a few lines of 
codes, the approach has exceeded state-of-the-art results on five datasets.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120622652

**Paper address**

https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html

## M3SDA

**title**

Moment Matching for Multi-Source Domain Adaptation

**Times**

2019

**Authors**

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, Bo Wang

**Abstract**

Conventional unsupervised domain adaptation (UDA) assumes that training data are 
sampled from a single domain. This neglects the more practical scenario where 
training data are collected from multiple sources, requiring multi-source domain 
adaptation. We make three major contributions towards addressing this problem. 
First, we collect and annotate by far the largest UDA dataset, called DomainNet, 
which contains six domains and about 0.6 million images distributed among 345 
categories, addressing the gap in data availability for multi-source UDA 
research. Second, we propose a new deep learning approach, Moment Matching for 
Multi-Source Domain Adaptation (M3SDA), which aims to transfer knowledge learned 
from multiple labeled source domains to an unlabeled target domain by 
dynamically aligning moments of their feature distributions. Third, we provide 
new theoretical insights specifically for moment matching approaches in both 
single and multiple source domain adaptation. Extensive experiments are 
conducted to demonstrate the power of our new dataset in benchmarking 
state-of-the-art multi-source domain adaptation methods, as well as the 
advantage of our proposed model. Dataset and Code are available at 
http://ai.bu.edu/M3SDA/

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120819629

**Paper address**

https://arxiv.org/abs/1812.01754

## CMSS

**title**

Curriculum manager for source selection in multi- source domain adaptation

**Times**

2020

**Authors**

Luyu Yang, Yogesh Balaji, Ser-Nam Lim, Abhinav Shrivastava

**Abstract**

The performance of Multi-Source Unsupervised Domain Adaptation depends significantly on the effectiveness of transfer from labeled source domain samples. In this paper, we proposed an adversarial agent that learns a dynamic curriculum for source samples, called Curriculum Manager for Source Selection (CMSS). The Curriculum Manager, an independent network module, constantly updates the curriculum during training, and iteratively learns which domains or samples are best suited for aligning to the target. The intuition behind this is to force the Curriculum Manager to constantly re-measure the transferability of latent domains over time to adversarially raise the error rate of the domain discriminator. CMSS does not require any knowledge of the domain labels, yet it outperforms other methods on four well-known benchmarks by significant margins. We also provide interpretable results that shed light on the proposed method.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120877511

**Paper address**

https://arxiv.org/abs/2007.01261

## LtC-MSDA

**title**

Learning to Combine: Knowledge Aggregation for Multi-Source Domain Adaptation

**Times**

2020

**Authors**

Hang Wang , Minghao Xu , Bingbing Ni , and Wenjun Zhang

**Abstract**

Transferring knowledges learned from multiple source domains to target domain is a more practical and challenging task than conventional single-source domain adaptation. Furthermore, the increase of modalities brings more difficulty in aligning feature distributions among multiple domains. To mitigate these problems, we propose a Learning to Combine for Multi-Source Domain Adaptation (LtC-MSDA) framework via exploring interactions among domains. In the nutshell, a knowledge graph is constructed on the prototypes of various domains to realize the information propagation among semantically adjacent representations. On such basis, a graph model is learned to predict query samples under the guidance of correlated prototypes. In addition, we design a Relation Alignment Loss (RAL) to facilitate the consistency of categories’ relational interdependency and the compactness of features, which boosts features’ intra-class invariance and inter-class separability. Comprehensive results on public benchmark datasets demonstrate that our approach outperforms existing methods with a remarkable margin. Our code is available athttps://github.com/ChrisAllenMing/LtC-MSDA.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/120978951

**Paper address**

https://arxiv.org/abs/2007.08801

## Dirt-T

**title**

A DIRT-T approach to unsupervised domain adaptation

**Times**

2018

**Authors**

Rui Shu, Hung H. Bui, Hirokazu Narui, & Stefano Ermon

**Abstract**

Domain adaptation refers to the problem of leveraging labeled data in a source domain to learn an accurate model in a target domain where labels are scarce or unavailable. A recent approach for finding a common representation of the two domains is via domain adversarial training (Ganin & Lempitsky,2015), which attempts to induce a feature extractor that matches the source and target feature distributions in some feature space. However, domain adversarial training faces two critical limitations: 1) if the feature extraction function has high-capacity, then feature distribution matching is a weak constraint, 2) in non-conservative domain adaptation (where no single classifier can perform well in both the source and target domains), training the model to do well on the source domain hurts performance on the target domain. In this paper, we address these issues through the lens of the cluster assumption, i.e., decision boundaries should not cross high-density data regions. We propose two novel and related models: 1) the Virtual Adversarial Domain Adaptation (VADA) model, which combines domain adversarial training with a penalty term that punishes violation of the cluster assumption; 2) the Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T)1 model, which takes the V ADA model as initialization and employs natural gradient steps to further minimize the cluster assumption violation. Extensive empirical results demonstrate that the combination of these two models significantly improve the state-of-the-art performance on the digit, traffic sign, and Wi-Fi recognition domain adaptation benchmarks.

**Content introduction**

https://blog.csdn.net/qq_41076797/article/details/121226438

**Paper address**

https://openreview.net/pdf?id=H1q-TM-AW