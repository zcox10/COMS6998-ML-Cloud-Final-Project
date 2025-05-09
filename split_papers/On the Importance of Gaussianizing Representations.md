<|startofpaper|>
## On the Importance of Gaussianizing Representations

Daniel Eftekhari 1 2 Vardan Papyan 3 1 2

## Abstract

The normal distribution plays a central role in information theory - it is at the same time the best-case signal and worst-case noise distribution, has the greatest representational capacity of any distribution, and offers an equivalence between uncorrelatedness and independence for joint distributions. Accounting for the mean and variance of activations throughout the layers of deep neural networks has had a significant effect on facilitating their effective training, but seldom has a prescription for precisely what distribution these activations should take, and how this might be achieved, been offered. Motivated by the information-theoretic properties of the normal distribution, we address this question and concurrently present normality normalization: a novel normalization layer which encourages normality in the feature representations of neural networks using the power transform and employs additive Gaussian noise during training. Our experiments comprehensively demonstrate the effectiveness of normality normalization, in regards to its generalization performance on an array of widely used model and dataset combinations, its strong performance across various common factors of variation such as model width, depth, and training minibatch size, its suitability for usage wherever existing normalization layers are conventionally used, and as a means to improving model robustness to random perturbations.

## 1. Introduction

The normal distribution is unique - information theory shows that among all distributions with the same mean

1 Department of Computer Science, University of Toronto, Toronto, Canada 2 Vector Institute, Toronto, Canada 3 Department of Mathematics, University of Toronto, Toronto, Canada. Correspondence to: Daniel Eftekhari &lt; defte@cs.toronto.edu &gt; .

Proceedings of the 42 nd International Conference on Machine Learning , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

and variance, a signal following this distribution encodes the maximal amount of information (Shannon, 1948). This can be viewed as a desirable property in learning systems such as neural networks, where the activations of successive layers equivocates to successive representations of the data.

Moreover, a signal following the normal distribution is maximally robust to random perturbations (Cover &amp; Thomas, 2006), and thus presents a desirable property for the representations of learning systems; especially deep neural networks, which are susceptible to random (Ford et al., 2019) and adversarial (Szegedy et al., 2014) perturbations. Concomitantly, the normal distribution is informationtheoretically the worst-case perturbative noise distribution (Cover &amp; Thomas, 2006), which suggests models gaining robustness to Gaussian noise should be robust to any other form of random perturbations.

We show that encouraging deep learning models to encode their activations using the normal distribution in conjunction with applying additive Gaussian noise during training, helps improve generalization. We do so by means of a novel layer - normality normalization - so-named because it applies the power transform, a technique used to gaussianize data (Box &amp; Cox, 1964; Yeo &amp; Johnson, 2000), and because it can be viewed as an augmentation of existing normalization techniques such as batch (Ioffe &amp; Szegedy, 2015), layer (Ba et al., 2016), instance (Ulyanov et al., 2016), and group (Wu &amp;He, 2018) normalization.

Our experiments comprehensively demonstrate the general effectiveness of normality normalization in terms of its generalization performance, its strong performance across various common factors of variation such as model width, depth, and training minibatch size, which furthermore serve to highlight why it is effective, its suitability for usage wherever existing normalization layers are conventionally used, and its effect on improving model robustness under random perturbations.

In Section 2 we outline some of the desirable properties normality can imbue in learning models, which serve as motivating factors for the development of normality normalization. In Section 3 we provide a brief background on the power transform, before presenting normality normalization in Section 4. In Section 5 we describe our experiments, analyze the results, and explore some of the properties of

models trained with normality normalization. In Section 6 we comment on related work and discuss some possible future directions. Finally in Section 7 we contextualize normality normalization in the broader deep learning literature, and provide a few concluding remarks.

## 2. Motivation

In this section we present motivating factors for encouraging normality in feature representations in conjunction with using additive random noise during learning. Section 5 substantiates the applicability of the motivation through the experimental results.

## 2.1. Mutual Information Game &amp; Noise Robustness

## 2.1.1. OVERVIEW OF THE FRAMEWORK

The normal distribution is at the same time the best possible signal distribution, and the worst possible noise distribution; a result which can be studied in the context of the Gaussian channel (Shannon, 1948), and through the lens of the mutual information game (Cover &amp; Thomas, 2006). In this framework, X and Z denote two independent random variables, representing the input signal and noise, and Y = X + Z is the output. The mutual information between X and Y is denoted by I ( X Y ; ) ; X tries to maximize this term, while Z tries to minimize it. Both X and Z can encode their signal using any probability distribution, so that their respective objectives are optimized for.

Information theory answers the question of what distribution X should choose to maximize I ( X Y ; ) . It also answers the question of what distribution Z should choose to minimize I ( X Y ; ) . As shown by the following theorem, remarkably the answer to both questions is the same - the normal distribution.

Theorem 2.1. (Cover &amp; Thomas, 2006) Mutual Information Game. Let X Z , be independent, continuous random variables with non-zero support over the entire real line, and satisfying the moment conditions E X [ ] = µ x , E X [ 2 ] = µ 2 x + σ 2 x and E Z [ ] = µ z , E Z [ 2 ] = µ 2 z + σ 2 z . Further let X ∗ , Z ∗ be normally distributed random variables satisfying the same moment conditions, respectively. Then the following series of inequalities holds

$$I \left ( X \ ; X \ + Z ^ { * } \right ) & \leq \quad \text{ of } \\ I \left ( X ^ { * } ; X ^ { * } + Z ^ { * } \right ) & \leq \quad \text{ (1)} \quad \text{ for } \\ I \left ( X ^ { * } ; X ^ { * } + Z \ \right ). \quad \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{$$

Proof. Without loss of generality let µ x = 0 and µ z = 0 . The first inequality hinges on the entropy power inequality. The second inequality hinges on the maximum entropy of the normal distribution given first and second moment constraints. See Cover &amp; Thomas (2006) for details.

$$\text{eion} \quad \text{ This leads to the following minimax formulation of the game} \\ \text{ssible} \quad \text{$\min} \max I \left ( X ; X + Z \right ) = \max _ { X } \min _ { Z } I \left ( X ; X + Z \right ), \ \text{(2)}$$

which implies that any deviation from normality, for X or Z , is suboptimal from that player's perspective.

## 2.1.2. RELATION TO LEARNING

How might this framework relate to the learning setting? First, previous works have shown that adding noise to the inputs (Bishop, 1995) or to the intermediate activations (Srivastava et al., 2014) of neural networks can be an effective form of regularization, leading to better generalization. Moreover, the mutual information game shows that, among encoding distributions, the normal distribution is maximally robust to random perturbations. Taken together these suggest that encoding activations using the normal distribution is the most effective way of using noise as a regularizer, because a greater degree of regularizing noise in the activations can be tolerated for the same level of corruption.

Second, the mutual information game suggests gaining robustness to Gaussian noise is optimal because it is the worstcase noise distribution. This suggests adding Gaussian noise - specifically - to activations during training should have the strongest regularizing effect. Moreover, gaining robustness to noise has previously been demonstrated to imply better generalization (Arora et al., 2018).

Finally, there exists a close correspondence between the mutual information between the input and the output of a channel subject to additive Gaussian noise, and the minimum mean-squared error (MMSE) in estimating (or recovering) the input given the output (Guo et al., 2005). This suggests that when Gaussian noise is added to a given layer's activations, quantifying the attenuation of the noise across the subsequent layers of the network, as measured by the meansquared error (MSE) relative to the unperturbed activations, provides a direct and measurable proxy for the mutual information between the activations of successive layers in the presence of noise.

## 2.2. Maximal Representation Capacity and Maximally Compact Representations

The entropy of a random variable is a measure of the number of bits it can encode (Shannon, 1948), and therefore of its representational capacity (Cover &amp; Thomas, 2006). The normal distribution is the maximum entropy distribution for specified mean and variance. This suggests that a unit which encodes features using the normal distribution has maximal representation capacity given a fixed variance budget, and therefore encodes information as compactly as possible. This may then suggest that it is efficient for a unit (and by extension layer) to encode its activations using the normal distribution.

## 2.3. Maximally Independent Representations

Previous work has explored the beneficial effects of decorrelating features in neural networks (Huang et al., 2018; 2019; Pan et al., 2019). Furthermore, other works have shown that preventing feature co-adaptation is beneficial for training deep neural networks (Hinton et al., 2012).

For any set of random variables, for example representing the pre-activation values of various units in a neural network layer, uncorrelatedness does not imply independence in general. But for random variables whose marginals are normally distributed, then as shown by Lemma B.1, uncorrelatedness does imply independence when they are furthermore jointly normally distributed. Furthermore, for any given (in general, non-zero) degree of correlation between the random variables, they are maximally independent - relative to any other possible joint distribution - when they are jointly normally distributed.

We use these results to motivate the following argument: for a given level of correlation, encouraging normality in the feature representations of units by using normality normalization, would lead to the desirable property of maximal independence between them; in the setting where increased unit-wise normality also lends itself to increased joint normality.

## 3. Background: Power Transform

Before introducing normality normalization, we briefly outline the power transform (Yeo &amp; Johnson, 2000) which our proposed normalization layer employs. Appendix C provides the complete derivation of the negative log-likelihood (NLL) objective function presented below.

Consider a random variable H from which a sample h = { h i } N i =1 is obtained. 1 The power transform gaussianizes h by applying the following function for each h i :

̸

̸

$$\psi \left ( h ; \lambda \right ) = \begin{cases} \frac { 1 } { \lambda } \left ( \left ( 1 + h \right ) ^ { \lambda } - 1 \right ), & h \geq 0, \lambda \neq 0 & \text{$\text{$simila$} \\ \log \left ( 1 + h \right ), & h \geq 0, \lambda = 0 & \text{accur} \\ \frac { - 1 } { 2 - \lambda } \left ( \left ( 1 - h \right ) ^ { 2 - \lambda } - 1 \right ), & h < 0, \lambda \neq 2 & \text{$\text{Newt} \\ - \log \left ( 1 - h \right ), & h < 0, \lambda = 2 & \text{$Subse} \\ & \text{$(3)$} \quad \text{to eac} \end{cases} \\ \text{The normalize $\i$ is obtained using maximum likelihood}$$

The parameter λ is obtained using maximum likelihood estimation (MLE) so that the transformed variable is as normally distributed as possible, by minimizing the following

1 In the context of normalization layers, N represents the number of samples being normalized; for example in batch normalization, N = BHW for convolutional layers, where B is the minibatch size, and H,W are respectively the height and width of the activation.

NLL 2 :

$$\dots \dots \cdots \cdots \\ \mathcal { L } \left ( h ; \lambda \right ) & = \frac { 1 } { 2 } \left ( \log \left ( 2 \pi \right ) + 1 \right ) + \frac { 1 } { 2 } \log \left ( \hat { \sigma } ^ { 2 } \left ( \lambda \right ) \right ) \\ & \quad - \frac { \lambda - 1 } { N } \sum _ { i = 1 } ^ { N } \log \left ( 1 + h _ { i } \right ), \\ & \quad \dots \dots$$

where ˆ ( µ λ ) = 1 N ∑ N i =1 ψ h ( i ; λ ) and ˆ σ 2 ( λ ) = 1 N ∑ N i =1 ( ψ h ( i ; λ ) -ˆ ( µ λ )) 2 .

## 4. Normality Normalization

To gaussianize a unit's pre-activations h , normality normalization estimates ˆ λ using the method we present in Subsection 4.1, and then applies the power transform given by Equation 3. It subsequently adds Gaussian noise with scaling as described in Subsection 4.2. These steps are done between the normalization and affine transformation steps conventionally performed in other normalization layers.

## 4.1. Estimate of ˆ λ

Differentiating Equation 4 w.r.t. λ and setting the resulting expression to 0 does not lead to a closed-form solution for ˆ λ , which suggests an iterative method for its estimation; for example gradient descent, or a root-finding algorithm (Brent, 1971). However, motivated by the NLL's convexity in λ (Yeo &amp; Johnson, 2000), we use a quadratic series expansion for its approximation, which we outline in Appendix D.

With the quadratic form of the NLL, we can estimate ˆ λ with one step of the Newton-Raphson method:

$$\hat { \lambda } = 1 - \frac { \mathcal { L } ^ { \prime } ( h ; \lambda = 1 ) } { \mathcal { L } ^ { \prime \prime } ( h ; \lambda = 1 ) },$$

where the series expansion has been taken around 3 λ 0 = 1 . The expressions for L ′ ( h ; λ = 1) and L ′′ ( h ; λ = 1) are outlined in Appendix D.

Appendix E provides empirical evidence substantiating the similarity between the NLL and its second-order series expansion around λ 0 = 1 , and furthermore demonstrates the accuracy of obtaining the estimates ˆ λ using one step of the Newton-Raphson method.

Subsequent to estimating ˆ λ , the power transform is applied to each of the pre-activations to obtain x i = ψ ( h i ; ˆ λ ) .

2 To simplify the presentation, we momentarily defer the cases λ = 0 and λ = 2 , and outline the NLL for h ≥ 0 only, as the case for h &lt; 0 follows closely by symmetry.

3 The previously deferred cases of λ = 0 and λ = 2 are thus inconsequential, in the context of computing an estimate ˆ λ , by continuity of the quadratic form of the series expansion for the NLL. However, these two cases still need to be considered when applying the transformation function itself.

We next discuss a few facets of the method.

Justification for the Second Order Method The justification for using the Newton-Raphson method for computing ˆ λ is as follows:

- · A first-order gradient-based method would require iterative refinements to its estimates of ˆ λ in order to find the minima, which would significantly affect runtime. In contrast, the Newton-Raphson method is guaranteed to find the minima of the quadratic loss in one step.
- · A first-order gradient-based method for computing ˆ λ would require an additional hyperparameter for the step size. Due to the quadratic nature of the loss, the NewtonRaphson method necessarily does not require any such additional hyperparameter.
- · The minibatch statistics ˆ µ and ˆ σ 2 are available in closedform. It is therefore natural to seek a closed-form expression for ˆ λ , which is facilitated by using the NewtonRaphson method.

Location of Series Expansion The choice of taking the series expansion around λ 0 = 1 is justified using the following two complementary factors:

- · ˆ = 1 λ corresponds to the identity transformation, and hence having λ 0 = 1 as the point where the series expansion is taken, facilitates its recovery if this is optimal.
- · It equivocates to assuming the least about the nature of the deviations from normality in the sample statistics, since it avoids biasing the form of the series expansion for the loss towards solutions favoring ˆ λ &lt; 1 or ˆ λ &gt; 1 .

## Order of Normalization and Power Transform Steps

Applying the power transform after the normalization step is beneficial, because having zero mean and unit variance activations simplifies several terms in the computation of ˆ λ , as shown in Appendix D, and improves numerical stability.

No Additional Learned Parameters Despite having increased normality in the features, this came at no additional cost in terms of the number of learnable parameters relative to existing normalization techniques.

Test Time In the case where normality normalization is used to augment batch normalization, in addition to computing global estimates for µ and σ 2 , we additionally compute a global estimate for λ . These are obtained using the respective training set running averages for these terms, analogously with batch normalization. At test time, these global estimates µ, σ 2 , λ are used, rather than the test minibatch statistics themselves.

## 4.2. Additive Gaussian Noise with Scaling

Normality normalization applies regularizing additive random noise to the output of the power transform; a step which is also motivated through the information-theoretic principles described in Subsection 2.1, and whose regularizing effect is magnified by having gaussianized pre-activations.

For each input indexed by i ∈ { 1 , . . . , N } , during training 4 we have y i = x i + z i · ξ · s , where x i is the i -th input's postpower transform value, z i ∼ N (0 1) , , ξ ≥ 0 is the noise factor, and s = 1 N ∥ x -¯ x ∥ 1 represents the zero-centered norm of the post-power transform values, normalized by the sample size N .

Importantly, scaling each of the sampled noise values z i for a given channel's minibatch 5 by the channel-specific scaling factor s , leads to an appropriate degree of additive noise for each of the channel's constituent terms x i . This is significant because for a given minibatch, each channel's norm will differ from the norms of other channels.

Furthermore, we treat s as a constant, so that its constituent terms are not incorporated during backpropagation. 6 This is significant because the purpose of s is to scale the additive random noise by the minibatch's statistics, and not for it to contribute to learning directly by affecting the gradients of the constituent terms.

Note that we employ the ℓ 1 -norm for x rather than the ℓ 2 -norm because it lends itself to a more robust measure of dispersion (Pham-Gia &amp; Hung, 2001).

Algorithm 1 provides a summary of normality normalization.

## 5. Experimental Results &amp; Analysis

## 5.1. Experimental Setup

For each model and dataset combination, M = 6 models were trained, each with differing random initializations for the model parameters. Wherever a result is reported numerically, it is obtained using the mean performance and one standard error from the mean across the M runs. The best performing models for a given dataset and model combination are shown in bold. Wherever a result is shown graphically, it is displayed using the mean performance, and its 95 %confidence interval when applicable. The training

4 We do not apply additive random noise with scaling at test time.

5 For clarity the present discussion assumes the case where normality normalization is used to augment batch normalization. However, the discussion applies equally to other normalization layers, such as layer, instance, and group normalization.

6 Implementationally, this is done by disabling gradient tracking when computing these terms.

## Algorithm 1 Normality Normalization

Input:

u

=

{

u

N

i

}

i

=1

Output:

v = { v i } N i =1

Learnable Parameters: γ, β

Noise Factor:

ξ ≥ 0

## Normalization :

$$& \i \text{normalization} \colon \\ & \hat { \mu } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } u _ { i } \\ & \hat { \sigma } ^ { 2 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left ( u _ { i } - \hat { \mu } \right ) ^ { 2 } \\ & h _ { i } = \frac { u _ { i } - \hat { \mu } } { \sqrt { \hat { \sigma } ^ { 2 } + \epsilon } } \\ & - \quad - \quad \cdot \quad \cdot$$

## Power Transform and Scaled Additive Noise :

$$\hat { \lambda } = 1 - \frac { \mathcal { L } ^ { \prime } ( h ; \lambda = 1 ) } { \mathcal { L } ^ { \prime \prime } ( h ; \lambda = 1 ) }$$

$$x _ { i } = \psi \left ( h _ { i } ; \hat { \lambda } \right )$$

$$- \frac { \mathcal { L } ^ { \prime } ( \mathfrak { h } ; \lambda = } } { \mathcal { L } ^ { \prime \prime } ( \mathfrak { h } ; \lambda = } } \\ \flat \left ( h _ { i \colon \hat { \lambda } } \right )$$

with gradient tracking disabled:

$$\text{$w$ given values} \\ \bar { x } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } x _ { i } \\ s = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | x _ { i } - \bar { x } | \\ \text{sample} \, z _ { i } \sim \mathcal { N } \left ( 0, 1 \right ) \\ y _ { i } = x _ { i } + z _ { i } \cdot \xi \cdot s \\ \cdot \, \dashrightarrow \quad - \quad$$

Affine Transform :

$$\underline { v _ { i } = \gamma \cdot y _ { i } + \beta }$$

configurations of the models 7 are outlined in Appendix F.

## 5.2. Generalization Performance

We evaluate layer normality normalization (LayerNormalNorm) and layer normalization (LayerNorm) on a variety of models and datasets, as shown in Table 1. A similar evaluation is done for batch normality normalization (BatchNormalNorm) and batch normalization (BatchNorm), shown in Table 2.

Normality Normalization is Performant LayerNormalNorm generally outperforms LayerNorm across multiple architectures and datasets, with a similar trend holding between BatchNormalNorm and BatchNorm.

Effective With and Without Data Augmentations Normality normalization is effective for models trained with (Table 1) and without (Table 2) data augmentations. This is of value in application areas such as time series analysis and fine-grained medical image analysis, where it is often not clear what data augmentations are appropriate.

7 Code is made available at https://github.com/ DanielEftekhari/normality-normalization .

Table 1. Validation accuracy across several datasets for a vision transformer (ViT) architecture (see training details for model specification), when using LayerNormalNorm (LNN) vs. LayerNorm (LN). Data augmentations were employed during training.

| DATASET        | LN           | LNN          |
|----------------|--------------|--------------|
| SVHN           | 94.61 ± 0.31 | 95.78 ± 0.21 |
| CIFAR10        | 89.97 ± 0.16 | 91.18 ± 0.13 |
| CIFAR100       | 66.40 ± 0.42 | 70.12 ± 0.22 |
| FOOD101        | 73.25 ± 0.19 | 79.11 ± 0.09 |
| I MAGENET TOP1 | 71.54 ± 0.16 | 75.25 ± 0.07 |
| I MAGENET TOP5 | 89.40 ± 0.11 | 92.23 ± 0.04 |

Table 2. Validation accuracy for several ResNet (RN) architecture and dataset combinations, when using BatchNormalNorm (BNN) vs. BatchNorm (BN). No data augmentations were employed during training.

| DATASET     | MODEL   | BN           | BNN          |
|-------------|---------|--------------|--------------|
| CIFAR10     | RN18    | 88.89 ± 0.07 | 90.41 ± 0.09 |
| CIFAR100    | RN18    | 62.02 ± 0.17 | 65.82 ± 0.11 |
| STL10       | RN34    | 58.82 ± 0.52 | 63.86 ± 0.45 |
| TINYIN TOP1 | RN34    | 58.22 ± 0.12 | 60.57 ± 0.14 |
| TINYIN TOP5 | RN34    | 81.74 ± 0.16 | 83.31 ± 0.13 |
| CALTECH101  | RN50    | 72.60 ± 0.35 | 74.71 ± 0.51 |
| FOOD101     | RN50    | 61.15 ± 0.44 | 63.51 ± 0.33 |

Figure 1. Normality normalization is effective for various normalization layers. Validation accuracy for ResNet34 architectures evaluated on the STL10 dataset. Each bar represents the performance of the ResNet34 architecture, when using the given normalization layer across the entire network. INN: InstanceNormalNorm, IN: InstanceNorm, GNN: GroupNormalNorm, GN: GroupNorm, BNN: BatchNormalNorm, BN: BatchNorm.

<!-- image -->

## 5.3. Effectiveness Across Normalization Layers

Figure 1 demonstrates the effectiveness of normality normalization across various normalization layer types. Here we further augmented group normalization (GroupNorm) to group normality normalization (GroupNormalNorm), and instance normalization (InstanceNorm) to instance normal-

ity normalization (InstanceNormalNorm).

Table 3 furthermore contrasts decorrelated batch normalization (Huang et al., 2018) with its augmented form decorrelated batch normality normalization, providing further evidence that normality normalization can be employed wherever normalization layers are conventionally used.

Table 3. As in Table 2, but for models using decorrelated BatchNormalNorm (DBNN) vs. decorrelated BatchNorm (DBN).

| DATASET   | MODEL   | DBN          | DBNN         |
|-----------|---------|--------------|--------------|
| CIFAR10   | RN18    | 90.66 ± 0.05 | 91.50 ± 0.03 |
| CIFAR100  | RN18    | 65.11 ± 0.06 | 67.53 ± 0.10 |
| STL10     | RN34    | 66.76 ± 0.29 | 69.36 ± 0.14 |

## 5.4. Effectiveness Across Model Configurations

Network Width Figure 2 shows that BatchNormalNorm outperforms BatchNorm across varying WideResNet architecture model widths. Of particular note is that BatchNormalNorm shows strong performance even in the regime of relatively small network widths, whereas BatchNorm's performance deteriorates. This may indicate that for smallwidth networks, which do not exhibit the Gaussian process limiting approximation attributed to large-width networks (Neal, 1996; Lee et al., 2017; Jacot et al., 2018; Lee et al., 2019), normality normalization provides a correcting effect. This could, for example, be beneficial for hardware-limited deep learning applications.

Figure 2. Normality normalization is effective for small and large width networks. Validation accuracy on the STL-10 dataset for WideResNet architectures with varying width factors when controlling for depth of 28 , when using BatchNormalNorm vs. BatchNorm.

<!-- image -->

Network Depth Figure 3 shows that BatchNormalNorm outperforms BatchNorm across varying model depths. This suggests normality normalization is beneficial both for small and large-depth models. Furthermore, the increased benefit to performance for BatchNormalNorm in deeper networks suggests normality normalization may correct for an increased tendency towards non-normality as a function of model depth.

Figure 3. Normality normalization is effective for networks of various depths. Validation accuracy on the STL10 dataset for WideResNet architectures with varying depths when controlling for a width factor of 2 , when using BatchNormalNorm vs. BatchNorm.

<!-- image -->

Training Minibatch Size Figure 4 shows that BatchNormalNorm maintains a high level of performance across minibatch sizes used during training, which provides further evidence for normality normalization's general effectiveness across a variety of configurations.

Figure 4. Normality normalization is effective across minibatch sizes used during training. Validation accuracy for ResNet18 architectures evaluated on the CIFAR10 dataset, with varying minibatch sizes used during training, when using BatchNormalNorm vs. BatchNorm.

<!-- image -->

## 5.5. Normality of Representations

Figure 5 shows representative Q-Q plots (Wilk &amp; Gnanadesikan, 1968), a method for assessing normality, for postpower transform feature values when using BatchNormalNorm, and post-normalization feature values when using BatchNorm. Figure 6 shows an aggregate measure of normality across model layers, derived from several Q-Q plots corresponding to different channel and minibatch combinations. The figures correspond to models which have been

Figure 5. Representative Q-Q plots of feature values for models trained to convergence with BatchNormalNorm (post-power transform, top row) vs. BatchNorm (post-normalization, bottom row), measured for the same validation minibatch (ResNet34/STL10). Left to right: increasing layer number. The x-axis represents the theoretical quantiles of the normal distribution, and the y-axis the sample's ordered values. A higher R 2 value for the line of best fit signifies greater gaussianity in the features. BatchNormalNorm induces greater gaussianity in the features throughout the model, in comparison to BatchNorm.

<!-- image -->

Figure 6. The average R 2 values for each model layer, derived from several Q-Q plots (see Figure 5) corresponding to 20 channel and 10 validation minibatch combinations, for models trained to convergence with BatchNormalNorm vs. BatchNorm. The plot demonstrates that normality normalization leads to greater gaussianity throughout the model layers.

<!-- image -->

trained to convergence. The plots demonstrate that normality normalization leads to greater gaussianity throughout the model layers.

## 5.6. Comparison of Additive Gaussian Noise With Scaling and Gaussian Dropout

Here we contrast the proposed method of additive Gaussian noise with scaling described in Subsection 4.2, with two other noise-based techniques.

The first is Gaussian dropout (Srivastava et al., 2014), where for each input indexed by i ∈ { 1 , . . . , N } , during training we have y i = x i · ( 1 + z i · √ 1 -p p ) , where x i is the i -th input's post-power transform value, z i ∼ N (0 1) , , and p ∈ (0 , 1] is the retention rate.

The second is additive Gaussian noise, but without scaling by each channel's minibatch statistics. This corresponds to the proposed method in the case where s is fixed to the mean of a standard half-normal distribution , i.e. 8 s = √ 2 π across all channels; and thus does not depend on the channel statistics.

Figure 7 shows that additive Gaussian noise with scaling is more effective than Gaussian dropout, giving further evidence for the novelty and utility of the proposed method. It is also more effective than additive Gaussian noise (without scaling), which suggests the norm of the channel statistics plays an important role when using additive random noise.

Figure 7. Additive Gaussian noise with scaling is effective. Validation accuracy for models trained with BatchNormalNorm (ResNet34/STL10), but with varying forms for the noise component of the normalization layer.

<!-- image -->

One reason why additive Gaussian noise with scaling may work better than Gaussian dropout, is because the latter scales activations multiplicatively, which means the effect of the noise is incorporated in the backpropagated errors. In contrast, the proposed method's noise component does not contribute to the gradient updates directly, because it is additive. This would suggest that models trained with normality normalization obtain higher generalization performance, because they must become robust to misattribution of gradient values during backpropagation, relative to the corrupted activation values during the forward pass.

## 5.7. Effect of Degree of Gaussianization

Here we consider what effect differing degrees of gaussianization have on model performance, as measured by the

8 This value for s precisely mirrors how it is calculated in Algorithm 1, since recall s = 1 N ∑ N i =1 | x i -¯ x | .

proximity of the estimate ˆ λ to its MLE solution, which was given by Equation 5.

We control the proximity to the MLE solution using a parameter α ∈ [0 , 1] in the following equation

$$\hat { \lambda } = 1 - \alpha \frac { \mathcal { L } ^ { \prime } ( h ; \lambda = 1 ) } { \mathcal { L } ^ { \prime \prime } ( h ; \lambda = 1 ) }, \quad \quad ( 6 ) ^ { \frac { \ddot { \mathbb { S } } } { \mathbb { S } } \, \mathbb { s } }$$

where α = 1 corresponds to the MLE, and decreasing values of α reduce the strength of the gaussianization.

Figure 8 demonstrates that the method's performance increases with increasing α , and obtains its best performance for α = 1 . This provides further evidence that increasing gaussianity improves model performance.

Figure 8. Increasing gaussianity improves model performance. Validation accuracy for models trained using BatchNormalNorm without noise (ResNet50/Caltech101), and with varying strengths for the gaussianization (parameterized by α ) when applying the power transform.

<!-- image -->

## 5.8. Controlling for the Power Transform and Additive Noise Components

Figure 9 demonstrates that both components of normality normalization - the power transform, and the additive Gaussian noise with scaling - each contribute meaningfully to the increase in performance for models trained with normality normalization.

## 5.9. Additional Experiments &amp; Analysis

We next describe several additional experiments and analyses which serve to further demonstrate the effectiveness of normality normalization, and to substantiate the applicability of the motivation we presented in Section 2.

Normality normalization induces robustness to noise at test time. Appendix A.1 &amp; Table 4 demonstrate that models trained using normality normalization are more robust to random noise at test time. This substantiates the applicability of the noise robustness framework presented in

Figure 9. Controlling for the effects of the power transform and the additive Gaussian noise with scaling components. Each subplot shows the performance of models trained with BatchNormalNorm with the use of additive Gaussian noise with scaling (BNN), and without (BNN w/o noise), while using BatchNorm (BN) as a baseline. Subplot titles indicate the model and dataset combination.

<!-- image -->

Motivation Subsection 2.1, and consequently of the benefit of gaussianizing learned representations.

Speed benchmarks. Appendix A.2 &amp; Figure 10 show that normality normalization increases runtime; with a large deviation at training time than at test time.

Normality normalization uniquely maintains gaussianity throughout training. Appendix A.3 &amp; Figure 11 show that, at initialization, layer pre-activations are close-to Gaussian regardless of the normalization layer employed; but that only models trained with normality normalization maintain gaussianity throughout training.

Normality normalization induces greater feature independence. Appendix A.4 &amp; Figure 12 demonstrate that normality normalization imbues models with greater joint normality and greater independence between channel features, throughout the layers of a model. This is of value in context of the benefit feature independence is thought to provide, which was explored in Motivation Subsection 2.3.

## 6. Related Work &amp; Future Directions

Power Transforms Various power transforms have been developed (Box &amp; Cox, 1964; Yeo &amp; Johnson, 2000) and their properties studied (Hernandez &amp; Johnson, 1980), for increasing normality in data. Box &amp; Cox (1964) defined a power transform which is convex in its parameter, but is only defined for positive variables. Yeo &amp; Johnson (2000) presented an alternative power transform which was further-

more defined for the entire real line, preserved the convexity property with respect to its parameter for positive input values (concavity in the parameter for negative input values), and additionally addressed skewed input distributions.

It is worth noting that many power transforms were developed with the aim of improving the validity of statistical tests relying on the assumption of normality in the data. This is in contrast with the present work, which uses an information-theoretic motivation for gaussianizing.

Gaussianization Alternative approaches to gaussianization, such as transforms for gaussianizing heavy-tailed distributions (Goerg, 2015), iterative gaussianization techniques (Chen &amp; Gopinath, 2000; Laparra et al., 2011), and copulabased gaussianization (Nelsen, 2006), offer interesting directions for future work. Non-parametric techniques for gaussianizing, for example those using quantile functions (Gilchrist, 2000), may not be easily amenable to the deep learning setting where models are trained using backpropagation and gradient descent.

Usage in Other Normalization Layers Works which have previously assumed normality in the pre-activations to motivate and develop their methodology, for example as seen in normalization propagation (Arpit et al., 2016), may benefit from normality normalization's explicit gaussianizing effect. It would also be interesting to explore what effect gaussianizing model weights might have, for example by using normality normalization to augment weight normalization (Salimans &amp; Kingma, 2016).

Adversarial Robustness It would be interesting to tie the present work with those suggesting robustness to ℓ 2 -norm constrained adversarial perturbations increases when training with Gaussian noise (Cohen et al., 2019; Salman et al., 2019). Furthermore, it has been suggested that adversarial examples and images corrupted with Gaussian noise may be related (Ford et al., 2019). This may indicate gaining robustness to Gaussian noise not only in the inputs, but throughout the model, can lead to greater adversarial robustness.

However, gaussianizing activations and training with Gaussian noise, may only be a defense in the distributional sense; exact knowledge of the weights (and consequently of the activation values), as is often assumed in the adversarial robustness setting, is not captured by the noise-based robustness framework, which is only concerned with distributional assumptions over the activation values. Nevertheless it does suggest that, on average, greater robustness may be attainable.

Neural Networks as Gaussian Processes Neal (1996) showed that in the limit of infinite width, a single layer neural network at initialization approximates a Gaussian process. This result has been extended to the multi-layer setting by Lee et al. (2017), and Jacot et al. (2018); Lee et al. (2019) suggest the Gaussian process approximation may remain valid beyond network initialization. However, these analyses still necessitate the infinite width limit assumption.

Subsequent work showed that batch normalization lends itself to a non-asymptotic approximation to normality throughout the layers of neural networks at initialization (Daneshmand et al., 2021). Given its gaussianizing effect, layers trained with normality normalization may be amenable - throughout training - to a non-asymptotic approximation to Gaussian processes. This could help further address the disparity in the analysis of neural networks in the infinite width limit, for example as in mean-field theory, with the finite width setting (Joudaki et al., 2023).

## 7. Conclusion

Among the methodological developments that have spurred the advent of deep learning, their success has often been attributed to their effect on the model's ability to learn and encode representations effectively, whether in the activations or in the weights. This can be seen, for example, by considering the importance of initializing model weights suitably, or by the effect different activation functions have on learning dynamics.

Seldom has a prescription for precisely what distribution a deep learning model should use to effectively encode its activations, and exactly how this can be achieved, been investigated. The present work addresses this - first by motivating the normal distribution as the probability distribution of choice, and subsequently by materializing this choice through normality normalization.

It is perhaps nowhere clearer what representational benefit normality normalization provides, than when considering that no additional learnable parameters, relative to existing normalization layers, were introduced. This highlights and precisely controls for the effect of - the importance of encouraging models to encode their representations effectively.

We presented normality normalization: a novel, principledly motivated, normalization layer. Our experiments and analysis comprehensively demonstrated the effectiveness of normality normalization, in regards to its generalization performance on an array of widely used model and dataset combinations, its consistently strong performance across various common factors of variation such as model width, depth, and training minibatch size, its suitability for usage wherever existing normalization layers are conventionally used, and through its effect on improving model robustness to random perturbations.

## Impact Statement

This work is of general interest to the machine learning and broader scientific community. There are many potential applications of the work, among which it is difficult to judiciously highlight one such possible application, in place of another.
<|endofpaper|>