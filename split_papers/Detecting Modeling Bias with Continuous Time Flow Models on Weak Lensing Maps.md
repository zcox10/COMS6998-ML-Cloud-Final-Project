<|startofpaper|>
## Detecting Modeling Bias with Continuous Time Flow Models on Weak Lensing Maps

Kangning Diao, a,b Biwei Dai, c and Uroˇ s Seljak b,d,e

- a Department of Astronomy, Tsinghua University, Beijing, 100084, China
- b Berkeley Center for Cosmological Physics, University of California, Berkeley, CA 94720, USA
- c School of Natural Sciences, Institute for Advanced Study, 1 Einstein Drive, Princeton, New Jersey 08540, USA
- d Physics Division, Lawrence Berkeley National Lab, 1 Cyclotron Road, Berkeley, CA 94720, USA
- e Department of Physics, University of California, Berkeley, CA 94720, USA

E-mail: dkn20@mails.tsinghua.edu.cn, biwei@ias.edu, useljak@berkeley.edu

Abstract. Simulation-based inference provides a powerful framework for extracting rich information from nonlinear scales in current and upcoming cosmological surveys, and ensuring its robustness requires stringent validation of forward models. In this work, we recast forward model validation as an out-of-distribution (OoD) detection problem within the framework of machine learning (ML)-based simulation-based inference (SBI). We employ probability density as the metric for OoD detection, and compare various density estimation techniques, demonstrating that field-level probability density estimation via continuous time flow models (CTFM) significantly outperforms feature-level approaches that combine scattering transform (ST) or convolutional neural networks (CNN) with normalizing flows (NFs), as well as NFbased field-level estimators, as quantified by the area under the receiver operating characteristic curve (AUROC). Our analysis shows that CTFM not only excels in detecting OoD samples but also provides a robust metric for model selection. Additionally, we verified CTFM maintains consistent efficacy across different cosmologies while mitigating the inductive biases inherent in NF architectures. Although our proof-of-concept study employs simplified forward modeling and noise settings, our framework establishes a promising pathway for identifying unknown systematics in the cosmology datasets.

## Contents

## 1 Introduction

1

| 2   | Methods and Datasets                                                      | Methods and Datasets                                                      | Methods and Datasets                                                      |   3 |
|-----|---------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|-----|
|     | 2.1                                                                       | Out-of-Distribution detection as a consistency test                       | Out-of-Distribution detection as a consistency test                       |   3 |
|     | 2.2                                                                       | Probability density as the test statistics                                | Probability density as the test statistics                                |   4 |
|     |                                                                           | 2.2.1                                                                     | Feature-level density estimation                                          |   5 |
|     |                                                                           | 2.2.2                                                                     | Field-level density estimation                                            |   5 |
|     | 2.3                                                                       | Weak lensing maps                                                         | Weak lensing maps                                                         |   7 |
|     |                                                                           | 2.3.1                                                                     | Dark matter only maps                                                     |   7 |
|     |                                                                           | 2.3.2                                                                     | Baryon effects                                                            |   8 |
| 3   | Results                                                                   | Results                                                                   | Results                                                                   |   8 |
|     | 3.1                                                                       | Testing the field-level density estimator on Gaussian random fields       | Testing the field-level density estimator on Gaussian random fields       |   8 |
|     | 3.2                                                                       | Detecting BCM maps as OoD                                                 | Detecting BCM maps as OoD                                                 |   9 |
|     | 3.3                                                                       | Performance at different cosmologies                                      | Performance at different cosmologies                                      |  11 |
|     | 3.4                                                                       | OoD as a model selection                                                  | OoD as a model selection                                                  |  12 |
|     | 3.5                                                                       | Impact of resolution and survey area                                      | Impact of resolution and survey area                                      |  13 |
|     | 3.6                                                                       | Normalizing flows v.s. Continuous-time Flow model                         | Normalizing flows v.s. Continuous-time Flow model                         |  14 |
| 4   | Conclusion                                                                | Conclusion                                                                | Conclusion                                                                |  15 |
| A   | Technical specifications of machine learning models employed in this work | Technical specifications of machine learning models employed in this work | Technical specifications of machine learning models employed in this work |  20 |
|     | A.1                                                                       | CNN feature compressor                                                    | CNN feature compressor                                                    |  20 |
|     | A.2                                                                       | RealNVP                                                                   | RealNVP                                                                   |  21 |
|     | A.3                                                                       | U-Net in CTFM                                                             | U-Net in CTFM                                                             |  21 |
|     | A.4                                                                       | GLOW                                                                      | GLOW                                                                      |  22 |

## 1 Introduction

Cosmology has entered an era of precision science, with current models predicting a wide range of observables at sub-percent level accuracy. Concurrently, the next generation of telescopes promises to deliver unprecedented volumes of high-quality, multi-observable data. Among these, weak gravitational lensing (WL) [1, 2], which is the subtle distortion of light from distant galaxies induced by intervening large-scale structures, stands out as a key probe for mapping the total matter distribution in the universe [e.g. 3-6]. Upcoming surveys by facilities such as LSST [7], Euclid [8], and Roman [9] are expected to revolutionize our understanding of cosmic origins, composition, and evolution.

A variety of summary statistics have been developed to analyze WL data, beginning with the traditional N-point correlation functions [10-14]. However, these methods are often hampered by issues such as incomplete information capture [15], the proliferation of statistical coefficients, large variances, and a higher sensitivity to outliers. To address these issues, researchers have proposed alternative approaches including correlation functions computed on transformed or marked fields [16, 17], peak counts [18, 19], void statistics [20], Minkowski

functionals [21, 22], scattering transforms (ST) [23-27], and features extracted via convolutional neural networks (CNN) and other neural network architectures [28-33]. Typically, the likelihoods associated with these summary statistics are modeled using either multivariate Gaussian approximations or simulation-based inference (SBI), yet these approaches remain susceptible to their ad hoc nature and potential information loss. Recently, the advent of advanced machine learning models and increased computational power has led to the development of normalizing flows for field-level likelihood modeling [e.g. 34, 35], offering significant improvements over traditional feature-level methods.

Despite these advances, a critical challenge persists: the robustness of the forward models that underpin the training data. Variations among different hydrodynamical simulations and baryon models, none of which have converged to a single, universally accepted description [e.g. 36], can introduce biases. Models trained on one simulation may perform poorly when applied to data from another [37], and there is no guarantee that any current forward model accurately represents the real universe. It is therefore imperative to detect whether a forward model deviates from actual observations. Questions naturally arise, such as which features are reliable, and which may be compromised by unmodeled or inaccurately modeled effects? How can we identify these discrepancies in high-dimensional, complex data? In the context of field-level inference, these issues are particularly important. The richer the information incorporated into the inference pipeline, the greater the risk of inadvertently including features that are poorly modeled, potentially leading to biased posteriors [e.g. Figure 5 in 38] or overconfident constraints.

Beyond their impact on inference, these biases also signal gaps in our understanding of the underlying physics. Discrepancies between forward models and observations may arise from diverse sources-including cosmological evolution, complex astrophysical processes, and unaccounted observational effects, and thus merit thorough investigation. Detecting such biases can be framed as a consistency test: given that the full range of forward model outputs forms a statistical distribution, the task becomes one of determining whether an observation is a member of that distribution, which is often called the out-of-distribution (OoD) detection. When the likelihood L is Gaussian, -2 log L follows the χ 2 -distribution and is widely used to assess how well the model fits the data in current survey analysis [e.g. 39-41]. Recently, [38] generalizes the test to high-dimensional field-level inference where the likelihood is no longer Gaussian. They employed wavelet decomposition to segregate information by scale, leveraging the relative robustness of large-scale structures in modeling, and used normalizing flows to learn the corresponding distributions at each scale. Moreover, continuous time flow models (CTFM), such as diffusion models [42-44] and flow matching (FM) models [45], have emerged as state-of-the-art techniques for learning high-dimensional distributions. These methods have seen extensive application in the generation and inference of various cosmological fields [e.g. 46-50]. In this work, we explore the use of CTFM for bias detection at the field level and compare its performance against normalizing flows applied at both the field and feature levels.

This paper is organized as follows. In Section 2 we detail our detection methodology, while Section 3 presents our primary results. We validate our field-level density estimation method on Gaussian random field in Section 3.1, and test the performance of different density estimation methods in Section 3.2. The consistency of our results across different cosmological models is verified in Section 3.3, and Section 3.4 discusses the application of out-of-distribution detection metrics for model selection, and Section 3.5 examines the scalability of our approach to large survey volumes. We further analyze the challenges posed by the inductive bias

Figure 1 . An illustration of OoD detection pipeline. A probability distribution function (PDF) of InD sample test statistics is calculated and shown in blue solid line, and an empirical threshold in grey dashed line is chosen to identify an OoD sample with t ( x ) smaller than the threshold, shown in orange dashed line.

<!-- image -->

inherent in NF models in Section 3.6. Finally, Section 4 summarizes our conclusions. We provides detailed model architectures in Appendix A.

## 2 Methods and Datasets

## 2.1 Out-of-Distribution detection as a consistency test

Detecting biases in forward modeling can be naturally framed as a consistency test between the model and the observation. When the forward model produces a full distribution of possible outputs, bias detection reduces to an OoD test, determining whether the observation is consistent with the model-generated distribution.

Following the posterior predictive test formalism [51], we introduce an arbitrary test statistic t ( x ), where x represents either simulated or observed data. Given an observation x obs and its inferred posterior distribution p θ ( | x obs ), the posterior predictive distribution is defined as

$$p ( x _ { \text{rep} } | x _ { \text{obs} } ) & = \int p ( x _ { \text{rep} } | \theta ) p ( \theta | x _ { \text{obs} } ) \, d \theta, & ( 2. 1 )$$

where x rep denotes a replication of the observation. This distribution encapsulates all plausible outputs of the forward model, and samples drawn from it are considered in-distribution (InD).

If the model accurately captures the data, the value of the test statistic t ( x ) computed for the observation should be consistent with the distribution of t ( x ) values computed for

replicated datasets. To quantify this consistency, we calculate the probability

$$p ( t ( x _ { \text{rep} } ) > t ( x _ { \text{obs} } ) ) = \int 1 _ { \{ t ( x _ { \text{rep} } ) > t ( x _ { \text{obs} } ) \} } \, p ( x _ { \text{rep} } | x _ { \text{obs} } ) \, d x _ { \text{rep} },$$

where 1 A ( x ) is the indicator function, which equals 1 if x ∈ A and 0 otherwise.

In practice, we draw N samples { x rep ,i } N i =1 from the posterior predictive distribution and compute the corresponding set of test statistics { t ( x rep ,i ) } N i =1 . If n out of these N samples satisfy t ( x rep ) &gt; t ( x obs ), then

$$p ( t ( x _ { \text{rep} } ) > t ( x _ { \text{obs} } ) ) \approx \frac { n } { N }.$$

Values of p t ( ( x rep ) &gt; t ( x obs )) approaching 0 or 1 indicate that x obs is likely an OoD sample relative to p ( x rep | x obs ), suggesting potential biases in the forward model. To operationalize this test, an empirical threshold t th is often introduced, as is illustrated in Figure 1. For example, if OoD samples are flagged when t ( x obs ) &gt; t th , the false positive rate (FPR) is estimated as p t ( ( x rep ) &gt; t th ); similarly, if a lower threshold is used, the FPR is given by p t ( ( x rep ) &lt; t th ). In cases where both upper and lower bounds are applied, the FPR is approximated as p t ( ( x rep ) &lt; t th , low ) + p t ( ( x rep ) &gt; t th , high ) . The choice of the bound depends on the choice of t ( x ).

This framework provides a rigorous means of assessing whether an observation is consistent with the predicted distribution of outputs, thereby serving as a diagnostic for biases in the forward modeling process.

## 2.2 Probability density as the test statistics

Although the test statistic t ( x ) can be defined arbitrarily, selecting an informative t ( x ) significantly enhances the performance of detecting an anomaly. In the absence of a specific model for the anomaly there is no optimal choice for the test statistic. When searching for unknown unknowns we must therefore rely on test statistics that are generic. In this context, probability density estimation p ( x ) is the most direct and intuitive choice, as values lower than the typical range for InD samples clearly indicate a low likelihood of sampling that particular observation from the modeled distribution. In the Gaussian likelihood setup, the density estimation is closely related to the χ 2 goodness-of-fit test, ln p ( x ) = -χ / 2 2, which is a widely used test for the validity of the model: a large value of χ 2 compared to the number of degrees of freedom indicates that the model is a poor fit to the data, and thus a model misspecification that needs to be addressed. In this paper we generalize this concept to the non-Gaussian likelihoods learned using ML.

However, evaluating probability density in high-dimensional spaces is a non-trivial task. Here, we introduce several approaches to address this challenge. These approaches can be broadly categorized into two groups. The first involves feature-level methods, in which a compressor reduces the high-dimensional sample to a low-dimensional feature vector; the probability density of this vector is then estimated using an NF trained on the forward-modeled dataset. The second group comprises field-level density estimation techniques, where we directly evaluate the density of the high-dimensional observable using two variants of CTFM, the diffusion model and the optimal transport flow matching (OTFM) model. The detailed structures of different kinds of neural networks and corresponding training configurations mentioned from here on are described in Appendix A.

## 2.2.1 Feature-level density estimation

In the feature-level density estimation, the high-dimensional field is first compressed into a low-dimensional feature vector, after which its probability density is estimated using a NF. Specifically, we employ two compressors: the Scattering Transform (ST) coefficients and a CNN trained with the VMIM loss [52], which provide optimal performance for adhoc and learned summary statistics, respectively. The real-valued non-volume preserving transformations (RealNVP) [53] is then used as the NF density estimator.

ST Compressor The ST coefficients, S 1 and S 2 , are defined as

$$I _ { 1 } ( j, l ) & = | x * \Psi \left ( j, l \right ) | * \Phi ( j ), \\ I _ { 2 } ( j _ { 1 }, l _ { 1 }, j _ { 2 }, l _ { 2 } ) & = | | x * \Psi \left ( j _ { 1 }, l _ { 1 } \right ) | * \Psi \left ( j _ { 2 }, l _ { 2 } \right ) | * \Phi ( j _ { 2 } ), \\ S _ { 1 } ( j, l ) & = \left \langle I _ { 1 } ( j, l ) \right \rangle, \\ S _ { 2 } ( j _ { 1 }, l _ { 1 }, j _ { 2 }, l _ { 2 } ) & = \left \langle I _ { 2 } ( j _ { 1 }, l _ { 1 }, j _ { 2 }, l _ { 2 } ) \right \rangle.$$

Here, x is the input field, ∗ denotes the convolution operation, Ψ represents the Morlet wavelet kernel (see e.g. Appendix B of [25] for details) and Φ is the Gaussian kernel to filter all small-scale fluctuations. The index j specifies the scale of the convolutional kernel with smaller j corresponding to more localized kernels while l defines its orientation. We select j = 0-3 and l = 0-3 to cover a broad range of scales and orientations, yielding 16 coefficients for S 1 ( j, l ) and 96 coefficients for S 2 ( j 1 , l 1 , j 2 , l 2 ), for a total feature vector length of 112. The ST coefficients are computed using Kymatio 1 [54].

CNN Compressor For the CNN-based compressor, we utilize a 34-layer ResNet [55] optimized with the VMIM loss:

$$( w ^ { * }, u ^ { * } ) = \arg \min _ { w, u } \mathbb { E } _ { p ( \theta, x ) } [ - \log p _ { u } ( \theta | f _ { w } ( x ) ) ],$$

where an auxiliary RealNVP with parameter u is used to estimate log p u ( θ f | ( x )), f w denotes the CNN with parameter w , and f w ( x ) is the compressed feature vector. The dimension of this feature vector is set to 128 to align with the ST feature vector, and we also performed experiments with different feature dimensionality and verify that this length has a negligible effect in the OoD detection performance.

Once the ST and CNN compressors have been applied, separate RealNVPs are trained on the resulting feature vectors to estimate their probability densities. The NF loss function is defined as

$$w ^ { * } = \arg \min _ { w } \mathbb { E } _ { p ( \theta, y ) } [ - \log p _ { w } ( y | \theta ) ],$$

with y representing the compressed feature vector and w is the parameters of the NF.

## 2.2.2 Field-level density estimation

Unlike feature-level methods, field-level density estimation avoids the information loss inherent in compression, and is therefore expected to perform better. In this work, we employ two variants of CTFM, a diffusion model and an OTFM model, as field-level density estimators.

1 https://github.com/kymatio/kymatio

Given any distribution p ( x ), CTFM estimates the probability density of a sample x by solving an ordinary differential equation (ODE) [56]. Specifically, such an ODE has the form

$$\frac { d } { d t } \phi _ { t } ( x ) = f ( \phi _ { t } ( x ), t ),$$

where the transformation ϕ : [0 , 1] × R d → R d is termed the flow and is initialized as ϕ 0 ( x ) = x . As the time t varies from 0 to 1, the flow transports the original distribution p ( x ) to a tractable target distribution p ϕ ( 1 ( x )), typically a d -dimensional standard Gaussian. For simplicity, we denote ϕ t ( x ) as x t . Assuming that the density p 1 ( x 1 ) is tractable, the density p ( x ) is given by

$$\log p ( x ) = \log p _ { 1 } ( x _ { 1 } ) + \int _ { 0 } ^ { 1 } \nabla \cdot f ( x _ { t }, t ) \, d t. \quad \quad \quad$$

The divergence ∇· f ( x t , t ) is estimated using the Skilling-Hutchinson trace estimator [57, 58]:

$$\nabla \cdot f ( x _ { t }, t ) \approx \mathbb { E } _ { \epsilon \sim p ( \epsilon ) } \left [ \epsilon ^ { T } \nabla f ( x _ {$$

where p ( ϵ ) is a distribution with zero mean and an identity covariance matrix. When f ( x t , t ) is implemented as a differentiable function, the term ϵ T ∇ f ( x t , t ) can be computed via automatic differentiation of ϵ T f ( x t , t ). In our experiments, we adopt a standard Gaussian for p ( ϵ ) and perform Euler integration with 1000 steps for Equation 2.8.

Diffusion models In diffusion models [42, 44], the corresponding ODE is derived from a diffusion process described by the stochastic differential equation (SDE) [43]

$$d x _ { t } = - \frac { 1 } { 2 } \beta _ { t } x _ { t } \, d t + \sqrt { \beta _ { t } } \, d b,$$

where b is the Brownian motion and β t is a predefined, monotonically increasing function satisfying β 0 = 0 and β 1 → + ∞ . This SDE gradually transforms p ( x ) into a standard Gaussian by incrementally adding noise and diminishing the influence of the initial sample. The associated probability flow ODE preserves the marginal density p t ( x t ) for all t [43] and thus fulfills the requirement for transforming p ( x ) into a standard Gaussian, enabling the use of Equation 2.8 to recover p ( x ). This probability flow ODE is given by [43, 59]

$$d x _ { t } = - \frac { 1 } { 2 } \left ( \beta _ { t } x _ { t } + \beta _ { t } \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) \right ) d t.$$

Integration of this ODE requires knowledge of the score function ∇ x t log p t ( x t ), which we approximate using a neural network s w ( x , t ) parameterized by weights w . The network is trained by minimizing the loss [44]

$$w ^ { * } & = \arg \min _ { w } \mathbb { E } _ { t } \mathbb { E } _ { x _ { 0 } \sim p _ { 0 } ( x ) } \mathbb { E } _ { x _ { t } \sim p _ { t } ( x | x _ { 0 } ) } \\ & \left [ \| s _ { w } ( x _ { t }, t ) - \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } | x _ { 0 } ) \| ^ { 2 } \right ].$$

Given that the noise ϵ in Equation 2.10 is Gaussian, we have [60]

$$\begin{array} { c } \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\ & \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } | x _ { 0 } ) = \frac { \sqrt { \alpha _ { t } } x _ { 0 } - x _ { t } } { 1 - \alpha _ { t } }, & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cd. & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd> & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd!. & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd. & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd: & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdts & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd! & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd.: & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdot & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd.: & \cd: & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd: & \cd: & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd.: & \cdts & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdts & \cdts & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd.: & \cd.: & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd.: & \cd> & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cd.: & \cd:/ & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \$$

with α t := exp ( -1 2 ∫ t 0 β s ds ) . Since the probability flow ODE and the SDE share the same marginal distribution p t ( x t ), they also share the same score function. By substituting the

neural network s w ( x , t ) in place of the true score in Equation 2.11, we obtain the diffusion model ODE for estimating p ( x ). Our implementation of the diffusion model, including the neural network, is based on the diffusers 2 package. We choose the linear schedule β t = 20 t in this work.

Optimal Transport Flow Matching The diffusion model ODE defined in Equation 2.11 requires the neural network to predict a non-constant score term over different t , which can pose challenges due to the complexity of the output. Optimal transport (OT) [61] addresses this issue by assuming a constant vector field f ( x t , t ) = x 1 -x 0 , where p ( x 1 ) is a high-dimensional standard Gaussian. This simplification improves fitting f ( x t , t ) with a neural network. However, directly training under this assumption is challenging without explicit ( x 0 , x 1 ) pairs. FM [45] overcomes this limitation by learning the OT vector field f ( x t , t ) without requiring explicit pairings.

In FM, given a sample x 0 drawn from p ( x ), the conditional distribution p t ( x x t | 0 ) is modeled as a Gaussian with mean µ t and standard deviation σ t , i.e.,

$$p _ { t } ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( \mu _ { t }, \sigma _ { t } ),$$

and the marginal density is obtained by

$$p _ { t } ( x _ { t } ) = \int p _ { t } ( x _ { t } | x _ { 0 } ) p ( x _ { 0 } ) \, d x _ { 0 }.$$

By choosing µ 0 = x 0 , σ 0 = 0, µ 1 = 0, and σ 1 = 1 , we ensure that p 0 ( x 0 ) = p ( x ) and p 1 ( x 1 ) is a standard Gaussian. Although an analytical expression for f ( x t , t ) under this setting is difficult to obtain, it can be learned by training a neural network s w ( x , t ) through minimizing the loss [45]

$$w ^ { * } & = \arg \min _ { w } \mathbb { E } _ { t } \mathbb { E } _ { x _ { 0 } \sim p _ { 0 } ( x ) } \mathbb { E } _ { x _ { t } \sim p _ { t } ( x _ { t } | x _ { 0 } ) } \\ & \left [ \| s _ { w } ( x _ { t }, t ) - \frac { d \sigma _ { t } ( x _ { 0 } ) } { \sigma _ { t } ( x _ { 0 } ) d t } ( x _ { t } - \mu _ { t } ) - \frac { d \mu _ { t } ( x _ { 0 } ) } { d t } \| ^ { 2 } \right ]. \\ \dots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots.$$

To align this loss with the OT ODE, one can set µ t = (1 -t ) x 0 and σ t = t , so that x t = (1 -t ) x 0 + t ϵ with ϵ ∼ N (0 , 1 ). Under these choices, Equation 2.15 reduces to

$$w ^ { * } = & \arg \min _ { w } \, \mathbb { E } _ { t } \mathbb { E } _ { x _ { 0 } \sim p _ { 0 } ( x ) } \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0, 1 ) } \\ & \left [ \| s _ { w } ( x _ { t }, t ) - ( \epsilon - x _ { 0 } ) \| ^ { 2 } \right ].$$

Since ϵ and x 1 follow the same distribution, this loss function effectively trains s w ( x t , t ) to approximate the difference x 1 -x 0 . Our implementation of OTFM is based on the torchcfm package [62] . 3

## 2.3 Weak lensing maps

## 2.3.1 Dark matter only maps

The dark-matter-only (DMO) weak lensing (WL) convergence maps used in this study are obtained from [28] and are generated from a suite of 80 N-body simulations under flat ΛCDM cosmologies. In these simulations, the baryon density, Hubble parameter, and spectral index

2 https://huggingface.co/docs/diffusers/index

3 https://github.com/atong01/conditional-flow-matching

are fixed at Ω b = 0 046, . h = 0 72, and . n s = 0 96, respectively, while Ω . m and σ 8 vary around Ω m = 0 26 and . σ 8 = 0 8. .

The N-body simulations employ 512 3 particles in a box of size 240 h -1 Mpc and are run using GADGET-2 [63]. WL convergence maps are computed via ray-tracing using a multiple-lens-plane algorithm [64] on snapshots spanning 0 &lt; z &lt; 1. From each snapshot, an 80 h -1 Mpc slice along the line-of-sight is extracted as a lens plane. By applying random rotations, flips, and shifts to the snapshots, 512 pseudo-independent maps are derived from each simulation. Further details on the map generation process can be found in [28].

The convergence maps are produced at a resolution of 1024 × 1024 pixels and are subsequently downscaled to various resolutions via spatial averaging. The noise considered in this work is galaxy shape noise, modeled as independent Gaussian noise in each pixel with a standard deviation given by

$$\sigma _ { g } = \frac { \sigma _ { \epsilon } } { \sqrt { 2 n _ { g } A _ { \text{pix} } } },$$

where σ ϵ ∼ 0 4 denotes the mean intrinsic ellipticity of galaxies, . n g is the galaxy number density, and A pix is the pixel area. We investigate three noise scenarios corresponding to n g = { 30 50 100 , , } arcmin -2 . The 30 arcmin -2 case represents the targeted noise level for surveys such as LSST or Euclid, while future space missions like Roman may achieve 50 arcmin -2 or higher. The 100 arcmin -2 scenario is an optimistic forecast for forthcoming space-based surveys.

## 2.3.2 Baryon effects

To better model the physical process, baryon models are used to post-process the N-body simulation output [30, 65]. The post-processing first identify all the halos with mass larger than 10 12 M ⊙ , then substitute the halo particles with spherical symmetric analytical density profile. The analytical halo profile derived from the Baryon Correction Model (BCM) [66] represents halos as composed of four components: the central galaxy, bound gas, ejected gas due to AGN feedback, and relaxed dark matter. It is characterized by four free parameters: M c (the halo mass that retains half of the total gas), M 1 0 , (the halo mass corresponding to a galaxy mass fraction of 0.023), η (the maximum distance to which gas is ejected from its parent halo), and β (the logarithmic slope that describes how the gas fraction scales with halo mass). Although this model omits the substructures and non-spherical shapes of halos, it has been argued that the morphological differences between simulated halos and these idealized spherical profiles are statistically negligible relative to the uncertainties in the power spectrum and peak counts measured in an HSC-like survey [65].

These post-processed snapshots go through the same pipeline to produce the BCM convergence map with the same resolution and noise. For each cosmology, 2048 BCM maps are generated, each with distinct baryon parameter { M ,M c 1 0 , , η, β } .

## 3 Results

## 3.1 Testing the field-level density estimator on Gaussian random fields

Gaussian random fields (GRFs) provide an ideal benchmark because their log-likelihood can be computed in closed form. Therefore, we first validates our density estimation formalism on GRFs. We generate GRFs whose power spectrum is a power law with additive white noise,

$$P ( k ) = \left ( k / 5 \right ) ^ { - 2 } + \sigma _ { n },$$

Figure 2 . Left : Distribution of the log-probability densities of Gaussian random fields (GRFs) evaluated with OTFM (dashed curves) and by the analytic expression (solid curves). Different colours correspond to different noise levels σ n . Right : Point-by-point comparison between the OTFM log density and the analytic value. For each noise level, the mean log density inferred analytically has been subtracted, so that only the residual variations are shown.

<!-- image -->

<!-- image -->

Figure 3 . Example probability distribution of test statistics t ( x ) = log p x ( ) of InD and OoD test dataset with noise level n g = 30. From left to right are t ( x ) obtained with different methods.

<!-- image -->

and test three noise amplitudes, σ n ∈ { 1 2 3 , , } . For each σ n we draw 120 000 realisations of , size 64 2 pixels and train an OTFM model; its performance is assessed on an independent set of 1024 samples, yielding the OTFM estimate log p OTFM .

The left panel of Figure 2 compares the distributions of log p OTFM with the analytic log density log p ana . The OTFM successfully recovers the log density of the GRF under various noise settings, with a relative scatter of ∼ 6 due to simplified integration scheme, insufficient noise realizations in Equation 2.9 and insufficient steps. The residual scatter shown in right panel of Figure 2 are almost independent of the noise level, indicating that OTFM delivers stable accuracy across datasets with different signal-to-noise ratios.

The present evaluation employs the simplest Euler integrator with 1000 time-steps. Replacing it with higher-order schemes (e.g. Runge-Kutta) or increasing the step count should further tighten the agreement between OTFM and the analytic benchmark.

## 3.2 Detecting BCM maps as OoD

In this subsection, we present main results from our test problem, where we try to detect the unmodeled baryonic effects when DMO simulations are available. We test the performance of each methods mentioned above on our fiducial cosmology ( σ , 8 Ω ) = (0 82 0 265). m . , . Unless stated otherwise, the InD samples consist of 304 maps of DMO maps at fiducial cosmology, while OoD samples are 1024 BCM maps with different baryon parameters at the same fiducial

Figure 4 . ROC curve of using different log p ( x ) as OoD detection test statistics with the fiducial cosmology and resolution 128 . 2

<!-- image -->

<!-- image -->

<!-- image -->

cosmology. This setting corresponding to an idealized δ -function like posterior, and the reported performance here is the upper limit, as a broadened posterior leads to wider range of distribution of test statistics, therefore worse performance. These maps have a physical resolution of 128 2 pixels. For each methods mentioned before, we compute the probability density of these maps as the t ( x ), because the density is the most natural OoD detector. Examples with noise level n g = 30 are shown in Figure 3, and a well-behaved OoD detector is expected to generate clearly separable density distributions for InD and OoD sets.

For field-level detection methods like diffusion model and OTFM model, we train the model unconditionally, as the total number of cosmologies is limited making it difficult to capture the dependency of cosmologies accurately. The output log density log p ( x ) is thus an average of the likelihood over the cosmology prior and can thus be viewed as the Bayesian evidence of the observation. For higher computational efficiency, we limited the input and output dimensions to 64 2 pixels, and the density of a 128 2 map is the average of 4 64 2 map density cut from the original map.

The metric used to evaluate the performance is Receiver Operating Characteristic (ROC) curve, which is a graphical tool used in classification tasks to illustrate the performance of a binary classifier across different decision thresholds. It plots the True Positive Rate (TPR) against the FPR for different classification thresholds, effectively showing the trade-off between correctly identifying positive instances and incorrectly flagging negative ones.

The Area Under the ROC Curve (AUROC) summarizes this plot into a single number ranging from 0 to 1. A larger AUROC indicates better overall classifier performance, with 0.5 representing a random guess and 1.0 indicating a perfect model. This metric is particularly useful for comparing different , as it remains invariant to class distribution and threshold selection.

For feature-level detection methods like CNN and ST, we adopt conditional RealNVP to estimate their conditional log density log p ( y | θ ) with compressed features y . The conditional probability performs better because of the additional information θ , which is confirmed by our tests: the conditional density at the best fit value of θ gives better AUROC than the average over θ , but the difference is small and we will ignore it in the following. We first compute the feature vectors from the 128 2 maps, and use these vectors as the input of the NF and the true cosmology as the condition θ .

4 In this work, the InD dataset is drawn from DMO maps at the Max-A-Priori (MAP) cosmology of the

Table 1 . OoD Detection AUROC of using different log p ( x ) as OoD detection test statistics with the fiducial cosmology and resolution 128 2 .

| Method               | n g = 30   | n g = 50   | n g = 100   |
|----------------------|------------|------------|-------------|
| Diffusion            | 0.85       | 0.90       | 0.94        |
| OT FlowMatching      | 0.87       | 0.92       | 0.95        |
| MultiscaleFlow[38] 4 | ≳ 0 . 65   | -          | -           |
| CNN                  | 0.54       | 0.55       | 0.60        |
| ST coefficient       | 0.73       | 0.77       | 0.81        |

Table 2 . OTFM AUROC at different cosmologies

| ( σ 8 , Ω m )         |   n g = 30 |   n g = 50 |   n g = 100 |
|-----------------------|------------|------------|-------------|
| (0.82,0.268)(default) |       0.87 |       0.92 |        0.95 |
| (0.717, 0.315)        |       0.93 |       0.96 |        0.98 |
| (0.766, 0.275)        |       0.88 |       0.93 |        0.96 |
| (0.768, 0.264)        |       0.87 |       0.92 |        0.94 |
| (0.875, 0.259)        |       0.86 |       0.9  |        0.93 |
| (0.864, 0.234)        |       0.83 |       0.87 |        0.9  |
| (0.842, 0.217)        |       0.79 |       0.85 |        0.87 |

The ROC curve of different methods is shown in Figure 4, and the corresponding AUROC is presented in Table 1. The field-level detections significantly outperform featurelevel detection, even if extra cosmology information is provided to feature-level methods. We further compared our field-level results to [38], concluding that our CTFM field-level approaches is much better than field-level NFs. For two CTFM field-level methods, OTFM is slightly better than diffusion as expected, for its regularized temporal evolution helps stabilize the neural network.

In the feature-level analysis, we found that ST coefficients significantly outperforms CNN, even if they have similar dimensions. Given that CNN is optimized by maximizing the cosmological information in the compressed features, it focuses on cosmology-dependent features and ignores other information which may be important for detecting model misspecification. This explains why CNN leads to more constraining power on cosmological parameters than ST coefficients in previous study [67], yet it performs poorly in OoD detection here in our experiment. Note that while CNN learned statistics are not able to distinguish InD data and OoD data effectively, it does not mean that CNN analysis is robust to modeling bias. Its inference can still be biased by model misspecification, since the latter can be degenerate with cosmological information.

## 3.3 Performance at different cosmologies

To confirm the robustness of our method across different cosmologies, we tested the OTFM density on six additional cosmologies, with their ( σ , 8 Ω ) values and results presented in m Table 2. The InD samples and OoD samples are constructed in the same way as in Section 3,

OoD maps, which gives a higher p ( x | θ ) than the true cosmology in our case for OoD maps. However, the difference in p ( x | θ ) between MAP and true cosmology InD maps is an order of magnitude smaller than the difference between InD and OoD p ( x | θ ). As a result, the AUROC obtained with true cosmology InD samples would be slightly higher but not significantly so.

Figure 5 . Left : the log density of the n g, true = 30 mock observation with different models are shown in solid line, and the dashed line represents the log density of InD samples for each model, different colors represents different models. Right : The test statistics distribution of n g, true = 30 samples with different model M .

<!-- image -->

except the cosmology is different. By examining the AUROC values, we find that OTFM achieves consistently high performance for cosmologies that are degenerate with the default cosmology (first four entries in Table 2). However, in the last three additional cosmologies, where the amplitude of fluctuations decreases, the detection accuracy declines. This is likely due to smaller-scale signals being increasingly obscured by Gaussian noise, diminishing the contrast between InD and OoD samples.

## 3.4 OoD as a model selection

A density-based OoD detector measures how much the density of an observation deviates from the typical value in the training set specified by a given model. Likewise, when we evaluate the deviation from multiple models using the same detection methods, this deviation serves as a metric of how well each model fits the observation. A typical example would be having access to multiple simulations that disagree with each other [37]. In this case we may use density estimation as the model selection, choosing the simulation that gives the highest density of the data.

To validate the performance of generative model likelihoods as model selectors, we test the OTFM model in a noise miscalibration scenario, as OTFM is the best test statistics according to the results in Section 3. Specifically, we consider the mock observation x mock , which consists of 304 DMO maps with shape noise n g, ture = 30arcmin -2 at fiducial cosmology. Our three candidate models are DMO simulations with n g = { 25 30 35 , , } arcmin -2 , and their final noise standard deviations are σ n ≈ { 0 0345 0 0315 0 0291 . , . , . } . We train an OTFM model for each candidate model M to serve as the density estimator log p ( x |M ).

We first calculated the log density of x mock with different models and find their the log density distribution overlaps with each other as is shown in the left panel of Figure 5, making it is hard to identify the best model. However, the log density distribution of x mock for mis-specified models significantly differ from that of InD samples, shed a light to model comparison with the difference to the typical values of the InD samples. In this case, we can select the best model by the false positive rate of identify them as OoD, or equivalently the log density deviation from the log density of typical set samples. Therefore, the selection metric is then defined as

$$t ( x _ { \text{mock} } | \mathcal { M } ) = | \log p ( x _ { \text{mock} } | \mathcal { M } ) \ - \ \mathbb { E } _ { x \sim p _ { \text{ln} } ( x ) } \left ( \log p ( x | \mathcal { M } ) \right ) |,$$

Figure 6 . ROC curve of OTFM density with different field-of-view and resolution, as is shown in the legend.

<!-- image -->

where p InD ( x ) is approximated by a dataset of model output. Intuitively, t grows if (i) the model is overly broad-so its typical-set likelihood is lower than that of the x mock -or (ii) the model assigns substantially lower likelihood to x mock than to its own typical samples. The right panel of Figure 5 shows the distribution of t ( x mock |M ) for different n g models, demonstrating that the correct n g model attains a significantly smaller t ( x |M ). With this metric OTFM successfully selects the correct model for all 304 x mock maps.

## 3.5 Impact of resolution and survey area

Our experiments in section 3 is performed on mock weak lensing maps with small fieldof-view (3 5deg . × 3 5deg) and fixed resolution (pixel size 1 64 arcmin). . . This map area is significantly smaller compared to current and upcoming weak lensing surveys which normally span thousands of deg , and the resolution is also much lower compared to some of the current 2 WL analysis that aims to study baryonic effect [68]. We expect our model performance to improve significantly with higher resolution [38] and larger area, and we explicitly verify this in this section.

In Figure 6 we show the OoD performance using OTFM field-level likelihood with different map area and resolutions. Here we follow the same OoD and InD setups as in Section 3, but we divide the map to subfields with 64 2 pixels, and model the likelihood of each subfields independently. The total likelihood of the map is approximated as the product of the likelihood of all subfields, ignoring the correlation between different subfields. When testing the method on larger area (60 deg ), we randomly sample 5 maps of size 3 5deg 2 . × 3 5deg . and combine their likelihood. We see from figure 6 that increasing the resolution improves the performance of detecting baryonic physics, with AUROC increases from 0.87 to 0.94. Furthermore, the OTFM model was able to perfectly identify all OoD samples when the map size is increased to 60 deg 2 .

Figure 7 . Samples of InD set, smoothed set, scaled set and masked set, respectively.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

NF

0.03

0.02

0.01

0.00

8000

InD

Smoothed

Scaled

Masked

8500

9000

9500

10000

t x

<!-- image -->

(

)

Figure 8 . Log probability density of field-level normalizing flow and OTFM model for InD samples and different kind of OoD samples.

## 3.6 Normalizing flows v.s. Continuous-time Flow model

Although we employed CTFM in this work, NF models with certain architectures, such as GLOW [69], are also capable of scaling to high-dimensional field-level density estimation. However, it has been observed that these NF models exhibit a preference for specific spatial structures, introducing systematic bias in the density estimation [70]. This bias renders NF suboptimal for OoD detection tasks. In our study, we confirmed this conclusion on our dataset and compared the performance of OTFM on the same test cases.

The InD samples consist of 304 DMO WL maps of size 128 2 generated under the fiducial cosmology. We then constructed three distinct types of strong spatial biases as OoD samples, as shown in Figure 7:

- · Smoothing : We smooth the field using a 2D Gaussian kernel with a width of (1.5 pix, 1.5 pix) along the (x,y) directions.
- · Scaling : We multiply the field values by a factor of 0.5.
- · Masking : We mask the field using a chessboard pattern. Specifically, we divide the field into 32 × 32 blocks, each of size 4 × 4 pixels, and evenly mask half of these blocks.

Each of these OoD sets contains 304 samples. We then employed both the GLOW and OTFM models to estimate the probability density as test statistics t ( x ); the corresponding PDFs are shown in Figure 8. It is evident that GLOW misestimates the OoD sets, whereas OTFM remains robust and successfully identifies all OoD samples. This limitation originates from the inductive bias inherent in the GLOW architecture, and the loss is less important

Density

than the architecture [70]. Although alternative test statistics, such as the 2-norm of the score function ∥ ∂ log p ( x ) /∂ x ∥ which serves as a metric for typicality, can mitigate this issue [71, 72], they still suffer from certain biases [73]. In conclusion, our findings confirm that CTFM generally outperforms NF, as it has less inductive biases from the model architecture, thus, the model can learn more information from the loss function without being limited by the model inductive bias.

## 4 Conclusion

Future cosmological surveys are poised to deliver high-quality, high-dimensional observables, and ML-based SBI has demonstrated remarkable efficacy in extracting information from complex, field-level data. Nonetheless, the fidelity of forward modeling remains pivotal for accurate posterior estimation. In this work, we reframe the validation of forward models as an OoD detection problem-a null test to determine whether an observation originates from the distribution produced by the forward model. Among the various summary statistics examined, the field-level probability density estimated via CTFM achieves the best performance.

Our test set comprises DMO WL maps and BCM WL maps generated under identical cosmologies. We evaluate the probability density using test statistics computed at different levels and by various methods. At the feature level, we employ ST and CNN as feature extractors and train NFs separately to assess the density. At the field level, we utilize two variants of CTFM-diffusion models and OTFM-as density estimators. Notably, CTFM significantly outperforms the feature-level approaches as well as the NF-based field-level density estimator MultiscaleFlow [38], as quantified by AUROC. This suggests that there is a lot of information at the field level that is lost if one compresses the data into O (100) features.

Moreover, our findings indicate that the magnitude of deviation from typical models not only serves as a null test for consistency with forward modeling but also functions as a metric for model selection. We further demonstrate that increasing the map resolution and map size significantly enhances the OoD performance - when the survey area reaches 60deg 2 we achieves a perfect characterization of our test OoD samples with n g = 30 shape noise. We also confirm that CTFM maintains consistent performance across different cosmologies. Finally, our results corroborate previous findings regarding the inductive bias of NF toward certain spatial patterns, while also revealing that CTFM is more robust against biases stemming from model architecture.

As a proof of concept, this study employs simplified forward modeling and a limited OoD test set. Future work could address these limitations by incorporating more realistic baryon models and observational systematics. Additionally, this work detects the modeling bias without addressing explicitly the interpretability of the results. Future work can focus on various splits of the data to identify the regions of strongest OoD detection, such as splits by scale, density etc.

## Acknowledgments

KD is supported by the National SKA Program of China (grant No. 2020SKA0110401) and NSFC (grant No. 11821303). BD acknowledges support from the Ambrose Monell Foundation, the Corning Glass Works Foundation Fellowship Fund, and the Institute for Advanced Study. This work is also supported by U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under Contract No. DE-AC02-05CH11231

at Lawrence Berkeley National Laboratory to enable research for Data-intensive Machine Learning and Analysis. KD thanks Tao Jing, Ce Sui, Bhuvnesh Jain, Matias Zaldarriaga, Richard Grumitt and Xiaosheng Zhao for for their helpful insights and comments.
<|endofpaper|>