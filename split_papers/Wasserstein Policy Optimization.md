<|startofpaper|>
## Wasserstein Policy Optimization

David Pfau 1 Ian Davies 1 Diana Borsa 1 Jo˜ ao G. M. Ara´jo u 1 Brendan Tracey 1 Hado van Hasselt 1

## Abstract

We introduce Wasserstein Policy Optimization (WPO), an actor-critic algorithm for reinforcement learning in continuous action spaces. WPO can be derived as an approximation to Wasserstein gradient flow over the space of all policies projected into a finite-dimensional parameter space (e.g., the weights of a neural network), leading to a simple and completely general closed-form update. The resulting algorithm combines many properties of deterministic and classic policy gradient methods. Like deterministic policy gradients, it exploits knowledge of the gradient of the action-value function with respect to the action. Like classic policy gradients, it can be applied to stochastic policies with arbitrary distributions over actions - without using the reparameterization trick. We show results on the DeepMind Control Suite and a magnetic confinement fusion task which compare favorably with state-of-theart continuous control methods.

## 1. Introduction

Reinforcement learning has made impressive progress in controlling complex domains with continuous actions such as robotics (Haarnoja et al., 2024), magnetic confinement fusion (Degrave et al., 2022) and game playing (Farebrother &amp;Castro, 2024). A significant portion of this progress can be attributed to policy optimization methods, which directly optimize the parameters of a policy by stochastic gradient ascent on the expected long term return (e.g., Schulman et al., 2015a; 2017; Abdolmaleki et al., 2018). While unbiased estimates of the policy gradient can be computed directly from returns (Williams, 1992), many practical deep reinforcement learning algorithms for continuous control employ an actor-critic approach (Barto et al., 1983), which also estimates a value function to diminish the variance of

1 Google DeepMind, London, UK. Correspondence to: David Pfau &lt; pfau@google.com &gt; .

Proceedings of the 41 st International Conference on Machine Learning , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

the policy update. The majority of these methods use a policy update derived from the classic policy gradient theorem for stochastic policies (Sutton et al., 1999), which applies to both discrete and continuous action spaces.

One notable exception is the class of algorithms derived from the deterministic policy gradient (DPG) theorem (Silver et al., 2014; Lillicrap, 2015). These use information about the gradient of the value in action space to update the policy, but are limited to deterministic policies, hindering exploration. Adding in stochasticity can be difficult to tune, and extensions that learn the variance (Heess et al., 2015; Haarnoja et al., 2018) rely on the reparameterization trick, which limits the class of policy distributions that can be used.

Here we present a new policy gradient method that uses gradients of the action value but can learn arbitrary stochastic policies. The method can be derived from the theory of Wasserstein gradient flows (Ambrosio et al., 2008), and projecting the nonparametric flow onto the space of parametric functions (e.g. neural networks) leads to a simple closed-form update that combines elements of stochastic and deterministic policy gradients. We give a high-level illustration of this in Fig. 1. The update is natural to implement as an actor-critic method. We call the resulting method Wasserstein Policy Optimization (WPO).

The paper is organized as follows. First, we review policy optimization for continuous control, and Wasserstein gradient flows. We then show how the latter can be used to derive WPO when applied to the former. We review prior work in this area and analyze the dynamics of WPO and how it relates to other policy gradients. We describe several extensions needed to convert WPO into a practical and competitive deep reinforcement learning algorithm. Finally, we compare its performance against several baseline methods on the DeepMind Control Suite (Tassa et al., 2018; Tunyasuvunakool et al., 2020) and a task controlling magnetic coils in a simulated magnetic confinement fusion device (Tracey et al., 2024). On a task where the task dimension can be varied arbitrarily, we find that WPO learns faster than baseline methods by an amount that increases as the task dimension grows, suggesting it may work well in very high dimensional action spaces. An open-source implementation

Figure 1. Conceptual illustration of how WPO combines elements of stochastic and deterministic policy gradient methods. Left: 'classic' policy gradient. Samples are taken from a stochastic policy. Each sample contributes a scalar Q π ( s a , ) factor to the gradient. Middle: deterministic policy gradient (DPG). A deterministic action is chosen and the policy gradient depends on the gradient of Q π ( s a , ) . Right: Wasserstein policy optimization (WPO). Samples are taken from a stochastic policy, as in classic policy gradient, but depend on the gradient of Q π with respect to the action, as in DPG.

<!-- image -->

of WPO is available in Acme (Hoffman et al., 2020) . 1

et al., 2018). This is especially critical for achieving good performance with deep reinforcement learning.

## 2. Method

## 2.1. Policy Gradient and Iteration Methods

We aim to find a policy π ( a s | ) which is a distribution over a continuous space of actions a ∈ R n conditioned on a state s ∈ R m that maximizes the expected long-term discounted return J [ π ] = E a t ∼ π, s t ∼P [ ∑ T t =0 γ r t t ] for a Markov decision process with transition distribution P ( s ′ | s a , ) and reward function r ( s a , ) , where r t is shorthand for r ( s t , a t ) . While 'classic' policy gradient methods come in many forms (Schulman et al., 2015b), they are mostly variants of the basic update ∇ J θ [ π θ ] = E a t ∼ π, s t ∼P [ ∑ T t =0 Ψ t ∇ θ log π θ ( a t | s t ) ] and only differ in the choice of scalar Ψ t . If Ψ t is entirely a function of the returns like ∑ T t ′ = t γ t ′ r t ′ , then the policy can be optimized directly, as in REINFORCE (Williams, 1992). If Ψ t is the action-value function Q π ( s t , a t ) or some transformation of the action-value function like the advantage function, or softmax of the action-value, then both a policy and value function must be estimated simultaneously, which is standard in actor-critic methods (Barto et al., 1983). Additionally, it is common to add some form of trust region or regularization to prevent the policy update from changing too much on any step (Schulman et al., 2015a; 2017; Abdolmaleki

1 https://github.com/google-deepmind/acme .

Note that the implementation in Acme is not the version used for the experiments in this paper. However we have run this implementation on DeepMind Control Suite tasks and found qualitatively similar performance to that reported here.

One notable exception to this is deterministic policy gradients (DPG) (Silver et al., 2014; Lillicrap, 2015; BarthMaron et al., 2018), which can be seen as the limit of the policy gradient update as the policy becomes deterministic. These algorithms were developed as early as the 1970s (Werbos, 1974) and were known under names such as 'action-dependent heuristic dynamic programming' (Prokhorov &amp; Wunsch, 1997) or 'gradient ascent on the value' (van Hasselt &amp; Wiering, 2007). As the name suggests, this only applies to deterministic policies π that map state vectors to a single action vector, so a t = π ( s t ) . Then the deterministic policy gradient has the form ∇ J θ [ π θ ] = E s t ∼P [ ∇ a t Q ( s t , a t ) ∇ θ π ( s t )] , where ∇ θ π ( s t ) is the Jacobian of the deterministic policy. The appearance of the gradient of the value in action space in the policy gradient is potentially useful in high-dimensional action spaces. However, the restriction to deterministic policies makes exploration difficult. While there are extensions to DPG for learning a stochastic policy, such as SVG(0) (Heess et al., 2015) and SAC (Haarnoja et al., 2018), they rely on the reparameterization trick, which limits their generality.

Separately, policy iteration algorithms (Howard, 1960; Sutton &amp; Barto, 2018) estimate the value of the current policy and then derive an improved policy based on these values. Iterating this converges to the optimal values and policy. These algorithms do not necessarily follow the gradient of the value. A modern example is MPO (Abdolmaleki et al., 2018), which updates its parametric policy (i.e., a neural network) towards the exponentiation of the current action values, using a target policy π a s ( | ) ∝ exp( Q ( s ,a ) τ )

and minimizing a KL divergence with respect to this target.

## 2.2. Wasserstein Gradient Flows

It is possible to derive a true policy gradient for stochastic policies with a form similar to DPG, based on Wasserstein gradient flows. The theory of gradient flows is discussed in depth in (Ambrosio et al., 2008), and we review the relevant results here. Although the discussion here is fully general, we keep the notation as close to the RL notation as possible to avoid confusion.

Consider an arbitrary functional J [ π ] of a probability density π ( a ) , and let δ J δπ be the functional derivative of J . Then the following PDE will converge to a minimum of J :

$$\frac { \partial \pi } { \partial t } = - \nabla _ { \mathfrak { a } } \cdot \left ( \pi \left ( - \nabla _ { \mathfrak { a } } \frac { \delta \mathcal { J } } { \delta \pi } \right ) \right ) \quad \ \ ( 1 ) \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

If we think of -∇ a δ J δπ as a velocity field, then this may be recognizable as the continuity equation from fluid mechanics, the drift term in the Fokker-Planck equation, or in a machine learning context as one way of writing the expression for the likelihood of a neural ODE (Chen et al., 2018). Problems in optimal transport can also be framed in terms of the continuity equation (Benamou &amp; Brenier, 2000). The 2-Wasserstein distance, conventionally defined as

$$W _ { 2 } ^ { 2 } ( \pi _ { 0 }, \pi _ { 1 } ) = \inf _ { \rho \in \Gamma ( \pi _ { 0 }, \pi _ { 1 } ) } \int \rho ( \mathbf a, \mathbf b ) | | \mathbf a - \mathbf b | | _ { 2 } d \mathbf a d \mathbf b \ ( 2 )$$

where Γ( π , π 0 1 ) is the space of all joint distributions with marginals π 0 and π 1 , can also be expressed as:

$$W _ { 2 } ^ { 2 } ( \pi _ { 0 }, \pi _ { 1 } ) = \inf _ { v _ { t } \in V ( \pi _ { 0 }, \pi _ { 1 } ) } \int _ { 0 } ^ { 1 } \mathbb { E } _ { \mathfrak { a } \sim \pi _ { t } } \left [ | | v _ { t } ( \mathfrak { a } ) | | _ { 2 } \right ] d t \ \ ( 3 ) \quad \ \text{which}$$

where V ( π , π 0 1 ) is the set of velocity fields v t such that, if π t = π 0 at t = 0 and ∂π t ∂t = -∇ · a ( π t ( ∇ a v t ( a ))) , then π t = π 1 at t = 1 . From this dynamic formulation, it can be shown that the flow in Eq. 1 is the steepest descent on the functional J in the space of probability densities under the metric induced locally by the 2-Wasserstein distance.

In our case J [ π ] is the expected return in an MDP and we want to maximize this quantity. The functional derivative (wrt π ) has the form:

$$\frac { \delta \mathcal { J } } { \delta \pi } ( s, \mathbf a ) = \frac { 1 } { 1 - \gamma } Q ^ { \pi } ( s, \mathbf a ) d ^ { \pi } ( s ) \quad \ \ ( 4 ) \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

where d π ( s ) = (1 -γ ) ∑ t γ t Pr( s t = ) s is the discounted state occupancy function. A derivation is given in Sec. A.1 of the appendix. This is the functional generalization of the policy gradient in the tabular setting (see Agarwal et al. (2021), Eq. 7). The d π ( s ) term typically emerges implicitly in the update as the sampling frequency when interacting with the environment, and (1 -γ ) -1 is just a constant. In what follows, we focus on per-state updates where these terms do not appear.

## 2.3. Wasserstein Policy Optimization

To convert the theoretical results in the previous section into a practical algorithm, we need to approximate the PDE in Eq. 1 with an update to a parametric function such as a neural network. Starting at π θ from the parametric family of functions, for a given infinitesimal dt and flow ∂π θ ∂t , we solve for the choice of infinitesimal ∆ θ which minimizes the KL divergence between the original distribution and the updated distribution. This gives the optimal direction to cancel out the flow, hence the minus sign in the KL below. It is well known that the KL divergence is approximated locally by a quadratic form with the Fisher information matrix (Pascanu &amp; Bengio, 2013):

$$\text{daily by a quadratic form with the Fisher information} \\ \text{matrix} (Pascanu & Bengio, 2013): \\ \text{D_KL} & \left [ \pi _ { \theta } \Big | \Big | \pi _ { \theta } + \frac { \partial \pi _ { \theta } } { \partial t } d t - \nabla _ { \theta } \pi _ { \theta } \Delta \theta \right ] \\ & \approx \begin{pmatrix} d t \\ - \Delta \theta \end{pmatrix} ^ { T } \begin{pmatrix} \mathcal { F } _ { t t } & \mathcal { F } _ { t \theta } ^ { T } \\ \mathcal { F } _ { t \theta } & \mathcal { F } _ { \theta \theta } \end{pmatrix} \begin{pmatrix} d t \\ - \Delta \theta \end{pmatrix} \\ \mathcal { F } _ { t t } & = \mathbb { E } _ { \pi } \left [ \frac { \partial \log \pi _ { \theta } ( \mathfrak { a } | s ) } { \partial t } \end{pmatrix} \\ \mathcal { F } _ { t \theta } & = \mathbb { E } _ { \pi } \left [ \frac { \partial \log \pi _ { \theta } ( \mathfrak { a } | s ) } { \partial t } \nabla _ { \theta } \log \pi _ { \theta } ( \mathfrak { a } | s ) \right ] \\ & = \int \frac { \partial \pi _ { \theta } ( \mathfrak { a } | s ) } { \partial t } \nabla _ { \theta } \log \pi _ { \theta } ( \mathfrak { a } | s ) d \mathfrak { a } \\ \mathcal { F } _ { \theta \theta } & = \mathbb { E } _ { \pi } \left [ \nabla _ { \theta } \log \pi ( \mathfrak { a } | s ) \nabla _ { \theta } \log \pi _ { \theta } ( \mathfrak { a } | s ) ^ { T } \right ] \\ \text{which is minimized at } \Delta \theta = \mathcal { F } _ { \theta \theta } ^ { - 1 } \mathcal { F } _ { t \theta }.$$

which is minimized at ∆ = θ F -1 θθ F tθ .

Next, we derive a simple expression for F tθ . We plug in ∂π θ ∂t = -∇ · a ( π θ ( a s | ) ∇ a Q π ( a s , )) for ascent on J [ π ] :

$$\text{at, if } \quad \nu \quad \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \\ \text{then} \quad \mathcal { F } _ { t \theta } & = \int \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf a | s ) \frac { \partial \pi _ { \theta } ( \mathbf a | s ) } { \partial t } d \mathbf a } \\ \mathbf r \, & \text{the} \quad \mathbf r \, & \quad = - \int \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf a | s ) \nabla _ { \mathbf a } \cdot \left ( \pi _ { \theta } ( \mathbf a | s ) \nabla _ { \mathbf a } Q ^ { \pi } ( s, \mathbf a ) \right ) d \mathbf a } \\ \text{d we } \quad \frac { - \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot } { - \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot }.$$

This derivation is expanded in Sec. A.2 in the appendix, but mainly follows from integration by parts. This leads to a simple closed-form update for parametric policies which we call the Wasserstein Policy Optimization update :

$$\tilde { \theta } _ { t + 1 } = \theta _ { t } + \mathcal { F } _ { \theta \theta } ^ { - 1 } \mathbb { E } _ { \pi } \left [ \nabla _ { \theta } \nabla _ { \mathbf a } \log \pi ( \mathbf a | \mathbf s ) \nabla _ { \mathbf a } Q ^ { \pi } ( \mathbf s, \mathbf a ) \right ] \ ( 6 )$$

A similar parametric approximation was derived for finding the ground state energy of quantum systems (Neklyudov et al., 2023). While this application is quite different, the derivation follows from the same theory of gradient flows, arriving at a very similar expression. To the best of our

Figure 2. Concrete WPO updates for a single-variate normal policy for two different action-value functions. In the left and middle plots we consider Q a ( ) = -a / 2 2 , with an obvious optimum at a = 0 , and a policy with µ = σ = 1 . The left plot shows the gradient on the mean at several sampled actions. These are averaged to produce the update, moving the mean towards the optimal action. The expected WPO update, as shown in the middle plot, is then ∆ µ = -µ and ∆ σ = -σ . Both the mean and variance will decrease (or, more generally, move probability mass to the optimal action), as shown. Conversely, in the right plot we consider Q a ( ) = a / 2 2 then ∆ µ = µ and ∆ σ = σ , and both the mean and variance will increase, as shown. In all cases, the expected value of the resultant policy is increased.

<!-- image -->

knowledge we are the first to use this update for policy optimization in reinforcement learning. This somewhat idealized update requires the full Fisher information matrix and lacks some extensions needed to make policy optimization methods work in practice. We describe how to extend this update into a practical deep RL algorithm in Sec. 5.

amount to a different way to update the critic, whereas WPO is a novel way to update the actor.

## 3. Related Work

The general idea of using the Wasserstein metric in reinforcement learning has appeared in many forms. In Abdullah et al. (2019) the 2-Wasserstein distance is used to define Wasserstein Robust Reinforcement Learning, an algorithm that trains a policy which is robust to misspecification of the environment for which it is being trained. The Wasserstein distance appears here as a constraint on the transition dynamics of the environment, rather than as a way of defining learning dynamics. In Moskovitz et al. (2020), the Wasserstein metric is used locally as an alternative preconditioner in place of the Fisher information matrix in natural policy gradient (Kakade, 2001), while the conventional form of the policy gradient is still used. The Wasserstein distance has also been explored as an alternative to the KL divergence as a regularizer to prevent the policy from changing too quickly (Pacchiano et al., 2020).

The Wasserstein metric has also been used to define distances between state distributions, for various ends (Ferns et al., 2004; He et al., 2021; Castro et al., 2022), whereas we are focused on gradient flows in the space of actions .

In a combination of the above aims, (Metelli et al., 2019) and (Likmeta et al., 2023) use the Wasserstein distance to define a method to propagate uncertainty across state-action pairs in the Bellman Equation, with the aim of using that quantified uncertainty to better deal with the explorationexploitation trade-off inherent to online reinforcement learning. In the context of actor-critic methods, this would

The idea of policy optimization as a Wasserstein gradient flow has appeared a few times in the literature. Richemond &amp;Maginnis (2018) show that policy optimization with an entropy bonus can be written as a PDE similar to the expression in Sec. 2.2 with an additional diffusion term, but they stop short of deriving an update for parametric policies. The work of Zhang et al. (2018) also formulates policy optimization as a regularized Wasserstein gradient flow, but arrives at a more complicated DPG-like update that uses the reparameterization trick, similarly to SVG(0) and SAC.

## 4. Analysis

## 4.1. Gaussian Case

To better understand the mechanics of the WPO update, we analyze the simplified case where the policy is a singlevariate normal distribution and the policy and value are not state-dependent. In this case we have:

$$\begin{array} { c } \dots \quad \kappa \quad \dots \quad \dots \quad \dots \\ \nabla _ { \mu } \log \pi ( a ) = \frac { a - \mu } { \sigma ^ { 2 } } = - \nabla _ { a } \log \pi ( a ) \,, \\ \nabla _ { \sigma } \log \pi ( a ) = \frac { ( a - \mu ) ^ { 2 } } { \sigma ^ { 3 } } - \frac { 1 } { \sigma } \,, \\ \mathcal { F } _ { \mu \mu } = \frac { 1 } { \sigma ^ { 2 } } \,, \mathcal { F } _ { \sigma \sigma } = \frac { 2 } { \sigma ^ { 2 } } \,, \mathcal { F } _ { \sigma \mu } = 0 \,. \\ \dots \quad \omega \quad \omega \dots \quad \omega \dots \quad \omega \dots \quad \dots \quad \dots \end{array}$$

Let ∆ µ θ and ∆ σ θ be the contributions to the update due to the gradients of the mean and variance, respectively, such that ∆ = ∆ θ µ θ +∆ σ θ . For the mean, we then have

$$\Delta _ { \mu } \theta & = \mathcal { F } _ { \mu \mu } ^ { - 1 } \mathbb { E } _ { \pi } \left [ \nabla _ { a } Q ( a ) \nabla _ { a } \nabla _ { \mu } \log \pi ( a ) \nabla _ { \theta } \mu \right ] \\ & = \sigma ^ { 2 } \mathbb { E } _ { \pi } \left [ \nabla _ { a } Q ( a ) \nabla _ { a } \frac { a - \mu } { \sigma ^ { 2 } } \nabla _ { \theta } \mu \right ] \\ & = \mathbb { E } _ { \pi } \left [ \nabla _ { a } Q ( a ) \nabla _ { \theta } \mu \right ] \,.$$

Figure 3. Concrete WPO learning for a one dimensional mixture of Gaussians policy for the non-concave action-value function Q a ( ) = -1 100 a 4 + a 2 . The left plot shows the action-value function and mixture of Gaussians policy. In the middle plot we show the evolution of the policy under a standard policy gradient update, both with samples from the policy and the change in the means of each mixture component. On the left we show the same evolution for WPO. WPO converges faster, is more stable around the optimum, and converges to both optima if the policy is initialized symmetrically.

<!-- image -->

<!-- image -->

This is very similar to the DPG update: ∇ µ Q µ ( ) ∇ θ µ , except that we take the gradient of the action value at a = µ , rather than sampling a ∼ π . While this similarity holds for the normal distribution, it is not necessarily true in general.

with WPO as well.

For the variance we have:

$$\text{For the variance we have} \colon \\ \Delta _ { \sigma } \theta & = \mathcal { F } _ { \sigma \sigma } ^ { - 1 } \mathbb { E } _ { \pi } \left [ \nabla _ { a } Q ( a ) \nabla _ { a } \nabla _ { \sigma } \log \pi ( a ) \nabla _ { \theta } \sigma \right ] \\ & = \frac { \sigma ^ { 2 } } { 2 } \mathbb { E } _ { \pi } \left [ \nabla _ { a } Q ( a ) \nabla _ { \sigma } \nabla _ { a } \log \pi ( a ) \nabla _ { \theta } \sigma \right ] \\ & = - \frac { \sigma ^ { 2 } } { 2 } \mathbb { E } _ { \pi } \left [ \nabla _ { a } Q ( a ) \nabla _ { \sigma } \frac { a - \mu } { \sigma ^ { 2 } } \nabla _ { \theta } \sigma \right ] \\ & = \mathbb { E } _ { \pi } \left [ \frac { a - \mu } { \sigma } \nabla _ { a } Q ( a ) \nabla _ { \theta } \sigma \right ] \\ \text{This update is a little more complicated. but on inspection it}$$

This update is a little more complicated, but on inspection it also is quite intuitive. The variance increases when ( a -µ ) and ∇ a Q a ( ) have the same sign. So, when we sample an action, then we increase the variance if the gradient of Q with respect to that action points even further away from the mean. If the gradient points back towards the mean, we will instead decrease the variance. Some special cases are illustrated in Figure 2.

Stochastic extensions of DPG such as SVG(0) (Heess et al., 2015) and soft actor-critic (SAC) (Haarnoja et al., 2018) use the reparameterization trick and define a gradient E η [ ∇ a Q ( s a , ) ∇ θ π ( s , η )] , where π denotes the action as deterministic function of the state, as well as a noise term η . For instance, we can define π ( s , η ) = µ ( s ) + σ ( s ) ◦ η , where η ∼ N ( 0 , I) and ◦ denotes the elementwise product. The mean and variance updates for the single-variate normal distribution are then respectively E [ ∇ a Q a ( ) ∇ θ µ ] and E [ η ∇ a Q a ( ) ∇ θ σ ] . For this choice of parameterization, η = ( a -µ /σ ) , which exactly coincides with WPO. SAC is similar to SVG(0), and extends this by including an entropy bonus to encourage exploration, as well as a mechanism based on double Q-learning (van Hasselt, 2010) to avoid value overestimations. Such extensions could be combined

Note that the natural WPO update is independent of parameterization, while SVG(0) is not, so in general the two updates may still differ. For instance, suppose we have a task that requires non-negative actions, and we decide to use an exponential policy with components a i ∼ exp( a /β i i ) /β i , where β i are learnable scale parameters for each action dimension. The Fisher for this distribution is diagonal with components 1 /β 2 i on the diagonal. Then, the natural WPO update with respect to β is

$$\text{update with respect to } \beta \text{ is} \\ \mathcal { F }$$

We can reparameterize the policy for SVG(0) with a standard exponential η with density p η ( = x ) = exp( -x ) and then define a = β ◦ η . Then, the SVG(0) update is

$$\mathbb { E } _ { \eta } \left [ \nabla _ { a }$$

where we used η = a/β , by definition. This is not generally the same as the WPO update.

Not only does the natural WPO update coincide with DPG augmented with the reparameterization trick in the Gaussian case , it also equals the standard policy gradient. Specifically,

$$\text{case, it also equals the standard policy gradient. Specifically,} \\ \Delta _ { \$$

Figure 4. Results from selected DeepMind Control Suite tasks. Full results are in Fig. 7 in the appendix.

<!-- image -->

where the last line is just the expected policy gradient update for the parameters of the mean. A similar derivation, with the same result, can be done for the variance; this can be found in Appendix A.3. If we add those contributions together, we get the standard policy gradient E π [ Q s, a ( )( ∇ µ log π s, a ( ) ∇ θ µ + ∇ σ log π s, a ( ) ∇ θ σ )] = E π [ Q s, a ( ) ∇ θ log π s, a ( )] .

## 4.2. Mixture of Gaussian Case

These equivalences can be extended to multivariate normal distributions quite straightforwardly. This counterintuitive result suggests that there is essentially only one correct way to update a Gaussian policy, and that the major differences between approaches will only become clear when going beyond simple normal distributions over actions.

Importantly, even if the expected natural WPO update coincides with the expected policy gradient update, the sampled updates could still have dramatically different variance. For instance, consider an action-value function that is (locally) linear in the actions, such that Q ( s a , ) = w s ( ) T a . Then ∇ a Q ( s a , ) = w s ( ) does not depend on the action, and because the action-value gradient is then the same for each sampled action the WPO update for the mean of the policy will have zero variance. In contrast, in this case the standard policy gradient update will have non-zero variance. This observation is consistent with Fig. 1: the variance in the WPO updates will be low when locally all the gradients point roughly in the same direction.

To understand the qualitative differences between different updates, we will have to move beyond the case of Gaussian policies. We consider the case of a one-dimensional mixture of Gaussians policy π a ( ) = ∑ i ρ i N ( a µ , σ | i i ) where ∑ i ρ i = 1 are the mixture weights. Because the assignment of a sample to a mixture component is a discrete latent variable, the reparameterization trick cannot be used exactly (though it can be approximated through relaxations such as the concrete/Gumbel-softmax distribution (Maddison et al., 2017; Jang et al., 2017)) and so we exclusively consider WPO and classic policy gradient and not DPG/SVG(0). In this case, the dynamics of learning are too complex for closed-form results as in the previous section, so we consider an illustrative numerical example, with the actionvalue function Q a ( ) = -1 100 a 4 + a 2 , which has maxima at ± √ 50 . We initialize the policy with two components with ρ i = 0 5 . , σ i = 10 and µ i = ± 1 . We used a batch size of 1024 and learning rate of 0.003. For WPO, rather than approximate the true Fisher information matrix, we rescale the gradients for ρ i , µ i and σ i by σ 2 i , a choice which is justified by the form of the FIM for a single Gaussian. We show results in Fig. 3. Despite being the same in expectation for the Gaussian case, policy gradient and WPO clearly have qualitatively different learning dynamics in the mixture-of-Gaussians case. WPO converges faster, is more stable around the minimum, and finds both local maxima (so long as the mixture components are initialized sym-

metrically). Notably, early in optimization, the variance of the policy actually increases for WPO, when the means of the mixture components are in the region with positive curvature, consistent with the intuition in Fig. 2.

## 5. Implementation

We make two modifications to the update in Eq. 6 to make WPO into a practical method. First, the use of the full natural gradient update is not practical for deep neural networks. It is tempting to simply drop it, but this could lead to serious numerical stability issues, as the ∇ a log π ( a s | ) term in the update blows up as the policy converges to a deterministic result. Note that this is different from classic policy gradient methods, where natural gradient descent accelerates learning but is not strictly necessary (Kakade, 2001).

We could in theory use approximate second order methods such as KFAC (Martens &amp; Grosse, 2015), but we have found that a much simpler approximation works well in practice. First, while the WPO update can be applied to arbitrary stochastic policies, we focus on normally-distributed policies π θ ( a s | ) = N ( a | µ θ ( s ) , Σ ( )) θ s where the covariance is constrained to be diagonal with elements σ 2 i ( s ) . The normal distribution with diagonal covariance has a diagonal Fisher information matrix with 1 σ 2 i for the mean element µ i and 2 σ 2 i for the standard deviation σ i .

Thus, rather than try to approximate the full Fisher information matrix, we may redefine the gradient of the log likelihood such that ¯ ∂ ∂µ i log N ( a | µ, Σ) = σ 2 i ∂ ∂µ i log N ( a | µ, Σ) and ¯ ∂ ∂σ i log N ( a | µ, Σ) = 1 2 σ 2 i ∂ ∂σ i log N ( a | µ, Σ) . While this ignores the contribution of the gradients of µ and Σ with respect to θ in the Fisher information matrix, it provides a qualitatively correct scaling that cancels out the tendency of the likelihood gradient to blow up as Σ → 0 . A similar heuristic was suggested in the original REINFORCE paper (Williams, 1992).

Secondly, we regularize the policy with a penalty on the KL divergence between the current and past policy. Regularization to prevent the policy from taking excessively large steps is standard practice in deep reinforcement learning (e.g., Schulman et al., 2015a; 2017). We similarly found that without regularization, WPO will prematurely collapse onto a deterministic solution and fail to learn, especially on the fusion task from Tracey et al. (2024). While a variety of forms of KL regularization are used in continuous control, we closely follow the form used in MPO (Abdolmaleki et al., 2018). We can express this as a soft constraint, and modify the loss so that at each step we take a step towards solving

$$\max _ { \pi } \mathbb { E } _ { \mathfrak { s }$$

or express it as a hard constraint

$$\begin{array} {$$

which can be implemented by treating the α in the soft constraint as a Lagrange multiplier and performing a dual optimization step. Here ¯ π denotes a previous state of the policy, e.g., a target network. In either case, the gradient of the KL penalty is computed conventionally - only the gradient of the reward uses the approximate Wasserstein gradient flow and variance rescaling. While the KL divergence could be replaced by the Wasserstein distance, which would be more mathematically consistent and has been explored elsewhere as a regularizer (Richemond &amp; Maginnis, 2018; Zhang et al., 2018; Pacchiano et al., 2020), we find the KL divergence works well in practice and leave it to future work to explore alternatives. We also note that while the KL penalty slows the convergence of WPO to a deterministic policy, it does not prevent it, as we show in Fig. 6.

## 6. Experiments

To evaluate the effectiveness of WPO, we evaluate it on the DeepMind Control Suite (Tassa et al., 2018; Tunyasuvunakool et al., 2020), a set of tasks in MuJoCo (Todorov et al., 2012). These tasks vary from one-dimensional actions, like swinging a pendulum, up to a 56-DoF humanoid. We additionally consider magnetic control of a tokamak plasma in simulation, a problem originally tackled by MPO in Degrave et al. (2022). On Control Suite, we compare WPO against both conceptually related and state-of-the-art algorithms which can be used in the same setting: Deep Deterministic Policy Gradient (DDPG; Lillicrap, 2015), Soft-Actor Critic (SAC; Haarnoja et al., 2018), and Maximum a Posteriori Policy Optimization (MPO; Abdolmaleki et al., 2018).

Our training setup is similar to other distributed RL systems (Hoffman et al., 2020): we run 4 actors in parallel to generate training data for the Control Suite tasks, and 1000 actors for the tokamak task. For WPO, the policy update uses sequences of states from the replay buffer, which may come from an old policy, but the actions are resampled from the current policy, making the algorithm effectively off-policy for states but on-policy for actions. MPO is implemented similarly. We used the soft KL penalty in Eq. 7 for Control Suite tasks, as we found the hard KL penalty did not noticeably improve results, but used the hard KL penalty for the fusion task for consistency with previously published results. Separate KL penalties with different weights were put on the mean and variance of the policy, as described in Song et al. (2019). We found that the penalty on the mean had little effect on stability and mainly slowed convergence, while the penalty on the variance noticeably helped stability.

Figure 5. Plots of reward from various agents on combined Humanoid Stand environments. Left to right: 1, 3 and 5 replicated environments (21, 65 and 105 action dimension). Solid line denotes the mean and the shaded region highlights the minimum and maximum over 5 seeds. As the number of replicas grows, WPO is able to learn faster than other methods by a larger margin.

<!-- image -->

Training hyperparameters are listed in Sec. B and an outline of the full training loop is given in Alg. 1 in the appendix. For each RL algorithm, the same hyperparameters were used for each control suite environment.

For the critic update we use a standard n -step TD update with a target network:

$$\delta _ { T D } = \left [ \sum _ { \tau = 0 } ^ { n } r _ { t + \tau } + \gamma ^ { n } \bar { V } ( \mathbf s _ { t + n }$$

where the target value ¯ ( V s t + n ) is approximated by sampling multiple actions from π . This target value could be the mean of the target action value network, the maximum, or something in between. In MPO, the softmax over samples is theoretically optimal, so we use that for Control Suite. We use the maximum for WPO on all Control Suite tasks, as that worked well on the hardest domains, but use the mean for both MPO and WPO on fusion tasks, as the performance is significantly better.

## 6.1. DeepMind Control Suite

Figure 4 shows results from a subset of the DeepMind Control Suite. We selected tasks which show the full range of WPO's performance from excelling to struggling. Learning curves for all Control Suite tasks are in Fig. 7 in the appendix. While no single algorithm uniformly outperformed all others on all tasks, it is notable that DDPG and SAC converged to lower rewards and occasionally struggled to take off on a number of tasks. This is even noticed for relatively low dimensional tasks (particularly with sparse reward). WPO robustly takes off and is in the same range as the best performing method across nearly all tasks. Through hyperparameter tuning, we noticed that SAC is particularly sensitive to the weighting of its entropy objective. This meant that finding generally hyperparameters across all tasks led to difficult trade-offs of stability and performance. We note that WPO and MPO demonstrated greater out-of-the-box generalisation across Control Suite. On the Humanoid CMU

domain, one of the highest dimensional tasks in the Control Suite, neither SAC nor DDPG took off at all, while WPO made progress on all tasks and, in the case of the Walk task, initially learned faster than MPO. On Dog, another high-dimensional domain, WPO converged to roughly the same final reward, but often took longer. Dog is the only domain where the state space is larger than the observation space, suggesting that WPO may have difficulty in partially observed settings. We experimented with several choices of nonlinearity, squashing function (see Sec. C.3 and Fig. 12), KL regularization weight, and critic bootstrapping update for WPO. This is still less than the years of experimentation which has gone into many other popular continuous control algorithms (Huang et al., 2022). This shows that, while DDPG or SAC may struggle to learn on certain tasks, WPO almost always makes some learning progress, and is often comparable to state-of-the-art methods for these tasks, even without significant tuning.

## 6.2. Combined Tasks

Methods that use action-value gradients may perform well in high-dimensional action spaces, but no Control Suite task goes beyond a few dozen dimensions. To evaluate the effectiveness of WPO in higher dimensional action spaces, we construct tasks that consist of controlling many replicas of a Control Suite environment simultaneously with a single centralized agent. Specifically, the action and observation vectors are concatenated, and rewards are combined with a SmoothMin operator (see Eq. 31 in the Appendix), biasing the reward towards lower performing replicas to encourage learning across all replicas. We use the same agent hyperparameters as we used in the standard control suite benchmark no matter the number of replicas. We selected Humanoid Stand as the base task due to its relatively high dimensional action space (21) and moderate difficultly.

The results in figure 5 demonstrate that as the number of replicas grows, WPO continues to be able to learn across

Figure 6. Reward and policy average standard deviation evolution throughout training on the fusion task discussed in Section 6.2

<!-- image -->

<!-- image -->

the environments. In the singleton case, all methods are qualitatively similar except SAC which converges to a lower reward than other learning algorithms. As the number of replicas grows, WPO takes off earlier in training than MPO, which takes off earlier than DDPG, with SAC being the slowest to take off even in the singleton case. This trend becomes more pronounced for larger numbers of replicas, even if the asymptotic performance is similar for all methods. This suggests that for tasks with hundreds of action dimensions, WPO may be able to learn faster than other methods.

## 6.3. Fusion

Degrave et al. (2022) used MPO to discover policies to control the magnetic coils of the TCV tokamak, a toroidal fusion experiment (Duval et al., 2024). The resulting policies successfully ran at 10kHz to hold the plasma stable. Follow-on work (Tracey et al., 2024) explored methods to improve the accuracy of RL-derived magnetic control policies. We run WPO on a variant of the shape 70166 task considered therein, modified to reward higher control accuracy (details in Sec. B.3). This task has 93 continuous measurements, a 19-dimensional continuous action space and lasts 10,000 environment steps, simulating a 1s experiment on TCV.

We compare the performance of WPO (with cube-root squashing as described in Sec. C.3) alongside MPO in Fig. 6. Both MPO and WPO used the default network and settings as in Tracey et al. (2024), including the hard KL regularization as in Eq. 8. WPO achieved a slightly higher reward than MPO. Interestingly, we see a notable difference in the adaptation of the policy variance between the two algorithms. WPO evolves in the direction of a deterministic policy as training evolves, as one would expect for a fully observed environment, while MPO maintains approximately constant policy variance on average. We see similar results in B.3 on the original shape 70166 task, where MPO and WPO achieve similar rewards with different policy adaptation behavior. These results show WPO as a viable alternative to

MPO in this complex real-world task.

## 7. Discussion

We derived a novel policy gradient method from the theory of optimal transport: Wasserstein Policy Optimization. The resulting gradient has an elegantly simple form, closely resembling DPG, but can be applied to learn policies with arbitrary distributions over actions. On the DeepMind Control Suite, WPO is quite robust, performing comparably to state-of-the-art methods on most tasks, despite little tuning. Results on magnetic confinement fusion domains show that WPO can be applied to diverse tasks beyond simulated robotics. Promising initial results suggest it can learn quickly on tasks with over 100 action dimensions. We hope these results can help unlock performant algorithms for very larger control problems, and inspire researchers to develop new, challenging high-dimensional benchmarks in continuous control with hundreds of action dimensions or more.

While we chose a particular instantiation of WPO for the experiments, the WPO update is general and can be the foundation for many different implementations. We have only begun to scratch the surface of how WPO can be applied, and we are hopeful that extensions such as non-Gaussian policies or more advanced critic updates (Bellemare et al., 2017) could fully exploit the advantages of this new method. This could greatly increase the performance and scope of problems to which WPO can be applied.
<|endofpaper|>