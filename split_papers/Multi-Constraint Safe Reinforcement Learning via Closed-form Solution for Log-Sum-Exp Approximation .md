<|startofpaper|>
## Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions

## Chenggang Wang

CGWANG-AUV@SJTU.EDU.CN

Shanghai Jiao Tong University, Shanghai, China

Xinyi Wang

XINYWA@UMICH.EDU

University of Michigan, Ann Arbor, MI, USA

Yutong Dong

755467293@SJTU.EDU.CN

Shanghai Jiao Tong University, Shanghai, China

Lei Song

SONGLEI 24@SJTU.EDU.CN

Shanghai Jiao Tong University, Shanghai, China

## Xinping Guan

XPGUAN@SJTU.EDU.CN

Shanghai Jiao Tong University, Shanghai, China

Editors: N. Ozay, L. Balzano, D. Panagou, A. Abate

## Abstract

The safety of training task policies and their subsequent application using reinforcement learning (RL) methods has become a focal point in the field of safe RL. A central challenge in this area remains the establishment of theoretical guarantees for safety during both the learning and deployment processes. Given the successful implementation of Control Barrier Function (CBF)-based safety strategies in a range of control-affine robotic systems, CBF-based safe RL demonstrates significant promise for practical applications in real-world scenarios. However, integrating these two approaches presents several challenges. First, embedding safety optimization within the RL training pipeline requires that the optimization outputs be differentiable with respect to the input parameters, a condition commonly referred to as differentiable optimization, which is non-trivial to solve. Second, the differentiable optimization framework confronts significant efficiency issues, especially when dealing with multi-constraint problems. To address these challenges, this paper presents a CBF-based safe RL architecture that effectively mitigates the issues outlined above. The proposed approach constructs a continuous AND logic approximation for the multiple constraints using a single composite CBF. By leveraging this approximation, a close-form solution of the quadratic programming is derived for the policy network in RL, thereby circumventing the need for differentiable optimization within the end-to-end safe RL pipeline. This strategy significantly reduces computational complexity because of the closed-form solution while maintaining safety guarantees. Simulation results demonstrate that, in comparison to existing approaches relying on differentiable optimization, the proposed method significantly reduces training computational costs while ensuring provable safety throughout the training process. This advancement opens up promising potential for applications in large-scale optimization problems.

Keywords: Safe reinforcement learning, composite control barrier functions, closed-form solution

## 1. Introduction

The safety of reinforcement learning (RL) during both training and deployment phases has garnered increasing attention Lavanakul et al. (2024); Vaskov et al. (2024); Buerger et al. (2024), particularly due to the safety-critical nature of many robotic systems. A core challenge lies in ensuring provable safety throughout these phases. Traditional RL methods commonly address safety by penalizing unsafe behaviors, which inevitably leads to the exploration of unsafe actions during training and fails to guarantee the safety of the learned policy during deployment. Recent solutions can be divided into two categories: constrained optimization-based methods and safety filter-based methods. For constrained optimization involving multiple safety constraints, Lagrangian-based safe RL methods Xu et al. (2021); Yao et al. (2024) are proposed to improve training efficiency with constraint satisfaction. Safety filter based methods typically rely on certificate functions such as Control Barrier Functions (CBFs), or Hamilton-Jacobi Reachability value functions. However, there remains a lack of efficient and generalizable approaches to ensure safety across all phases of the RL process.

The CBF-based approach Wang et al. (2017); Ames et al. (2019); Agrawal and Panagou (2021); Wang et al. (2023); Xiao and Belta (2022) theoretically ensures safety for control strategies and has been widely applied to various control-affine robotic systems, such as autonomous vehicles Wang et al. (2023), bipedal robots Csomay-Shanklin et al. (2021) and etc. The core idea involves formulating safety constraints for the control strategy, defining a safe set through these constraints, and deriving forward invariance conditions for the safe set to impose decision-variable constraints that ensure safety. These constraints are then integrated into an optimization problem to generate safe strategies. Typically, the safety optimization is based on system models and nominal controllers derived from control theory. Building on this framework, learning-based methods can replace nominal controllers, leveraging the powerful approximation capabilities and superior task performance of learning techniques Cheng et al. (2019).

Integrating safety optimization into the pipeline of control policy learning can be framed as a decision-focused learning paradigm Shah et al. (2022). In this framework, the prediction phase is handled by a RL policy network, followed by downstream safety optimization to generate the final safe strategy and evaluate its performance. This end-to-end approach requires the safety optimization process to be differentiable, which is often challenging due to issues like solution discontinuity Ferber et al. (2020) and gradient approximation Wilder et al. (2019). Recent works in decisionfocused learning address these challenges through various methods: using surrogates to replace the original optimization problem and learning loss functions Wilder et al. (2019) or constructing differentiable optimization tools Amos and Kolter (2017); Pineda et al. (2022); Agrawal et al. (2019). For control-affine systems, safety behavior optimization benefits from linear relaxation of decision variables via Nagumo's theorem Ames et al. (2019), which avoids the complexity of differentiable nonlinear programming Pineda et al. (2022) or mixed-integer programming Ferber et al. (2020). This allows the use of differentiable Quadratic Programming (QP) solvers Amos and Kolter (2017); Agrawal et al. (2019).

Recent research on differentiable QP-based safe control Emam et al. (2022); Ma et al. (2022); Amos et al. (2018); Romero et al. (2024) primarily focuses on three aspects: (1) addressing the impact of constraint parameters, such as environmental changes on safety strategies, e.g., Ma et al. (2022), by adjusting the classK functions within safety constraints via differentiable QP; (2) constructing linear MPC problems Amos et al. (2018) and tuning receding horizon parameters during optimization through differentiable QP to enhance task performance Romero et al. (2024); and (3)

imitating safe behaviors Xiao et al. (2023) or integrating safe QP as the final layer of RL policy networks Emam et al. (2022) to generate safe optimization strategies. Differentiable QP frameworks offer several notable advantages. First, they enable a decoupled design of task policy learning and safety correction, thereby facilitating the seamless integration of various learning methodologies. Second, the end-to-end learning of optimized strategies often yields superior task performance compared to hierarchical learning frameworks that incorporate safety corrections post-policy training. Despite these benefits, differentiable QP frameworks are not without limitations. Differentiable optimization is inherently complex, particularly for problems involving discrete decision variables. Furthermore, each gradient update requires solving an optimization problem and subsequently differentiating through it, which can result in significant computational cost.

Given these observations, this work focuses on safe RL with CBF-based optimization and addresses the computational complexity associated with differentiable optimization. Given that safetycritical applications often involve multiple constraints within the optimization problem, a continuous Log-Sum-Exp approximation is employed to transform multiple constraints into a single composite constraint. Utilizing this composite constraint, the closed-form solution of the corresponding QP is derived and integrated into the final layer of the RL policy network, which enables an end-to-end training pipeline with analytical computation, effectively serving as a surrogate for the differentiable QP. The proposed framework significantly reduces the computational cost associated with computing the derivatives of the differentiable QP output with respect to its input parameters Amos and Kolter (2017); Agrawal et al. (2019), offering an efficient and scalable solution for training in large-scale optimization problems.

## 2. Preliminaries

## 2.1. Safe policy via CBF-based QP

Consider a control-affine system

$$\dot { x } = f ( x ) + g ( x ) u,$$

where x ∈ R n denotes system state, u ∈ R m denotes control input (policy). In this paper, we consider f : R n → R n , g : R n → R n × m are bounded Lipschitz continuous vector fields and f, g are known for safety guarantee. This consideration is common, since most of the mechanical systems can be formulated as the control-affine form including manipulators, autonomous vehicles, drones, bipedal robots, and etc. For the safety of control-affine systems at the dynamical level, CBFs have successful applications. The safety is related to the desired safe sets which can be defined by continuous differentiable functions h i ( x ) : R n → R , i = 1 , . . . , I :

$$\mathcal { C } _ { i } \stackrel { \triangle } { = } \left \{ x \in \mathbb { R } ^ { n } \, \colon h _ { i } ( x ) \geq 0 \right \},$$

$$\partial \mathcal { C } _ { i } \stackrel { \triangle } { = } \left \{ x \in \mathbb { R } ^ { n } \, \colon h _ { i } ( x ) = 0 \right \},$$

$$\text{Int} \mathcal { C } _ { i } \stackrel { \triangle } { = } \left \{ x \in \mathbb { R } ^ { n } \, \colon h _ { i } ( x ) > 0 \right \}.$$

The set C i is forward invariant if for any initial state x (0) ∈ C i , x ( ) t ∈ C i , ∀ t ∈ [0 , ∞ ) . The system is safe if all C i , i = 1 , . . . , I are forward invariant.

Given the dynamics in (1) and safety requirement, the forward invariance condition based on CBFis formulated as follows: Let C i be the 0-superlevel set of a continuously differentiable function

h i : R n → R . The function h i is a CBF for (1) w.r.t. C i if there exists extended class K function α and u ∈ R m such that

$$L _ { f } h _ { i } ( x ) + L _ { g } h _ { i } ( x ) u \geq - \alpha ( h _ { i } ( x ) ),$$

where

L h

f

i

(

x

) =

∂h

i

(

x

)

∂h

i

(

x

)

∂x

f

(

x , L

)

g

h

i

(

x

) =

∂x g x

(

)

w.r.t f, g

.

Since all the I th safe constraints are linear on u , the control optimization based on QP can be formulated as

$$\text{ted as} u _ { s } = & \arg \min _ { u \in \mathbb { R } ^ { m } } \frac { 1 } { 2 } \| u - \bar { u } \| _ { 2 } ^ { 2 } \\ & \text{s.t. } L _ { f } h _ { i } ( x ) + L _ { g } h _ { i } ( x ) u \geq - \alpha ( h _ { i } ( x ) ), i = 1, \dots, I,$$

where ¯ u denotes the nominal controller designed for the original task objective. The optimization (6) minimally corrects the nominal controller when ¯ u violates the safety constraints, resulting in the safe policy u s . More details are referred to Lemma 2 and 3 in Breeden and Panagou (2023) or Theorem 1 in Aali and Liu (2022), with an assumption for nonempty feasible set of u s .

## 2.2. Soft Actor-Critic

As an off-policy RL algorithm, the Soft Actor-Critic (SAC) Haarnoja et al. (2018) leverages its high sample efficiency and entropy regularization features to offer performance advantages in RL methods for continuous action spaces. The entropy objective is optimized by

$$\pi ^ { * } & = \arg \max \sum _ { t } \mathbb { E } _ { ( x _ { t }, u _ { t } ^ { \phi } ) \sim \rho _ { \pi } } \left [ r ( x _ { t }, u _ { t } ^ { \phi } ) + \alpha _ { e } H ( \pi ( \cdot | x _ { t } ) ) \right ], \\ \pi ^ { * } & = \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \end{cases}$$

The SAC algorithm utilizes an AC approach, where the critic is represented by a Q-function parameterized by θ , and the actor is represented by a policy π parameterized by φ . The critic loss J Q ( θ ) aims to minimize the difference between the Q-values generated by the critic and the sum of the rewards plus the expected value of the next state's value function:

$$J _ { Q } ( \theta ) = \mathbb { E } _ { ( x _ { t }, u _ { t } ^ { \phi } ) \sim D _ { r } } & \left [ \frac { 1 } { 2 } \left ( Q _ { \theta } ( x _ { t }, u _ { t } ) - \left ( r ( x _ { t }, u _ { t } ^ { \phi } ) \\ + & \gamma \mathbb { E } _ { x _ { t + 1 } \sim p } \left [ V _ { \hat { \theta } } ( x _ { t + 1 } ) \right ] \right ) \right ) ^ { 2 } \right ], \\ \text{is the replay buffer, and } \hat { \theta } \text{ represents the target } Q \text{-network parameters. The replay buffer}$$

where D r is the replay buffer, and ˆ θ represents the target Q-network parameters. The replay buffer D r provides a diverse set of experiences, enabling the critic to learn from a broad range of past states and actions, which enhances sample efficiency. The target Q-network parameters ˆ θ ensure stable updates by serving as a slowly updating reference.

The entropy term is included to promote exploration and prevent premature convergence to suboptimal policies, which is given by:

$$H ( \pi ( \cdot | x _ { t } ) ) = - \log \pi _ { \phi } \left ( u _ { t } ^ { \phi } | x _ { t } \right ). \\ \text{autources actions that maximize both the expected reward and the entropy.}$$

The policy loss encourages actions that maximize both the expected reward and the entropy, leading to an effective balance between performance and exploration, which is given by

$$J _ { \pi } ( \phi ) = \mathbb { E } _ { x _ { t } \sim \mathcal { D } _ { r } } \left [ \mathbb { E } _ { u _ { t } ^ { \phi } \sim \pi _ { \phi } } [ \alpha _ { e } \log \pi _ { \phi } ( u _ { t } ^ { \phi } | x _ { t } ) - Q _ { \theta } ( x _ { t }, u _ { t } ^ { \phi } ) ] \right ].$$

One of the primary advantages of constructing a differentiable optimization framework is the ability to decouple the design of the safety layer, enabling seamless integration into the policy network of actor-critic (AC)-based RL methods. Therefore, the SAC serves as a candidate when implementing a differentiable QP layer. The policy loss is given by

$$J _ { \pi } ( \phi ) & = \mathbb { E } _ { x _ { t } \sim \mathcal { D } _ { r } } \\ & \left [ \mathbb { E } _ { u _ { t } ^ { \phi } \sim \pi _ { \phi } } [ \alpha _ { e } \log \pi _ { \phi } ( u _ { t } ^ { \phi } | x _ { t } ) - Q _ { \theta } ( x _ { t }, u _ { t } ^ { \phi } + u _ { t } ^ { C } ) ] \right ], \\ \iota _ { t } ^ { C } & \text{ is the compensation term computed by differentiable QP layer.}$$

where u C t is the compensation term computed by differentiable QP layer.

## 3. Main results

## 3.1. Composite CBF for multiple constraints

To solve the CBF-based optimization under multiple constraints, the constraints are regarded as the intersection of safe sets defined by these CBFs. Each safe constraint h i is defined by a 0-superlevel set and their intersection is defined as

$$\bigcap _ { i = 1, \dots, I } C _ { i } & = \{ x \in \mathbb { R } ^ { n } \colon h _ { i } ( x ) \geq 0 \}, \\. \quad. \quad. \quad.$$

where I denotes the number of the safety constraints.

In other words, the intersection of sets captures the logical AND relationship between multiple safety constraints, which is denoted as

$$x \in \bigcap _ { i = 1, \dots, I } C _ { i } & \iff x \in C _ { 1 } \, \text{AND} \, x \in C _ { 2 } \cdots \, \text{AND} \, x \in C _ { I }. \\ \text{--} \quad. & \quad \cdots \quad. & \quad. & \quad. & \quad. & \quad.$$

When there are multiple constraints, the complexity of the QP problem increases, generally making it impossible to derive a closed-form solution, thus requiring numerical optimization methods such as active set or interior point methods. However, inspired by existing literature Molnar and Ames (2023) solving complex safety specifications, this paper employs a Log-Sum-Exp approximation technique to transform multiple constraints into a single constraint, thereby enabling a closed-form solution for the safe QP.

The approximated composite single CBF is constructed as:

$$h ( x ) = - \frac { 1 } { \kappa } \ln \left ( \sum _ { i = 1 } ^ { I } e ^ { - \kappa h _ { i } ( x ) } \right ), \\ \intertext { a n e c a d k }.$$

whose Lie derivatives are expressed by:

$$L _ { f } h ( x ) = \sum _ { i = 1 } ^ { I } \lambda _ { i } ( x ) L _ { f } h _ { i } ( x ), \ \ L _ { g } h ( x ) = \sum _ { i = 1 } ^ { I } \lambda _ { i } ( x ) L _ { g } h _ { i } ( x ), \\ \dots$$

where

$$\lambda _ { i } ( x ) = e ^ { - \kappa ( h _ { i } ( x ) - h ( x ) ) },$$

with ∑ i ∈ I λ i ( x ) = 1 and κ &gt; 0 .

Since an equivalent substitution for the constraints of optimization problem (6) is min h i ( x ) ≥ 0 , i = 1 , · · · , I . The composite CBF in (14) shares the following property.

Lemma 1: Molnar and Ames (2023) Consider sets C i in (2) and their intersection in (12). Continuous function h x ( ) in (14) under approximates min i =1 , ··· ,I h i ( x ) ≥ 0 with bounds:

$$\min _ { i = 1, \cdots, I } h _ { i } ( x ) - \frac { \ln I } { \kappa } \leq h ( x ) \leq \min _ { i = 1, \cdots, I } h _ { i } ( x ) \quad \forall x \in \mathbb { R } ^ { n },$$

such that lim κ →∞ h x ( ) = min i =1 , ··· ,I h i ( x ) . The corresponding set C = { x ∈ R n : h x ( ) ≥ } 0 lies inside the intersection, C ⊆ i =1 , ··· ,I C i , such that lim κ →∞ C = i =1 , ··· ,I C i .

⋂ ⋂ See Proof of Theorem 4 in Molnar and Ames (2023). h x ( ) ≥ 0 guarantees min i =1 , ··· ,I h i ( x ) ≥ 0 , indicating all constraints h i ≥ 0 , i = 1 , · · · , I are satisfied.

## 3.2. Closed-form solution for CBF-based QP

The safety-oriented framework offers a QP-based optimization approach to modify a nominal policy to ensure safety. The nominal policy ¯ u , typically designed to achieve a specific task objective, can be derived from model-based control or generated through RL. Based on the established composite CBF h x ( ) in (14), the optimization problem ensuring system safety can be formulated as the following QP:

$$u _ { s } ( x ) = \underset { u \in \mathbb { R } ^ { m } } { \arg \min } \frac { 1 } { 2 } \| u - \bar { u } ( x ) \| _ { 2 } ^ { 2 }$$

subject to

$$L _ { f } h ( x ) + L _ { g } h ( x ) u \geq - \alpha ( h ( x ) ).$$

When the nominal policy satisfies the safety constraint, the constraint (19) is inactive, and the safe policy aligns with the nominal policy. However, when the nominal policy violates the safety constraint, the QP seeks a safe policy that satisfies the constraints while deviating minimally from the nominal policy. The purpose of transforming multiple constraints into a composite CBF is to derive a closed-form solution for the safe policy of the optimization (18). The closed-form solution can be obtained by referring to the following theorem.

Theorem 1: Let C be the 0-superlevel set of a continuously differentiable function h : R n → R , and let ¯( u x ) : R n → R m be a nominal controller. If h is a composite CBF for (1) on the set C ⊆ ⋂ i =1 , ··· ,I C i with the corresponding function α ∈ K e ∞ , then the optimization problem in (18) is feasible for any x ∈ R n and has a closed-form solution given by

$$u _ { s } ( x ) = \bar { u } ( x ) + \max \{ 0, \eta ( x ) \} L _ { g } h ( x ) ^ { \top }$$

where the function η : R n → R is defined as

/negationslash

$$\eta ( x ) = \begin{cases} - \frac { L _ { f } h ( x ) + L _ { g } h ( x ) \bar { u } ( x ) + \alpha ( h ( x ) ) } { \| L _ { g } h ( x ) \| _ { 2 } ^ { 2 } } & \text{if } L _ { g } h ( x ) \neq 0, \\ 0 & \text{if } L _ { g } h ( x ) = 0. \end{cases}$$

See proof of Theorem 2 in Alan et al. (2023). Theorem 2 provides a sufficient but not necessary condition for a safe solution, offering an analytical form for solving the QP associated with a single constraint. This formulation eliminates the need to invoke a QP solver, significantly reducing the computational cost. Therefore, this advantage motivates its integration with the RL framework.

Furthermore, the closed-form solution provides the significant advantage of circumventing the requirement for differentiable optimizations within the RL framework, thereby substantially simplifying the gradient computation in the safe policy generation and alleviating the complexity associated with gradient-based optimization, as will be elaborated in the next subsection.

Figure 1: An illustration of an end-to-end training safe RL framework.

<!-- image -->

## 3.3. Safety layer via closed-form solution in RL framework

In conventional RL architectures, the final layer of the control policy network typically consists of a fully connected layer, particularly for continuous control actions in affine systems. The output is bounded by the final activation function, such as the hyperbolic tangent, to ensure bounded action outputs. For safe policy generation, an intuitive approach is to correct the RL-derived control policy by adjusting it through safety-oriented mechanisms, such as correcting the control policy to a safe policy using a CBF-based QP Cheng et al. (2019). However, in this approach, the reward from the safe output cannot backpropagate to the RL network, due to the absence of a gradient pathway connecting the safe policy to the RL policy. Recently, decision-focused learning have proposed architectures based on differentiable optimization, embedding optimizable and differentiable structures to achieve an end-to-end learning pipeline, thus enabling CBF-QP-based safe learning, as illustrated in Figure 1. This raises a critical challenge: each training step requires solving a batch of QP problems for the policy loss function, along with calculating the gradient of each QP output concerning the QP parameters, making it computationally expensive and challenging to large-scale multi-constraint problems. To address this issue, we integrate a closed-form solution for the safe policy directly into the RL policy generation pipeline. This approach leverages a composite singleconstraint approximation to handle multi-constraint scenarios, alongside explicit QP solutions to circumvent forward optimization and its gradient backpropagation. We replace the final layer in RL policy generation with an analytically computed 'safety layer', which, due to its analytical properties, can be integrated into any actor-critic RL method. An illustration of safe policy networks in actor-critic framework is shown in Figure 2 with different safety layers. The proposed framework is demonstrated in Figure 2(a), where the closed-form solution (20) and (21) are integrated into the final layer before safety policy generation. As a comparison, Figure 2(b) demonstrates that taking nominal policy as the input, the differentiable QP layer compute the forward solution with multiple constraints, which is potentially infeasible and computationally expensive.

We illustrate the proposed approach using the SAC method, where the loss functions in this framework are given by

$$J _ { Q } ( \theta ) = \mathbb { E } _ { ( x _ { t }, u _ { s } ^ { \phi } ) \sim \mathcal { D } _ { R } } \left [ \frac { 1 } { 2 } \left ( Q _ { \theta } ( x _ { t }, u _ { s } ^ { \phi } ) - \left ( r ( x _ { t }, u _ { s } ^ { \phi } ) + \gamma \mathbb { E } _ { x _ { t + 1 } \sim p } \left [ V _ { \tilde { \theta } } ( x _ { t + 1 } ) \right ] \right ) \right ) ^ { 2 } \right ],$$

Figure 2: An illustration of safe policy networks with different safety layers. Subfigure (a) demonstrates the proposed framework where N constraints are composited to h x ( ) using a continuous Log-Sum-Exp approximation. The safety layer is analytical based on the closed-form solution of the composite CBF-based optimization. Subfigure (b) demonstrates the existing framework with differentiable QP layer. The safety layer solves the forward CBF-based optimization and computes the gradient during backpropagation.

<!-- image -->

$$V _ { \bar { \theta } } ( x _ { t } ) = \mathbb { E } _ { u _ { s } ^ { \phi } \sim \pi _ { \phi } } \left [ Q _ { \bar { \theta } } ( x _ { t }, u _ { s } ^ { \phi } ) - \alpha _ { e } \log \pi _ { \phi } ( u _ { s } ^ { \phi } | x _ { t } ) \right ], \\ \Gamma _ { r } \quad. \quad.$$

$$J _ { \pi } ( \phi ) = \mathbb { E } _ { x _ { t } \sim \mathcal { D } _ { r } } \left [ \mathbb { E } _ { u _ { s } ^ { \phi } \sim \pi _ { \phi } } [ \alpha _ { e } \log \pi _ { \phi } ( u _ { s } ^ { \phi } | x _ { t } ) - Q _ { \theta } ( x _ { t }, u _ { s } ^ { \phi } ) ] \right ], \\ \mathfrak { e } \, \pi _ { \phi } \, \text{ denotes the policy generated by the entire policy network, including both the fully con-}$$

where π φ denotes the policy generated by the entire policy network, including both the fully connected layers and the QP-based adjustment. In this case, u φ s ∼ π φ would mean that the sample u φ s is drawn from the distribution defined by the entire policy network, which inherently includes the safety layer for the QP adjustment.

## 4. Experiment

In this section, we aim to validate the capability of the proposed method to ensure safety during training while achieving faster training efficiency compared to the differentiable QP-based approach. The testing environment is designed as a reachability task, where the agent's objective is to reach the goal position while avoiding obstacles. To illustrate the incorporation of multiple constraints in the safety-oriented optimization, the collision-free constraints are defined as follows:

$$h _ { i } ( x ) = \| p - p _ { i, \text{obs} } \| ^ { 2 } - r _ { \text{safe} } ^ { 2 } \geq 0, \quad i = 1, \dots, I,$$

where p = [ p , p x y ] /latticetop denotes the position of the agent, p i, obs denotes the position of the i th obstacle, and r safe is the safe radius to avoid collision. Furthermore, within the safety constraints based on CBFs, the selection of the classK function α significantly affects the conservativeness of safety enforcement and κ affects the the approximation error of min operation. In this study, we adopt a trade-off value to balance safety and performance with α = 5( ) · , κ = 2 .

To demonstrate the safety of the proposed method, we present the following metrics during training: min h , i i = 1 , . . . , I and the composite h for each episode with I = 3 . Moreover, the min h , i i ∈ I and composite h across all steps of training episodes, and the trajectories for the deployment phase after training are also demonstrated. The results are illustrated in the Figure 3 and Figure 4.

Figure 3: Performance of safe RL training and testing with the proposed method. Subfigure (a) illustrates min h , i i = 1 , . . . , I and the composite h for each episode during training. The composite h x ( ) under approximates min h i and maintains positive in each episode. Subfigure (b) illustrates the successful trajectories during testing. The blue area in Subfigure (b) contains three obstacles, each with a different safe radius size. The colored squares denote the initial positions. The red circle represents the target area, while the colored lines indicate the testing trajectories, each starting from different initial positions near the origin.

<!-- image -->

As shown in Figure 3(a), the minimum value of h i remains consistently greater than 0 throughout the entire training process, indicating that the system successfully achieves safe training and policy learning. Furthermore, the red line h x ( ) serves as an under-approximation of the blue dashed line, which is consistent with the conclusion of Theorem 1. Figure 3(b) illustrates the trajectories reaching the target area, where safety is ensured across 200 trials.

A more detailed perspective is provided in Figure 4, which illustrates the evolution of h i and h across all steps in each episode (different color of curves) over 1000 training iterations (with a maximum step limit of 200). The time intervals during which the h i curves near the safety set boundary exhibit 'flattening' correction, depending on the different positions of obstacles. During these intervals, the condition L h x f ( ) + L h x u x g ( )¯( ) + α h x ( ( )) &lt; 0 holds. Therefore, the safe policy is actively filtered and corrected based on (20) and (21), ensuring that h remains positive for all time.

Table 1: Comparison of different approaches

| Method               |   ATTS ( s ) with I = 3 |   ATTS ( s ) with I = 10 |   ATTS ( s ) with I = 30 |
|----------------------|-------------------------|--------------------------|--------------------------|
| Closed-form solution |                   0.018 |                    0.024 |                    0.043 |
| CBF Batch QP         |                   0.13  |                    0.25  |                    0.4   |
| CBF CVXPYlayer       |                   0.84  |                    1.45  |                    2.26  |

## WANG WANG DONG SONG GUAN

Figure 4: Evolution of h , h 1 2 , h 3 and the composite h over 1000 episodes. The safe learning during training is guaranteed.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

As previously noted, the closed-form solution eliminates the need for differentiable QP solvers, thereby reducing computational costs. This represents another significant advantage of the proposed method. In the same scenario for collision avoidance of multiple obstacles, we compared the proposed method with the differentiable QP solver Batched-QPFunction Amos and Kolter (2017) and the CVXPYLayer Agrawal et al. (2019) in terms of computational performance, which are commonly used in similar works within CBF-based safe learning Emam et al. (2022); Ma et al. (2022); Jiang et al. (2024); Romero et al. (2024). The comparative results are summarized in Table 1, where the performance metric is the average solving time per time step (ATTS) during the RL training process. In addition, scenarios with 10 and 30 constraints are also tested to validate the scalability of the proposed method in solving larger-scale safe RL problems.

As demonstrated in Table 1, the proposed method exhibits a computational speed advantage of at least one order of magnitude because of the close-form nature. The CVXPYlayer-based method, while supporting disciplined parametrized programming, exhibits the lowest solving efficiency due to the lack of support for batch solving and the requirement for gradient computation in QP. The advantage in training time makes it potentially effective for optimization in large-scale safe RL problems.

## 5. Conclusion

This paper addresses the challenges of ensuring multiple safety constraints and improving training efficiency in safe RL. We propose a safe RL framework based on the closed-form solution of composite CBF. The framework constructs a composite CBF by using the Log-Sum-Exp approximation of the min function to integrate multiple safety constraints in the optimization problem. It also inherits the safety guarantees based on the composite CBF defining the safe set. By serving as a surrogate of the differentiable QP architecture with a closed-form solution, the proposed method significantly enhances training efficiency. Comparative experiments demonstrate that the proposed method is up to 7 times faster than the current state-of-the-art differentiable batch QP solvers, and at least 46 times faster than the differentiable convex optimization layers CVXPYlayer, showcasing its potential for solving optimization in large-scale safe RL problems. Future work will further investigate the composite CBF-QP under explicit input constraints, with a focus on guaranteeing feasibility and improving efficiency within the framework of differentiable optimization.

## Acknowledgments

This work was supported by National Natural Science Foundation of China under Grant 62303316, in part by the Science Center Program of National Natural Science Foundation of China under Grant 62188101, in part by the Fellowship of China National Postdoctoral Program for Innovative Talents under Grant BX20240224, and the Oceanic Interdisciplinary Program of Shanghai Jiao Tong University (project number SL2022MS010).
<|endofpaper|>