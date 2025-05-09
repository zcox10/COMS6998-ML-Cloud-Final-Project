<|startofpaper|>
## Neural Network Verification for Gliding Drone Control: A Case Study

Colin Kessler 1 2 , GLYPH&lt;12&gt; , Ekaterina Komendantskaya , Marco Casadio , Ignazio 1 1 Maria Viola , Thomas Flinkow , Albaraa Ammar Othman , Alistair 2 3 1 Malhotra , and Robbie McPherson 1 1

1 Heriot-Watt University and Edinburgh Centre for Robotics, UK ck2049@hw.ac.uk 2 School of Engineering, University of Edinburgh, UK 3 Maynooth University, Maynooth, Ireland

Abstract. As machine learning is increasingly deployed in autonomous systems, verification of neural network controllers is becoming an active research domain. Existing tools and annual verification competitions suggest that soon this technology will become effective for realworld applications. Our application comes from the emerging field of microflyers that are passively transported by the wind, which may have various uses in weather or pollution monitoring. Specifically, we investigate centimetre-scale bio-inspired gliding drones that resemble Alsomitra macrocarpa diaspores. In this paper, we propose a new case study on verifying Alsomitra -inspired drones with neural network controllers, with the aim of adhering closely to a target trajectory. We show that our system differs substantially from existing VNN and ARCH competition benchmarks, and show that a combination of tools holds promise for verifying such systems in the future, if certain shortcomings can be overcome. We propose a novel method for robust training of regression networks, and investigate formalisations of this case study in Vehicle and CORA. Our verification results suggest that the investigated training methods do improve performance and robustness of neural network controllers in this application, but are limited in scope and usefulness. This is due to systematic limitations of both Vehicle and CORA, and the complexity of our system reducing the scale of reachability, which we investigate in detail. If these limitations can be overcome, it will enable engineers to develop safe and robust technologies that improve people's lives and reduce our impact on the environment.

Keywords: Neural Network Control · Bioinspired Robots · Verification of Cyber-Physical Systems · Machine Learning.

## 1 Introduction

A recent research trend in drone design concerns the development of gliding microdrones, which could serve a function as airborne sensors and remain aloft for extended periods of time [16, 18, 21, 35]. Current research focuses on the

aerodynamics of seeds that have exceptional wind dispersal mechanisms such as, for example, the Taraxacum (dandelion) [9] and Alsomitra (Javan cucumber). This case study focuses specifically on Alsomitra -inspired drones (Figs. 1,4) as the aerodynamics underlying the flight of this diaspore is unique in the plant kingdom, enhancing the dispersal mechanism provided by the wind by an efficient gliding flight. This allows one of the heaviest seeds (314 mg) [4] to reach a similar descent velocity than some of the lightest seeds such as the dandelion (0.6 mg) [9]. Because of this unique feature, several authors have considered this diaspore as a bioinspiration for microdrones [26,35]. Such drones could function as distributed sensors in the atmosphere, for weather monitoring or detecting pollutants [16, 18,21,26,35]. This could be particularly useful for environmental monitoring and meteorology, with research and regulations moving towards incorporating drone observations to improve weather predictions [11, 36]. It has been demonstrated that such systems are capable of sustained flight with active control and internal electronics [18], although more work on effective actuation and control methods is needed in the future.

Neural networks (NNs) have been widely investigated for drone control, for both quadcopters [3, 25] and fixed-wing designs [31, 33]. The control of small passive gliders is a relatively unexplored field, with the most relevant works involving larger aircraft [1, 33] or without continuous control [18]. For our application, we consider NN control since it has been shown to achieve accurate and robust control for systems with uncertain dynamics [17], it is particularly applicable to controlling swarms [30], and improvements to low-order aerodynamic modelling [24] facilitates easier simulations of such drones. This approach could facilitate particularly lightweight and low-cost drones - such as with analogue network circuits printed on flexible substrates acting as the body of the gliding drone [28, 32]. One could alternatively consider uncontrolled flying sensors [16,21,35] or traditional approaches such as state-space or model predictive control. However, one should consider that such systems will need to be verifiably safe with regard to people, other air users, and the environment [36]. These drones could collide with each other, veer into unsafe airspace, fall into an endangered ecosystem, or otherwise cause harm. The utility of uncontrolled flyers would be hampered by such issues, unless they can be made biodegradable. Traditional control methods may be applied, but NN methods have advantages in that they can be made data-driven and adaptive, and printed NN circuits could lead to lighter designs than digital microcontrollers.

## 1.1 Contributions

Our first aim is the introduction of a novel case study (outlined in Sect. 4) in the verification of Alsomitra -inspired drone controllers (our modelling methods are explained in Sect. 3), that differs significantly from existing benchmarks. Unlike VNN-COMP benchmarks such as ACAS Xu, our study involves regression control and continuous dynamic equations. Compared to ARCH-COMP benchmarks such as QUAD, our system involves differential equations that are far more complex in terms of the number of non-linear terms. Moreover, unlike the

Fig. 1. An artist's impression of a swarm of gliding drones inspired by Alsomitra seeds [7].

<!-- image -->

majority of ARCH-COMP cases, this problem does not have as natural a notion of the start, goal, safe, and unsafe states; and thus requires an out-of-the-box approach to property specification.

We propose our ideal formalisation of the problem in Sect. 4.1, and distil the formalisation down to properties that can be handled with available tools (Marabou [20] implemented with Vehicle [10], and CORA [2]) in Sects. 7 and 8. The choice is motivated by the fact that each can be seen as a representative of a set of tools that come from the research communities of VNN-COMP [5] and ARCH-COMP [15], respectively. We present a new implementation of adversarial training for Lipschitz robustness applied to regression training for our controllers in Sect. 5, and present the results of verifying those properties with our robust networks in Sects. 7.3 and 8.2.

Our second aim is to present the lessons learnt from investigating this case study, to help inform the development of relevant tools for similar real-life cyberphysical projects in the future (Sect. 9). The main lesson learnt is that no single existing tool ticks all the desirable boxes. Moreover, each individual tool we chose would benefit from further development in several aspects that are crucial for real-life models. Concretely:

- -On the Vehicle side, the verification properties that arise in the presented study are more complex than the usual VNN-COMP benchmarks in at least three ways.
- · Firstly, the constraints on the input vector are more complex: instead of constraining individual vector elements by constants (as e.g. in a ≤ x i ≤ b ), as is the case in the majority of benchmarks including ACAS XU [19], the constraints establish relation between different vector elements, as e.g. in x i ≤ cx j . This changes mathematical interpretation of the verification problem: it no longer boils down to defining a hyper-rectangle (or other constant shape) on the input space and propagating it through the network layers, but gives a more general case of linear programming that works on arbitrary input space constraints. Not every VNN-COMP [5] verifier will be able to deal with such verification properties: Marabou is one of the most general tools in this family of tools and this case study suggests this generality may play a bigger role in the future.

- C. Kessler et al.
- · Secondly, for verification of Lipschitz robustness, we implemented relational properties , i.e. properties that compare different outputs of a neural network. These properties are not natively supported by Marabou or Vehicle yet, and required some additional plumbing. On-going implementation of support for relational verification in Marabou will be useful for cases such as this.
- · Finally, some novelty of our verification approach is derived from the fact that, unlike most benchmarks in VNN-COMP, our models are regression models, rather than verification models. Some of the methods for training and verification are specialised to classification tasks only, and we predict that this has to change with occurrence of new engineering-inspired benchmarks.
- -On the CORA side, the system outlined in this study required several workarounds in order to compute reachability:
- · The complexity of our system of equations [24] far exceeds that of all ARCH-COMP [15] benchmarks, in the number of non-linear terms. This would cause the Jacobian and Hessian matrices to far exceed the maximum number of terms supported by MATLAB , and fail to run. The equations were simplified (Sect. 3.2) by constraining the pitch angle and using an angle-of-attack definition, solving the complexity issue, but (for any reasonably large initial set) the reachable set still tended to expand exponentially after relatively few timesteps. This was solved by dividing the initial set into smaller subsets, computing reachable sets for each, and combining the results.
- · CORA expects a NN controller that takes the system variables as inputs, with relatively few layer types supported [2]. Certain parameters occupy wider ranges than others (for example, θ ∈ -[ 0 93 . , -0 07] . , x ∈ [0 48 . , 41 7] . ) but unlike Vehicle, input normalisation (keeping all inputs between 0 and 1) is not supported. This is problematic since we intend to observe the effect of adversarial training, for which the input ranges should to be normalised, such that PGD attacks occur in ϵ ranges that are not imbalanced between input dimensions. A workaround was found by training an adversarial network on normalised data, then implementing normalisation layers to the start and end of the network.
- · Unlike similar ARCH-COMP benchmarks such as QUAD, our notion of a goal region is less obvious. We want the drone to adhere to the target trajectory in x and y , so define a goal state as a region around that trajectory.

Although this study considers only one modification of gliding drones, most of the paper's conclusions will be common between their different modifications, such as e.g. dandelion-inspired drones, and the lessons learnt can be broadly applied to other continuous control tasks. All relevant files are publicly available here.

## 2 Background

## 2.1 Neural Network Control

For our control method, we will use the common closed-loop negative feedback method, an overview of which can be seen in Fig. 2. In simple terms, the controller in a drone is given information about its current state (such as position, relative to some desired state) as input, and outputs a command to an actuator which affects how the drone flies. The controller can be considered as an equation linking the system states to an actuation force that changes the states over time according to the system dynamics, where the controller design affects how the drone behaves. If a traditional control theory approach is difficult (such as if the dynamics are highly complex) or a data-driven approach is desireable (if collecting data is easier than modelling the system, or if adaption based on new data is required), an engineer might consider implementing a NN controller.

Fig. 2. Overview of a negative feedback control system. For each control iteration, an error signal is calculated by subtracting the current system state (feedback) from the desired system state (input). A controller computes an actuation based on this error, which is applied to a simulated or real system (plant), resulting in some new output state.

<!-- image -->

## 2.2 Verification Tools

The case study will rely on the following three groups of neural network verification (NNV) tools. The first group concerns verification of infinite time-horizon properties of controllers in isolation from verification of the overall system dynamics. The most famous benchmark in the domain is ACASXu, and the representative verifier is Marabou [20]; other tools, such as ERAN [27], Pyrat [23] or αβ -CROWN [37] could be interchangeably used for the verification tasks in which Marabou is deployed in this paper; we refer the reader to VNN-Comp [5] for an in depth discussion of existing tools in this category. In addition, we use Vehicle [10], a higher-level interface on top of Marabou, and take advantage of its facility in bridging the embedding gap [8] between the physical domains and vector representation of data.

The second group of methods considers the neural controller together with the overall system dynamics to ensure that the entire system avoids unsafe states, see Fig. 3. This class of problems is also known under the umbrella term reachability verification and representative examples include e.g. POLAR-Express [34],

Fig. 3. General form of reachability specifications - dots represent the system at successive control time steps, and arrows represent the continuous trajectory of the system. Any trajectory starting in the initial set should never intersect an unsafe set, and always finish in the goal set.

<!-- image -->

and CORA [2], see [25] for an exhaustive overview of the mainstream tools in this category. Representative benchmarks include simple dynamic problems such as the inverted pendulum, and more complex problems such as the quadcopter, space docking, and 2-wheeled obstacle avoidance. Each benchmark has a predetermined set of dynamic equations and a NN controller, with a mix of supervised learning (through behaviour-cloning) and reinforcement learning. The limitations of these benchmarks are in the complexity of the networks (large networks require reduction methods), complexity of the equations (systems are either linear, or relatively simple non-linear differential equations), and verification of complex properties (no tools can successfully verify the Spacecraft Docking benchmark as of the most recent results [15]).

Finally, an important group of methods for practical NNV cases comes from machine learning domain, under the umbrella term of property-driven training (PDT) . These methods allow to optimise a given neural network for satisfying a desired verification property, with a view of improving the verification success [6,12,14]. Although methods in this group vary, they usually deploy a form of training with projected gradient descent (PGD) [22]. PGD methods involve finding the worst-case perturbed example in a region around a data point, which can then be implemented as a loss function during training:

$$\min _ { \theta } \, E _ { ( x, y ) \sim \mathcal { D } } \, \left [ \max _ { \delta \in \Delta } \, \mathcal { L } ( f _ { \theta } ( x + \delta ), y ) \right ]$$

where θ represents the parameters of the NN; ( x, y ) ∼ D are input-label pairs sampled from the data distribution D ; E is the expected value, averaging the loss over all samples in the data distribution D ; δ ∈ ∆ is the adversarial perturbation constrained within a feasible set ∆ (e.g., ∥ ∥ δ p ≤ ϵ ) and L is the loss function (e.g., RMSE, MAE) measuring the discrepancy between the predicted output f θ ( x + ) δ and the true label y .

The inner maximisation, which identifies the worst-case adversarial perturbation δ ∈ ∆ , is performed using PGD that iteratively adjusts δ by ascending the gradient of the loss function with respect to the input, followed by projection back onto the feasible set ∆ (e.g., ensuring ∥ ∥ δ p ≤ ϵ ).

The outer minimisation, aimed at optimising the neural network parameters θ to minimise the adversarial loss, is achieved using gradient descent.

## 3 Modelling Methodology

## 3.1 Alsomitra macrocarpa

A dynamics model (Fig. 4) was derived from [24] resulting in a system of equations for falling plates with displaced centre of mass (CoM), as defined in Sect. 3.2. Based on experimental measurements, our model accurately describes the falling trajectories of Alsomitra seeds by inferring aerodynamic forces from the angle of attack [24]. The flight characteristics are highly dependant on the CoM displacement ( e x , Fig. 4), providing us with a convenient actuation method for an Alsomitra -inspired drone.

Fig. 4. (a) An Alsomitra seed [7]. (b) A two-dimensional approximation of an Alsomitra seed, with centre of mass (CoM) displaced by ℓ CM (nondimensional form e x = ℓ CM /ℓ ). (c) Effect of various e x on gliding trajectories; according to a quasi-steady aerodynamic model ( [24], Sect. 3.2). As the CoM is displaced the trajectory behaviour is affected significantly.

<!-- image -->

## 3.2 Equations

The following equations describe falling plates with a displaced centre of mass [24], with six system variables ( x 1 ... 6 , Equations 11 ... 16), involving mechanical ( , ℓ m g , , ρ f , I ) and aerodynamic ( C 0 CP , C 1 CP , C 2 CP , C 1 L , C 2 L , C 0 D , C 1 D , C π/ 2 D , C R , α 0 , δ ) constants chosen to match that of Alsomitra seeds [7]. Several intermediate terms are included for simplicity (Equations 1 ... 10)., and a more detailed overview can be seen in Appendix 10.1

$$\tan \alpha = ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell ) / x _ { 1 } \approx x _ { 2 } / x _ { 1 }$$

$$f = ( 1 - \tanh ( ( \alpha - \alpha _ { 0 } ) / \delta ) ) / 2$$

$$- C _ { L } = f ( | \alpha | ) C _ { L } ^ { 1 } \sin ( | \alpha | ) + ( 1 - f ( | \alpha | ) ) C _ { L } ^ { 2 } \sin ( 2 \, | \alpha | ) \quad \quad ( 3 )$$

$$C _ { D } = f ( | \alpha | ) ( C _ { D } ^ { 0 } + C _ { D } ^ { 1 } \sin ^ { 2 } ( | \alpha | ) ) + ( 1 - f ( | \alpha | ) ) C _ { D } ^ { \pi / 2 } \sin ^ { 2 } ( | \alpha | ) \quad ( 4 )$$

$$\ell _ { \text{CP} } / \ell = f ( | \alpha | ) ( C _ { \text{CP} } ^ { 0 } - C _ { \text{CP} } ^ { 1 } \alpha ^ { 2 } ) + C _ { \text{CP} } ^ { 2 } [ 1 - f ( | \alpha | ) ] ( 1 - | \alpha | / ( \pi / 2 ) ) \quad ( 5 )$$

$$L _ { T } = \frac { 1 } { 2 } \rho _ { f } \ell C _ { L } \sqrt { x _ { 1 } ^ { 2 } + ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell ) ^ { 2 } } \, ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell, x _ { 1 } )$$

$$L _ { \text{R} } = - \frac { 1 } { 2 } \rho _ { f } \ell ^ { 2 } C _ { \text{R} } x _ { 3 } \left ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell, x _ { 1 } \right )$$

$$D = - \frac { 1 } { 2 } \rho _ { f } \ell C _ { \text{D} } \sqrt { x _ { 1 } ^ { 2 } + ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell ) ^ { 2 } } \left ( x _ { 1 }, x _ { 2 } - x _ { 3 } y _ { 1 } \ell \right ) \text{ \quad \ } ( 8 )$$

$$\tau _ { T } = - \frac { 1 } { 2 } \rho _ { f } \ell \sqrt { x _ { 1 } ^ { 2 } + ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell ) } \left [ C _ { L } x _ { 1 } + C _ { D } ( x _ { 2 } - x _ { 3 } y _ { 1 } \ell ) \right ] \left ( \ell _ { C P } - \ell _ { C M } \right ) \quad ( 9 )$$

$$\tau _ { \text{R} } = - \frac { 1 } { 1 2 8 } \rho _ { f } \ell ^ { 4 } C _ { \text{D} } ^ { \pi / 2 } x _ { 3 } \left | x _ { 3 } \right | \left [ \left ( 2 y _ { 1 } + 1 \right ) ^ { 4 } \pm \left ( 2 y _ { 1 } + 1 \right ) ^ { 4 } \right ] \quad \ \ ( 1 0 )$$

mx ˙ 1 = ( m + πρ ℓ / f 2 4 ) x x 3 2 -( πρ ℓ / f 2 4) x ℓ 2 3 CM + L x ′ T + L x ′ R + D x ′ -mg ′ sin x 4 (11)

( m + πρ ℓ / f 2 4 ) ˙ x 2 = -mx x 3 1 +( πρ ℓ / f 2 4) ˙ x ℓ 3 CM + L y ′ T + L y ′ R + D y ′ -mg ′ cos x 4 (12)

$$I \dot { x } _ { 3 } = \tau _ { T } + \tau _ { R }$$

$$\dot { x _ { 4 } } = x _ { 3 }$$

$$\dot { x } _ { 5 } = x _ { 1 } \cos x _ { 4 } - x _ { 2 } \sin x _ { 4 }$$

$$\dot { x } _ { 6 } = x _ { 1 } \sin x _ { 4 } + x _ { 2 } \cos x _ { 4 }$$

## 4 Verification Task

Our Alsomitra model from Sect. 3.2 is used as the basis of a feedback control system with a NN controller, as described in Sect. 2.1. In our case, the plant is the aerodynamic model, and the desired input is a linear reference trajectory in x 5 and x 6 (translational x and y ):

$$x _ { 6 } = - x _ { 5 }$$

The feedback signal consists of the six system states, and the CoM displacement is actuated by a controller aiming to follow the target trajectory (Fig. 5).

Fig. 5. As a control problem, we consider an Alsomitra -inspired microdrone and attempt to follow a linear trajectory in two dimensions.

<!-- image -->

As per the ARCH-COMP airplane and pendulum benchmarks [15], the neural network controller is trained using behaviour cloning. All simulations ran for a total of 20 s, with a model timestep of 0.01 s and a control timestep of 0.5 s. A PID controller actuates y 1 ( e x ) based on an error in x 6 , and the gains are tuned manually until the control system performs well for a range of starting x 6 positions. For each controller actuation (24 per simulation, for nine simulations), the system states, x 6 error, and PID actuation are recorded for use as training data. This data is imported to Python for standard regression learning, and networks are exported in .onnx format for evaluation (Fig. 6) and verification. All networks have 6 inputs, 3 hidden layers with 6, 4, and 1 nodes respectively with ReLU activation functions, and 1 output.

## 4.1 Formalisation

The core of this case study lies in examining the challenges in adopting the existing NNV methods in this new domain. Our ideal formalisation of the problem would be as follows. We consider a hybrid program where the six system states x , ..., x 1 6 change over continuous time t according the dynamics model shown in Sect. 3.2, and a NN controller acts to change the system state discretely every 0.5 s. For any starting state x , ..., x 1 6 (0) , after some time t ∗ the trajectory of the drone will always be within some small distance y ∗ of the target trajectory (ideally, x 6 = -x 5 ). This boils down to the following ideal verification property:

Fig. 6. PID and basic NN controller performance on an Alsomitra -inspired drone. The naive network is trained on regression data obtained from simulations with the PID controller, and the resulting performance is similar but not perfect.

<!-- image -->

$$\forall t \geq t ^ { * }, \forall x _ { 1 }, \dots, x _ { 6 } ( 0 ) \in \mathbb { R } \, \colon | x _ { 6 } ( t ) + x _ { 5 } ( t ) | \leq y ^ { * } \, \text{ \quad \ \ } ( 1 8 )$$

There are several features that distinguishes this system from standard NNV benchmarks, and we aim to explain the technical implications of these challenges for existing verification technologies, and propose ways in which these challenges can be overcome:

- 1. The system dynamics are continuous, therefore unlike standard control verification benchmarks (such as ACAS Xu [5]), control is modelled as a regression task as oppose to classification.
- 2. Unlike the ARCH-COMP benchmarks [25] that have a pre-defined notion of safe and unsafe state, these gliding drones do not have a notion of safety in the sense of a pre-defined coordinate region. A safe state is instead defined in a relational way, as adhesion to certain safe trajectory.
- 3. Unlike many ARCH-COMP benchmarks, our verification task requires modelling with an infinite time horizon. Each drone could stay airborne for an arbitrary duration of time, depending on the surrounding airflow.
- 4. As defined by our model, the dynamics of gliding drones are more complex than what is currently handled by the ARCH-COMP benchmarks and tools, and in particular it is more complex than the dynamics handled by tools that can verify infinite-time horizon systems, such as KeyMaeraX [29].

The available verification tools (Sect. 2.2) do not allow us to formalise this idealised goal directly, since CORA does not support infinite time, and Marabou does not support differential equations. As a result, we simplified this general task as two simpler tasks (in the first case sacrificing the analysis of the overall system dynamics, and in the second case the infinite-time horizon and relational notion of the target state):

- 1. The NN will never command the drone to deviate significantly from the target trajectory. This task was implemented in Marabou, using the Vehicle specification language since it facilitates complex property definition.
- 2. Given an interval of initial positions and a finite time horizon, the NNcontrolled drone will always reach a goal region, defined as a region around the target trajectory within this finite time frame. This task was verified in CORA.

We note that task 1 resembles in some way a robustness property [6], except for now we deal with a regression NN and robustness relative to a line rather than a given data point.

## 5 Robustness Training for Regression

Since robustness is critically important for drone safety, it seems reasonable to attempt a form of adversarial training based on PGD methods for our controller. The guiding hypothesis was that a general improvement in NN robustness should lead to improved verification performance. Since our neural network is a regression model, the classification-based training methods surveyed in Sect. 2.2 could not easily be used without modification. We therefore had to modify the PGD algorithm to use an RMSE loss function instead of cross-entropy, and modify other aspects of the adversarial training algorithm that relied on the presence of discrete classes.

We focused on two notion of robustness, standard and Lipschitz robustness [6]. Given x ∗ ∈ D and constants ϵ, δ, L ∈ R ,

$$\forall x \in R ^ { n } \, \colon \| x - x ^ { * } \| \leq \epsilon \, \Longrightarrow \, \| f ( x ) - f ( x ^ { * } ) \| \leq \delta \, \text{ \quad \ \ } ( 1 9 )$$

$$\forall x \in R ^ { n } \, \colon \| x - x ^ { * } \| \leq \epsilon \implies \| f \left ( x \right ) - f \left ( x ^ { * } \right ) \| \leq L \, \| x - x ^ { * } \| \quad \ \ ( 2 0 )$$

Since the latter has been proven to be strictly stronger than the former in [6], we implemented a form of PGD training with a Lipschitz loss function. During each training epoch, the algorithm finds the worst-case adversarial example ( x ∗ , f ( x ∗ ) ) in an ϵ -ball around each training point ( x , f ( x ) ).To optimise the regression model for Lipschitz robustness, we dynamically compute the highest value of L from the training and adversarial points (according to Eq. 20), which is summed to the training RMSE loss, penalising the network for large gradients about each data point. This is expected to result in a network with a smoother and therefore more robust output, at the expense of some accuracy.

## 6 Property-Driven Training (PDT)

A different PDT method using differentiable logics was independently developed and applied to this case study in a separate submission [13]. As that submission does not include verification experiments, we will evaluate several models optimised for Properties 1, 2, 4, and 5 in [13] for the sake of comparison. The

models are listed in Table 1. In the table, adversarial results represent just by one model trained according to Sect. 5; but DL2 and Gödel Logic models are optimised specifically for Properties 1, 2, 4, and 5 respectively. A value of y ∗ = 2 was chosen for properties 1, 2, and 4 in training, in order to keep the properties relatively strict whilst avoiding counterexamples in the dataset.

## 7 Vehicle Implementation

Task 1 was broken down into five simpler specifications to be implemented in Vehicle (a detailed introduction to which can be found in [10]), to ensure the NNs control the drone as desired in various ways. Global properties relating to the controller's output relative to the target trajectory are introduced in Sect. 7.1, and a local robustness property is introduced in Sect. 7.2. Global properties are verified for all inputs bounded by the training data (representing the entire parameter space over which our controller is trained), and the local property is evaluated about ϵ -balls from the traiing data.

## 7.1 Global Property Specifications

Our first goal is to ensure the controller never causes the drone to deviate from the target trajectory. To establish a performance criteria, properties 1-4 include y ∗ (a threshold distance from the target trajectory, Eqs. 17, 18), such that a critical y ∗ can be found per network per property where verification succeeds. For example, for properties 1 and 2, a lower critical y ∗ would indicate a controller that better adheres to the target trajectory:

- 1. If the drone is above the line by some threshold y ∗ , the NN output will always make the drone pitch down (Listing 1)

$$x _ { 6 } \geq - x _ { 5 } + y ^ { * } \Rightarrow f ( x ) \geq 0. 1 8 7$$

- 2. If the drone is below the line by some threshold y ∗ , the NN output will always make the drone pitch up

$$x _ { 6 } \leq - x _ { 5 } - y ^ { * } \Rightarrow f ( x ) \leq 0. 1 8 7$$

Our third property is reversed, where a larger y ∗ would indicate better adherence to a larger region around the target trajectory:

- 3. If the drone is close to the line by some threshold y ∗ , and at an intermediate pitch angle, the NN output will always be intermediate (Listing 2)

$$- x _ { 5 } - y ^ { * } \leq x _ { 6 } \leq - x _ { 5 } + y ^ { * } \wedge - 0. 7 8 6 \leq x _ { 4 } \leq - 0. 7 4 7 \Rightarrow 0. 1 8 4 \leq f ( x ) \leq 0. 1 9$$

Our fourth property is more complex, and represents a desireable behaviour not present in the data:

```
            Neural Network Verification for Gliding Drone Control: A C

    droneFarAboveLine : UnnormalisedInputVector -> Bool
    droneFarAboveLine x =
            x! d_y >= - x! d_x + ystar

    @property
    property1 : Bool
    property1 = forall x. validInput x and droneFarAboveLine x =>
      alsomitra x! e_x >= 0.187


                           Listing 1: Property 1 implemented in Vehicle.

    intermediatePitch : UnnormalisedInputVector -> Bool
    intermediatePitch x =
            -0.786 <= x! d_theta <= -0.747

    closeToLine : UnnormalisedInputVector -> Bool
    closeToLine x =
            x! d_y >= -x! d_x - ystar and
            x! d_y <= - x! d_x + ystar

    @property
    property3 : Bool
    property3 = forall x. validInput x and intermediatePitch x
    and closeToLine x => 0.184 <= alsomitra x! e_x <= 0.19


                           Listing 2: Property 3 implemented in Vehicle.
```

- 4. If the drone is above and close to the line, pitching down quickly and moving fast, the NN output will always make the drone pitch up

$$- x _ { 5 } \leq x _ { 6 } \leq - x _ { 5 } + y ^ { * } \wedge \ x _ { 3 } \leq - 0. 1 2 \wedge \ x _ { 2 } \leq - 0. 3 \Rightarrow f ( x ) \leq 0. 1 8 7 \ ( 2 4 )$$

```
y : -vvx
y1 = forall x. validInput x and droneFarAboveLine x =>
itra x! e_x >= 0.187


               Listing 1: Property 1 implemented in Vehicle.

diatePitch : UnnormalisedInputVector -> Bool
diatePitch x =
 -0.786 <= x! d_theta <= -0.747

Line : UnnormalisedInputVector -> Bool
Line x =
 x! d_y >= -x! d_x - ystar and
 x! d_y <= - x! d_x + ystar

ty
y3 : Bool
y3 = forall x. validInput x and intermediatePitch x
seToLine x => 0.184 <= alsomitra x! e_x <= 0.19


               Listing 2: Property 3 implemented in Vehicle.


the drone is above and close to the line, pitching down quickly and moving
st, the NN output will always make the drone pitch up

-x5 <= x6 <= -x5 + y*   ^\  x3 <= -0.12   ^\  x2 <= -0.3 => f(x) <= 0.187  (24)

our Vehicle code, alsomitra represent the NN, validInput represents the
space bounded by the training data, and the parameter ystar is defined
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

In our Vehicle code, alsomitra represent the NN, validInput represents the input space bounded by the training data, and the parameter ystar is defined during runtime.

## 7.2 Local Robustness Specification

Our fifth property is an evaluation of robustness around ϵ -balls with respect to the training dataset, as defined in Sect. 5. A detailed introduction to ϵ -ball robustness for image classification implemented in Vehicle can be found in [10]. Similarly to properties 1-4, we are interested in finding at what threshold L value ( L ∗ ) does each network pass verification. Similarly to properties 1-4, we expect the verification results for this property to depend on the strictness of L , which we consider as the parameter L ∗ . However, due to Marabou limitations, this was evaluated with respect to the training dataset, where for each network Property 5 was evaluated for each training point, given fixed L ∗ and ϵ values. Additionally, the distance between points was computed with L ∞ and the input distance could not be included in the formula, leading use to use a different robustness definition:

- 5. For any given input point x , the network output f ( x ∗ ) of any perturbed point x ∗ within an ϵ -ball around x , will have a distance less than or equal to L /ϵ ∗

to f ( x ) (Listing 3)

$$\forall x \in R ^ { n } \, \colon \| x - x ^ { * } \| \leq \epsilon \, \Longrightarrow \, \| f ( x ) - f ( x ^ { * } ) \| \leq L ^ { * } / \epsilon \, \quad ( 2 5 )$$

This definition is less strict than our definition of Lipschitz robustness (Eq. 20), since it is effectively equivalent to standard robustness (Eq. 19) where L /ϵ ∗ = δ . This means that any counterexample to Property 5 will also violate Lipschitz robustness where L = L ∗ , but not the other way around. Since we train for the stronger definition (Sect. 5), we expect to see improved robustness with regard to this weaker definition.

In our Vehicle code, parameters epsilon and Lipschitz are defined during runtime, and n is inferred from the training data (provided in idx format [10]).

```
                                                                                                                                                                                                       
                                                                                                                                                                                                       

                                                                                                                                                                                                        <?xml>,                                                                                                                                                                                                        </?xml>,
```

Listing 3: Property 5 implemented in Vehicle. In this case, the states are defined in normalised terms to avoid scaling issues. Instead of calling the network twice to evaluate f ( x ) and f ( x ∗ ) , the NN is doubled in onnx format so that two sets of inputs and outputs can be evaluated at once.

## 7.3 Verification Results

Table 1. Critical y ∗ values for properties 1-4 (Sect. 7.1), for naive and adversarially trained NNs (Sect. 5), and for PDT NNs (Sect. 6). For properties 1, 2, and 4, a lower y ∗ indicates a controller that better adheres to the target trajectory, and the inverse for Property 3.

| Property      | Naive    | Adversarial   |   DL2 |   Gödel Logic |
|---------------|----------|---------------|-------|---------------|
| 1 (21)        | 46       | 30            |    30 |            27 |
| 2 (22)        | 42       | 42            |    42 |            42 |
| 3 (22) 4 (23) | Failed 0 | Failed 0      |     0 |             0 |

Table 2. Verification success rates ( % ) of Property 5 (Sect. 7.2) for naive and adversarial NNs (Sect. 5), per L ∗ values and ϵ , with respect to the training dataset. A higher success rate means that the NN is robust with respect to more of the training data points. As ϵ increases we are increasing the radius for perturbation around each training point, and a decreasing L ∗ results in a stricter maximum gradient threshold. Empty cells represent properties that timed out before verifying 100 data points.

| Naive   | L ∗       | L ∗      | L ∗     | L ∗    |
|---------|-----------|----------|---------|--------|
| Naive   | 0 . 00001 | 0 . 0001 | 0 . 001 | 0 . 01 |
| 0.00001 | 100       | 100      | 100     | 100    |
| 0.0001  | 96 . 1    | 100      | 100     | 100    |
| 0.001   | 17 . 0    | 98 . 5   | 100     | 100    |
| 0.01    | -         | -        | -       | 100    |

L ∗

| Adversarial   | 0 . 00001   | 0 . 0001   | 0 . 001   |   0 . 01 |
|---------------|-------------|------------|-----------|----------|
| 0.00001       | 100         | 100        | 100       |      100 |
| 0.0001        | 99 . 7      | 100        | 100       |      100 |
| 0.001 ϵ       | 23 . 9      | 99 . 7     | 100       |      100 |
| 0.01          | 0           | 13 . 6     | 92 . 5    |      100 |

These results provide interesting insights from an engineering perspective. From Table 1 there is a clear improvement in performance for Property 1 when implementing adversarial training and PDT, suggesting that our approaches have been successful. However, Properties 1 and 2 only succeed with very large

values of y ∗ -our controllers only adhere to a region around the target trajectory so wide as to be useless in real applications. Properties 3 and 4 failed and succeeded, for all y ∗ values, for all the tested networks, suggesting that they are particularly difficult and easy to verify respectively. Table 2 shows a marginal improvement in robustness performance for our adversarial network, suggesting that our approach has been moderately successful.

## 8 CORA Implementation

## 8.1 Reachability Specification

Since our case study is based on the QUAD benchmark from the ARCH competition [25], our reachability specification is defined similarly. The initial set is:

$$x _ { 1 } = 1, x _ { 2 } = 0, x _ { 3 } = 0, x _ { 4 } = 0, x _ { 5 } = 0, x _ { 6 } \in [ 1. 4 3, 4. 2 9 ] \text{ \quad \ \ (2 6 ) }$$

The reachability goal is for the drone to always be within a distance y ∗ = 2 of the target trajectory after 20 s. To compute reachability, CORA uses set representations (such as zonotopes [2]) and set operations to over-approximate the continuous time reachable set in discrete time steps. Additional considerations include the initial set representation, time step size, controller type, and reachability algorithm - in our case we use a zonotope, 0.01 s, a NN controller, and conservative linearization.

## 8.2 Reachability Results

<!-- image -->

Fig. 7. Reachable regions in x 5 and x 6 for naive and adversarial NNs implemented as controllers, from the initial set defined in Eq. 26. The naive NN fails to reach a region bounded by y ∗ = 2 after 20 s, and the adversarial NN succeeds.

<!-- image -->

<!-- image -->

This result shows significant improvement from the adversarial network compared to the naive (these networks are identical to those from Tables 1 and 2), with the adversarial network adhering much closer to the target trajectory. This would suggest that our adversarial training has been successful, but we note

that whilst these networks are identical in terms of structure, training data, and training epochs, the discrepancy in this plot could be partly due to differences in regression performance.

## 9 Discussion

## 9.1 Limitations and Lessons Learned

A use case such as ours requires multiple tools to verify, each with benefits and drawbacks. Although standard methods apply, modifications are needed for proper implementation, such as with robustness training (Sect. 5). Vehicle does not have integration with Python, and our dynamics model and CORA are in Matlab, requiring the use of three programming languages for our case study. Marabou does not support multiple network calls, so a workaround involving doubling the NN (in .onnx format) was required to evaluate robustness in Vehicle (3). Additionally, for our local robustness property, certain Marabou queries timed out after very few data points (Table 2). Correct handling of normalisation was found to be very important, since the implemented robustness training and verification methods require normalisation but CORA does not support normalisation. To evaluate reachability in CORA with normalised networks required another workaround, incorporating the normalisation arithmetic as extra layers in the network. Furthermore, CORA could not evaluate reachability for the full system of equations described in Sect. 3.2, due to their complexity and the size of the starting set.

Implementing CORA for our application was difficult due to set explosions, where the complexity of the reachability computation would cause an exponential expansion in the set, causing a crash. To reduce the complexity of the equations, the angle of attack definition was simplified ( 1). To solve the initial set size issue, the initial set was divided in x 6 , and the resulting subsets combined to a final reachable set. The reachability timesteps still needed to be small (0.01 s) to avoid set explosions; resulting in long computation times for full reachable sets (over 8 hours in some cases). All reachable sets only involve a starting interval in x 6 (starting values are constants in every other dimension), since increasing the starting interval size in multiple dimensions would cause set explosions. These limitations in CORA were found to stem from the complexity of the partial derivative matrices (Jacobian and Hessian), especially due to a large number of nonlinear terms. CORA could be improved significantly by faster approximations of these derivatives, such as with finite differences instead of symbolic methods, but performing such calculations over sets was found to be non-trivial. ReLU activation functions were also found to reduce computation times compared to sigmoid.

Another consideration for future work is the trade-off between robustness and regression performance for such a controller. For our verification implementations we assume that each NN is capable of controlling the drone relatively well, but that may not always have been the case. A better comparison could be made using NNs with equivalent regression performance on a test set of data, using a

coefficient such as R 2 . Additionally, the effect of PDT [13] on our system was not fully investigated, partly due to differences in network structure and unresolved issues with the Marabou solver.

## 9.2 Conclusion

In this paper, we introduce a novel case study in the verification of gliding drones (Sect. 4). We developed a two-dimensional model for Alsomitra -inspired drones in Sect. 3, presented an ideal verification formalisation in Sect. 4.1, reduced the formalisation into properties manageable by current tools (Vehicle and CORA, Sects. 7 and 8), and presented verification results in Sects. 7.3 and 8.2. We noted challenges that this class of problems presents, among which are its more complex dynamics, under-defined notion of safe and unsafe states, preference for infinite-time horizon guarantees, and its different learning-as-regression regimes. We have shown that in principle, a combination of existing verification tools (Sect. 2.2) and novel training methods (Sects. 5 and 6) could be effectively adopted in order to enable future inclusion of this class of benchmarks into NNV portfolios. We note, however, that verification tasks like this motivate strongly development of tools that cross the boundaries of ARCH-COMP and VNN-COMP on the one hand, and incorporate these tools more smoothly with machine learning toolboxes on the other hand; cohering perhaps with the general agenda of building more complex programming language interfaces for such more complex verification tasks [8]. Technical problems such as insufficient support for normalisation (as reported here) can make a difference between verification success and failure, yet they are often over-looked in papers and tools that are dedicated to implementing NNV algorithms. Whilst we can define and verify for the behaviours that we want to a certain extent, the state of tools makes it difficult to say exactly how well NNs will fulfil their role - even for our relatively simple (two-dimensional, non-turbulent) drone system. For example, Table 1 suggests that our NNs are only guaranteed to adhere to a very large region around the line, and Fig. 7 shows reachable regions but for which the initial set is relatively small (zero-width in 5 dimensions). All of these issues will likely be found on any comparable regression task with complex dynamic equations. If these limitations can be overcome, it will help enable engineers to develop safe and robust control and modelling methods for technologies that improve people's lives and reduce our impact on the environment.
<|endofpaper|>