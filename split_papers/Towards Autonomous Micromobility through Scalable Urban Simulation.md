<|startofpaper|>
## Towards Autonomous Micromobility through Scalable Urban Simulation

Wayne Wu 1 ∗ Honglin He 1 ∗ Chaoyuan Zhang 2 Jack He 1 Seth Z. Zhao 1 Ran Gong 1 Quanyi Li 1 Bolei Zhou 1 1 University of California, Los Angeles 2 University of Washington https://metadriverse.github.io/urban-sim/

Figure 1. Autonomous micromobility. In public urban spaces, various mobile machines (circular images) are essential for short-distance travel. However, urban environments are complex and contain varied terrain and challenging situations (rectangular images). To bridge this gap, we present a scalable urban simulation solution to advance autonomous micromobility. Images are from our Urban-Tra-City data.

<!-- image -->

## Abstract

Micromobility, which utilizes lightweight mobile machines moving in urban public spaces, such as delivery robots and mobility scooters, emerges as a promising alternative to vehicular mobility. Current micromobility depends mostly on human manual operation (in-person or remote control), which raises safety and efficiency concerns when navigating busy urban environments full of unpredictable obstacles and pedestrians. Assisting humans with AI agents in maneuvering micromobility devices presents a viable solution for enhancing safety and efficiency. In this work, we present a scalable urban simulation solution to advance autonomous micromobility. First, we build URBAN-SIM - a high-performance robot learning platform for large-scale training of embodied agents in interactive urban scenes. URBAN-SIM contains three critical modules: Hierarchical Urban Generation pipeline, Interactive Dynamics Generation strategy, and Asynchronous Scene Sampling scheme, to improve the diversity, realism, and efficiency of robot learning in simulation. Then, we propose URBAN-BENCH - a suite of essential tasks and benchmarks to gauge various capabilities of the AI agents in achieving autonomous micromobility. URBAN-BENCH includes eight tasks based on three core skills of the agents: Urban Locomotion, Urban Navigation, and Urban Traverse. We evaluate four robots with heterogeneous embodiments, such as the wheeled and legged robots, across these tasks. Experiments on diverse terrains and urban structures reveal each robot's strengths and limitations.

* Co-first authors.

## 1. Introduction

Micromobility becomes a promising urban transport way for short-distance travel [6, 55]. It includes a range of lightweight machines that have a mass of no more than 350 kg and operate at speeds not exceeding 45 kph [47] in public spaces. These machines encompass mobile robots with different forms, such as wheeled, quadruped, wheeled-legged, and humanoid robots, and assistive mobility devices for elderly and disabled people, such as electric wheelchairs and mobility scooters. They can accommodate various users' needs in individual travel and parcel delivery. The appeal of micromobility lies in its provision of a flexible, sustainable, cost-effective, and on-demand transport alternative, which enhances urban accessibility [50, 65] and reduces reliance on vehicles for short-distance trips [15, 72].

Current road designs predominantly cater to full-sized vehicles [25]. Micromobility machines thus have to move through intricate urban public spaces, such as sidewalks, alleys, and plazas, which contain unpredictable terrains, various obstacles, and dense pedestrian traffic. Traditional micromobility machines rely on either onboard control (like wheelchairs) or teleoperation by humans (like food delivery bots [4]) to navigate complex urban spaces. However, humans and their driven mobile machines face critical safety concerns from human fatigue and limited situational awareness. As reported by FARS [76], over 6,000 vulnerable road users died on U.S. streets in 2018, a 14% increase over 2015 and a 27% increase over 2014. Humans are prone to distractions that can lead to collisions with road hazards. On the other hand, human-driven machines have low operation efficiency , as they require high labor costs and have limited agility. For instance, in teleoperated systems for parcel delivery [4, 5], robots require continuous human monitoring, which limits the number of robots that can be operated simultaneously. Also, given the complexity of the urban environment, human teleoperators may find it challenging to move swiftly through a hustling street.

Autonomous micromobility harnesses embodied AI agents for decision-making and maneuvering, providing a viable way to improve safety and efficiency. Existing AI solutions are mainly targeted at specific abilities of robots, such as obstacle avoidance [70] and parkour [13]. However, micromobility tasks require agents to have versatile capabilities facing various complex and challenging terrains and situations (bottom row in Figure 1), i.e ., traversing varied terrains (stairs, slopes, and rough surfaces), moving on traversable paths in open spaces, and avoiding both static and dynamic obstacles. Current AI solutions, focused on isolated tasks, are thus incapable of conducting complex micromobility tasks. Apart from that, existing robot learning and simulation platforms are insufficient for agent training on micromobility. They either have simple training scenes with no contextual environments [44, 51] or have low training performances without environment parallelization on GPUs [21, 38, 83]. For example, IsaacGym [44] has superior performance but simple environments, while CARLA [21] provides rich town scenes but has low end-to-end training efficiency. However, for micromobility tasks, on the one hand, robots should learn situational awareness by interacting with large-scale scene contexts, such as urban facilities and pedestrians; on the other hand, robots need a high-performance training platform to sample diverse scenes to obtain strong generalizability. Yet, 'large-scale training' with abundant diverse scenes and 'high-performance training' are contradictory in the existing robot learning platforms. Current platforms can not balance these two demands and thus lack sufficient support for autonomous micromobility tasks.

In this work, we present a scalable urban simulation solution to advance autonomous micromobility. This solution consists of two critical components: a robot learning platform URBAN-SIM , and a suite of tasks and benchmarks URBAN-BENCH . It forges a path to autonomous micromobility by enabling large-scale training and evaluation of varied embodied AI agents in complex urban environments.

First, we propose URBAN-SIM - a high-performance robot learning platform for autonomous micromobility. It can automatically construct infinite diverse and realistic interactive urban scenes for large-scale robot learning while providing more than 1,800 fps high training performance with large-scale parallelization in a single Nvidia L40S GPU. URBAN-SIM has three key designs: 1) The Hierarchical Urban Generation pipeline, which can construct an infinite number of static urban scenes, from street block to ground division to building and infrastructure placements to terrain generation. This pipeline remarkably enhances the diversity of training environments. 2) The Interactive Dynamics Generation strategy, which can provide rich dynamics of pedestrians and cyclists that are responsive to robots in real-time during training. This strategy highly improves the realism of dynamic agents while maintaining the performance in our large-scale, distributed robot learning workflows. 3) The Asynchronous Scene Sampling scheme, which can train robots on thousands of various urban scenes on GPUs in parallel. This scheme significantly enhances the training performance , especially for large-scale scenes, achieving more than 26.3 % relative improvement compared to synchronous approaches with the same training steps. URBAN-SIM is built on top of Nvidia's Omniverse [53] and PhysX 5 [54] to provide realistic scene rendering and physics simulation.

Though the goal of autonomous micromobility is to move from point A to B in an urban environment, it requires the multifaceted capabilities of the agent. Thus, we construct URBAN-BENCH - a suite of essential tasks and benchmarks to train and evaluate different capabilities of an agent. We first construct a set of tasks for the agent to

acquire two orthogonal skills in micromobility: Urban Locomotion and Urban Navigation . For urban locomotion, an agent must learn various movement skills to tackle different ground conditions, i.e ., flat surfaces, slopes, stairs, and rough terrain. We define four tasks for urban locomotion based on these ground conditions. For urban navigation, an agent needs to develop different operational skills to manage various scenarios, i.e ., unobstructed ground, static obstacles, and dynamic obstacles. We define three tasks for urban navigation based on these scene conditions. Furthermore, real-world micromobility often requires kilometerscale navigation in complex urban spaces; it remains extremely challenging to tackle this problem. Thus, we define Urban Traverse as a new task with a substantially long time horizon, where a mobile robot needs to make tens of thousands of actions at a kilometer-scale distance. We further introduce a human-AI shared autonomous approach to tackle the task. It is designed with a flexible architecture that ranges from full human control to complete AI management of the workflow, allowing us to explore various labor division modes between humans and AI agents in the urban traverse task.

We construct comprehensive benchmarks across four robots with heterogeneous mechanical structures for all 8 defined tasks. Experimental results demonstrate that all URBAN-BENCH tasks are challenging for existing solutions. By presenting well-defined challenges beyond the capabilities of current solutions, URBAN-BENCH can serve as a unified benchmark that facilitates the future development of autonomous micromobility. Furthermore, through training in complex urban environments, qualitative results indicate that agents have developed interesting and surprising skills based on their mechanical structures. For instance, humanoid robots learn to maneuver through narrow spaces by sidestepping, while wheeled robots learn to navigate around stairs by detouring. Finally, we demonstrate our work's strong scale-up ability, which is essential for learning skills in autonomous micromobility.

## 2. Related Work

## 2.1. Micromobility

Conventional mobility solutions [10], such as cars and buses, primarily operate on structured roadways, suited for medium to long-distance commutes. However, these systems often struggle with last-mile connectivity, where efficient transport is needed for the final leg of a journey, such as moving people from transit hubs to destinations or delivering parcels directly to recipients' doorsteps. Micromobility [6, 55], emerging in Europe and North America in the late 1900s [29, 48], offers a practical solution for shortdistance travel in urban spaces. It relies on lightweight and low-speed devices, such as electric wheelchairs and e-mobility scooters for personal transport [42], or small robots for parcel delivery [19], providing flexible, sustainable, and cost-effective alternatives to private vehicles. This approach reduces emissions [66], alleviates congestion [46], and enhances accessibility [65], especially in densely populated areas.

Recently, a few AI-driven solutions [28, 81] have been introduced in micromobility, focusing on device-sharing systems [73] and scene understanding [85], including fleet management, demand prediction, as well as road change and hazard detection. While these improve operational efficiency, they do not tackle the core challenge of enabling autonomous travel from point A to B in urban spaces. Current solutions lack the embodied intelligence essential for real-time decision-making, which is crucial for tasks like assistive mobility and autonomous delivery.

## 2.2. Simulation Platforms for Robot Learning

Simulation platforms have rapidly advanced over the past decades, offering scalable training for embodied agents and robots, as well as safe evaluation before real-world deployment [16, 17, 36, 59, 71]. Existing platforms mainly focus on two types of environments: 1) indoor environments [57, 62], such as homes and offices, and 2) driving environments [30, 34], like roadways and highways. In indoor environments, platforms like AI2-THOR [31], Habitat [62], iGibson [67], OmniGibson [37], and ThreeDWorld [22] are tailored for tasks like indoor navigation, object rearrangement, and manipulation, which differ greatly from micromobility scenarios in complex urban spaces. In driving environments, platforms like GTA V [45], CARLA [21], DriverGym [33], Nuplan [12], and MetaDrive [38] support medium to long-distance driving tasks, focusing on vehicle-centric road scenarios rather than urban public spaces like sidewalks and alleys, which are crucial for micromobility tasks.

Some recent works have constructed detailed urban spaces [23, 80, 84, 88]. However, these focus mainly on algorithm evaluation [23, 80] or scene generation [84, 88], and lack support for interactive robot training, which requires efficient scene sampling, physical simulation, and real-time dynamics. Recently, task-oriented robot learning platforms, such as IsaacGym [44], IsaacSim [52], and IsaacLab [51], built on Nvidia ecosystem, have shown impressive training efficiency with high visual and physical realism. However, these platforms are mainly suited for repetitive tasks in uniform environments, like locomotion and manipulation, and often neglect contextual scene simulation needed for complex, long-horizon micromobility tasks.

## 2.3. Robot Autonomy Tasks

Recent advances in robotics and embodied AI have significantly enhanced specific skills for robot autonomy, particularly in locomotion [32] and navigation [20]. In locomotion, the main goal is to enable robots to move efficiently across

## (a) Hierarchical Urban Generation

Stage 1 - Block Connection

<!-- image -->

<!-- image -->

Stage 2 - Ground Planning

<!-- image -->

Stage 4 - Object Placement

Stage 3 - Terrain Generation

<!-- image -->

## (b) Interactive Dynamics Generation

Realistic Interactions of Humans  and Robots

<!-- image -->

## (c) Asynchronous Scene Sampling

Large-scale Robot Training in Parallel across Diverse Environments on GPUs

<!-- image -->

Figure 2. URBAN-SIM : a robot learning platform for autonomous micromobility. (a) Hierarchical Urban Generation. It generates an infinite number of diverse scenes through four progressive stages. (b) Interactive Dynamics Generation. GPU-based generation of realistic agent-scene and agent-agent interactions on the fly. (c) Asynchronous Scene Sampling. An asynchronous sampling scheme to enable high-efficiency training on varied scenes with rich context information.

diverse terrains. Considerable progress has been achieved in tasks categorized by different mechanical structures ( e.g ., bipedal [39], quadrupedal [8], multilegged [14]) or unique abilities ( e.g ., parkour [13], whole-body control [41], jumping [69]). In navigation, the focus is on guiding robots to specific destinations while avoiding obstacles. Research has proposed various tasks categorized by goals and conditions, such as point navigation [9], object navigation [87], and social navigation [75]. However, these tasks address isolated skills and struggle to meet micromobility's demands, which require unique and versatile abilities for complex urban environments. A few pioneering studies have explored long-horizon outdoor navigation tasks, but they are limited to case-specific robots [35, 49] and scenarios [64, 70], lacking the generalizability needed for micromobility tasks. In this work, we evaluate holistic tasks across different robots, from foundational abilities like locomotion and navigation to comprehensive tasks like traverse, which are essential for advancing autonomous micromobility in urban environments.

## 3. URBAN-SIM : A Robot Learning Platform for Autonomous Micromobility

To support robot learning in complex urban scenes, an ideal simulation platform needs to have two important features: large-scale - the platform should provide a vast array of diverse scenes with realistic interactions; and highperformance - the platform should support high-efficiency scene sampling for training. In this section, we introduce URBAN-SIM -a robot learning platform for autonomous micromobility, which can balance the contradiction between scale and performance. It supports infi-

nite urban scene generation with arbitrary size and achieves high-performance training with more than 1,800 fps sampling rate in a single GPU. We highlight three key designs of URBAN-SIM : the Hierarchical Urban Generation pipeline (Section 3.1), which ensures the diversity of static scenes on a large scale; the Interactive Dynamics Generation strategy (Section 3.2), which ensures the realism of dynamics on a large scale; and the Asynchronous Scene Sampling scheme (Section 3.3), which ensures highefficiency training on complex urban environments.

## 3.1. Hierarchical Urban Generation

The diversity of simulation environments is essential for the robustness and generalizability of robot training, especially in deep learning approaches. Following recent advancements in procedural generation in games [68], we introduce a hierarchical urban generation pipeline to procedurally create complex urban scenes, from macroscale street blocks to microscale terrains, enabling infinite variations of diverse scenes with arbitrary sizes (from a street corner to a city).

As shown in Figure 2 (a), this pipeline includes four progressive stages: 1) In block connection, street blocks ( e.g ., straight, curve, roundabout, diverging, merging, intersection, and T-intersection) are sampled and connected to form diverse road networks. 2) In ground planning, we divide urban public areas into functional zones ( e.g ., sidewalks, crosswalks, plazas, buildings, and vegetation) using randomized parameters for each area's dimensions. 3) In terrain generation, we employ the Wave Function Collapse (WFC) [27] algorithm to generate typical urban terrains - flat ( e.g ., pathway on grass), stair ( e.g ., front steps), slope ( e.g ., assistive ramps), and rough ( e.g ., cracked sidewalks) - each with adjustable parameters like step height or ramp angle, providing diverse ground conditions. 4) In object placement, static objects ( e.g ., trees and bus stops) are placed adaptably within the functional areas according to their sizes, creating varied obstacle layouts. To ensure the coverage of objects, we have compiled a repository of over 15,000 high-quality 3D assets of urban objects. This pipeline enables the creation of enormous static urban scenes with diverse street layouts, functional divisions, obstacles, and terrains in a breeze 1 .

## 3.2. Interactive Dynamics Generation

Beyond static scene diversity, the realism of dynamic agents, i.e ., vehicles, pedestrians, and other mobile machines, is crucial for simulated urban environments. To form realistic dynamics, the environmental agents should be interactive, with both the static scenes and other dynamic agents. A naive approach uses multi-agent path planning algorithms like ORCA [79] to optimize agents' trajectories,

1 Empowered by the UI of Omniverse [53], users can easily modify the scenes generated by our pipeline further, to cater to specific needs.

avoiding collisions and deadlocks. However, these methods pre-compute trajectories, preventing real-time interaction with the trained agent, and run only on the CPU, causing inefficiencies when integrated with GPU-based platforms due to the frequent CPU-GPU data transfer during training.

To address these issues, we follow Waymax [26] and JaxMARL [61] in upgrading ORCA with JAX [11] for multiagent path planning on GPUs without any CPU bottlenecks. This method enables parallelization across multiple environments for simultaneous collision avoidance with static and dynamic objects and interaction with the trained agent. Specifically, we first generate a 2D occupancy map labeling obstacles, roadways (for vehicles), and traversable areas (for pedestrians and mobile machines), then sample random start and end points for each agent. Using ORCA for initial trajectories, we adjust agents' positions in real-time based on proximity and relative velocity, all on GPUs. We illustrate the realistic interactions between agents and environments and other agents in Figure 2 (b). This strategy enables the creation of dynamic environments with realistic interactions on the fly in robot training.

## 3.3. Asynchronous Scene Sampling

So far, we can generate diverse scenes with realistic dynamics. However, the complexity of these scenes, with numerous objects and dense physical interactions, poses new challenges for the training performance, especially in learning long-horizon behaviors for robots with high degrees of freedom. Recent robot learning platforms like IsaacGym [44] and IsaacLab [51] achieve high performance through environment parallelization on GPUs. These platforms are designed for tasks that require extensive repetitive training in uniform environments with enormous trial and error, such as locomotion and manipulation. In micromobility tasks, however, rather than uniform environments, robots must make decisions based on varied environments with rich contextual information, such as ground paving, obstacles semantics, and pedestrian movements. Thus, existing synchronous scene sampling in [44, 51] will encounter huge barriers facing micromobility tasks, where the essential is not the repetitive training in uniform environments but the multi-faceted training in enormous varied environments .

To solve this problem, we propose an asynchronous scene sampling scheme, which can remarkably enhance training efficiency by training simultaneously on thousands of non-uniform environments with various static layouts, obstacles, dynamics, terrains, and episodes of agents. Specifically, as illustrated in Figure 3, all assets are initially loaded into a cache, from which environments randomly sample assets to create diverse settings simultaneously. Observations, rewards, and actions for each environment are fully vectorized on the GPU, enabling efficient parallel training of agents across multiple environments. Figure 2 (c) visualizes the parallel training on varied

environments simultaneously with the asynchronous scene sampling scheme. This approach significantly accelerates model convergence and reduces training time, essential for context-aware micromobility tasks.

Figure 3. Scene sampling diagram. (Left) Assets Cache that stores all assets in urban scenes. (Right) With a random sampling of assets, parallel environments can be constructed on GPU.

<!-- image -->

Performance benchmarking. Using the asynchronous scene sampling scheme, we can enable parallelization with any number of unique environments, depending on the GPU used. On a single GPU, parallelized training can be conducted across 256 environments, achieving performance ranging from 1,800 to 2,600 fps with RGBD sensors, depending on the specific scenario. Note that, due to the scalable nature of our platform, the sampling rate can be continually increased by adding more GPUs. Please refer to the Appendix for detailed performance benchmarks.

## 4. URBAN-BENCH : A Suite of Essential Tasks for Autonomous Micromobility

In this section, we introduce URBAN-BENCH , a suite of essential tasks and benchmarks that capture high-frequency scenarios in autonomous micromobility. Based on the data from users of micromobility, we first summarize several key Human Needs (Section 4.1) as the basis of the task definition. The real-world demands for micromobility devices mainly ask for two primary skills: Urban Locomotion (Section 4.2) - moving stably across diverse terrains, including flat, slope, stair, and rough surfaces, and Urban Navigation (Section 4.3) - moving efficiently in spaces with varying conditions like unobstructed pathways, static, and dynamic obstacles. Furthermore, we define a long-horizon task, Urban Traverse (Section 4.4), where robots must navigate urban spaces at kilometer scales. To tackle this challenging task, we introduce a pilot approach human-AI shared autonomy - leveraging the power of both humans and AI agents. We will present benchmark results for these tasks in Section 5.

## 4.1. Tasks Grounded in Human Needs

The selection of tasks in URBAN-BENCH is informed by urban mobility studies and infrastructure assessments, highlighting their practical importance. U.S. Department of Transportation (DOT) reports [77] indicate the prevalence of diverse terrains like ramps, stairs, and uneven surfaces in public spaces, so it is necessary to have various locomotion capabilities, including slope traversal , stair climbing , and rough terrain traversal . Besides, the National Household Travel Survey (NHTS) [78] indicates that a significant portion of urban travel involves short trips on sidewalks and plazas, where micromobility devices must navigate both unobstructed pathways and crowded zones. This underscores the need for safe and efficient clear pathway traversal , and static and dynamic obstacle avoidance . Based on these scene conditions, we define a set of essential tasks of urban locomotion and navigation.

## 4.2. Urban Locomotion

In urban locomotion, the embodied AI agent controls the robot's locomotion, ensuring stable and efficient movement across various terrains such as flat surfaces, slopes, and stairs. We define four tasks for urban locomotion (Figure 4 (a)) based on different ground conditions as below:

LOC LocoFlat → Flat Terrain Traversal: Traversing stable, flat surfaces commonly found on sidewalks and pedestrian zones. This is necessary for basic mobility in city spaces designed for foot traffic.

LOC LocoSlope → Incline Ascent and Descent: Moving up and down ramps and inclined surfaces with varying slope angles. This is essential in urban areas where slopes and accessibility ramps are common.

LOC LocoStair → Stair Ascent and Descent: Ascending and descending stairs with varying heights. This is critical in urban spaces where ramps are unavailable, allowing access to multi-level areas.

LOC LocoRough → Uneven Terrain Traversal: Maintaining stability on uneven surfaces like cobblestones or damaged sidewalks. This is important for robust movement in urban environments with irregular, worn-down paths.

## 4.3. Urban Navigation

In urban navigation, the embodied AI agent handles local navigation, determining how the robot should move to stay within traversable areas while avoiding obstacles and pedestrians. We define three tasks for urban navigation (Figure 4 (b)) based on different scene conditions as below:

NAV NavClear → Traversable Area Finding: Moving across open, unobstructed ground, avoiding non-walkable areas like mud or bushes. This is essential for efficient navigation on open plazas and trails on lawns.

NAV NavStatic → Static Obstacle Avoidance: Navigating around stationary urban obstacles such as benches, trash bins, and signposts. This is vital for safely maneuvering in crowded city environments with fixed structures.

NAV NavDynamic → Dynamic Obstacle Avoidance: Adjusting paths to avoid moving obstacles like pedestrians and cyclists. This is crucial in urban spaces with high foot traffic, ensuring safe interactions with moving entities.

## (a) Urban Locomotion

<!-- image -->

LocoFlat

(b) Urban Navigation

<!-- image -->

NavClear

<!-- image -->

LocoRough

<!-- image -->

LocoStair

<!-- image -->

LocoSlope

<!-- image -->

(c) Urban Traverse

<!-- image -->

Traverse

Figure 4. URBAN-BENCH : a suite of essential tasks for autonomous micromobility. Simulation environments of eight essential tasks of (a) Urban Locomotion, (b) Urban Navigation, and (c) Urban Traverse.

<!-- image -->

## 4.4. Urban Traverse

In kilometer-scale urban traverse, the embodied AI agent's goal is to reach the target point as efficiently as possible, minimizing travel time while ensuring safety in the journey. We define the urban traverse task (Figure 4 (c)) as below:

TRA Traverse → Urban Traverse: Moving from point A to point B with a distance of more than 1 km within a complex urban environment safely and efficiently. A challenging real-world setting for micromobility.

Human-AI shared autonomous approach. We propose a human-AI shared autonomous approach as a pilot study to address this task, combining AI capabilities with human interventions. In this approach, we structure the robot control into three layers: high-level decision-making, mid-level navigation, and low-level locomotion. With the layered architecture, we decompose the complex urban traverse task into a series of subtasks, with AI managing mid-level and low-level routine tasks, and humans making high-level decisions and intervening in risky situations. This approach allows a flexible transition between human and AI control. Humans can manage the entire process if needed, while AI can manage the entire operation using an extra rulebased/AI-based decision model to direct the dispatch of urban navigation and locomotion models. We evaluate these control variants to study micromobility performance at the kilometer scale in Section 5. Please refer to the Appendix for a detailed discussion of this approach.

## 5. Benchmarks

We benchmark four tasks in urban locomotion, three tasks in urban navigation, and one long-horizon task in urban traverse. We describe the benchmarks below regarding the

Settings (Section 5.1) of robots, data, and models, as well as the analysis of the Results (Section 5.2) of benchmarks. These benchmarks will be maintained and updated as time goes on to cover more robots, tasks, and models, as we aim to build a standard evaluation platform to facilitate research in autonomous micromobility and robot learning in urban spaces. Please see the Appendix for more details, including data, training parameters, evaluation metrics, etc .

## 5.1. Settings

Robots. Weevaluate four representative robots, each with distinct mechanical structures, to gain insights and demonstrate the general applicability of the proposed platform. The robots selected for this study include a wheeled robot (COCO Robotics' delivery robot), a quadruped robot (Unitree Go2), a wheeled-legged robot (Unitree B2-W), and a humanoid robot (Unitree G1) 2 .

Data. We construct 4 datasets in URBAN-SIM : UrbanNav is used for the training and testing of urban navigation; Urban-Loc is used for the training and testing of urban locomotion; Urban-Tra-Standard and Urban-Tra-City are used for the testing of urban traverse.

Models. For the urban navigation and locomotion task, we formulate it as a Markov Decision Process (MDP) [60], where the AI learns to optimize its navigation or locomotion policy using the reinforcement learning algorithm Proximal Policy Optimization (PPO) [63]. For each robot, we train and test three models for urban navigation tasks on UrbanNav and four models for urban locomotion on Urban-Loc (except wheeled devices), which form a 24-model matrix .

2 It is simple to import new robots in URBAN-SIM .

For the urban traverse task, we construct 4 control modes, spanning from the full human to full AI: Human - a full human control mode; Human-AI-Mode-1 - a human AI shared control mode with the dispatch of foundational navigation and locomotion models; Human-AI-Mode-2 - a human AI shared control mode with the dispatch of foundational navigation models and a general locomotion model; AI - a full AI control model.

## 5.2. Results

Urban locomotion benchmark. Table 1 brings the following insights: 1) Quadruped robot achieves optimal smoothness : The quadruped robot consistently demonstrates the best Smoothness scores across all terrains, highlighting its stability and controlled movement, even on challenging surfaces like stairs and rough ground. 2) Wheeledlegged robot excels in versatility : Leveraging its hybrid leg-wheel design, the wheeled-legged robot leads in both distance traversal ( X -displacement and Time to Fall) and keeping Balance, enabling it to cover diverse urban terrains efficiently. 3) Humanoid robot shows stability on even surfaces : The Humanoid robot achieves the best Balance performance on both flat and inclined ground, indicating its capability for navigation in even urban environments.

Table 1. Urban Locomotion benchmark. Different colors indicate the best performance of different metrics among three robots: Balance; X -displacement; Time to Fall (TTF); Smoothness.

<!-- image -->

| Metrics                                         | LOC LocoFlat                                                          | LOC LocoSlope                                                        | LOC LocoStair                                                        | LOC LocoRough                                                        |
|-------------------------------------------------|-----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Quadruped Robot                                 | Quadruped Robot                                                       | Quadruped Robot                                                      | Quadruped Robot                                                      | Quadruped Robot                                                      |
| Balance (%) ↑ X -dis. (m) ↑ TTF (s) ↑ Smooth. ↓ | 100 . 00 ± 0 . 00 19 . 58 ± 0 . 41 20 . 00 ± 0 . 00 7 . 85 ± 0 . 04   | 90 . 56 ± 3 . 13 4 . 63 ± 0 . 23 19 . 50 ± 0 . 44 5 . 18 ± 0 . 07    | 91 . 89 ± 2 . 07 9 . 20 ± 0 . 36 19 . 58 ± 0 . 39 8 . 11 ± 0 . 12    | 72 . 18 ± 4 . 76 4 . 88 ± 0 . 14 18 . 31 ± 0 . 25 10 . 02 ± 0 . 09   |
| Wheeled-Legged Robot                            | Wheeled-Legged Robot                                                  | Wheeled-Legged Robot                                                 | Wheeled-Legged Robot                                                 | Wheeled-Legged Robot                                                 |
| Balance (%) ↑ X -dis. (m) ↑ TTF (s) ↑ Smooth. ↓ | 100 . 00 ± 0 . 00 19 . 62 ± 0 . 15 20 . 00 ± 0 . 00 210 . 43 ± 0 . 07 | 95 . 57 ± 3 . 31 12 . 54 ± 0 . 34 19 . 95 ± 0 . 02 253 . 24 ± 0 . 28 | 83 . 01 ± 2 . 37 16 . 73 ± 0 . 27 19 . 07 ± 0 . 17 236 . 52 ± 0 . 18 | 85 . 04 ± 2 . 16 18 . 24 ± 0 . 22 19 . 13 ± 0 . 11 231 . 96 ± 0 . 14 |
| Humanoid Robot                                  | Humanoid Robot                                                        | Humanoid Robot                                                       | Humanoid Robot                                                       | Humanoid Robot                                                       |
| Balance (%) ↑ X -dis. (m) ↑ TTF (s) ↑ Smooth. ↓ | 100 . 00 ± 0 . 00 16 . 61 ± 0 . 50 20 . 00 ± 0 . 00 40 . 94 ± 0 . 15  | 95 . 67 ± 2 . 24 7 . 16 ± 0 . 22 19 . 91 ± 0 . 03 57 . 69 ± 0 . 31   | 80 . 98 ± 4 . 32 13 . 99 ± 0 . 27 19 . 03 ± 0 . 36 42 . 36 ± 0 . 19  | 82 . 45 ± 3 . 15 16 . 28 ± 0 . 31 19 . 02 ± 0 . 33 53 . 67 ± 0 . 24  |

Urban navigation benchmark. Table 2 brings the following insights. 1) Wheeled robot excels in clear pathway navigation : The wheeled robot achieves the highest Success Rate (97.60%) and Route Completion (98.61%) in the NavClear task, highlighting its suitability for open, predictable urban environments. 2) Quadruped robot leads in safety metrics : The quadruped robot outperforms others in tasks with obstacles, achieving the lowest Collision rates (0.08 in NavSta and 0.13 in NavDyn ) and the highest percentage On Walkable Regions. This demonstrates its stabil- ity in complex, obstacle-rich environments. 3) Humanoid robot performs best in complex scenarios : The humanoid robot shows the highest Success Rates and Route Completion in tasks with static and dynamic obstacles, indicating its flexibility in navigating crowded urban spaces.

Table 2. Urban navigation benchmark. Different colors indicate the best performance of different metrics among four robots: Success Rate; Route Completion; On Walkable Region; SPL; Collision.

<!-- image -->

| Metrics                                                                              | NAV NavClear NAV                                                     | NavStatic                                                                          | NAV NavDynamic                                                                                                                        |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Wheeled Robot                                                                        | Wheeled Robot                                                        | Wheeled Robot                                                                      | Wheeled Robot                                                                                                                         |
| Success Rate (%) ↑ Route Completion (%) ↑ On Walkable Region (%) ↑ SPL ↑ Collision ↓ | 97 . 60 ± 0 . 92 98 . 61 ± 1 . 28 74 . 38 ± 0 . 99 0 . 48 ± 0 . 05 - | 51 . 95 ± 2 . 63 53 . 11 ± 2 . 92 81 . 88 ± 1 . 00 0 . 24 ± 0 . 04 0 . 31 ± 0 . 09 | 48 . 82 ± 3 . 26 50 . 04 ± 3 . 02 84 . 82 ± 1 . 49 0 . 23 ± 0 . 01 0 . 35 ± 0 . 04                                                    |
| Quadruped Robot                                                                      | Quadruped Robot                                                      | Quadruped Robot                                                                    | Quadruped Robot                                                                                                                       |
| Success Rate (%) ↑ Route Completion (%) ↑ On Walkable Region (%) ↑ SPL ↑ Collision ↓ | 90 . 29 ± 3 . 25 94 . 28 ± 2 . 16 93 . 96 ± 3 . 38 0 . 37 ± 0 . 05 - | 76 . 13 ± 3 . 07 77 . 47 ± 2 . 99 85 . 81 ± 1 . 67 0 . 36 ± 0 . 04 0 . 08 ± 0 . 02 | 77 . 14 ± 2 . 57 77 . 63 ± 2 . 12 88 . 20 ± 2 . 17 0 . 36 ± 0 . 05 0 . 13 ± 0 . 02                                                    |
| Wheeled-Legged Robot                                                                 | Wheeled-Legged Robot                                                 | Wheeled-Legged Robot                                                               | Wheeled-Legged Robot                                                                                                                  |
| Success Rate (%) ↑ Route Completion (%) ↑ On Walkable Region (%) ↑ SPL ↑ Collision ↓ | 79 . 94 ± 3 . 06 80 . 44 ± 2 . 97 67 . 93 ± 0 . 85 0 . 37 ± 0 . 03 - | 42 . 97 ± 4 . 14 44 . 33 ± 3 . 74 62 . 17 ± 2 . 95 0 . 19 ± 0 . 02 0 . 15 ± 0 . 04 | 31 . 06 ± 3 . 77 33 . 95 ± 3 . 21 63 . 29 ± 2 . 71 0 . 14 ± 0 . 02 0 . 19 ± 0 . 02 79 . 23 ± 2 . 71 80 . 26 ± 2 . 92 65 . 85 ± 1 . 94 |
| Humanoid Robot                                                                       | Humanoid Robot                                                       | Humanoid Robot                                                                     | Humanoid Robot                                                                                                                        |
| Success Rate (%) ↑ Route Completion (%) ↑ On Walkable Region (%) ↑ SPL ↑ Collision ↓ | 80 . 47 ± 2 . 29 80 . 92 ± 1 . 36 65 . 86 ± 1 . 56 0 . 37 ± 0 . 01 - | 77 . 86 ± 3 . 54 79 . 72 ± 2 . 76 86 . 89 ± 1 . 73 0 . 37 ± 0 . 03 0 . 13 ± 0 . 03 | 0 . 38 ± 0 . 03 0 . 15 ± 0 . 04                                                                                                       |

Urban traverse benchmark. We evaluate a quadruped robot on a kilometer-scale urban traverse task using the Urban-Tra-Standard dataset with three control modes. As shown in Figure 6, the AI mode achieves the lowest human intervention but exhibits the poorest completeness and safety. Conversely, the Human mode achieves the highest completeness and safety but at a significantly higher labor cost. The two human-AI shared autonomy modes balance completeness and cost while maintaining moderate safety. Future research in urban traverse should aim to move the dot closer to the origin with minimal dot size, indicating optimized completeness, cost, and safety. Please refer to the Appendix for the complete benchmark of urban traverse.

Emerging robot behaviors. Through large-scale training in diverse urban environments, different robots obtain movement skills that exploit their unique mechanical structures, as shown in Figure 5: quadruped robots, known to be proficient at stair climbing, can traverse challenging terrain directly to reach the goal; wheeled robots prefer detouring over even surfaces to reduce the risk of getting stuck, despite the longer path; Wheeled-legged robots benefit from

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Figure 5. Emerging behaviors. The results of evaluating different robots in the same environment. After training in diverse urban scenes, robots with distinct structures have developed their unique movement skills.

<!-- image -->

8 to 256, a substantial performance gap (the colored areas) emerges, showing the strong scalability of our platform for diverse scene training. Further, as seen in Figure 7 (Right), the performance remarkably improves as the number of training scenes increases from 1 to 1,024, rising from 5.1 % to 83.2 % (Success Rate). The result highlights the importance of large-scale training on a greater variety of scenes.

Figure 6. Comparison of different control modes in urban traverse. X-axis: Attempts to Success - the number of failures before reaching the goal points (completion ability). Y-axis: Human Cost - time of human takeover of the control (labor cost). Size of circle: Collision Times to obstacles and pedestrians (safety property). indicate four control modes.

their hybrid design and show the ability to partially descend on slopes and stairs simultaneously; The humanoid robot, with greater degrees of freedom, can sidestep through narrow spaces efficiently.

## 6. Evaluation of Scalability

We try to address a fundamental question underlying the strengths demonstrated in this work: How does the scalability of our urban simulation contribute to autonomous micromobility?

The proposed asynchronous scene sampling scheme in URBAN-SIM enables high-performance, large-scale robot training in diverse urban environments with realistic interactions. We compare it to synchronous sampling, as used in IsaacLab [51], where all scenes in a batch are identical. In our asynchronous approach, however, each scene in a batch is unique. Furthermore, to assess the impact of large-scale training, we vary the number of training scenes from 1 to 1,024 and observe performance changes. All experiments are conducted using the NavStatic task.

As shown in Figure 7 (Left), asynchronous sampling performs the same as synchronous sampling with only one scene. However, as unique training scenes increase from

Figure 7. Effectiveness of scalable urban simulation. (Left) Comparison between synchronous and synchronous scene sampling. X-axis: training steps; Y-axis: Success Rate. Different colors indicate training scene numbers - 1, 8, or 256. (Right) Scaling-up ability. X-axis: training scene number; Y-axis: Success Rate and Route Completion.

<!-- image -->

<!-- image -->

## 7. Conclusion

We introduce a scalable urban simulation solution to advance research in autonomous micromobility. This solution comprises a high-performance robot learning platform, URBAN-SIM , and a suite of essential tasks and benchmarks, URBAN-BENCH . Through experiments, we evaluate various capabilities of AI agents across different tasks and demonstrate the platform's scalability for large-scale training in urban environments. Looking ahead, we plan to support real-world deployments of models trained on our platform. Our strategy includes building a sim-to-real pipeline based on ROS2 and enabling an integrated workflow for model training, evaluation, and deployment.

Acknowledgements. The project was supported by the NSF grants CNS-2235012 and IIS-2339769, and ONR grant N000142512166. We extend our gratitude for the excellent assets, including 3D objects from Objaverse-XL, 3D humans from SynBody, and robots from IsaacLab.
<|endofpaper|>