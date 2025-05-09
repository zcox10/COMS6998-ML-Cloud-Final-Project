<|startofpaper|>
## T2I-R1:

## Reinforcing Image Generation with Collaborative

## Semantic-level and Token-level CoT

Dongzhi Jiang ∗ 1 , Ziyu Guo ∗ 2 , Renrui Zhang ∗† 1 , Zhuofan Zong , Hao Li 1 3 1

Le Zhuo 1 3 , , Shilin Yan, Pheng-Ann Heng , Hongsheng Li 2

1 CUHK MMLab 2 CUHK MiuLar Lab 3 Shanghai AI Laboratory {dzjiang, ziyuguo, renruizhang}@link.cuhk.edu.hk

hsli@ee.cuhk.edu.hk

∗ Equal Contribution

† Project Leader

## Abstract

Recent advancements in large language models have demonstrated how chain-ofthought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1 , a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generation CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: https://github.com/CaraJ7/T2I-R1 .

## 1 Introduction

The emergence of advanced Large Language Models (LLMs) [45, 47, 63, 72], such as OpenAI o1 [48] and DeepSeek-R1 [17], has demonstrated considerable reasoning capabilities across domains including mathematics [1, 20, 42] and coding [6, 2, 23]. Through reinforcement learning (RL) [55, 56], these models analyze problems progressively with a comprehensive Chain-of-Thought (CoT) [66, 27, 19, 25, 77, 18] before providing answers, significantly enhancing output accuracy.

The CoT reasoning strategies have also been extended to the visual domain. Recent Large Multi-modal Models (LMMs) [5, 43, 75, 78] have adapted the paradigm to accommodate the visual understanding task [40, 77, 26]. These advanced LMMs can jointly process images and their associated textual queries, performing step-by-step analyses of visual details and integrating them with reasoning steps to derive final answers. Concurrently, CoT-like reasoning has been initially investigated in the visual generation task, particularly in autoregressive text-to-image generation. The pioneering work, 'Image Generation with CoT' [19], regards the progressive generation of the image tokens as a kind of CoT analogous to that of the text tokens, and proposes to optimize this intermediate process to enhance the image quality.

## CoT in Image Understanding

Figure 1: The Illustration of CoT in Image Understand and Generation Tasks. In the image understanding task, the CoT is the textual reasoning process. In the autoregressive visual generation task, we identify two levels of CoT: the semantic-level and token-level CoT. The semantic-level CoT is the high-level planning prior to the image generation, in the form of text. The token-level CoT is the intermediate patch-by-patch generation process, focusing on the local pixel details within a patch, in the form of image tokens.

<!-- image -->

Despite these advances, the exploration of CoT for image generation remains preliminary. Unlike image understanding, image generation requires the complex interpretation of cross-modal alignment and the synthesis of fine-grained visual details. To address these challenges, we identify two distinct levels of CoT reasoning that can be leveraged to enhance image generation, as illustrated in Fig. 1:

- · Semantic-level CoT is the textual reasoning about the image to generate, which is introduced prior to the image generation. The semantic-level CoT designs the global structure of the image, e.g., the appearance and location of each object. In case the prompt requires reasoning shown in Fig. 2, the semantic-level CoT also helps to deduce the objects to generate. Optimizing the semantic-level CoT could explicitly manage the planning and reasoning of the prompt before the subsequent image tokens generation, making the generation easier.
- · Token-level CoT is the intermediate patch-by-patch generation process of the image, as originally introduced in [19]. This process could be viewed as a form of CoT as it outputs each subsequent token conditioned on all previous tokens within a discrete space, similar to the textual CoT. Unlike semantic-level CoT, token-level CoT focuses on low-level details like pixel generation and maintaining visual coherence between adjacent patches. Optimizing the token-level CoT can enhance both the generation quality and the alignment between the prompt and the resulting images.

Despite recognizing these two levels of CoT, a critical question remains unaddressed: How can we enhance and coordinate them for text-to-image generation? Current mainstream generative models [58, 61, 53, 28] are trained exclusively on generation targets, lacking the explicit textual understanding required for semantic-level CoT reasoning. Although introducing a separate model (e.g., an LLM) specifically for prompt interpretation [9] is technically feasible, this approach would significantly increase computational costs, complexity, and deployment challenges. Recently, a trend has arisen to merge visual understanding and generation within a single model. Building upon LMMs, these unified LMMs (ULMs) [67, 70, 79, 7] could not only understand the visual inputs but also generate images from text prompts. However, their two capabilities are still decoupled, typically pre-trained in two independent stages, with no clear evidence that the understanding capabilities can

Janus-Pro

<!-- image -->

Figure 2: Visualization of the Image Generation Process of T2I-R1. All the prompts need reasoning or contain an uncommon scenario. We observe that T2I-R1 successfully deduces the true intention behind the prompt or provides a sensible imagination for the uncommon scenario (highlighted in the text) to produce a satisfying result compared with the baseline model, Janus-Pro.

benefit generation. Given these potentials and issues, we start from a ULM and enhance it to unite both the semantic-level and token-level CoT into one framework for text-to-image generation.

To fulfill our target, we introduce BiCoT-GRPO , an RL method to jointly optimize the two levels of CoT for ULM. We opt for RL instead of supervised fine-tuning (SFT) for two reasons: First, the ULM has possessed the fundamental ability needed for the semantic-level and token-level CoT; our goal is only to elicit the fusion of these two abilities by guiding the model's self-exploration. Second, RL methods have proven highly effective for enhancing reasoning capabilities, which are essential for both levels of CoT. Specifically, we first instruct the ULM to imagine and plan the image based on the prompt to obtain the semantic-level CoT. Then, we feed it into the ULM as the condition for the subsequent image generation for token-level CoT. We simultaneously generate multiple images from each prompt and then compute group-relative reward to optimize both levels of CoT within the same iteration. Unlike understanding tasks, where clearly defined rules for rewards exist, image generation lacks such standardized rules. Therefore, we propose to utilize an ensemble of diverse vision experts [68, 64, 38, 19] as reward models. This reward design serves two critical purposes: it evaluates generated images from multiple dimensions to ensure reliable quality assessment, while also functioning as a regularization method to prevent the ULM from hacking a single reward model.

Through the proposed reasoning strategies, we obtain T2I-R1 , the first reasoning-enhanced text-toimage model combining the semantic-level and token-level CoT. Empirical results show that our approach outperforms baseline models by 13% and 19% improvements on the T2I-CompBench and WISE benchmark, and even surpasses the previous state-of-the-art model FLUX.1. Qualitative analysis reveals that our method empowers the model to generate more human-aligned results by reasoning about the true intentions behind the prompt and demonstrates enhanced robustness when dealing with uncommon scenarios.

Our contributions are summarized as follows:

- 1. We identify a dual-level reasoning process in the autoregressive image generation task by introducing the semantic-level and token-level CoT, which decouple high-level image planning from low-level pixel generation for more reliable generation.
- 2. We develop BiCoT-GRPO, a new reinforcement learning framework that jointly optimizes both levels of CoT reasoning, seamlessly integrating the understanding capabilities of ULMs for image generation. For reward modeling, we investigate a robust reward system utilizing an ensemble of vision experts.
- 3. Our resulting model, T2I-R1, that incorporates both levels of CoT using BiCoT-GRPO, demonstrates significant performance improvements and surpasses FLUX.1 across multiple established benchmarks.

## 2 Related Work

Unified Generation and Understanding LMM. Recently, the effort to unify image generation and understanding in a single LMM has attracted much attention. Building upon large language models (LLMs), it is natural for the LMMs to understand the image and output the text [46, 30, 81, 16, 76]. However, the method of how to generate an image from a LMM is still under exploration. The image generation method diverges into different branches. One line of the method relies on an exterior image generation model to complete generation [11, 60, 59, 34, 62, 13, 80, 29]. The generator often utilizes text-to-image diffusion models [53, 49] due to its powerful generation capability. To deliver the generation information, the LMM passes either the implicit conditional feature or the explicit image prompt to the generator. For example, EMU [60] first trains the LMM to output CLIP [51] image features identical to that input to the LMM. Then, a pretrained UNet [54] of Stable Diffusion [53] receives the output feature as the condition to generate an image. Another line of the method seeks to train the LMM to generate discrete tokens produced by VQGAN [12] to eliminate the need for an additional generator. [65, 32] directly adopts the VQGAN encoder as the image tokenizer for LMM. However, the VQGAN encoder is only pretrained on the image reconstruction task and thereby generates visual tokens less helpful for image understanding. To improve the understanding capability, [67, 7, 41, 36] proposes to tackle the understanding and generation tasks with different vision encoders separately. The CLIP encoder deals with image input for understanding, while the VQGAN encoder is responsible for generation. Moreover, some works [69, 50, 57] attempt to empower the vision encoder with both the understanding and the generation capability. VILAU [69] trains a vision encoder with both the contrastive loss [51] for text-image understanding and reconstruction loss [12] for image detail preserving. Thanks to the joint pretraining, the vision encoder could generate text-aligned discrete visual tokens. The LMM is then trained to receive the discrete tokens for image understanding and predict them for image generation.

Reinforcement Learning for Large Reasoning Models. The emergence of OpenAI o1 [48] has gained tremendous attention in developing the reasoning capability of large language models. Later, DeepSeek-R1 [17] proposes a rule-based reward and GRPO training method. The introduced method instructs the model to perform an extensive reasoning process before generating the final answer. The reward only focuses on the correctness of the final answer and the following of the pre-defined format. Recently, a number of works have applied this method to multi-modal large language models [5, 43, 73, 75, 10, 22] with task-specific rewards like correctness and IoU [39]. This training paradigm largely helps various reasoning-intensive tasks [52, 26, 18] like mathematical problem-solving [20, 42, 40, 77, 78] and code generation [6, 2, 23].

## 3 Method

## 3.1 Preliminary

Recently, the employment of reinforcement learning has been the dominant approach to elicit the reasoning capability of the large models. [56] introduces GRPO, enhancing PPO by eliminating the value function and estimating the advantage in a group-relative manner. For a specific prompt-answer pair ( p, a ) , a group of G individual responses { o i } G i =1 is sampled from the old policy π θ old . Each response is then input to a reward function to obtain the individual reward R i . Then, the advantage of the i -th response is calculated by normalizing the rewards {R } i G i =1 of the group:

$$A _ { i } = \frac { \mathcal { R } _ { i } - \text{mean} ( \{ \mathcal { R } _ { i } \} _ { i = 1 } ^ { G } ) } { \text{std} ( \{ \mathcal { R } _ { i } \} _ { i = 1 } ^ { G } ) }.$$

GRPO adopts a clipped objective similar to PPO. Besides, a KL penalty term between the current policy π θ and the reference model π θ ref is directly added in the loss function:

$$\stackrel { \cdot } { \cdot } \quad \cdot \quad \cdot \quad \cdot \quad \cdot \\ \mathcal { J } _ { \text{GRPO} } ( \theta ) & = \mathbb { E } _ { ( q, a ) \sim D, \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \omega } } ( \cdot | q ) } \\ & \left [ \frac { 1 } { \sum _ { i = 1 } ^ { G } | o _ { i } | } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } \left ( \min \left ( r _ { i, t } ( \theta ) \hat { A } _ { i }, \text{clip} \left ( r _ { i, t } ( \theta ), 1 - \varepsilon, 1 + \varepsilon \right ) \hat { A } _ { i } \right ) - \beta D _ { \text{KL} } ( \pi _ { \theta } | | \pi _ { \text{ref} } ) \right ) \right ],$$

<!-- image -->

G

Figure 3: Framework of BiCoT-GRPO. In step 1, we instruct the model to generate the semanticlevel CoT based on the image prompt. In step 2, images are generated conditioned on both the image prompt and semantic-level CoT, with the intermediate generation process serving as token-level CoT. The resulting images are evaluated by an ensemble of vision experts to obtain rewards. We generate N images from each prompt to compute the group-relative reward and perform GRPO training.

where r i,j ( θ ) is the ratio between the probabilities of π θ and π θ old for outputting the current token:

$$r _ { i, j } ( \theta ) = \frac { \pi _ { \theta } ( o _ { i, j } \, | \, q, o _ { i, < j } ) } { \pi _ { \theta _ { \theta d } } ( o _ { i, t } \, | \, q, o _ { i, < j } ) }.$$

In text reasoning tasks like mathematical problem solving, the model is instructed to follow the pre-defined template to output the reasoning process and final answer. The reward functions are rule-based rewards that only check the correctness of the final answer and the output format.

## 3.2 Semantic-level and Token-level CoT

In the autoregressive text generation tasks of LLMs and LMMs, CoT occurs in the textual reasoning format. However, in autoregressive image generation tasks, we identify two distinct types of CoT that could enhance the image generation at different abstraction levels:

Semantic-level CoT. Semantic-level CoT is defined as the textual reasoning that precedes image generation, serving as an overall semantic planning stage for the intended image. This process mirrors human artistic creation: when given a brief prompt, an artist first thinks about the scene construction, considering object attributes, spatial relationships, and interactions. In addition to the planning for common prompts, we also observe the semantic-level CoT benefits two other scenarios. If the prompt does not directly depict the object to generate, the semantic-level CoT can reason about the true intention from the user's prompt, providing more aligned images. As illustrated in Fig. 2, the semantic-level CoT reasons that the flower cultivated in the country where Amsterdam is located is tulip. Without this semantic-level CoT, Janus-Pro fails to provide valid results. Additionally, the semantic-level CoT demonstrates importance when handling unusual or potentially ambiguous scenes. In the bottom example of Fig. 2, when given the prompt 'A pig on the bottom of a train' , semanticlevel CoT introduces the action 'lying' for the pig, creating a more sensible scenario. In contrast, direct generation without this interpretive imagination creates significant confusion for Janus-Pro. Formally, each semantic-level CoT s i is composed of | s i | text tokens { s i, 1 , s i, 2 , ..., s i, s | i | } .

Token-level CoT. Unique to the image generation task, a token-level step-by-step thinking exists in the image generation process. The generation of image tokens much resembles a chain of thought: the image tokens are generated patch by patch, where the current patch is generated based on the previous ones. We define the sequential generation of image tokens as token-level CoT. This process parallels how an artist progressively fills a canvas, with the generated patches forming a visual reasoning chain that maintains coherence across the image. This chain of patches is later reshaped to a 2D grid G ∈ R h × × w c and input to an image decoder D to obtain the image. Unlike semantic-level CoT, which addresses global planning, token-level CoT focuses on local details and visual coherence across the image space. Formally, each token-level CoT t i consists of M image tokens { t i, 1 , t i, 2 , ..., t i,M } , where M represents the resolution of the generated image, i.e., M = h × w .

## 3.3 BiCoT-GRPO

GRPO has been proven to be highly effective for exploring the reasoning capability of the LLMs and LMMs. To accommodate both semantic-level and token-level CoT in image generation, we propose BiCoT-GRPO, where the model reasons twice in a single generation process. We instruct the model to first perform semantic-level CoT for global planning, and then dive into the local details by performing token-level CoT.

However, compared with the task of text generation, a great pipeline challenge is posed for incorporating two levels of CoT for image generation. Limited by the training paradigm, most current ULMs cannot generate interleaved images and text themselves. A manual signifier is often needed to instruct the model on which task to perform, either text generation or image generation. For Janus-Pro to generate an image, which is the ULM we use in this work, we need to manually concatenate an image start token ( &lt;img\_start&gt; ) to explicitly instruct the model to start generating image tokens.

To tackle this problem, we propose a novel pipeline to facilitate ULM in generating images with two levels of CoT, as shown in Fig. 3. Specifically, our pipeline is composed of a two-step generation process. The first step is to generate the semantic-level CoT. We input the image prompt and instruct the model to imagine and reason about the details of the image to generate semantic-level CoT { s i } G i =1 . The second stage focuses on the token-level CoT generation. We input the image prompt, the generated semantic-level CoT in the first stage, and the image start token to the ULM for generating image tokens { t i } G i =1 . Then, the image tokens are input to the image decoder to obtain the image I . Since there exist two types of CoT in our method, first the semantic-level CoT and then the token-level CoT. Each response o i is composed of two parts, namely o i = ( s , t i i ) . In this sense, the r i,j ( θ ) is converted to:

$$r _ { i, j } ( \theta ) = \frac { \pi _ { \theta } ( o _ { i, j } \, | \, q, o _ { i, < j } ) } { \pi _ { \theta _ { \alpha d } } ( o _ { i, j } \, | \, q, o _ { i, < j } ) } = \begin{cases} \frac { \pi _ { \theta } ( s _ { i, j } | q, s _ { i, < j } ) } { \pi _ { \theta _ { \alpha d } } ( s _ { i, j } | q, s _ { i, < j } ) }, & 0 \leq j \leq | s _ { i } | \\ \frac { \pi _ { \theta } ( t _ { i, j } | q, s _ { i, t _ { i, < j } ) } } { \pi _ { \theta _ { \alpha d } } ( t _ { i, j } | q, s _ { i, t _ { i, < j } ) } }, & | s _ { i } | < j \leq | s _ { i } | + M \end{cases}$$

Then, we update the ULM by maximizing Equation 2. In practice, we incorporate the token-level policy gradient loss in [74], where the loss term is normalized over all the generated tokens to balance the reward on overly long semantic-level CoT.

## 3.4 Ensemble of Generation Rewards

Unlike DeepSeek-R1 with the rule-based reward, assessing the images based on pre-defined rules is infeasible. The assessment of the image includes various aspects, including the aesthetic appeal and objects' existence, attributes, and relationships. Considering the complexity, we introduce an ensemble of vision experts to judge the generated image from multiple aspects. Meanwhile, the use of multiple reward functions also serves as a regularization method to prevent the ULM from hacking into a specific reward model. As shown in Fig. 4, the ensemble contains the following experts:

Human Preference Model. Human preference models (HPMs), such as HPS [68] and ImageReward [71], are trained to simulate human aesthetic preferences. These models are developed using datasets of human rankings on synthetic images, where annotators evaluate and compare generated outputs. During inference, these models assess both the aesthetic quality and prompt alignment of

Figure 4: Illustration of the Ensemble of Generation Rewards. We use GPT-4o mini to extract the objects and their attributes before training. Each specialized reward model receives customized information inputs for the reward calculation. We take the average of all the rewards as final reward.

<!-- image -->

a generated image, producing a composite human preference score R HPM. This expert provides a holistic reward signal from a general perspective.

Object Detector. Another option of the reward model is an object detector, e.g., GroundingDINO [38] and YOLO-world [8]. These open-vocabulary detection models accept an image along with object queries as input and output both the spatial positions and confidence scores for detected objects. This kind of vision expert serves as an ideal tool to evaluate the object's existence and relationship concerning space and numbers. For implementation, we extract all objects obj i i K =1 from the training image prompts, where K represents the total number of objects. We then query the object detector to identify these objects within the generated image. For each object, we assign a binary existence score (1 if detected, 0 otherwise) and average these scores across all objects in the prompt. If the prompt contains a spatial relationship, we further leverage the detected location to validate its correctness. We calculate the relative distance and intersection over union (IoU) between the objects for the spatial score R spatial . If the number of the object n obj i is specifically pointed out in the prompt, we compare the number with the detected number of the object ˆ n obj i . The reward from the object detector R Det is determined as:

$$\mathcal { R } _ { \text{Det} } = \begin{cases} \begin{array} { c } \alpha \mathcal { R } _ { \text{spatial} } + ( 1 - \alpha ) \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbb { I } ( o b j _ { i } \text{ detected} ), & \text{if spatial relationship in the prompt}, \\ \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \mathbb { I } ( n _ { o b j _ { i } } = \hat { n } _ { o b j _ { i } } ), & \text{if number in the prompt}, \\ \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \mathbb { I } ( o b j _ { i } \text{ detected} ), & \text{else}, \end{array} \end{cases}$$

where R spatial is 1 if the relative distance between the objects are larger than a threshold and the direction is right. If the direction is wrong, the reward is 0. Otherwise, we use the IoU as the spatial reward. We set α as 0.6 to encourage the correctness of the spatial relationship.

Visual Question Answering Model. The visual question answering (VQA) models are trained to answer questions based on the image input. The VQA models include earlier models prior to LLM, e.g., BLIP [33] and GIT [64], and LMMs like LLaVA [35]. We leverage these models to judge the existence and attributes of the objects. For example, if the image prompt is a red dog and a yellow cat , we first reformat each individual object obj i with its attribute as a question to the VQA model, i.e., a red dog? and a yellow cat? . Then, we record the probability for the model to answer Yes as

<!-- image -->

Show a plant that is a symbol of  good fortune in Irish culture, and is known for its three-lobed leaves

<!-- image -->

A bird grooming its feathers

<!-- image -->

The animal that emerges from a cocoon, symbolizing transformation

<!-- image -->

A chameleon perfectly camouflaged against a brown leaf

<!-- image -->

<!-- image -->

A symbol of  imperial China, a sprawling complex of  palaces and temples in Beijing

<!-- image -->

Generate an image of  a bird and a dog, with the smaller animal on top and the larger

<!-- image -->

A typical dish from the country where Naples is located

A chameleon perfectly camouflaged against a green leaf

<!-- image -->

A specific type of  camera used in the 19th century for early photography

<!-- image -->

National Emblem of  the country where New York is located

Figure 5: Visualization Results. We provide the image generation results of the same prompt from four models: base model, the model with only semantic-level CoT optimized, the model with only token-level CoT optimized, and the model with both levels of CoT optimized.

P i Yes and No as P i No . The reward for a prompt is calculated as:

$$\mathcal { R } _ { \mathrm V Q A } = \frac { 1 } { K } \sum _ { i } \frac { P _ { \mathrm Y e s } ^ { i } } { P _ { \mathrm Y e s } ^ { i } + P _ { \mathrm N o } ^ { i } }.$$

Output Reward Model. Lastly, we also employ the output reward model (ORM) proposed in [19] as a reward model. The ORM is fine-tuned from an LMM (e.g., LLaVA-OneVision [30]) specifically for evaluating the alignment between the prompt and the image. The fine-tuning is to instruct the model to output Yes if the image perfectly aligns with the image and output No otherwise. Therefore, we calculate R ORM using the methodology similar to R VQA, except that we input the whole image prompt to the ORM instead of reformatting the prompt.

We can choose one or multiple reward functions illustrated above, and take the average as the final reward for a specific sample. The detailed experiments of reward model in shown in Table 3.

## 4 Experiment

## 4.1 Experimental Setup

Training Settings. Our training dataset comprises text prompts sourced from the training set of T2I-CompBench [21] and [19], totaling 6,786 prompts with no images. Prior to training, we use GPT4o mini to extract the objects and their attributes from the prompts to facilitate computing the rewards. We use Janus-Pro-7B as the base model. We use a learning rate of 1e-6 and a beta of 0.01. For the reward model, we choose HPS [68] as the human preference model, GroundingDINO [38] as the object detector, and GIT [64] as the VQA model. For the ORM, we finetune LLaVA-OneVision-7B in the same manner as [19].

Table 1: T2I-CompBench Result. The best score is in blue , with the second-best score in green .

| Model                       | Attribute Binding     | Attribute Binding     | Attribute Binding     | Object Relationship   | Object Relationship   | Complex ↑             |
|-----------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                             | Color ↑               | Shape ↑               | Texture ↑             | Spatial ↑             | Non-Spatial ↑         | Complex ↑             |
| Diffusion Models            | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      |
| StructureDiffusion [14]     | 0.4990                | 0.4218                | 0.4900                | 0.1386                | 0.3111                | 0.3355                |
| Composable Diffusion [37]   | 0.4063                | 0.3299                | 0.3645                | 0.0800                | 0.2980                | 0.2898                |
| Attend-and-Excite [3]       | 0.6400                | 0.4517                | 0.5963                | 0.1455                | 0.3109                | 0.3401                |
| PixArt- α [4]               | 0.6690                | 0.4927                | 0.6477                | 0.2064                | 0.3197                | 0.3433                |
| CoMat [24]                  | 0.7827                | 0.5329                | 0.6468                | 0.2428                | 0.3187                | 0.3680                |
| SD-v1.5 [53]                | 0.3758                | 0.3713                | 0.4186                | 0.1165                | 0.3112                | 0.3047                |
| SD-XL-base-1.0 [49]         | 0.5879                | 0.4687                | 0.5299                | 0.2131                | 0.3119                | 0.3237                |
| FLUX.1 [28]                 | 0.7407                | 0.5718                | 0.6922                | 0.2863                | 0.3127                | 0.3703                |
| AutoRegressive Models       | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models |
| Show-o [70]                 | 0.56                  | 0.41                  | 0.46                  | 0.20                  | 0.30                  | 0.29                  |
| Show-o + PARM [19]          | 0.75                  | 0.56                  | 0.66                  | 0.29                  | 0.31                  | 0.37                  |
| EMU3 [65]                   | 0.7544                | 0.5706                | 0.7164                | -                     | -                     | -                     |
| Janus-Pro-7B (Baseline) [7] | 0.6359                | 0.3528                | 0.4936                | 0.2061                | 0.3085                | 0.3559                |
| T2I-R1 (Ours)               | 0.8130                | 0.5852                | 0.7243                | 0.3378                | 0.3090                | 0.3993                |

Table 2: WISE Result. The best score is in blue , with the second-best score in green .

| Model                       | Cultural ↑            | Spatio-Temporal       | Spatio-Temporal       | Natural Science       | Natural Science       | Natural Science       | Overall               |
|-----------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                             |                       | Time ↑                | Space ↑               | Biology ↑             | Physics ↑             | Chemistry ↑           |                       |
| Diffusion Models            | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      | Diffusion Models      |
| PixArt-Alpha [4]            | 0.45                  | 0.50                  | 0.48                  | 0.49                  | 0.56                  | 0.34                  | 0.47                  |
| playground-v2.5 [31]        | 0.49                  | 0.58                  | 0.55                  | 0.43                  | 0.48                  | 0.33                  | 0.49                  |
| SD-v1-5 [53]                | 0.34                  | 0.35                  | 0.32                  | 0.28                  | 0.29                  | 0.21                  | 0.32                  |
| SD-XL-base-0.9 [49]         | 0.43                  | 0.48                  | 0.47                  | 0.44                  | 0.45                  | 0.27                  | 0.43                  |
| FLUX.1-dev [28]             | 0.48                  | 0.58                  | 0.62                  | 0.42                  | 0.51                  | 0.35                  | 0.50                  |
| AutoRegressive Models       | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models | AutoRegressive Models |
| Emu3 [65]                   | 0.34                  | 0.45                  | 0.48                  | 0.41                  | 0.45                  | 0.27                  | 0.39                  |
| Show-o [70]                 | 0.28                  | 0.40                  | 0.48                  | 0.30                  | 0.46                  | 0.30                  | 0.35                  |
| VILA-U [69]                 | 0.26                  | 0.33                  | 0.37                  | 0.35                  | 0.39                  | 0.23                  | 0.31                  |
| Janus-1.3B [67]             | 0.16                  | 0.26                  | 0.35                  | 0.28                  | 0.30                  | 0.14                  | 0.23                  |
| Janus-Pro-7B (Baseline) [7] | 0.30                  | 0.37                  | 0.49                  | 0.36                  | 0.42                  | 0.26                  | 0.35                  |
| T2I-R1 (Ours)               | 0.56                  | 0.55                  | 0.63                  | 0.54                  | 0.55                  | 0.30                  | 0.54                  |

Benchmark. We test on T2I-CompBench [21] and WISE [44] to validate the effectiveness of our method. T2I-CompBench comprises 6,000 compositional text prompts evaluating three categories (attribute binding, object relationships, and complex compositions) and six sub-categories (color binding, shape binding, texture binding, spatial relationships, non-spatial relationships, and complex compositions). WISE consists of 1,000 text prompts spanning three categories (cultural common sense, spatial-temporal reasoning, and natural science) for evaluating world knowledge of the text-toimage models. To correctly generate an image, the model needs to reason about what the exact object or scenario is depicted in the prompt. We slightly modify the reasoning instruction on the WISE benchmark for more aligned results. We follow the official evaluation setting of the two benchmarks.

## 4.2 Main Results

We compare T2I-R1 with leading text-to-image diffusion and autoregressive models on the T2ICompBench and WISE benchmarks (in Table 1 and 2). We also provide the qualitative results in Fig. 5. Our method demonstrates substantial improvements over the baseline model, with average enhancements of 13% and 19% on T2I-CompBench and WISE, respectively. On T2I-CompBench, the most significant gains appear in attribute binding, with an average improvement of 19%. For the

Table 3: T2I-CompBench Results with Different Reward Models. 'Det' stands for object detector.

| Model        | Reward Model   | Reward Model   | Reward Model   | Reward Model   | Attribute Binding   | Attribute Binding   | Attribute Binding   | Object Relationship   | Object Relationship   | Complex ↑   | Visual Quality ↑   |
|--------------|----------------|----------------|----------------|----------------|---------------------|---------------------|---------------------|-----------------------|-----------------------|-------------|--------------------|
| Model        | HPM            | Det            | VQA            | ORM            | Color ↑             | Shape ↑             | Texture ↑           | Spatial ↑             | Non-Spatial ↑         | Complex ↑   | Visual Quality ↑   |
| Janus-Pro-7B | -              | -              | -              | -              | 0.6359              | 0.3528              | 0.4936              | 0.2061                | 0.3085                | 0.3559      | -                  |
| -            | ✓              | -              | -              | -              | 0.8134              | 0.6048              | 0.7311              | 0.2383                | 0.3012                | 0.3899      | -                  |
| -            | -              | ✓              | -              | -              | 0.7422              | 0.5140              | 0.6494              | 0.3044                | 0.3100                | 0.3872      | -                  |
| -            | -              | -              | ✓              | -              | 0.8171              | 0.6019              | 0.7307              | 0.2969                | 0.3088                | 0.4052      | 0.218              |
| -            | -              | -              | -              | ✓              | 0.7819              | 0.5638              | 0.7010              | 0.3301                | 0.3103                | 0.3959      | 1.775              |
| -            | ✓              | ✓              | -              | -              | 0.8210              | 0.6074              | 0.7440              | 0.3189                | 0.3076                | 0.4005      | 1.942              |
| T2I-R1       | ✓              | ✓              | ✓              | -              | 0.8130              | 0.5852              | 0.7243              | 0.3378                | 0.3090                | 0.3993      | 2.063              |
| -            | ✓              | ✓              | ✓              | ✓              | 0.7599              | 0.5742              | 0.6902              | 0.2796                | 0.3070                | 0.3921      | -                  |

WISE benchmark, improvements are more evenly distributed across categories. When compared to the more powerful state-of-the-art diffusion models, T2I-R1 achieves superior or comparable results across both benchmarks. Notably, on T2I-CompBench, our method leads in five of six subtasks, with an exceptional performance in the spatial subtask (0.3378), surpassing previous SOTA results by over 5%. Similarly, for WISE, T2I-R1 excels in four of seven subtasks and achieves the highest overall score of 0.54, outperforming the robust FLUX.1-dev by 4%. Remarkably, our approach consistently achieves the leading results across all subtasks in both benchmarks when compared to other autoregressive models.

Remarkably, the improvement on T2I-Compbench benefits from the planning ability brought by the semantic-level CoT, which designs the complex scenarios before generation. While the enhancement of WISE is due to the reasoning capability from the semantic-level CoT, which deduces the true object or place depicted behind the prompt. The token-level CoT plays an important role in faithfully following the design to generate the image and ensure its visual appeal. As shown in Fig. 5, without the semantic-level CoT, the model fails to fully understand what the object or place is to generate, providing unaligned results. When lacking token-level CoT, we observe multiple artifacts in the images, showcasing relatively low image quality.

## 4.3 Reward Analysis

In this section, we experiment with the choice of reward functions and their combinations. We hope to provide some insights into how to choose the reward functions and combine them. Our results are shown in Table 3. We first experiment with the individual reward model. As shown in the table, HPM ( ) demonstrates superior performance in attribute binding but shows limited effectiveness H in supervising object relationships, likely due to its weak relation comprehension capabilities. The object detector ( O ) yields the least improvement in attribute binding among all other tested reward models, which aligns with expectations since our detector-based reward functions do not explicitly evaluate attributes. Any improvements observed stem solely from enhanced object existence ratios in the prompts. We observe that VQA model ( V ) and ORM ( O ) are both effective reward models with distinct strengths: the VQA model excels at improving attribute binding, while ORM demonstrates superior performance in object relationships. Then we experiment with multiple reward models. We start from the composition of HPM and object detector ( H + O ), and progressively incorporate other reward models. Our findings indicate that both the HPM-object detector combination ( H + O ) and the three-model integration of HPM, object detector, and VQA ( H + O + V ) deliver balanced and satisfactory results across both attribute binding and relationship modeling tasks.

To obtain the optimal choice of reward models, we conduct a human study to evaluate the visual quality. Specifically, we select four options of reward models ( V O H , , + O , and H + O + V ) to generate an image from the same prompt. Then we ask humans to rank the four images and score them according to the rank (rank 1 for 3 points, rank 2 for 2 points, and so on). We randomly choose 30 prompts from each of the subtasks from the T2I-CompBench. The result is shown in the visual quality column in Table 3. We observe that ensemble rewards achieve better visual quality, with H + O + V obtaining slightly superior results. This improvement could be attributed to the implicit regularization provided by multiple rewards, preventing overfitting to a single reward model. Conversely, individual reward models fail to provide satisfactory quality despite high benchmark scores. To ensure visual appeal, we adopt the ensemble of three reward models ( H + O + V ) for our final model.

Table 4: Ablation Experiments on the Effectiveness of the Two Levels of CoT.

| Model        | Optimized CoT   | Optimized CoT   | T2I-CompBench   | T2I-CompBench   | T2I-CompBench   | WISE      | WISE              | WISE      | Diversity ↑   |
|--------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------|-------------------|-----------|---------------|
| Model        | Semantic-level  | Token-level     | Color ↑         | Shape ↑         | Texture ↑       | Culture ↑ | Spatio-Temporal ↑ | Science ↑ | Diversity ↑   |
| Janus-Pro-7B |                 |                 | 0.6359          | 0.3528          | 0.4936          | 0.3000    | 0.4232            | 0.3467    | 6.976         |
| -            | ✓               |                 | 0.8082          | 0.5684          | 0.7219          | 0.4900    | 0.5599            | 0.4367    | 8.177         |
| -            |                 | ✓               | 0.7752          | 0.5849          | 0.7451          | 0.3500    | 0.4732            | 0.3900    | 6.255         |
| T2I-R1       | ✓               | ✓               | 0.8130          | 0.5852          | 0.7243          | 0.5600    | 0.5855            | 0.4633    | 8.203         |

Token-level CoT Only

<!-- image -->

Semantic-level + Token-level CoT

<!-- image -->

a black squirrel and a brown nut

<!-- image -->

a fabric bag and a glass vase

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

a key on the right of  a dog

<!-- image -->

Figure 6: Visualization Result of the Image Diversity of a Single Prompt. We showcase the result of only token-level CoT optimized and both semantic-level and token-level CoT optimized.

## 4.4 Ablation Study

We validate the effectiveness of incorporating both semantic-level and token-level CoT. We first validate the effectiveness of the semantic-level CoT by comparing it with a baseline method that generates images using only the token-level CoT optimized with the GRPO method. This is the default text-to-image generation setting in Janus, whose result is shown in the third row in Table 4. Comparing the third and fourth row in the table, we find that semantic-level CoT generally brings performance improvements across both benchmarks tested. We witness a particularly significant gain on the WISE benchmark. This enhanced performance can be attributed to the textual reasoning capabilities inherent in semantic-level CoT. As illustrated in Fig. 5, our method could first clearly reason about the objects or phenomena described in the prompt through semantic-level CoT. This effectively decouples the reasoning and generation processes and thereby facilitates superior results. We also observe that training solely with token-level CoT substantially reduces the diversity of generated images, as demonstrated in Fig. 6. To quantify this effect, we evaluate image diversity by reusing the generated images from T2I-CompBench, where each prompt generates ten images. We compute the Vendi Score [15] across the ten images for each prompt. Results indicate that GRPO training without semantic-level CoT decreases the diversity score, whereas incorporating semantic-level CoT significantly improves diversity through varied textual planning.

We also consider another situation: the semantic-level CoT is incorporated in image generation, but GRPO only optimizes the semantic-level CoT without the token-level CoT. This can be viewed as

only enhancing the model's high-level planning capabilities. The second row of Table 4 presents the result. The results show that optimizing semantic-level CoT exclusively yields smaller improvements compared to the joint optimization approach. Additionally, we find that optimizing both CoT types produces images with much better aesthetic quality compared with optimizing semantic-level CoT only. This indicates the necessity to jointly optimize both levels of CoT.

## 5 Conclusion

In this paper, we introduce T2I-R1, the first reasoning-enhanced text-to-image model powered by a bi-level CoT reasoning process. We identify both the semantic-level CoT for high-level planning and the token-level CoT for patch-by-patch generation. We further integrate them through our proposed BiCoT-GRPO, a reinforcement learning framework incorporating two levels of CoT within the same training step. By leveraging a ULM capable of both visual understanding and generation, our approach eliminates the need for separate specialized models while achieving significant performance improvements, +13% on T2I-CompBench and +19% on the WISE benchmark, surpassing even FLUX.1. Our qualitative analysis demonstrates that T2I-R1 better understands complex prompts, reasons about user intentions, and handles uncommon scenarios with greater robustness, establishing a new paradigm for reasoning-centric generative systems.
<|endofpaper|>