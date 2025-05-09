<|startofpaper|>
## Visual Test-time Scaling for GUI Agent Grounding

Tiange Luo 1 2 , Lajanugen Logeswaran 2 , † Justin Johnson 1 , † Honglak Lee 1 2 , , † University of Michigan 1 LG AI Research 2 † equal advising

## Abstract

We introduce RegionFocus, a visual test-time scaling approach for Vision Language Model Agents. Understanding webpages is challenging due to the visual complexity of GUI images and the large number of interface elements, making accurate action selection difficult. Our approach dynamically zooms in on relevant regions, reducing background clutter and improving grounding accuracy. To support this process, we propose an image-as-map mechanism that visualizes key landmarks at each step, providing a transparent action record and enables the agent to effectively choose among action candidates. Even with a simple region selection strategy, we observe significant performance gains of 28+% on Screenspot-pro and 24+% on WebVoyager benchmarks on top of two state-of-the-art open vision language model agents, UI-TARS and Qwen2.5VL, highlighting the effectiveness of visual test-time scaling in interactive settings. We achieve a new state-of-theart grounding performance of 61.6% on the ScreenSpotPro benchmark by applying RegionFocus to a Qwen2.5VL-72B model. Our code will be released publicly at https://github.com/tiangeluo/RegionFocus.

## 1. Introduction

Graphical user interface (GUI) agents have become increasingly pivotal in modern computing, powering applications ranging from automated web browsing to intuitive operating system navigation [1, 21]. With the proliferation of large-scale vision-language models (VLMs), researchers have sought to harness both textual and visual information to build more capable interactive systems [2]. While many existing frameworks rely heavily on text-based reasoning [39, 46] or simple visual grounding [11, 22], realworld GUIs often contain a substantial number of irrelevant elements-such as menu bars, ads, and extraneous buttons-that can overwhelm purely textual or naive visual approaches. This mismatch between text-heavy inference and the visual complexity of GUIs leads to frequent errors (e.g., clicking the wrong button or navigating to an unintended section). Since these tasks are typically high-level,

Figure 1. Overview . When GUI agents encounter execution errors, we instruct the model to focus on a specific point of interest and extract multiple sub-regions around this focal point (1). The agent then independently generates candidate actions for each subregion. Actions that interact with a specific coordinate are marked with pink-star landmarks (e.g., 'click') to visually indicate the relevant location (2). We retain the pink-star landmarks to track interaction history of RegionFocus for diverse exploration (3).

<!-- image -->

such low-level mistakes accumulate and ultimately result in higher failure rates and poorer overall performance.

Recent research on GUI agents typically falls into two main categories: those relying on textual cues for planning and reasoning, and those incorporating visual information through VLMs. Text-based approaches often generate text labels or bounding boxes for each visual element to guide agent actions [22, 46]. However, they can struggle with visually entangled tasks where textual descriptions are ambiguous, incomplete, or fail to capture crucial visual features (e.g., floating windows), even when using accessibility trees. On the other hand, vision-based pipelines [11, 28] often rely heavily on the VLM's ability to ground visual elements. We observe that many errors arise from inadvertently clicking empty or incorrect interface components, underscoring the limitations of existing single-inference visual grounding methods using VLMs. Once an error occurs, there is no feedback loop to correct it, causing mistakes to compound throughout the process.

Motivated by these shortcomings, we propose a visual test-time scaling framework, RegionFocus, designed to narrow the GUI model's attention to salient interface regions when execution errors occur or other conditions are trig- gered (e.g., VLM self-judgment). Specifically, as illustrated in Figure 1(1), we leverage the VLM's capability to identify points of interest and combine this with bounding-box proposals generated either from fixed-ratio masks or segmentation models such as SAM [16]. For each sub-region, the agent independently predicts actions based solely on the local context (Figure 1(2)), subsequently aggregating the top candidate actions to form a refined, single-step response. Furthermore, interactions with web or OS interfaces allow our method to zoom into targeted areas, enhancing the resolution of selected regions for more careful examination. RegionFocus works as a modular plug-in for GUI agents without affecting the original workflow.

In order to keep track of regions visited by RegionFocus, we introduce an image-as-map mechanism to record temporal information. In this approach, elements previously considered by the agent are annotated in the UI screenshot with visual landmarks (e.g., the pink-stars in Figure 1 (3)). These landmarks prevent the agent from revisiting regions it has already examined and guide the agent towards unexplored areas. These markers do not interfere with the model's regular inference process which leverages unaltered webpage screenshots, and are only used in the RegionFocus component. Once the agent navigates to a new page, all landmarks are refreshed to have a new RegionFocus history.

In addition to representing RegionFocus history, we also leverage image-as-map in the action aggregation process (Figure 1 (2)), where candidate actions are represented using landmarks for selecting an optimal action. We find image-as-map to be highly effective in representing both temporal information (e.g., previously visited regions) and spatial information (e.g., multiple action candidates) compared to alternative representations such as element coordinates represented in the form of text alone. This is particularly crucial for distinguishing between screen elements in close proximity, which is challenging to reason about based on a text representation of the coordinates alone.

With our proposed visual test-time scaling framework, we help existing models-such as UI-TARS [28] and QWen2.5-VL [3] -achieve better performance in both web-based and desktop interfaces. In particular, we demonstrate substantial performance gains on benchmarks including ScreenSpot-Pro [18] for OS-level GUI navigation, as well as WebVoyager [13] for browser automation. Through our experiments, we show that even a simple fixed-ratio bounding-box generation approach yields pronounced improvements over baseline systems, underscoring the efficacy of focusing the model's attention on visually relevant regions. Our empirical studies further indicate that imageas-map consistently outperforms text-based representations for VLM agents. Overall, our findings highlight the value of visual test-time scaling as a simple yet powerful extension to existing VLM-based GUI agents.

## 2. Related Work

## 2.1. GUI Agents

Recent advancements in Large Language Models (LLMs) and Vision Language Models (VLMs) have significantly enhanced GUI automation, enabling agents to effectively interact with diverse graphical environments through textual and visual modalities [8, 11, 15, 21, 24, 27, 31, 33, 34, 38, 40]. Prior studies generally adopt two distinct approaches: (1) text-based reasoning, which leverages structured representations such as HTML or accessibility trees [5, 17, 47], extracting structured interface information [22, 43] and supplementary textual details for input into LLMs/VLMs [46]; and (2) vision-based inference, relying on VLMs to directly interpret GUI elements [11, 28]. While text-based techniques efficiently handle structured information, they often struggle with visually complex or ambiguous interfaces [13, 18, 36], resulting in inaccuracies and reduced reliability. Similar observations have been reported in [39]. Conversely, visual grounding approaches may inadvertently interact with irrelevant or empty regions due to overly broad visual attention. In contrast to prior methods using entire interface screens as input [6, 14, 29], our approach explicitly separates planning from visual grounding via a novel visual test-time scaling framework. This framework selectively targets salient GUI regions through precise boundingbox proposals and integrates an innovative 'image-as-map' strategy, maintaining contextual coherence throughout interactions.

## 2.2. Test-Time Scaling in AI Agents

Test-time scaling involves dynamically adjusting computational resources during inference to enhance model performance [32, 35, 41, 42]. This approach allows AI agents to allocate additional processing power to challenging tasks, thereby improving decision-making and accuracy. Inspired by advancements in test-time scaling for LLMs, several studies have extended similar principles to GUI agents. For example, during the inference, [45] leverages intermediate action histories, [25] collects external information during inference, and [44] incorporates reflection mechanisms into AI agents. Despite their success in improving performance, these methods do not utilize the unique advantages of visual information. In this paper, we propose a preliminary approach toward visual test-time scaling, dynamically adjusting the image focus region and employing an 'image-asmap' technique to encode historical information for more effective GUI agent inference.

## 2.3. Visual Image Attention

Visual image attention mechanisms have a rich history of enabling AI models to selectively focus on pertinent regions within visual inputs [9, 12, 20, 37]. Such mechanisms are

Figure 2. Overview of integrating RegionFocus into GUI agent pipelines. The standard inference process (blue arrow) takes input information and continuously predicts subsequent actions. When the GUI agent encounters errors, RegionFocus (green arrow) activates, proposing focal points to extract targeted sub-regions. Actions are then predicted individually for these sub-regions and aggregated into a single refined action for standard inference.

<!-- image -->

especially critical for GUI-based agents, where accurately identifying and interacting with interface elements amidst visually cluttered environments is essential. Our proposed approach introduces a region-focused mechanism utilizing predefined bounding boxes that progressively refine their attention through historical recording, incrementally concentrating focus more precisely on target elements. This iterative refinement shares conceptual similarities with prior recurrent models [4, 23]; however, our method leverages VLMs to achieve refinement [26], uniquely integrating this process directly into GUI agents' test-time scaling. Accurately generating these attention regions directly via VLMs remains an open direction for future research.

## 3. Method

To address the limitations of current GUI agent frameworks, we propose a visual test-time scaling approach that enhances the robustness and accuracy of VLM agents interacting with complex graphical user interfaces. Unlike traditional methods, which uniformly treat all interface elements, our method dynamically adjusts the model's focus by selectively emphasizing visually salient regions whenever potential errors are detected. This approach significantly reduces misclicks and navigation mistakes.

Crucially, our framework operates entirely at infer- ence time, enabling straightforward integration into existing VLM agents without requiring retraining or architectural modifications. In the following sections, we first describe the integration process for our pipeline with existing GUI agents, followed by a detailed explanation of the design principles and implementation of each component.

## 3.1. Overview

Figure 2 illustrates our proposed pipeline. On top of the standard interaction pipeline, our approach introduces an error-triggered refinement mechanism. Specifically, when the agent encounters a trigger condition -such as clicking on non-interactive elements (e.g., selecting empty space instead of the intended date option) or repeating unsuccessful actions (e.g., repeatedly typing 'Los Angeles' without successfully clicking the correct button)-the RegionFocus component is activated.

During this refinement stage, the agent initially predicts a focal point near the intended target element. Based on this approximate location, it generates a bounding box likely to encapsulate the target element (Section 3.2). For each region defined by the bounding boxes, the agent independently predicts candidate actions. Finally, the agent aggregates a single action to be executed based on the predictions for each region.

Additionally, we maintain an annotated history of previously examined focal points on the same UI image using an image-as-map representation (Section 3.3). This visual history guides the agent in avoiding redundant searches and helps it progressively focus on the correct target. Interactive predictions involving specific coordinates are visually marked (e.g., using pink-star landmarks) to help the model verify the correctness of its selections.

## 3.2. Visual Region Focus

The core of our pipeline is a dynamic visual adaptation mechanism activated during inference. When initial action predictions result in errors-such as clicks on noninteractive or empty regions-the model dynamically adjusts its visual attention. Specifically, it refines attention by generating bounding-box proposals around visually salient regions, leading to more precise single-step actions.

Trigger Condition We define two primary types of triggering conditions. The first relies on environmental feedback obtained from direct interaction with dynamic GUI environments, such as interactive webpages. Errors, like clicks on non-interactive elements, can be easily detected based on environment feedback (e.g., webpage change). The second type of trigger occurs in cases where environmental interaction isn't possible and we are dealing with static screenshots (e.g., ScreenSpot-Pro [19] scenario). Here, the VLM evaluates predicted actions, shifting its role from merely making predictions to explicitly evaluating the correctness of actions. Although the VLM's judgments aren't always perfectly accurate, they significantly help identify and mitigate errors shown in our experiments.

Bounding-box Proposal Empirically, VLM agents reliably produce focal points near target elements but struggle to directly predict accurate bounding boxes [10]. Instead, we derive bounding boxes from these focal points rather than predicting them directly. Although advanced segmentation models (e.g., SAM [16]) could provide more precise bounding boxes, we currently focus exclusively on leveraging the GUI agent itself. This ensures that future enhancements in GUI agents directly benefit our approach.

We employ a heuristic approach, defining bounding boxes with fixed dimensions and aspect ratios (e.g., 0 5 . width by 0 5 . height) centered around the focal point. If a bounding box exceeds the image boundary, it is adjusted to remain fully within the screenshot. This simple yet effective strategy strikes a good balance between accuracy and computational efficiency.

Action Candidate Prediction Given the regions identified by the bounding boxes extracted from the previous stage, the agent predicts an action for each region. If the

Figure 3. Image-as-Map records temporal information. We place numbered pink stars in the image as visual landmarks to indicate previously focused points for the GUI agent. Each time a RegionFocus attempt fails (i.e., no action takes effect), we add a new pink star at the attempted location. Once an action successfully takes effect, we refresh the history and remove any existing landmarks.

<!-- image -->

agent can interact with the environment (e.g., in the case of a webpage), a zoomed-in, high resolution view of the region is provided to the agent. In this case, at least one side of the region can be made to match the original full image resolution via zooming in. If environment interaction is not possible, we simply crop the region from the initial image and upsample it for prediction.

Action Aggregation After the GUI agent independently analyzes each bounding-box proposal and generates candidate actions, we select a single action to serve as the next step in the inference pipeline based on these candidates. For coordinate-based actions (e.g., 'Click (x, y)' or 'Scroll (x, y) down'), we visually mark the candidate action coordinates on the snapshot, as shown in Figure 1 (2). 1 This process significantly reduces the model's workload by simplifying how textual coordinates are mapped and interpreted on the image. Empirically, we observe that incorporating these visual markers leads the model to select action candidates more accurately.

## 3.3. Image-as-Map

When RegionFocus is triggered multiple times for a given UI image, it is important for the agent to generate diverse focal points and avoid revisiting previously explored regions. Initially, we attempted to represent the region focus history using textual coordinates but found this approach ineffective, as the agent often revisited similar focal points despite explicit prompts to avoid them.

To address this, we propose an image-as-map representation, where we visually encode previously examined focal points as landmarks (e.g., pink stars) directly onto UI snapshots (Figure 3) to record past action points. Unlike textbased histories, this visual representation more effectively

1 Landmark annotations are only used for actions that involve interacting with a specific point and element in the current view.

Table 1. Comparison of various models on ScreenSpot-Pro. [19]. Our proposed RegionFocus helps the UI-TARS-72B [28] model achieve a 31 8 . %improvement, while Qwen2.5-VL-72B [3] sees a 28 9 . %gain, thereby achieving state-of-the-art performance. Additionally, integrating RegionFocus into UI-TARS-7B allows it to surpass the performance of the substantially larger UI-TARS-72B model.

| Agent Model     | Development   | Development   | Development   | Creative   | Creative   | Creative   | CAD   | CAD   | CAD   | Scientific   | Scientific   | Scientific   | Office   | Office   | Office   | OS   | OS   | OS   | Avg   | Avg   | Avg   |
|-----------------|---------------|---------------|---------------|------------|------------|------------|-------|-------|-------|--------------|--------------|--------------|----------|----------|----------|------|------|------|-------|-------|-------|
| Agent Model     | text          | icon          | avg           | text       | icon       | avg        | text  | icon  | avg   | text         | icon         | avg          | text     | icon     | avg      | text | icon | avg  | text  | icon  | avg   |
| QwenVL-7B       | 0.0           | 0.0           | 0.0           | 0.0        | 0.0        | 0.0        | 0.0   | 0.0   | 0.0   | 0.7          | 0.0          | 0.4          | 0.0      | 0.0      | 0.0      | 0.0  | 0.0  | 0.0  | 0.1   | 0.0   | 0.1   |
| GPT-4o          | 1.3           | 0.0           | 0.7           | 1.0        | 0.0        | 0.6        | 2.0   | 0.0   | 1.5   | 2.1          | 0.0          | 1.2          | 1.1      | 0.0      | 0.6      | 0.0  | 0.0  | 0.0  | 1.3   | 0.0   | 0.8   |
| SeeClick        | 0.6           | 0.0           | 0.3           | 1.0        | 0.0        | 0.6        | 2.5   | 0.0   | 1.9   | 3.5          | 0.0          | 2.0          | 1.1      | 0.0      | 0.5      | 2.8  | 0.0  | 1.5  | 1.8   | 0.0   | 1.1   |
| Qwen2-VL-7B     | 2.6           | 0.0           | 1.3           | 1.5        | 0.0        | 0.9        | 0.5   | 0.0   | 0.4   | 6.3          | 0.0          | 3.5          | 3.4      | 1.9      | 3.0      | 0.9  | 0.0  | 0.5  | 2.5   | 0.2   | 1.6   |
| OS-Atlas-4B     | 7.1           | 0.0           | 3.7           | 3.0        | 1.4        | 2.3        | 2.0   | 0.0   | 1.5   | 9.0          | 5.5          | 7.5          | 5.1      | 3.8      | 4.4      | 5.6  | 0.0  | 3.1  | 5.0   | 1.7   | 3.7   |
| ShowUI-2B       | 16.9          | 1.4           | 9.4           | 9.1        | 0.0        | 5.3        | 2.5   | 0.0   | 1.9   | 13.2         | 7.3          | 10.6         | 15.3     | 7.5      | 13.5     | 10.3 | 2.2  | 6.6  | 10.8  | 2.6   | 7.7   |
| CogAgent-18B    | 14.9          | 0.7           | 8.0           | 9.6        | 0.0        | 5.6        | 7.1   | 3.1   | 6.1   | 22.2         | 1.8          | 13.4         | 13.0     | 0.0      | 6.5      | 5.6  | 0.0  | 3.1  | 12.0  | 0.8   | 7.7   |
| Aria-UI         | 16.2          | 0.0           | 8.4           | 23.7       | 2.1        | 14.7       | 7.6   | 1.6   | 6.1   | 27.1         | 6.4          | 18.1         | 20.3     | 1.9      | 16.1     | 4.7  | 0.0  | 2.6  | 17.1  | 2.0   | 11.3  |
| UGround-7B      | 26.6          | 2.1           | 14.7          | 27.3       | 2.8        | 17.0       | 14.2  | 1.6   | 11.1  | 31.9         | 2.7          | 19.3         | 31.6     | 11.3     | 27.9     | 17.8 | 0.0  | 9.7  | 25.0  | 2.8   | 16.5  |
| Claude Comp.Use | 22.0          | 3.9           | 12.6          | 25.9       | 3.4        | 16.8       | 14.5  | 3.7   | 11.9  | 33.9         | 15.8         | 25.8         | 30.1     | 16.3     | 26.2     | 11.0 | 4.5  | 8.1  | 23.4  | 7.1   | 17.1  |
| OS-Atlas-7B     | 33.1          | 1.4           | 17.7          | 28.8       | 2.8        | 17.9       | 12.2  | 4.7   | 10.3  | 37.5         | 7.3          | 24.4         | 33.9     | 5.7      | 27.4     | 27.1 | 4.5  | 16.8 | 28.1  | 4.0   | 18.9  |
| UGround-V1-7B   | -             | -             | 35.5          | -          | -          | 27.8       | -     | -     | 13.5  | -            | -            | 38.8         | -        | -        | 48.8     | -    | -    | 26.1 | -     | -     | 31.1  |
| UI-TARS-7B      | 58.4          | 12.4          | 36.1          | 50.0       | 9.1        | 32.8       | 20.8  | 9.4   | 18.0  | 63.9         | 31.8         | 50.0         | 63.3     | 20.8     | 53.5     | 30.8 | 16.9 | 24.5 | 47.8  | 16.2  | 35.7  |
| + RegionFocus   | 59.7          | 15.9          | 38.5          | 59.6       | 11.9       | 39.6       | 30.5  | 7.8   | 24.9  | 67.4         | 30.0         | 51.2         | 69.5     | 30.2     | 60.4     | 45.8 | 21.3 | 34.7 | 55.2  | 18.7  | 41.2  |
| UI-TARS-72B     | 63.0          | 17.3          | 40.8          | 57.1       | 15.4       | 39.6       | 18.8  | 12.5  | 17.2  | 64.6         | 20.9         | 45.7         | 63.3     | 26.4     | 54.8     | 42.1 | 15.7 | 30.1 | 50.9  | 17.5  | 38.1  |
| + RegionFocus   | 72.1          | 26.9          | 50.2          | 68.7       | 22.4       | 49.3       | 35.5  | 25.0  | 33.0  | 77.1         | 30.9         | 57.1         | 72.9     | 45.3     | 66.5     | 63.6 | 27.0 | 46.9 | 64.0  | 28.0  | 50.2  |
| Qwen2.5-VL-7B   | 45.5          | 1.4           | 24.1          | 32.8       | 6.3        | 21.7       | 22.3  | 6.2   | 18.4  | 50.7         | 7.3          | 31.9         | 52.5     | 15.1     | 43.9     | 36.4 | 10.1 | 24.5 | 39.3  | 6.6   | 26.8  |
| + RegionFocus   | 52.6          | 4.8           | 29.4          | 42.9       | 7.7        | 28.2       | 31.0  | 3.1   | 24.1  | 56.9         | 10.9         | 37.0         | 64.4     | 26.4     | 55.7     | 43.0 | 15.7 | 30.6 | 48.0  | 9.9   | 33.5  |
| Qwen2.5-VL-72B  | 66.2          | 13.8          | 40.8          | 64.6       | 15.4       | 44.0       | 47.7  | 12.5  | 39.1  | 78.5         | 29.1         | 57.1         | 74.6     | 37.7     | 66.1     | 60.7 | 22.5 | 43.4 | 64.9  | 20.2  | 47.8  |
| + RegionFocus   | 75.3          |               |               | 76.3       |            |            |       |       |       |              |              |              |          | 60.4     | 80.9     |      |      | 57.1 |       |       |       |
|                 |               | 25.5          | 51.2          |            | 30.8       | 57.2       | 71.6  | 28.1  | 60.9  | 87.5         | 39.1         | 66.5         | 87.0     |          |          | 74.8 | 36.0 |      | 78.6  | 34.1  | 61.6  |

conveys temporal information, allowing the agent to reason directly over the image and avoid revisiting the same areas. Note that the visual landmark annotation process used for action aggregation is also a form of the imageas-map strategy, capturing spatial information. Landmarkannotated snapshots are used only in the RegionFocus process, whereas the original inference pipeline (e.g., for the initial action prediction) receives unaltered UI images. We maintain these highlighted landmarks on the page until an action takes effect (i.e., causes a meaningful state change), at which point the history is refreshed. Our empirical results show that this image-as-map strategy consistently outperforms text-based methods, especially in more complex, multi-step GUI tasks. We also observe that it helps the agent distinguish between two GUI elements that are very close to each other, as shown in Figure 8 (2).

## 4. Experiments

In the experimental sections below, we integrate our proposed pipeline with UI-TARS [28] and Qwen2.5-VL [3], two recently proposed GUI agent models that autonomously interacts with GUI screenshots, which have demonstrated exceptional performance in various GUI tasks. We evaluate our pipeline across both OS operation tasks and Web Interaction. For our main experiments, we adopt a fixed bounding box approach (Section 3.2), using only the agent model itself. We also include ablation studies using SAM [16], which provides more accurate bounding boxes. Additional experimental details are provided in Appendix A.

## 4.1. ScreenSpot-Pro

ScreenSpot-Pro [19] is a recently introduced benchmark specifically designed to evaluate GUI grounding capabilities in complex, high-resolution professional desktop environments. These environments typically involve screenshots larger than 3 k × 2 k (see Figure1(b) in their paper for specific configuration details). The benchmark's emphasis on intricate, large-scale interfaces makes it an ideal platform for assessing our proposed pipeline, which incorporates a mechanism to zoom into local regions for more detailed analysis.

Specifically, ScreenSpot-Pro comprises expertannotated tasks across 23 applications spanning five domains and three operating systems, thereby providing an extensive assessment of model performance. Tasks are categorized by functional domains, including Development, Creative, CAD, Scientific, Office, and Operating Systems, and are further divided into text-based and icon/widget-based grounding challenges. This structure facilitates a nuanced evaluation of grounding capabilities, particularly in tasks that require precise localization and interaction with small or visually similar GUI elements. The benchmark enforces stringent metrics, measuring grounding accuracy based on whether the model-predicted location falls within the bounding box of the target element.

Because ScreenSpot-Pro only provides static screenshots-without an OS environment for interaction-we employ a VLM-based judge to trigger RegionFocus. Specifi-

Agent Model

Claude

GPT-4o

WebVoyager

Qwen2.5-VL-7B

+

RegionFocus

Qwen2.5-VL-72B

+

RegionFocus

UI-TARS-7B

+

RegionFocus

UI-TARS-72B

Allrecipes

45.9%

±

3 4%

.

56.3%

±

51.1%

47.2%

49.2

%

28.6%

43.4

%

17.8%

35.6

%

16.2%

1 3%

.

±

±

±

±

±

±

±

±

2 2%

.

4 4%

.

7 2%

.

1 5%

.

1 3%

.

1 3%

.

2 2%

.

2 6%

.

Amazon

58.6%

±

4 2%

.

53.7%

52.9%

49.1%

53.6

%

56.2%

60.6

%

30.9%

39.0

%

38.6%

±

2 5%

.

±

1 4%

.

±

7 5%

.

±

4 4%

.

±

4 3%

.

±

1 5%

.

±

1 4%

.

±

2 4%

.

±

2 0%

.

Apple

58.1%

±

4 0%

56.6%

62.8%

47.3%

%

67.1

53.0%

%

69.7

17.1%

%

31.8

45.9%

±

±

±

±

±

±

±

±

±

.

1 3%

.

2 3%

.

2 5%

.

3 6%

.

0 9%

.

4 0%

.

1 3%

.

2 7%

.

1 2%

.

ArXiv

55.0%

±

7 0%

60.5%

52.0%

14.9%

51.7

%

32.6%

45.4

%

20.9%

37.2

%

64.9

%

±

±

±

±

±

±

±

±

±

.

0 0%

.

1 3%

.

2 1%

.

1 3%

.

2 4%

.

2 0%

.

2 3%

.

0 0%

.

4 1%

.

GitHub

56.9%

±

1 4%

.

57.7%

59.3%

23.9%

35.0

%

61.5%

67.3

%

32.5%

58.1

%

43.2%

±

3 7%

.

±

3 7%

.

±

4 8%

.

±

0 3%

.

±

4 3%

.

±

3 8%

.

±

2 5%

.

±

1 5%

.

±

2 0%

.

Booking

19.0%

±

1 3%

.

43.9%

32.6%

10.0%

±

±

±

30.0

%

24.8%

3 5%

.

2 7%

.

0 9%

.

±

±

37.3

%

7.6%

0 %

.

2 9%

±

±

15.2

%

.

2 9%

.

1 3%

±

40.2%

.

1 3%

.

±

2 3%

.

ESPN

46.2%

±

1 3%

44.0%

±

47.0%

±

39.2%

±

39.6

%

±

49.8%

±

59.8

%

±

45.0%

±

62.8

%

±

47.2%

±

.

2 7%

.

1 3%

.

7 2%

.

6 4%

.

3 8%

.

1 4%

.

1 3%

.

0 0%

.

3 8%

.

Coursera

68.2%

±

1 3%

.

65.1%

57.9%

46.4%

±

±

±

71.1

%

72.9%

±

±

78.2

%

70.7%

±

±

74.2

%

58.9%

±

±

2 8%

.

2 7%

.

0 3%

.

4 8%

.

1 9%

.

1 4%

.

1 2%

.

1 4%

.

1 6%

.

+

RegionFocus

57.5

%

±

1 1%

.

55.8

%

±

1 4%

.

52.6

%

±

1 3%

.

50.3%

±

3 6%

.

58.9

%

±

3 7%

.

44.7

%

±

2 6%

.

69.7

%

±

1 3%

.

82.2

%

±

1 5%

.

Table 2. Comparison of various models on WebVoyager [13]. For each automatic evaluation, we run GPT evaluator three times to calculate the performance mean and standard deviation. We evaluated our method on UI-TARS [28] and Qwen2.5-VL [3], consistently observing performance improvements.

| Agent Model                                              | Cambridge Dictionary                                          | BBC News                                                      | Google Flights                                                | Google Map                                                    | Google Search                                                 | Huggingface                                                   | Wolfram Alpha                                                 | Overall                                                       |
|----------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|
| Claude GPT-4o WebVoyager                                 | 71.3% ± 3 . 6% 82.2% ± 1 . 3% 71.3% ± 1 . 3%                  | 66.7% ± 4 . 8% 54.8% ± 2 . 4% 60.3% ± 2 . 8%                  | 15.1% ± 5 . 5% 28.6% ± 0 . 0% 51.6% ± 1 . 4%                  | 55.3% ± 1 . 4% 56.9% ± 2 . 8% 64.3% ± 2 . 8%                  | 72.9% ± 1 . 3% 63.6% ± 1 . 3% 77.5% ± 2 . 7%                  | 53.5% ± 4 . 7% 42.6% ± 3 . 6% 55.8% ± 2 . 3%                  | 51.5% ± 5 . 4% 65.2% ± 2 . 2% 60.9% ± 2 . 2%                  | 52.8% ± 1 . 4% 55.5% ± 0 . 8% 57.1% ± 0 . 2%                  |
| Qwen2.5-VL-7B + RegionFocus Qwen2.5-VL-72B + RegionFocus | 21.1 % ± 4 . 8% 17.3% ± 0 . 3% 63.7% ± 1 . 3% 68.9 % ± 1 . 3% | 45.1% ± 3 . 0% 52.9 % ± 0 . 9% 45.9% ± 1 . 6% 54.4 % ± 2 . 9% | 10.0% ± 0 . 9% 12.8 % ± 4 . 8% 17.7% ± 1 . 5% 34.6 % ± 4 . 4% | 30.2 % ± 1 . 5% 17.1% ± 1 . 2% 31.2% ± 1 . 5% 42.2 % ± 1 . 5% | 10.0% ± 1 . 3% 18.3 % ± 2 . 4% 11.5% ± 0 . 0% 20.3 % ± 0 . 0% | 41.4% ± 0 . 3% 60.0 % ± 2 . 8% 38.3% ± 2 . 9% 51.5 % ± 4 . 9% | 51.3 % ± 3 . 8% 40.8% ± 1 . 2% 48.9% ± 2 . 5% 56.0 % ± 1 . 1% | 32.5% ± 1 . 3% 41.1 % ± 1 . 2% 42.4% ± 0 . 5% 52.7 % ± 1 . 1% |
| UI-TARS-7B + RegionFocus UI-TARS-72B + RegionFocus       | 57.1 % ± 1 . 3% 55.8% ± 0 . 0% 72.9% ± 1 . 4% 75.9 % ± 1 . 3% | 41.3% ± 2 . 7% 52.4 % ± 2 . 4% 39.3% ± 3 . 4% 49.8 % ± 3 . 6% | 10.3% ± 1 . 4% 28.6 % ± 4 . 8% 33.9% ± 2 . 1% 49.9 % ± 2 . 1% | 17.5% ± 0 . 0% 29.3 % ± 2 . 4% 27.2% ± 5 . 2% 59.9 % ± 1 . 4% | 49.6% ± 1 . 3% 60.5 % ± 2 . 3% 60.6% ± 1 . 2% 65.8 % ± 1 . 2% | 40.7% ± 1 . 4% 45.7 % ± 1 . 3% 24.9% ± 1 . 6% 59.6 % ± 1 . 3% | 38.4% ± 1 . 3% 44.2 % ± 1 . 3% 48.3% ± 2 . 3% 60.2 % ± 1 . 3% | 33.2% ± 0 . 5% 44.7 % ± 0 . 5% 44.1% ± 0 . 5% 59.5 % ± 0 . 1% |

cally, when the agent predicts a point, we will highlight that point in the input screenshot with pink-star landmarks and ask the model itself to assess the correctness of that point. If deemed incorrect, we initiate the RegionFocus process.

Results In Table 1, we summarize the reported grounding accuracy of various methods evaluated on ScreenSpotPro. For fair comparisons, we employ the official test code released by ScreenSpot-Pro for evaluation. We report the original numbers from the UI-TARS paper in Table 1.

From Table 1, we can see that RegionFocus consistently improves performance across all categories for both text and icon grounding when compared to the base model. UI-TARS-72B + RegionFocus achieves a 31 7 . % improvement over the base UI-TARS-72B model. Moreover, the UI-TARS-7B + RegionFocus variant outperforms the UITARS-72B model overall, demonstrating the effectiveness of our approach. Furthermore, RegionFocus further helps QWen2.5-VL-72B achieve the state-of-the-art performance, 61 6 . %.

Figure 4 illustrates how our pipeline works in ScreenSpot-Pro: we first ask the agent to judge the initial action prediction. If it is incorrect, we initiate RegionFocus by zooming into the regions predicted by the agent, predicting actions within each region, and finally aggre-

Figure 4. Qualitative results - Screenspot-Pro. In one example from our evaluation, the agent successfully rejects the initial action via self-VLM-judge and proposes a correct grounding point based on the zoomed-in view. More qualitative results are listed in Appendix B.

<!-- image -->

gating all actions into a single outcome. Because we used the agent model itself to judge whether a prediction is correct or incorrect, our results show that although the VLM may generate incorrect coordinates initially, it can still reli-

Initial Action

RegionFocus

RegionFocus

<!-- image -->

Initial Action

Task : Find a Popular recipe for a chocolate chip cookie and list the ingredients and preparation steps.

<!-- image -->

<!-- image -->

Task : Check the Dataset Viewer for ai2lumos/lumos\_complex\_qa\_plan\_onetime on Hugging face. what is the content corresponding to user in the first message?

Action Aggregation Initial Action Action Aggregation Initial Action Figure 5. Qualitative Results - RegionFocus. In these two examples, we illustrate how RegionFocus reduces background noise by emphasizing salient regions of an image. The mouse pointer indicates the agent's initial action prediction, which is suboptimal in both cases. Left pair of images : The green window in the second image marks the zoomed-in region. By focusing on this region, we naturally cut out the distracting portion of the first image. Right pair of images : The second image is zoomed in, significantly reducing distracting details. This allows the agent to focus on the relevant information-even though the distracting region from the first image is still visible. Task : Find a Popular recipe for a chocolate chip cookie and list the ingredients and preparation steps. Task : Check the Dataset Viewer for ai2lumos/lumos\_complex\_qa\_plan\_onetime on Hugging face. what is the content corresponding to user in the first message?

Initial Action

Action Aggregation

Action Aggregation

Initial Action

: Search the latest article about space exploration on BBC News and

<!-- image -->

Task

: Compare the prices and flight durations for economy class flights from Oslo to

Dubai, departing on March 8, 2025, and show options with no more than two layovers.

Task : Search the latest article about space exploration on BBC News and summarize its key points.

<!-- image -->

<!-- image -->

<!-- image -->

Task : Compare the prices and flight durations for economy class flights from Oslo to Dubai, departing on March 8, 2025, and show options with no more than two layovers.

Figure 6. Qualitative results - image-as-map. These examples demonstrate how action aggregation, enhanced by the proposed image-asmap, helps distinguish subtle coordinate differences between target elements. The mouse pointer indicates the agent's initial predictions, which were incorrect in both cases. Each star-like landmark is generated during the RegionFocus process before action aggregation. Left pair of images : The two landmarks at the top left correspond to the home and search buttons. Right pair of images : the landmarks correspond to different options in a dropdown menu.

ably judge whether those click points are correct with the help of image-as-map. Then, such an incorrect prediction can be corrected by the RegionFocus process. This generation-verification gap has also been noted in recent literature [7, 30].

## 4.2. WebVoyager

WebVoyager [13] is a benchmark designed to evaluate autonomous web agents' capabilities in performing complex, open-ended tasks through multimodal interactions with real-world websites. Distinct from previous web agent benchmarks, WebVoyager comprises 643 semiautomatically generated tasks across 15 popular, real-world websites such as Amazon, Apple, ArXiv, and Google Maps. This selection ensures a diverse range of interactions re- flecting everyday web browsing scenarios. Tasks in WebVoyager require agents to process visual information from rendered screenshots and textual cues from web elements, enabling nuanced evaluation of multimodal reasoning and navigation skills. Furthermore, the benchmark introduces an automatic evaluation protocol utilizing GPT-4V, achieving 85.3% agreement with human judgment, thereby offering a reliable assessment of agent performance.

In this scenario, the agent actively interacts with the web environment. We employ a Playwright-controlled Chrome browser to navigate webpages, with the VLM agent determining the appropriate action based on each webpage screenshot. After task execution, we use the official evaluation setting, where a GPT-based judge reviews the last 15 screenshots along with an optional textual response to deter-

Figure 7. Histogram of step differences :

<!-- image -->

BaseModel + Region-

Focus vs. BaseModel alone. BaseModel is UI-TARS-72B.

mine whether the task has been successfully accomplished.

Results For comparative analysis, Table 2 presents the task success rates of various web agents evaluated on the WebVoyager benchmark. RegionFocus consistently improves performance across all types of websites-including 'Booking' and 'Search'-highlighting the effectiveness of integrating RegionFocus into the GUI agent for web browsing. It also brings consistent improvements over two opensource model, UI-TARS and Qwen2.5-VL. Please note that our model performance was impacted by online interaction constraints-such as bot blocking and intermittent VPN issues. By resolving these factors, we can further boost the model's overall performance.

We present several qualitative examples of WebVoyager's performance in Figures 5, 6. In Figure 5 left, the agent initially fails by clicking the 'ingredients' button, which appears in the search bar despite being on the correct page. By highlighting the relevant region with a green bounding box, RegionFocus naturally filters out background noise and draws attention to the primary content. In Figure 5 right, RegionFocus zooms in on the sub-region of interest, enlarging key content and making it easier for the agent to locate the target content. Figure 6 left shows a case where the agent initially clicks an unrelated element. Our pipeline then corrects this mistake by proposing two closely positioned buttons. The image-as-map mechanism allows the agent to distinguish between these nearly identical elements, even when their coordinates differ only slightly. Finally, Figure 6 right illustrates a scenario where the agent mistakenly clicks on an empty area close to the desired element. Once again, RegionFocus highlights the correct button, helping the agent choose it accurately.

More Analysis Figure 7 shows the distribution of step differences between the combined BaseModel + RegionFo-

<!-- image -->

<!-- image -->

1) SAM provides Bounding Box

2) Image-as-Map tells small coordinates difference

Figure 8. Ablation studies. (1) RegionFocus can natively use the SAM to generate bounding boxes. (2) how image-as-map helps highlight subtle differences of different GUI elements.

cus approach and the BaseModel alone over 400 trajectories. Only the actual browser-interactive steps are counted, excluding RegionFocus overhead. Here the BaseModel is UI-TARS-72B. As shown, BaseModel + RegionFocus generally yields more steps on average ( 19 74 . % steps), correlating with a overall 34 3 . %higher success rate. RegionFocus is triggered 5 8 . times on average for each trajectory in Web Browsing. On average, RegionFocus is triggered 5 8 . times per Web Browsing trajectory. Furthermore, in 32 3 . % of cases RegionFocus is triggered only once, yet a single trigger yields an impressive 83 7 . % increase in the success of those trajectories.

## 4.3. Ablation Studies

We conduct ablation studies on the entire pipeline, including our 'image-as-map' design choice and the use of a predefined bounding box based on the point predicted by the agent. We also demonstrate that by leveraging SAM [16] and increasing the number of trajectory steps, performance can be further improved. For the sake of computation, we employed the UI-TARS-7B-

Table 3. Ablation study results. We tested on a subset of WebVoyager, and the score is higher the better.

| Agent Model     |   Overall |
|-----------------|-----------|
| image-as-map    |      43.2 |
| Text-as-History |      37.2 |
| Fixed-BBox      |      43.2 |
| Predict-Region  |      28.1 |
| SAM             |      46.5 |

DPO model for these ablation studies on a subset of the WebVoyager benchmark. The results are shown in Table 3, where 'image-as-map' and 'Fixed-BBox' refers to our same 7B model configuration with RegionFocus enabled and a maximum limit of 100 action steps.

Text-based RegionFocus History Representation By using image-as-map, we can directly provide visual location information to the agent, helping it distinguish even minor differences and thereby enhancing its perception. For instance, as shown in Figure 8 (2), we color-code click points in the image to denote the image-as-map mechanism, while the corresponding coordinates are listed in the text box on

Figure 9. Ablation study - inference steps : higher limits yield further improvements, though the benefits gradually decay.

<!-- image -->

the right. Notably, although the textual coordinate difference is within only five pixels, the resulting action can vary significantly. In this ablation study, we use a text representation for both history tracking and action aggregation, which leads to significant degradation compared to our image-asmap representation.

Proposing regions directly In our pipeline (Section 3), rather than having the agent model directly propose a bounding box, we first prompt it to identify a point of interest and then generate a predefined bounding box around that point. This design choice stems from the observation that agent models often struggle to accurately predict bounding boxes on their own. To validate this, we conducted an experiment in which the model was required to predict both the upper-left and bottom-right corner coordinates, which were then used to crop the UI image. As shown by the 'Predict-Region' results, this approach led to a marked decrease in performance.

SAM As discussed in Section 3, our pipeline can naturally leverage segmentation models that take point inputs as indicator, such as SAM [16]. For example, in Figure 8 (1), we provide a point generated by the model for RegionFocus, despite the fact that the point itself is referring to a non-interactive empty area. Nevertheless, SAM is able to produce a bounding box that includes the correct region, showcasing its suitability in such cases. The effectiveness of incorporating point-based segmentation models into RegionFocus is further validated by the results in Table 3.

Test-time Thinking Budget Our quantitative analysis in Figure 7 shows that incorporating RegionFocus naturally increases the number of steps taken by the agent. Motivated by this, we investigate whether raising the inferencestep limit beyond 100-or removing it altogether-yields further performance gains. Here, inference steps refer to actual browser-interactive actions, excluding RegionFocus overhead. Due to computational constraints, we extended the limit to 300 steps and also evaluated lower bounds of 10 and 50 steps. All the experiments are conducted with the UI-TARS-7B model. As shown in Figure 9, increasing the maximum to 300 steps improved the 7B model's performance from 43 2 . to 45 3 . , although most trajectories terminated before reaching the 300-step ceiling. Moreover, the incremental benefit gradually decays as the maximum inference-step threshold grows.

## 5. Conclusion

We introduced RegionFocus , a visual test-time scaling approach that dynamically zooms in on relevant interface regions to address the clutter and ambiguity of modern GUIs. By integrating an image-as-map mechanism that marks key landmarks, our method provides transparent action records and improves coordinate-based action predictions. Experiments on Screenspot-pro and WebVoyager show substantial performance gains-even with a simple fixed-ratio bounding box strategy-highlighting the power of visual test-time scaling in enhancing interactive AI systems.
<|endofpaper|>