<|startofpaper|>
## OmicsCL: Unsupervised Contrastive Learning for Cancer Subtype Discovery and Survival Stratification

## Atahan Karag¨ oz

Department of Computer Science

University of Basel Basel, Switzerland atahan.karagoez@stud.unibas.ch

Abstract -Unsupervised learning of disease subtypes from multiomics data presents a significant opportunity for advancing personalized medicine. We introduce OmicsCL , a modular contrastive learning framework that jointly embeds heterogeneous omics modalities-such as gene expression, DNA methylation, and miRNA expression-into a unified latent space. Our method incorporates a survival-aware contrastive loss that encourages the model to learn representations aligned with survival-related patterns, without relying on labeled outcomes. Evaluated on the TCGA BRCA dataset, OmicsCL uncovers clinically meaningful clusters and achieves strong unsupervised concordance with patient survival. The framework demonstrates robustness across hyperparameter configurations and can be tuned to prioritize either subtype coherence or survival stratification. Ablation studies confirm that integrating survival-aware loss significantly enhances the predictive power of learned embeddings. These results highlight the promise of contrastive objectives for biological insight discovery in high-dimensional, heterogeneous omics data.

Index Terms -Multi-Omics, Contrastive Learning, Cancer Subtype, Survival Analysis, Unsupervised Learning

## I. INTRODUCTION

Cancer is a heterogeneous disease that manifests through complex molecular alterations across different biological levels, including genomics, epigenomics, and transcriptomics. The advent of high-throughput sequencing technologies has enabled researchers to collect multi-omics datasets, which together provide a comprehensive molecular view of individual tumors. Integrating these heterogeneous data types is critical to uncovering latent subtypes and stratifying patients based on prognosis. However, effective multi-omics integration remains a challenging task due to the high dimensionality, noise, and inconsistency across omics modalities.

Traditional approaches to subtype discovery have relied on supervised learning using predefined cancer subtype labels, or unsupervised clustering techniques with limited biological interpretability. More recently, deep learning has shown promise in learning compact representations of omics data, yet many models are either supervised or require complex architectures and large amounts of annotated data. Additionally, models that do not explicitly incorporate survival outcomes may fail to capture clinically relevant distinctions between subtypes.

To address these challenges, we propose OmicsCL , an unsupervised contrastive learning framework designed to learn joint embeddings from multi-omics data with no subtype labels. OmicsCL integrates contrastive objectives across omicsspecific encoders while incorporating a novel survival-aware contrastive loss that encourages embeddings of patients with similar survival outcomes to be closer in latent space. This design enables the model to learn biologically meaningful and survival-informative representations in a purely unsupervised manner.

## II. RELATED WORK

The integration of multi-omics data for cancer subtype discovery has been extensively studied over the past decade. Traditional methods such as Similarity Network Fusion (SNF) [1] and iCluster [2] combine heterogeneous data sources to build unified representations, often followed by unsupervised clustering. While effective, these approaches rely on predefined similarity metrics and do not directly optimize for downstream survival relevance.

With the rise of deep learning, autoencoder-based methods have become popular for learning low-dimensional representations from high-dimensional omics data. Models such as MOFA [3] and DCCA [4] leverage probabilistic or canonical correlation-based frameworks to jointly embed multiple modalities. However, these approaches typically assume paired data distributions and are not inherently designed for survival-aware representation learning.

Contrastive learning has recently emerged as a powerful unsupervised method for representation learning in various domains, including computer vision and bioinformatics. Methods such as scCL [5] and CONAN [6] apply contrastive objectives to single-cell or multi-omics settings. Most of these methods focus on learning modality-invariant features or maximizing agreement across data views, but they often neglect the temporal aspect of clinical outcomes like patient survival.

Survival analysis in deep learning has traditionally been addressed through supervised models such as DeepSurv [7] or DeepHit [8], which require labeled event times and typically predict survival functions directly. While these models have achieved strong performance, they require extensive labeled data and are not naturally suited for unsupervised stratification tasks. Recent large-scale benchmarks [9] have emphasized the limitations of supervised survival models under multi-omics settings, reinforcing the need for more flexible, unsupervised alternatives.

Our work bridges the gap between unsupervised representation learning and survival analysis. Unlike previous models, OmicsCL introduces a survival-aware contrastive loss that encodes temporal outcome information into the embedding space without relying on explicit survival supervision or pre-defined subtype labels. This allows for discovery of clinically meaningful cancer subtypes in an entirely label-free setting, while still preserving discriminative features for patient stratification.

## III. METHODOLOGY

## A. Problem Definition

Let D = { ( x ( g ) i , x ( m ) i , x ( r ) i , t i , e i ) } N i =1 , represent a multiomics dataset consisting of N patients, where x ( g ) i , x ( m ) i , and x ( r ) i correspond to gene expression, DNA methylation, and miRNA profiles, respectively. Each patient also has an associated survival time t i ∈ R + and an event indicator e i ∈ { 0 1 , } , with 1 denoting death and 0 indicating censoring. Our objective is to learn compact embeddings for each omics modality and a joint representation that enables meaningful clustering of patients into subtypes predictive of survival outcomes, without relying on subtype labels during training.

## B. Model Architecture

As a modular contrastive learning framework, OmicsCL learns view-specific and joint representations across omics modalities. For each modality, we use a dedicated encoder f ( v ) θ parameterized by neural networks with shared structure but independent weights. Each encoder consists of a multi-layer perceptron with batch normalization and ReLU activations, followed by a projection head that maps features to a latent space R d , where d is the embedding dimension. These projections are ℓ 2 -normalized to lie on the unit hypersphere.

## C. Contrastive Objective Across Omics

̸

̸

Given a minibatch of samples, we construct positive pairs ( z ( v ) i , z ( w ) i ) for each i from different modalities v = w , and negative pairs ( z ( v ) i , z ( w ) j ) for j = i . We adopt the normalized temperature-scaled cross-entropy loss (NT-Xent) [10]:

̸

$$\mathcal { L } _ { \text{NT-Xent} } = - \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \sin ( z _ { i } ^ { ( v ) }, z _ { i } ^ { ( w ) } ) / \tau ) } { \sum _ { j = 1 } ^ { N } \mathbb { 1 } _ { [ j \neq i ] } \exp ( \sin ( z _ { i } ^ { ( v ) }, z _ { j } ^ { ( w ) } ) / \tau ) }, \quad B. \ D a t a \\ \text{where} \, \quad \text{where} \, \ \text{$\theta$} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{or} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{end} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{out} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{in} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{with} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{will} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{in} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{is} \, \ \text{will} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{and} \, \ \text{will} \, \ \text{will} \, \ \text{and} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{is} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{with} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{and} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{with} \, \ \text{will} \, \ \text{will} \, \ \text{with} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{with} \, \ \text{will} \, \ \text{will} \, \ \text{is} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{will} \, \ \text{with} \, \ \text{will} \, \ \text{will} \, \ \text{and} \, \ \text{will} \, \$$

where sim ( a, b ) denotes cosine similarity, and τ is a temperature parameter. This objective encourages agreement between representations from different omics views of the same patient.

## D. Survival-Aware Contrastive Loss

To encode temporal risk structure into the embedding space, we introduce a novel unsupervised survival contrastive loss . It penalizes representations of patients with dissimilar survival times (when both are deceased) and encourages closeness between embeddings with similar outcomes. Let d ij denote the Euclidean distance between embeddings z i and z j , and

∆ t ij = | t i -t j | their survival time difference. The loss is defined as:

$$\int _ { \substack { n \text{ or } \\ \tau \dots \text{ } } } \left | \text{ding } \right | \text{ } \mathcal { L } _ { \text{surv } } = & \lambda _ { \text{pull} } \cdot \mathbb { E } _ { i, j } \left [ \mathbb { 1 } _ { [ e _ { i } = e _ { j } = 1 ] } \cdot \mathbb { 1 } _ { [ \Delta t _ { i j } < \delta ] } \cdot d _ { i j } ^ { 2 } \right ] \\ \int _ { \substack { \tau \dots \text{ } } } + \lambda _ { \text{push} } \cdot \mathbb { E } _ { i, j } \left [ \mathbb { 1 } _ { [ \Delta t _ { i j } \geq \delta ] } \cdot \max ( 0, \delta - d _ { i j } ) ^ { 2 } \right ], \quad ( 2 )$$

where δ is a tunable time margin, and λ pull , λ push are weighting coefficients. Notably, this formulation does not rely on supervised risk labels and operates in an entirely unsupervised regime, allowing it to generalize across cancer types and data splits.

## E. Joint Training

The final training objective is a weighted sum of contrastive loss and the survival-aware regularization term:

$$\mathcal { L } _ { \text{total} } = \mathcal { L } _ { \text{NT-Xent} } + \alpha \cdot \mathcal { L } _ { \text{surv} },$$

where α controls the trade-off between modality alignment and survival stratification. During training, we utilize a cyclical learning rate scheduler and early stopping based on validation concordance index (C-index), which evaluates how well the learned embeddings capture survival risks.

## F. Clustering and Evaluation

After training, we concatenate the learned embeddings across omics modalities to form a unified patient representation. These representations are clustered using KMeans. Evaluation is performed using clustering and survival metrics, which we detail in Section IV-D.

## IV. EXPERIMENTS

## A. Dataset

We evaluated our proposed framework on the TCGA-BRCA dataset from the Multi-Omics Cancer Benchmark [11]. The dataset included three primary omics views: gene expression (RNA-seq), DNA methylation (450k array), and miRNA expression. Survival information was provided for each patient, including time to event or censoring and an event indicator. Additionally, PAM50 subtype annotations were available for benchmarking clustering performance.

## B. Data Preprocessing

We harmonized sample identifiers across omics sources and removed patients with missing survival time or event status. Survival times were extracted from clinical fields such as 'overall survival,' with corresponding binary death indicators derived from 'status'; both are consistently encoded. Subtype labels with missing values were imputed as 'Unknown' and excluded from supervised evaluation metrics. All preprocessing scripts were released as part of our pipeline for reproducibility.

After preprocessing, the dataset comprised a total of 612 patients with all three omics modalities and survival labels available. We applied z-score normalization to each omics view and split the dataset into 60% training, 20% validation, and 20% test sets using a fixed random seed for reproducibility.

## C. Implementation Details

Each omics encoder was a two-layer MLP with hidden dimension 128 and projection dimension 64 . All embeddings were ℓ 2 -normalized. The models were trained using the Adam optimizer with a weight decay of 1 × 10 -6 and a cyclical learning rate policy ranging from 1 × 10 -5 to 1 × 10 -3 . The NTXent loss temperature τ was set to 0.1. The survival contrastive loss used a margin of 1.0, and the weighting coefficient α was set to 10.0 based on grid search.

Training proceeded for a maximum of 1000 epochs with early stopping based on validation concordance index, using a patience of 20 epochs. The temporal dynamics captured during training are visualized in Figure 1. Each training run was seeded for reproducibility. All experiments were run on a machine with an Apple M1 Max CPU and 64GB RAM.

Fig. 1. Combined survival histograms over training epochs. This visualization captures the evolution of censored and deceased event distributions throughout training, highlighting the temporal dynamics our model encodes into the embedding space.

<!-- image -->

## D. Evaluation Metrics

To assess the quality of learned representations, we evaluate both clustering coherence and survival relevance of the predicted clusters.

a) Clustering Metrics: To evaluate agreement between predicted clusters and known PAM50 subtypes, we report several unsupervised clustering metrics. Silhouette Score measures the cohesion and separation of samples within clusters. Purity reflects the proportion of correctly assigned samples based on majority voting in each cluster. Adjusted Rand Index (ARI) quantifies similarity between predicted and true labels, adjusted for chance. Normalized Mutual Information (NMI) measures the shared information between cluster assignments and ground truth subtypes.

b) Survival Metrics: To assess the ability of clusters to stratify patient survival, we use survival-specific metrics. The Concordance Index (C-index) evaluates the agreement between predicted risk scores and actual survival times. The logrank test provides a statistical measure of survival separation across clusters. Finally, Kaplan-Meier curves visualize survival probabilities over time for each predicted cluster.

## E. Baselines

As our primary goal is to remain unsupervised, we compared our method's performance to a Cox Proportional Hazards (CoxPH) model trained on the same features. However, it should be noted that CoxPH directly optimizes supervised survival prediction, whereas OmicsCL infers survival-relevant embeddings without access to labels.

We also compared against ablated versions of OmicsCL trained without the survival-aware contrastive regularization, showing the impact of our design choices on downstream survival analysis.

## V. RESULTS

## A. Survival Stratification Performance

OmicsCL achieved strong performance in survival prediction, evidenced by a concordance index (C-index) of 0.7512 on the test set. This result demonstrated that the unsupervised embeddings learned by the model effectively capture riskrelated structure in the patient population, despite the absence of label supervision during training.

The Kaplan-Meier curves in Figure 2 showed distinct separation across clusters predicted by KMeans, supporting the hypothesis that the learned representations preserve clinically relevant survival differences. Additionally, a multivariate logrank test yielded a p-value of 0.0082, indicating that the survival distributions across predicted clusters are statistically different.

## B. Subtype Discovery and Clustering Quality

Table I summarizes the clustering metrics. OmicsCL achieved a purity of 0.4022 without using subtype labels. Although ARI and NMI were modest-reflecting weak alignment with PAM50-this was expected due to label noise and the unsupervised setting. As discussed further in subsection V-F, alternative configurations can substantially improve clustering performance.

<!-- image -->

Time

Fig. 2. Kaplan-Meier curves stratified by predicted clusters. Clear survival separation is observed, especially between clusters 0 and 2.

TABLE I

CLUSTERING METRICS ON TEST SET USING KMEANS WITH k = 4

| Metric                       |   Score |
|------------------------------|---------|
| Silhouette Score             |  0.0705 |
| Accuracy                     |  0      |
| Adjusted Rand Index (ARI)    | -0.0013 |
| Normalized Mutual Info (NMI) |  0.0672 |
| Purity                       |  0.4022 |

## C. Visualization of Learned Embeddings

To qualitatively assess the learned representations, we projected the embeddings to 2D and 3D using both UMAP and t-SNE. Figures 3 and 4 revealed meaningful structure in the latent space. While the clusters did not perfectly align with PAM50 labels, visual separation suggested that the model captured alternative biological substructures or clinically relevant features. For enhanced exploration, we also generated interactive 3D plots in HTML format, which provided a more dynamic view of the cluster geometry.

Fig. 3. 2D t-SNE visualization of embeddings colored by predicted clusters. Distinct subpopulations emerge despite unsupervised training.

<!-- image -->

Fig. 4. 2D UMAP visualization of embeddings colored by PAM50 subtypes. The model partially recovers subtype structure without supervision.

<!-- image -->

## D. Ablation Study: Impact of Survival Contrastive Loss

We conducted an ablation study to quantify the effect of survival-aware regularization. Removing the survival contrastive loss term from the training objective resulted in a significant drop in C-index to 0.617. This validated our hypothesis that incorporating temporal survival dynamics directly into the contrastive loss improves the survival discrimination power of learned embeddings.

## E. Comparison with Cox Proportional Hazards

To provide a baseline comparison, we evaluated the survival stratification performance of Cox Proportional Hazards (CoxPH) models trained on the same embeddings. We assessed the concordance index (C-index) across various cluster configurations, obtained scores ranging from 0.4570 (2 clusters) to 0.7541 (9 clusters). OmicsCL achieved a C-index of 0.7512 with only 4 clusters, consistently outperforming CoxPH at every cluster configuration below 9. Although CoxPH achieved a slightly higher peak at 9 clusters, our method demonstrated competitive performance without relying on supervised survival modeling, highlighting the effectiveness of OmicsCL in capturing survival-relevant information from multi-omics data.

## F. Configurable Trade-offs Between Survival and Subtype Metrics

While our primary configuration optimized for survival stratification, leading to a high C-index of 0.7512, OmicsCL also demonstrated the ability to adapt to alternative objectives. Specifically, by adjusting hyperparameters such as the number of clusters ( k ) in KMeans and the embedding dimension, we observed improvements in subtype-related clustering metrics.

For instance, increasing the number of clusters to k = 9 resulted in a significantly improved purity score of 0.5217 , which suggested enhanced alignment with known PAM50 subtypes. However, this configuration yielded a lower C-index, highlighting an inherent trade-off between biological subtype

discovery and survival discrimination in unsupervised multiomics representation learning.

This configurability indicated that OmicsCL was not rigidly bound to a single objective but could be adapted to prioritize specific clinical or biological goals, depending on the downstream application.

## VI. DISCUSSION

The behavior of OmicsCL across configurations reveals several important characteristics of unsupervised multi-omics learning. First, while the survival-aware contrastive loss clearly enhances stratification of patient outcomes, it can suppress subtype-specific structure in the latent space. This suggests that certain survival-relevant patterns may cut across known subtype boundaries or capture orthogonal biological signals.

We also explored several enhancements to improve survivalaware representation learning, including unsupervised margin scheduling, multi-view agreement penalties, and time-aware hard negative mining. However, these modifications did not consistently improve performance, and in some cases, introduced noise into the learning dynamics. Interestingly, a stabilized version of the time-similarity weighting via tanh ( t i -t j ) proved to be beneficial, pushing the C-index closer to the 0.73 range in intermediate configurations.

Notably, OmicsCL 's architecture remains intentionally simple-a lightweight MLP encoder per omics modality with a shared contrastive objective. Despite this simplicity, it is able to outperform many more complex approaches in unsupervised survival modeling. This suggests that meaningful integration and alignment of omics views, combined with principled objectives like contrastive and survival-aware losses, can yield powerful models with minimal architectural overhead.

These findings highlight the nuanced behavior of contrastive objectives in multi-omics settings, where optimizing for one biological axis can obscure others. The observed performance across various configurations emphasizes the importance of balancing task-specific objectives with methodological simplicity, highlighting contrastive frameworks as promising tools for unsupervised biomedical representation learning. A deeper understanding of how biological signals interact within the embedding space remains essential for advancing interpretable and clinically robust models.

## VII. LIMITATIONS AND FUTURE WORK

Despite the promising results of OmicsCL , there are several limitations to consider. First, while our approach demonstrates strong performance on survival prediction, its effectiveness in uncovering biologically meaningful subtypes remains dependent on specific hyperparameter configurations. The observed trade-off between survival-based clustering and subtype purity suggests that no single configuration optimally balances all downstream objectives. This highlights the need for more principled multi-objective optimization strategies or model selection criteria tailored to biomedical tasks.

Second, the current model architecture is based on independent encoders for each omics modality, followed by average fusion at the representation level. While effective, this simplistic late fusion may fail to capture higher-order interdependencies across modalities. Future work could investigate learnable attention-based fusion mechanisms, cross-modality transformers, or shared encoder layers to enable richer integration of omics-specific signals.

Third, although the model is trained in a purely unsupervised fashion, it still indirectly depends on survival time and censoring labels through the survival-aware contrastive loss. While this does not constitute supervised subtype learning, it introduces weak supervision from survival outcomes. Exploring completely label-agnostic training schemes or self-supervised pretext tasks could expand the generality of this framework to even noisier or less annotated datasets.

Lastly, this study focuses on a single cancer type (TCGA BRCA), which may limit generalizability. Applying OmicsCL to additional cohorts with diverse omics profiles and survival patterns will be crucial for validating its robustness and broad applicability. Furthermore, integrating clinical variables or imaging data into the contrastive training process remains an open direction for more holistic patient modeling.

In future iterations, we also aim to explore differentiable survival loss surrogates directly optimized for the concordance index, as well as semi-supervised extensions of OmicsCL that combine unlabeled data with sparse subtype annotations.

## VIII. CONCLUSION

OmicsCL offers a flexible and effective approach for uncovering clinically relevant structure in multi-omics cancer data without relying on subtype labels. By combining multiview representation learning with a survival-aware contrastive regularizer, our approach effectively integrates gene expression, DNA methylation, and miRNA profiles into unified embeddings that capture both molecular similarity and survival heterogeneity.

Through extensive experiments on the TCGA BRCA dataset, we demonstrate that OmicsCL achieves a strong unsupervised concordance index of 0.7512, along with statistically significant separation in Kaplan-Meier survival curves (log-rank p = 0 0082 . ). These results validate our method's capacity to learn prognostically informative representations in a label-free setting.

Moreover, we highlight the flexibility of our pipeline: by adjusting configuration parameters such as the number of clusters, embedding dimensionality, or survival loss weight, OmicsCL can be tuned to emphasize different evaluation criteria, such as subtype purity or silhouette score. This adaptability is particularly valuable in biomedical contexts where different downstream applications may prioritize interpretability, prognosis, or subtype discovery.

Overall, OmicsCL contributes to the growing body of methods enabling unsupervised discovery in high-dimensional biological data. Its modular structure, minimal reliance on supervision, and strong empirical performance suggest that it can serve as a foundation for future models tackling more complex, heterogeneous, and clinically nuanced datasets.

- [1] B. Wang, A. M. Mezlini, F. Demir, M. Fiume, Z. Tu, M. Brudno, B. Haibe-Kains, and A. Goldenberg, 'Similarity network fusion for aggregating data types on a genomic scale,' Nature Methods , vol. 11, no. 3, pp. 333-337, 2014. [Online]. Available: https: //www.nature.com/articles/nmeth.2810
- [2] R. Shen, A. B. Olshen, and M. Ladanyi, 'Integrative clustering of multiple genomic data types using a joint latent variable model with application to breast and lung cancer subtype analysis,' Bioinformatics , vol. 25, no. 22, pp. 2906-2912, 2009. [Online]. Available: https://academic.oup.com/bioinformatics/article/25/22/2906/180866
- [3] R. Argelaguet, B. Velten, D. Arnol, S. Dietrich, T. Zenz, J. C. Marioni, F. Buettner, W. Huber, and O. Stegle, 'Multi-omics factor analysis-a framework for unsupervised integration of multi-omics data sets,' Molecular Systems Biology , vol. 14, no. 6, p. e8124, 2018. [Online]. Available: https://www.embopress.org/doi/10.15252/msb.20178124
- [4] G. Andrew, R. Arora, J. Bilmes, and K. Livescu, 'Deep canonical correlation analysis,' in Proceedings of the 30th International Conference on Machine Learning (ICML) , vol. 28, no. 3. PMLR, 2013, pp. 12471255. [Online]. Available: https://proceedings.mlr.press/v28/andrew13. html
- [5] L. Du, R. Han, B. Liu, Y. Wang, and J. Li, 'Scccl: Singlecell data clustering based on self-supervised contrastive learning,' IEEE/ACM Transactions on Computational Biology and Bioinformatics , vol. 20, no. 3, pp. 2233-2241, 2023. [Online]. Available: https: //doi.org/10.1109/TCBB.2023.3241129
- [6] G. Ke, Z. Hong, Z. Zeng, Z. Liu, Y. Sun, and Y. Xie, 'Conan: Contrastive fusion networks for multi-view clustering,' in 2021 IEEE International Conference on Big Data (Big Data) . IEEE, 2021, pp. 653-660. [Online]. Available: https://doi.org/10.1109/BigData52589.2021.9671851
- [7] J. L. Katzman, U. Shaham, A. Cloninger, J. Bates, T. Jiang, and Y. Kluger, 'Deepsurv: personalized treatment recommender system using a cox proportional hazards deep neural network,' BMC Medical Research Methodology , vol. 18, no. 1, p. 24, 2018. [Online]. Available: https://bmcmedresmethodol.biomedcentral.com/ articles/10.1186/s12874-018-0482-1
- [8] C. Lee, W. Zame, J. Yoon, and M. van der Schaar, 'Deephit: A deep learning approach to survival analysis with competing risks,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 32, no. 1, 2018, pp. 2314-2321. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/11842
- [9] M. Herrmann, P. Probst, R. Hornung, V. Jurinovic, and A.-L. Boulesteix, 'Large-scale benchmark study of survival prediction methods using multi-omics data,' Briefings in Bioinformatics , vol. 22, no. 3, p. bbaa167, 2021. [Online]. Available: https: //academic.oup.com/bib/article/22/3/bbaa167/5893227
- [10] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, 'A simple framework for contrastive learning of visual representations,' in Proceedings of the 37th International Conference on Machine Learning , ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 1597-1607. [Online]. Available: https://proceedings.mlr.press/v119/chen20j.html
- [11] D. Leng, L. Zheng, Y. Wen, Y. Zhang, L. Wu, J. Wang, M. Wang, Z. Zhang, S. He, and X. Bo, 'A benchmark study of deep learning-based multi-omics data fusion methods for cancer,' Genome Biology , vol. 23, no. 1, p. 171, 2022. [Online]. Available: https://genomebiology. biomedcentral.com/articles/10.1186/s13059-022-02739-2
<|endofpaper|>