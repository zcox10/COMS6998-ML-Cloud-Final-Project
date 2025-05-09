<|startofpaper|>
## Investigating Task Arithmetic for Zero-Shot Information Retrieval

## Marco Braga

## Pranav Kasela

m.braga@campus.unimib.it

Department of Informatics, Systems and Communication DISCo, University of Milano-Bicocca Milano, Italy DAUIN Dipartimento di Automatica e Informatica, Politecnico di Torino Turin, Italy

## Alessandro Raganato

alessandro.raganato@unimib.it Department of Informatics, Systems and Communication DISCo, University of Milano-Bicocca Milano, Italy

## Abstract

Large Language Models (LLMs) have shown impressive zero-shot performance across a variety of Natural Language Processing tasks, including document re-ranking. However, their effectiveness degrades on unseen tasks and domains, largely due to shifts in vocabulary and word distributions. In this paper, we investigate Task Arithmetic, a technique that combines the weights of LLMs pretrained on different tasks or domains via simple mathematical operations, such as addition or subtraction, to adapt retrieval models without requiring additional fine-tuning. Our method is able to synthesize diverse tasks and domain knowledge into a single model, enabling effective zero-shot adaptation in different retrieval contexts. Extensive experiments on publicly available scientific, biomedical, and multilingual datasets show that our method improves state-of-the-art re-ranking performance by up to 18% in NDCG@10 and 15% in P@10. In addition to these empirical gains, our analysis provides insights into the strengths and limitations of Task Arithmetic as a practical strategy for zero-shot learning and model adaptation. We make our code publicly available at https://github.com/DetectiveMB/Task-Arithmetic-for-ZS-IR.

## CCS Concepts

- · Information systems → Language models .

## Keywords

Task Arithmetic, Domain-specific and Multilingual IR, Zero-Shot

## ACMReference Format:

Marco Braga, Pranav Kasela, Alessandro Raganato, and Gabriella Pasi. 2025. Investigating Task Arithmetic for Zero-Shot Information Retrieval. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13-18, 2025, Padua, Italy. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3726302.3730216

<!-- image -->

This work is licensed under a Creative Commons Attribution 4.0 International License. SIGIR '25, July 13-18, 2025, Padua, Italy © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1592-1/2025/07 https://doi.org/10.1145/3726302.3730216

pranav.kasela@unimib.it Department of Informatics, Systems and Communication DISCo, University of Milano-Bicocca Milano, Italy

## Gabriella Pasi

gabriella.pasi@unimib.it Department of Informatics, Systems and Communication DISCo, University of Milano-Bicocca Milano, Italy

## 1 Introduction

Large Language Models (LLMs) have achieved state-of-the-art performance in a wide range of tasks in several research fields [39, 49], including Information Retrieval (IR) [68]. By learning representations from massive unlabeled corpora, LLMs can be employed in document re-ranking [42, 45], query expansion [16, 34, 48], and synthetic data generation [3, 11]. Notably, these models often excel in zero-shot scenarios [1, 14, 59], effectively handling unseen tasks or domains without additional supervised fine-tuning. This zero-shot capability has driven their widespread use in document re-ranking [8, 35, 69], enabling robust retrieval even in the absence of domain-specific training data. Despite these advantages, domain mismatch remains a critical challenge that can significantly hamper the effectiveness of the model [58]. The BEIR benchmark [30, 54], which spans diverse tasks and domains, provides a heterogeneous framework for evaluating zero-shot IR performance. A common strategy is to pre-train a model on a large-scale IR dataset (e.g., MS-MARCO [41]) and then apply it zero-shot to unseen domains [30]. Although this approach often achieves strong performance, developing a single LLM that robustly generalizes to every domain remains very challenging [32, 54], as domain-specific adaptation typically requires large labeled datasets and considerable computational resources [61]. To mitigate these costs, Parameter-Efficient Fine-Tuning (PEFT) approaches have been proposed [12, 24, 25], which adapt a small subset of parameters while leaving most of the model frozen. Although PEFT significantly reduces the scale of updates and can be effective under limited data, labeled instances are still needed for each target domain, which makes it less suitable for frequent domain shifts or truly zero-shot scenarios. Meanwhile, the proliferation of publicly available LLMs, fine-tuned for diverse tasks, domains, and languages, on open-source platforms such as HuggingFace [29], offers a new opportunity to reuse existing models rather than training new ones from scratch. In this context, Task Arithmetic [27] emerges as a promising method. In fact, by merging the parameters of two or more fine-tuned LLMs through a simple addition or subtraction, Task Arithmetic transfers domain knowledge into a target model without further gradient-based optimization. More specifically, a Task Vector is defined as the difference

Figure 1: Proposed approach: Given a pre-trained LLM Θ 0 and its domain-finetuned version Θ 𝐷 , we compute the Task Vector 𝜏 𝐷 as their parameter difference. To build a domainspecific IR model Θ ′ , we add 𝜏 𝐷 to an IR-finetuned model Θ 𝑇 .

<!-- image -->

between the parameters of a domain-fine-tuned model and its original pre-trained version, allowing domain-specific information to be injected or removed in parameter space with minimal overhead.

In this work, we investigate how Task Arithmetic can be applied to Information Retrieval, emphasizing domain and language transfer under zero-shot conditions. As shown in Figure 1, we begin with a pre-trained LLM and leverage domain-specialized vectors derived from fine-tuned models, combining them with an IR-focused baseline to produce a new domain-aware model. Our extensive evaluation covers eight publicly available datasets spanning scientific, biomedical, and multilingual tasks. We experiment with six LLMs with parameter counts ranging from 66 million to 7 billion, including encoder-only, encoder-decoder, and decoder-only architectures. Inasmuch as all specialized models are publicly available, our method remains reproducible, computationally efficient, and it does not require additional training thus minimizing carbon footprint. Our results show that Task Arithmetic consistently improves upon strong IR baselines, with gains of up to 18% in NDCG@10 and 15% in P@10, highlighting the practical value of reusing existing domain-trained models and underlying how Task Arithmetic is a lightweight but powerful strategy for zero-shot domain and language adaptation in IR.

## 2 Methodology

In this Section, we first present related work on model merging and introduce Task Arithmetic (Section 2.1). We then explain how we adapt this framework for zero-shot IR (Section 2.2).

## 2.1 Related Work and Motivation

Adapting LLMs to specialized domains typically requires resourceintensive fine-tuning. To address frequent domain shifts or labelscarce settings, recent studies explore weight interpolation and model merging [38, 60], leveraging the observation that models fine-tuned on related tasks may reside in compatible regions of parameter space [2, 17]. This allows arithmetic operations, such as averaging parameters, to preserve or even boost performance [22, 28, 33]. Among these methods, Task Arithmetic [27] is particularly appealing for the fact that it requires no training. Each domainor task-specific fine-tuning run is represented as a Task Vector, defined by the difference between the fine-tuned model's parameters and those of the pre-trained model; adding or subtracting these vectors transfers knowledge without further updates [6, 18, 20, 26, 43]. Unlike approaches requiring adapters [10, 13, 24, 31] or gating modules [32, 37, 53], Task Arithmetic retains the original network architecture and remains cost-effective. In the context of zero-shot Information Retrieval (IR), Task Arithmetic remains relatively underexplored. Common approaches often rely on domain-specific adapters or fine-tuning [4, 5, 63-65], which are effective but still require training data and overhead. In contrast, Task Arithmetic simply reuses existing domain-fine-tuned LLMs, readily available on open-source platforms, and integrates them plug-and-play into an IR model. This approach is particularly appealing given the recent proliferation of LLMs fine-tuned for diverse domains and languages; by merging their specialized capabilities, we can construct a domain-aware IR model with minimal computational cost.

## 2.2 Task Arithmetic for Zero-Shot IR

In this Section we detail how Task Arithmetic can be applied in the context of Information Retrieval, particularly for domain and language transfer in zero-shot conditions. Let Θ 0 = {( 𝜃 1 ) 0 , . . . , ( 𝜃 𝑁 ) 0 } denote the parameters of a pre-trained LLM. Fine-tuning this model on a generic IR task (e.g., on the MS-MARCO benchmark [41]) produces Θ 𝑇 = {( 𝜃 1 ) 𝑇 , . . . , ( 𝜃 𝑁 𝑇 ) } , while fine-tuning Θ 0 on a specific domain yields Θ 𝐷 = {( 𝜃 1 ) 𝐷 , . . . , ( 𝜃 𝑁 𝐷 ) } . We define the Task Vector 𝜏 𝐷 for domain 𝐷 as follows:

$$\tau _ { D } = \{ \tau _ { 1 }, \dots, \tau _ { N } \}, \ \text{ where } \ \tau _ { i } = ( \theta _ { i } ) _ { D } - ( \theta _ { i } ) _ { 0 }. \quad ( 1 )$$

This vector 𝜏 𝐷 represents the domain-specific shift in the parameter space. To create a domain-aware IR model Θ ′ , we add 𝜏 𝐷 to the IR-tuned model Θ 𝑇 :

$$\Theta ^ { \prime } = \{ \theta _ { i } ^ { \prime } = ( \theta _ { i } ) _ { T } + \alpha \tau _ { i } \} _ { i = 1 } ^ { N }.$$

The scaling factor 𝛼 ∈ R controls how much of the domain vector is injected. If 𝛼 = 0, then Θ ′ defaults to Θ 𝑇 . Setting 𝛼 &gt; 0 adds the specialized knowledge, while 𝛼 &lt; 0 subtracts it. In a fully zeroshot scenario, which does not require additional labelled data, 𝛼 is equal to one. If, instead, a small development set is available, the hyperparameter 𝛼 can be optimized.

In summary, our framework involves three models: i) a publicly available pre-trained LLM, i.e. Θ 0, ii) the Θ 0 LLM fine-tuned on a specific domain, i.e. Θ 𝐷 , and iii) the Θ 0 LLM fine-tuned for IR, i.e. Θ 𝑇 . In many publicly released models, Θ 𝐷 is trained using a Language Modeling (LM) or Masked Language Modeling (MLM) objective, while Θ 𝑇 is fine-tuned to specific IR tasks like re-ranking or passage retrieval. As shown in Figure 1, we follow a three-step procedure: (1) Task Vector Generation : Using Equation 1, we compute 𝜏 𝐷 by subtracting the pre-trained model's weights Θ 0 from the domain-fine-tuned model's weights Θ 𝐷 . This step captures the domain shift in parameters. (2) Task Vector Integration : Following Equation 2, we add the task vector 𝜏 𝐷 (scaled by 𝛼 ) to the IR-specific model Θ 𝑇 , obtaining the Θ ′ model. This operation seamlessly transfers domain knowledge into the IR model without needing further backpropagation or labeled domain data. (3) Zero-Shot Evaluation : Finally, we directly evaluate the adapted model Θ ′ on IR tasks in the target domain. As no training steps are required, we can readily test multiple domains, languages, or tasks by simply substituting different Task vectors.

## 3 Experimental Setup

In this Section, we describe the datasets, evaluation metrics, and models used to assess the effectiveness of our proposed approach.

Table 1: Effectiveness of all models on Biomedical and Scientific domains. Best results are highlighted in boldface.

| Model        | Model                                       | SciFact   | SciFact   | SciFact   | SciFact   | NFCorpus   | NFCorpus   | NFCorpus   | NFCorpus   | SCIDOCS   | SCIDOCS   | SCIDOCS   | SCIDOCS   | TREC-COVID   | TREC-COVID   | TREC-COVID   | TREC-COVID   |
|--------------|---------------------------------------------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|-----------|-----------|-----------|-----------|--------------|--------------|--------------|--------------|
| Re-ranker    | Variant                                     | P@10      | NDCG@3    | NDCG@10   | MAP@100   | P@10       | NDCG@3     | NDCG@10    | MAP@100    | P@10      | NDCG@3    | NDCG@10   | MAP@100   | P@10         | NDCG@3       | NDCG@10      | MAP@100      |
| BM25         | BM25                                        | .091      | .637      | .691      | .649      | .247       | .404       | .343       | .154       | .086      | .156      | .165      | .112      | .734         | .764         | .688         | .085         |
| Llama-2-7B   | Θ 0 : Pre-trained                           | . 099     | .701      | .748      | .702      | .260       | .431       | .363       | .165       | . 099     | .179      | . 191     | . 130     | .838         | .845         | .782         | .095         |
| Llama-2-7B   | Θ 𝐷 : Domain-specific (MedTuned)            | .097      | .699      | .742      | .701      | .250       | .410       | .349       | .158       | .095      | .175      | .184      | .124      | .812         | .810         | .761         | .093         |
| Llama-2-7B   | Θ 𝑇 : MS-MARCO (RankingLlama)               | . 099     | . 724     | . 770     | . 731     | . 265      | . 448      | . 373      | . 170      | .096      | . 182     | .188      | .129      | .860         | . 869        | .810         | .098         |
| Llama-2-7B   | Θ ′ : Task Arithmetic ( 𝛼 = 1)              | .096      | .718      | .757      | .723      | .265       | .445       | .370       | .167       | .095      | .179      | .185      | .126      | .858         | .867         | .801         | .098         |
| Llama-2-7B   | Θ ′ : Task Arithmetic (optimized 𝛼 = 0 . 8) | .097      | . 728     | .765      | .730      | .262       | .442       | .365       | .165       | .097      | . 182     | .189      | .129      | . 866        | .867         | . 812        | . 099 *      |
| T5-Large     | Θ 0 : Pre-trained                           | .089      | .639      | .691      | .650      | .247       | .404       | .343       | .154       | .076      | .144      | .150      | .102      | .738         | .757         | .688         | .086         |
| T5-Large     | Θ 𝐷 : Domain-specific (SciFive)             | .091      | .637      | .691      | .649      | .247       | .404       | .343       | .154       | .065      | .111      | .121      | .082      | .596         | .582         | .551         | .076         |
| T5-Large     | Θ 𝑇 : MS-MARCO (Mono-T5)                    | . 095     | . 706     | . 743     | . 709     | . 266 *    | . 431      | . 368 *    | . 167      | .095      | .170      | .182      | .124      | .784         | 805          | .735         | .092         |
| T5-Large     | Θ ′ : Task Arithmetic ( 𝛼 = 1)              | .092      | .688      | .721      | .692      | .257       | .420       | .356       | .161       | .096      | . 176     | .185      | .124      | .816         | . 818        | . 765        | .096         |
| T5-Large     | Θ ′ : Task Arithmetic ( 𝛼 = 0 . 9)          | .092      | .699      | .727      | .699      | .259       | .423       | .359       | .162       | . 098 *   | . 176     | . 187 *   | . 126     | . 818        | .805         | .759         | . 097 *      |
| T5-base      | Θ 0 : Pre-trained                           | .091      | .638      | .691      | .649      | .247       | .404       | .343       | .154       | .081      | .152      | .156      | .105      | .710         | .761         | .671         | .084         |
| T5-base      | Θ 𝐷 : Domain-specific (SciFive)             | .090      | .638      | .691      | .648      | .248       | .400       | .343       | .154       | .059      | .110      | .115      | .080      | .646         | .662         | .598         | .079         |
| T5-base      | Θ 𝑇 : MS-MARCO (Mono-T5)                    | .096      | .681      | .726      | .684      | .258       | . 424      | . 359      | . 162      | .090      | .163      | .173      | .118      | .762         | .782         | .712         | .089         |
| T5-base      | Θ ′ : Task Arithmetic ( 𝛼 = 1)              | .089      | .645      | .686      | .649      | .250       | .405       | .345       | .156       | .070      | .131      | .136      | .094      | .764         | .825         | .726         | .088         |
| T5-base      | Θ ′ : Task Arithmetic ( 𝛼 = 0 . 7)          | . 098     | . 702     | . 748     | . 708 *   | . 259      | . 424      | .358       | . 162      | . 091     | . 171     | . 176     | . 120     | . 798        | . 836        | . 753        | . 095 *      |
| RoBERTa-base | Θ 0 : Pre-trained                           | .090      | .633      | .686      | .646      | .248       | .405       | .344       | .154       | .072      | .139      | .142      | .096      | .612         | .681         | .579         | .077         |
| RoBERTa-base | Θ 𝐷 : Domain-specific (BioMed RoBERTa)      | .091      | .639      | .691      | .649      | .241       | .408       | .340       | .154       | .071      | .138      | .140      | .096      | .600         | .657         | .561         | .077         |
| RoBERTa-base | Θ 𝑇 : MS-MARCO (msmarco-RoBERTa)            | .095      | .655      | .707      | .662      | .258       | .425       | .359       | .162       | .087      | .165      | .170      | .116      | .776         | .818         | .732         | .090         |
| RoBERTa-base | Θ ′ : Task Arithmetic ( 𝛼 = 1)              | .092      | .649      | .700      | .659      | .250       | .408       | .347       | .156       | .078      | .153      | .156      | .108      | .784         | . 822        | .734         | .089         |
| RoBERTa-base | Θ ′ : Task Arithmetic ( 𝛼 = 0 . 3)          | . 096     | . 669     | . 720     | . 676     | . 259      | . 432      | . 361      | . 165      | . 088     | . 169     | . 173     | . 118     | . 806 *      | .821         | . 757 *      | . 093 *      |
| DistilBERT   | Θ 0 : Pre-trained                           | .091      | .635      | .689      | .647      | .246       | .407       | .345       | .156       | .082      | .153      | .159      | .108      | .712         | .745         | .666         | .084         |
| DistilBERT   | Θ 𝐷 : Domain-specific (Bio-DistilBert)      | .093      | .661      | .706      | .663      | .251       | .405       | .346       | .155       | .083      | .151      | .160      | .108      | .736         | .789         | .697         | .086         |
| DistilBERT   | Θ 𝑇 : MS-MARCO (msmarco-distilbert)         | .093      | .657      | .703      | .662      | . 258      | .418       | .357       | .161       | .087      | .159      | .168      | .115      | .794         | .808         | .744         | .091         |
| DistilBERT   | Θ ′ : Task Arithmetic ( 𝛼 = 1)              | .090      | .641      | .689      | .650      | .249       | .406       | .346       | .156       | .076      | .141      | .147      | .099      | .710         | .745         | .675         | .083         |
| DistilBERT   | Θ ′ : Task Arithmetic ( 𝛼 = 0 . 5)          | . 095     | . 671     | . 720     | . 677     | .257       | . 429      | . 359      | . 163      | . 088     | . 162     | . 171     | . 116     | . 806        | . 849        | . 765        | . 094 *      |

The evaluation spans scientific, biomedical, and multilingual scenarios to show the potential of Task Arithmetic for zero-shot IR.

## 3.1 Datasets and Evaluation Metrics

Weevaluate our approach on eight publicly available datasets across diverse domains. Four of these datasets are drawn from the BEIR benchmark [54]: TREC-COVID [56] and NFCorpus [9] address biomedical retrieval, while SCIDOCS [19] and SciFact [57] focus on scientific citation prediction and fact-checking, respectively. The remaining four datasets concern language-specific retrieval: GermanQuAD [40], which targets question answering in German, and three subsets (English, French, and Spanish) from the MIRACL multilingual IR challenge [67]. We focus on the biomedical and scientific domains, as they represent areas where both domain-specific and IR-focused models are available, thus enabling the application of Task Arithmetic. In contrast, we exclude BEIR datasets based on Wikipedia since the pre-trained language models used in our experiments have already been trained on Wikipedia [21, 36, 46, 55], and we cannot add domain-specific knowledge through Task Arithmetic. Retrieval effectiveness is measured using P@10, NDCG@3, NDCG@10 and MAP@100. Statistical significance is assessed via a Bonferroni-corrected two-sided paired student's t-test at 99% confidence. In all result tables, the symbol * indicates a statistically significant improvement over the best performing baseline.

## 3.2 Models and Baselines

or language-specific fine-tuned model from those of its original pre-trained version. Specifically, we use LLama2-MedTuned-7b [51], SciFive [44], BioMed-RoBERTa [23], and Bio-DistilBERT [50] for the biomedical and scientific domains, as well as MT5-base-german , MT5-base-spanish , MT5-base-french , and MT5-base-english [15] for language adaptations. These Task Vectors are then added to models fine-tuned on MS-MARCO ( RankingGPT-Llama2-7b [66], MonoT5 [42], msmarco-RoBERTa [47], msmarco-distilbert [47], and MT5-base-msmarco [7]), a widely adopted passage retrieval benchmark [41]. This setup facilitates direct comparisons with the same models specialized for either a specific language or domain (i.e. Θ 𝐷 ), or IR (i.e. Θ 𝑇 ). In a fully zero-shot scenario, we set 𝛼 = 1. Furthermore, in a setting where few labeled data are available, we tune the scaling factor 𝛼 from 0.1 to 1.0 in steps of 0.1, selecting the optimal value based on the highest average retrieval performance over two development sets: the official NFCorpus split and a 20% subset of SciFact training queries. Since GermanQuAD and MIRACL do not provide development sets, we apply a fully zero-shot scenario by not optimizing the value of 𝛼 , i.e. we put 𝛼 = 1. We report both the results with the optimized and not-optimized 𝛼 . All re-ranking experiments begin by retrieving the top 100 documents via BM25. Following a two-stage retrieval paradigm, the final rankings are then computed using a weighted sum of BM25 and LLM scores, with 𝜆 𝐵𝑀 25 and 𝜆 𝐿𝐿𝑀 optimized in [ 0 1 , ] on the NFCorpus and SciFact development sets. We take the average score for all remaining datasets, i.e. 𝜆 𝐵𝑀 25 = 𝜆 𝐿𝐿𝑀 = 0 5. .

For our experiments, we focus on two-stage retrieval and we use six different pre-trained language models (i.e. Θ 0) spanning multiple retrieval paradigms (bi-encoder, cross-encoder, LLM) and neural architectures (encoder-only, encoder-decoder, decoder-only). In details, we use DistilBERT [52] and RoBERTa-base [36] as encoder-only bi-encoders, T5-base T5-Large , [46], and MT5-base [62] as encoder-decoder cross-encoders, and LLama-2-7b [55] as a decoder-only LLM. For each pre-trained model, we compute Task Vectors by subtracting the weights of a publicly available domain-

## 4 Results and Discussion

Table 1 presents the performance of our approach on the biomedical and scientific datasets, evaluated with five different models. In the initial evaluation, we fix 𝛼 = 1. Under this setting, Task Arithmetic outperforms the MS-MARCO fine-tuned baselines only on TREC-COVID with RoBERTa-base, T5-base, and T5-Large, and on SCIDOCS with T5-Large. These findings suggest the need for a small amount of labeled data to optimize the value of 𝛼 . The

Marco Braga, Pranav Kasela, Alessandro Raganato, and Gabriella Pasi

| Model     | Model                          | GermanQuAD   | GermanQuAD   | GermanQuAD   | GermanQuAD   | MIRACL Spanish   | MIRACL Spanish   | MIRACL Spanish   | MIRACL Spanish   | MIRACL French   | MIRACL French   | MIRACL French   | MIRACL French   | MIRACL English   | MIRACL English   | MIRACL English   | MIRACL English   |
|-----------|--------------------------------|--------------|--------------|--------------|--------------|------------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|-----------------|------------------|------------------|------------------|------------------|
| Re-ranker | Variant                        | P@10         | NDCG@3       | NDCG@10      | MAP@100      | P@10             | NDCG@3           | NDCG@10          | MAP@100          | P@10            | NDCG@3          | NDCG@10         | MAP@100         | P@10             | NDCG@3           | NDCG@10          | MAP@100          |
|           | BM25                           | .059         | .381         | .437         | .397         | .135             | .248             | .270             | .215             | .052            | .125            | .174            | .139            | .107             | .251             | .302             | .247             |
| MT5-base  | Θ 0 : Pre-trained              | .050         | .275         | .336         | .296         | .094             | .173             | .191             | .152             | .035            | .095            | .127            | .106            | .075             | .175             | .212             | .174             |
| MT5-base  | Θ 𝐷 : Language specific        | .051         | .296         | .353         | .316         | .097             | .172             | .191             | .153             | .045            | .099            | .143            | .113            | .079             | .180             | .225             | .187             |
| MT5-base  | Θ 𝑇 : MS-MARCO (en)            | .069         | .451         | .513         | .463         | .190             | .342             | .379             | .301             | .071            | .175            | .234            | .186            | .140             | .330             | .398             | .325             |
| MT5-base  | Θ ′ : Task Arithmetic ( 𝛼 = 1) | . 071 *      | . 477 *      | . 537 *      | 487 *        | . 200 *          | . 373 *          | . 405 *          | . 325 *          | . 081 *         | . 215 *         | . 278 *         | . 220 *         | .151 *           | .366 *           | 435 *            | .358 *           |

Table 2: Effectiveness of all models on Language Transfer. Best results are highlighted in boldface.

remainder of this section focuses on the results obtained when 𝛼 is optimized. Across all datasets and metrics, the Task Arithmetic-based model ( Θ ′ ) consistently outperforms BM25, indicating its potential for effective domain adaptation in IR. For bi-encoders (DistilBERT and RoBERTa-base), Task Arithmetic yields consistent improvements over all baselines, with one exception: DistilBERT on the NFCorpus shows a minor drop in P@10 compared to the MSMARCO fine-tuned counterpart ( Θ 𝑇 ). Nevertheless, our approach achieves a statistically significant improvement in MAP@100 on TREC-COVID, surpassing every baseline. For cross-encoders (T5base and T5-Large), Task Arithmetic outperforms MonoT5 ( Θ 𝑇 ) on SCIDOCS and TREC-COVID, while producing comparable results on SciFact and NFCorpus. Notably, our method shows significant gains in MAP@100 on TREC-COVID, on the T5 variant, and yields statistically significant improvements in NDCG@10 on TREC-COVID and SCIDOCS. Regarding the decoder-only model (LLama-2), our approach ( Θ ′ ) achieves superior performance on TREC-COVID compared to all baselines and remains competitive on SCIDOCS. Interestingly, the pre-trained LLama-2 ( Θ 0) attains the highest P@10 and NDCG@10 on SciFact and SCIDOCS, which likely reflects the extensive domain knowledge acquired during large-scale pre-training [55]. Finally, it is worth noting that the optimal scaling factor 𝛼 exceeds 0.3 for all models and surpasses 0.7 for T5 variants and LLama-2, indicating that Task Arithmetic injects non-trivial domain knowledge into these retrieval models.

Table 2 extends the results to multilingual IR by using MT5-base as a cross-encoder for language-specific tasks. Both our proposed approach ( Θ ′ ) and the IR-specific baseline ( Θ 𝑇 ) outperform all other baselines on each metric. Notably, our method shows statistically significant improvements over the IR-specific model by up to 18% in NDCG@10, highlighting its effectiveness in adapting to new language settings. The pre-trained MT5-base ( Θ 0) and its languagespecific variant ( Θ 𝐷 ) do not outperform BM25, suggesting that these models are not inherently optimized for IR tasks. In contrast, our approach successfully injects language-specific knowledge into the multilingual IR model ( Θ 𝑇 ), thereby enhancing its retrieval capabilities. These results further support Task Arithmetic as a lightweight but powerful strategy for zero-shot adaptation in multilingual IR.

## 5 Ablation Study

Weconductanablation study to examine scaling factor impact while applying Task Arithmetic. Table 3 presents NDCG@10 scores on NFCorpus and SciFact development sets for 𝛼 values from 0.1 to 1.0 in increments of 0.1. The results reveal that retrieval performance varies considerably with changes in 𝛼 . On SciFact, for instance, T5base improves from .640 at 𝛼 = 1 0 to .722 at . 𝛼 = 0 7, representing a . notable difference of approximately 12%. In general, while values in

Table 3: Ablation study about the impact of the scaling factor.

| Model            | Dataset          | 0.1       | 0.2            | 0.3            | 0.4       | 0.5        | 0.6       | 0.7            | 0.8        | 0.9         | 1         |
|------------------|------------------|-----------|----------------|----------------|-----------|------------|-----------|----------------|------------|-------------|-----------|
| LLama 2-7B       | SciFact          | .631 .235 | .643           | .660 .239      | .675 .241 | .687 . 242 | .705 .241 | . 712          | .711 . 242 | .705 .241   | .704 .241 |
|                  | NFCorpus SciFact | .728      | .238 .732      | .727           | .731 .308 | .730 .308  | .732 .306 | .239 .728 .306 | .730 . 311 | . 737 . 311 | .710 .309 |
| T5-Large T5-base | NFCorpus SciFact | .307 .712 | .307 .710 .301 | .307 .712 .304 | .711 .306 | .709 .306  | .718 .311 | . 722 . 313    | .708 .312  | .673 .303   | .640 .298 |
| RoBERTa-base     | SciFact NFCorpus | .713      | .717           | . 725          | .723      | .722 .333  | .710      | .715           | .706       | .695        | .687      |
|                  | SciFact          | .334      | .331           | .332           | .332      | . 724      | . 334     | .330           | .323       | .315        | .307      |
|                  |                  |           | .720           | .723           | .722      |            | .712      | .705           |            | .688        | .652      |
| DistilBERT       | NFCorpus         | .723 .334 | . 336          | .333           | .333      | .333       | .329      | .325           | .699 .306  | .297        | .286      |

the 0.7-0.9 range often yield strong results for models such as T5base and T5-Large, no single 𝛼 value is consistently optimal across all models or datasets. This variability underscores the importance of model-specific calibration. Moreover, 𝛼 = 1 0 rarely provides the . best performance, suggesting that excessive emphasis on domain parameters can overshadow the IR-specific knowledge embedded in IR-tuned weights. RoBERTa-base and DistilBERT sometimes peak at moderate values between 0.3 and 0.5, whereas T5-based models and LLama-2 commonly favor higher settings. These observations indicate that a small grid search over 𝛼 can be effective in identifying a tradeoff between the IR and the domain-specific knowledge.

## 6 Conclusion

In this paper, we investigate Task Arithmetic as a training-free method for zero-shot domain and language adaptation in IR, leveraging publicly available domain- and IR-specific LLMs. To this aim, we evaluate Task Arithmetic with six LLMs, including encoderonly, encoder-decoder, and decoder-only architectures, across scientific, biomedical, and multilingual datasets. Our analysis shows that the proposed approach consistently improves the IR-specific model's performance across the board, reaching gains of up to 18% in NDCG@10. These findings underscore Task Arithmetic as a lightweight yet powerful strategy for IR applications, particularly when computational resources or labeled data are limited.
<|endofpaper|>