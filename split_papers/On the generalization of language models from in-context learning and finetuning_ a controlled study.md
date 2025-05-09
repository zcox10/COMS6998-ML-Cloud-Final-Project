<|startofpaper|>
<!-- image -->

## On the generalization of language models from in-context learning and finetuning: a controlled study

Andrew K. Lampinen *,1 , Arslan Chaudhry *,1 , Stephanie C.Y. Chan *,1 , Cody Wild , Diane Wan , Alex Ku 1 1 1 , Jörg Bornschein , Razvan Pascanu 1 1 , Murray Shanahan 1 and James L. McClelland 1,2

* Equal contributions, 1 Google DeepMind, 2 Stanford University

Large language models exhibit exciting capabilities, yet can show surprisingly narrow generalization from finetuning - from failing to generalize to simple reversals of relations they are trained on, to missing logical deductions that can be made from trained information. These failures to generalize from fine-tuning can hinder practical application of these models. However, language models' in-context learning shows different inductive biases, and can generalize better in some of these cases. Here, we explore these differences in generalization between in-context- and fine-tuning-based learning. To do so, we constructed several novel datasets to evaluate and improve models' ability to generalize from finetuning data. The datasets are constructed to isolate the knowledge in the dataset from that in pretraining, to create clean tests of generalization. We expose pretrained large models to controlled subsets of the information in these datasets - either in context, or through fine-tuning - and evaluate their performance on test sets that require various types of generalization. We find overall that in data-matched settings, in-context learning can generalize more flexibly than fine-tuning (though we also find some qualifications of prior findings, such as cases when fine-tuning can generalize to reversals embedded in a larger structure of knowledge). We build on these findings to propose a method to enable improved generalization from fine-tuning: adding in-context inferences to finetuning data. We show that this method improves generalization across various splits of our datasets and other benchmarks. Our results have implications for understanding the inductive biases of different modes of learning in language models, and practically improving their performance.

## Introduction

Language models (LMs) pretrained on huge corpora of internet text become efficient in-context learners; they can generalize from a few examples of a task to answer new instances (Brown et al., 2020; Team et al., 2023). Pretrained LMs can also be fine-tuned for downstream tasks using relatively few examples-though achieving good generalization from fine-tuning often requires hundreds to thousands of examples (e.g. Kirstain et al., 2022; Vieira et al., 2024). Indeed, the generalization from fine-tuning on a particular example can be surprisingly limited; for example, LMs fine-tuned on a statement like 'B's mother is A' fail to generalize to answer 'Who is A's son?' (Berglund et al., 2024). However, LMs can readily answer questions about these types of reverse relations in context (e.g. Lampinen et al.,

2024b). Do in-context learning and fine-tuning therefore result in different patterns of generalization (cf. Chan et al., 2022b; Russin et al.; Shen et al., 2023)? If so, how does this affect how we should adapt models to downstream tasks? In this paper, we explore these questions.

To do so, we construct controlled synthetic datasets of factual knowledge. We design these datasets to have complex and self-consistent structures, but avoid any overlap with knowledge that may be present in pretraining corpora. We create train and test splits of these datasets that involve different types of generalization: reversals or chaining multiple logical inferences into a syllogism. We then evaluate how well large pretrained language models generalize to these test sets through fine-tuning, or through in-context learning-via placing the entire training set (or

large subsets thereof) in context. We also explore various methods of improving generalization, such as data augmentation. Overall, we find that across various datasets, in-context learning (ICL) generalizes better than finetuning, however, finetuning generalization can be improved, and in fact made better than ICL (on the original data), by spending more train time compute to augment the training dataset using in-context inferences.

Our contributions are as follows:

- · We study the distinct patterns of generalization that pretrained LMs exhibit from incontext learning and finetuning.
- · We find that in-context learning over the entire training dataset often generalizes better than finetuning does, when evaluated on systematic holdouts like reversals, syllogistic inferences, compositions, etc.
- · We propose to bridge this gap via dataset augmentation-prompting an LM to generate augmentations of the dataset in-context, and then adding these augmented data to the training set.
- · We show that dataset augmentations can bridge the gap to yield improved generalization from fine-tuning.
- · We also propose a fine-tuning method that breaks up correlations among sentences, thereby improving the benefits of dataset augmentation.

## Datasets

We evaluate learning and generalization across several different datasets which are crafted to respectively isolate features of the generalization problem, or situate it within a broader set of learning challenges. We also draw on datasets from prior work.

## Simple reversals and syllogisms

We first test our methods with two simple datasets 1 that contain independent examples of reversal and syllogistic reasoning.

1 These datasets are adapted from the nonsense NLI and Syllogism datasets from Lampinen et al. 2024b.

Simple reversals: Each training example consists of a brief context (e.g. 'Did you know that') followed by a single sentence containing a comparison of two entities, e.g. 'femp are more dangerous than glon.' There are a hundred such facts provided (with the dimension of comparison for each sampled from a set of 28 features, e.g. 'brighter', 'heavier,' etc.), each of which is repeated across 10 different training articles with a randomly sampled context statement in each. The test set consists of forced choices between the correct reversal and a contradictory relation, e.g.: 'glon are less/more dangerous than femp.'

Simple syllogisms: There are 69 training examples, each of which contains a piece of context (which entities are referred to) and two statements that form a syllogism. For example: 'The relationships between glon, troff and yomp are: All glon are yomp. All troff are glon.' The test examples test whether the model makes a correct inference from that syllogistic form, by providing a context of two entities from the conclusion, and then scoring all possible statements about their relationship involving the quantifiers and relations in the dataset. Following Lampinen et al. (2024b), we omit the partial negative form (some X are not Y) pattern; thus, there are six such possible statements (the product of 'all', 'some', and 'no' together with the two directions each of those relations could take). For example, the context for the test example corresponding to the syllogism above would be 'The relationships between yomp and troff are:' and the correct answer would be 'All troff are yomp.'

## Reversal curse paper

We use the reversal dataset proposed by Berglund et al. (2024) that contains one-sentence descriptions of fictional celebrities. Dataset examples can have the celebrity name (e.g. 'Daphne Barrington') preceding the description (e.g. 'the director of "A Journey Through Time."') or vice-versa. During finetuning, the training dataset is presented in one order, say where the name precedes the description, and then tested on the reversed order, where the name follows the description.

Following Berglund et al. (2024), we use two

independent sets of celebrities - 'A' and 'B'. During finetuning, we present the celebrities from the set ' A' with names preceding the description, and from set 'B' where names and descriptions appear in both orders but in separate examples. Overall, the train set consists of 3600 examples. The test set evaluates whether the model can infer the celebrity names from the set 'A' given only their descriptions. To add distractors in test examples, we include three randomly selected incorrect celebrity names in the list of options to score.

## Semantic structure benchmark

Figure 1 | A semantic structure with a hierarchy of properties and relations. (Reproduced with permission from Rogers and McClelland (2008).)

<!-- image -->

The benchmark is built around a relational semantic hierarchy that allows deductive inferences and abstractions. This hierarchy is based on realworld categories and relations, and draws inspiration from prior work in cognitive science that has studied learning of rich semantic hierarchies (see Fig. 1 for an intuitive example that composes part of the real-world structure). We similarly create a hierarchical knowledge structure involving 110 categories of animals and objects, and properties (1-6 per category, plus inherited ones) and relations. This underlying structure is derived from the real world.

In order to make the structure novel to the pretrained models, however, we replace all nouns, adjectives, and verbs with nonsense terms. This removes the potential overlap with pretraining data, thus ensuring that the data properties follow certain real-world distributional features, but are not contaminated.

The use of nonsense terms does, however, potentially pose tokenization challenges; however, we ameliorate these by generating short 4-5 letter nonsense words using plausible combinations of phonemes for English (sampled via Gao et al., 2023). Moreover, we show in the next sections that in practice the models can readily make inferences about these nonsensical entities in context, so use of unfamiliar words is not the primary bottleneck on model performance.

Train set: For training, we assemble train-set facts about this semantic hierarchy into short vaguely-wikipedia-like synthetic articles with varied formatting and styles, as well as some QA examples (to ensure that fine-tuning on the data doesn't degrade question answering capabilities). We ensure that all facts that are necessary for the test questions (below) are presented in at least one training document. However, some facts will be redundantly presented across multiple documents. We create a total of 2200 training documents of 4-20 sentences in length.

Test sets: Within the semantic structure, our test sets accommodate reversals of relations (gruds [dogs] are a type of abmes [mammals] =&gt; abmes include gruds; cf. Berglund et al., 2024), syllogism-like deductive inferences (e.g. gruds [dogs] are abmes [mammals]; abmes are rony [warm-blooded] =&gt; gruds are rony), and longer deductions. Specifically, we focus on the following splits (in rough difficulty order):

- · Rephrasings of trained facts that don't change the direction: used for validation that the material is learned.
- · Reversals of trained facts.
- · Syllogisms over trained facts.
- · Category holdouts : only one fact about a category is present in training: what its parent category is. All possible inferences from that fact are tested. This overlaps with some aspects of the syllogism splits, except that the information about the target category is strictly more limited, thus limiting other cues

that could aid generalization.

When creating the evaluation questions, we choose difficult distractors for the incorrect answers, by choosing entities or properties that have the target relationship to a different entity in the dataset. For example, if the correct answer is 'gruds are rony' one of the distractors might be 'gruds are zept' where there is some other valid statement in the dataset such as 'telk are zept.' Thus, it is not possible to guess simply by the local context of the words which answer is correct.

## Methods

## Evaluation

We perform evaluation using multiple-choice likelihood scoring. We do not provide the answer choices in context.

## Fine-tuning

Our fine-tuning experiments mainly involve tuning Gemini 1.5 Flash (Team et al., 2024) on our datasets. We generally fine-tune with batch size 8 or 16 and learning rate 3 · 10 -4 for 200-1000 steps (depending on dataset size and loss).

## In-context evaluation

To perform full-dataset in-context evaluation, we concatenate the documents in a training dataset and provide them as context to the (instruction tuned) model. We then prompt the model to use that context to answer the question. We randomly subsample the documents by a factor of 8x when evaluating models in-context on the largest datasets, as there is some redundancy in the datasets, and we find that models experience interference as the context length scales.

## Dataset augmentation

Our key approach to dataset augmentation is to take advantage of the in-context generalization of models in order to improve the coverage of the fine-tuning dataset. We approach this via several methods, but all chiefly aim at the goal of spending training time compute for in-context inference to create more finetuning data, in order to improve generalization out-of-context at test time.

Specifically , we consider two types of augmentation: a local strategy that tries to increase the flexibility with which particular pieces of information can be used, and global strategies that attempt to relate between different pieces of information. Each of these strategies uses distinct context and prompts (Appx. A.1).

Local (sentence) augmentation: We prompt an LM to augment each training data point (e.g. sentence) to enhance the flexibility with which the model encodes it. Our prompt includes examples of rephrasings and reversals.

Global (document) augmentation: We concatenate the full training dataset as context, then provide a particular document and prompt the model to generate inferences by linking between that document and the rest of the documents in context. This results in a longer reasoning trace of relevant inferences.

## Sentence-splitting

Some datasets, such as the fictional celebrities dataset by Berglund et al. (2024) and our semantic structure dataset, contain documents comprising multiple sentences. We find that splitting these documents at sentence-level, into multiple finetuning examples, leads to a much improved finetuning performance - even after accounting for the total dataset size and gradient steps. We explore two ways of splitting a document into sentences 1) Independent Splitting : wherein all sentences are split independently into separate training examples, 2) Cumulative Splitting : wherein a document is split such that for any target sentence in an example, all preceding sentences of the document are added to the context of that example. For an 𝑛 -sentence document, both methods produce 𝑛 training examples, although with different context sizes. We analyze the effect of sentence splitting on the model generalization in appendix B.1. In the following sections, we assume independent sentence splitting unless stated otherwise.

## Experiments

## Reversal Curse

Figure 2 | Reversal curse paper results.

<!-- image -->

In Fig. 2, we first explore the generalization issues in the context of the reversal curse phenomenon and dataset released by Berglund et al. (2024). We replicate the authors' finding that finetuning on the forward directions does not produce generalization to the reversals. The authors of that work noted briefly that in-context the models could perform better at this task. We study this more systematically by presenting the entire dataset in context, and find that the model performs nearly at ceiling on the reversals-thus providing a strong demonstration of the benefits of in-context learning over finetuning. Finetuning with data augmented with in-context inferences yields similarly high test performance. Simple finetuning, on the other hand, has near zero accuracy as the finetuned model always prefers those (incorrect) celebrity name completions that were seen as targets during training, regardless of the context. Finally, a pretrained model performs near chance on the test set, indicating a lack of contamination.

## Simple nonsense reversals

We then correspondingly test the models on the simple nonsense version of our reversal dataset (cf. Fig 3). We find a weaker, but still noticeable, benefit of ICL over finetuning in this setting, and again stronger benefits of augmented finetuning. The difference in benefits compared to the experiments above may be due to differences in the plausibility of the relationships in question, e.g. the possibility that nonsense words interfere to some degree with the model's reasoning over longer contexts (see Appx. B.2).

## Simple syllogisms

We next test the simple syllogisms dataset (cf. Fig 3). Again, the pretrained model performs at chance, indicating a lack of contamination. Finetuning on the dataset does yield some non-trivial above-chance generalization; perhaps because for certain flavors of syllogism the logical inferences are compatible with simpler linguistic patterns -e.g., tuning on sequences like 'All X include Y.' ' All Y include Z.' might make the model more likely to predict ' All X include Z' from pure associations. However, following simpler associative patterns is not valid for most syllogistic forms, e.g. if the universal quantifier is replaced by the existential in the examples above. Perhaps because of this, in-context learning yields much stronger performance; using in-context learning to augment the training dataset further improves performance overall.

## Semantic structure benchmark

We then test the broader semantic structure benchmark (Fig. 4) which integrates multiple kinds of generalization into a richer learning setting. In this setting we evaluate performance on: a) rephrased training statements (that preserve the direction of the relation) denoted as 'train' in the figure, b) reversals, c) syllogistic inferences, and d) propositions about held out categories. Across these settings, we find an overall benefit of ICL over finetuning, though the magnitude of this benefit varies depending on the split. We find some improvements in generalization to even rephrased information from training, and more dramatic improvements on reversals and syllogisms. However, the category-holdout split remains difficult, and improvements from ICL are minimal. Furthermore, we continue to find that augmenting the finetuning data with in-context inferences improves performance from finetuning, in many cases outperforming even ICL. (N.B., in this setting we do not use sentence splitting for the

Figure 3 | On our simple reversal (left) and syllogism (right) datasets, in-context learning outperforms finetuning. Moreover, augmenting the fine-tuning dataset produces strong improvements in model performance. Pretrained models perform near chance, showing that the datasets are not guessable based on superficial features.

<!-- image -->

finetuning baseline, as we find it impairs performance and we want to compare to the strongest baselines; results with sentence-splitting for the baseline, along with other ablations, can be found in Appx. B.4.)

Our results in this section also highlight an important nuance to prior results on the reversal curse (Berglund et al., 2024). When the information being tested forms part of a broader coherent structure of knowledge (such as our semantic structure), finetuning alone does exhibit some above-chance generalization to reversals. This generalization may be due to the fact that other information in the training set can support the reversed conclusion; e.g. if the reversal is from 'birds include eagles' to 'eagles are a type of bird,' but the training set also includes statements 'eagles have wings,' that information may enable an alternative route to inferring the reversal (associatively if not logically). Nevertheless, in-context learning and augmented finetuning continue to perform substantially better than finetuning alone.

## Process knowledge

Wealso performed some preliminary experiments exploring learning and generalization of processtype (rather than semantic) knowledge. Specifically, we focused on executing a simple pseudo- mathematical procedure which is a function to function transform similar to taking a derivative, but with uncommon rules. We created a traintest compositional generalization split in which certain combinations of rules are seen in training, but other combinations are held out for testing. The overall pattern of results is generally consistent with those above-with ICL outperforming finetuning, and augmentations improving finetuning performance. However, process knowledge requires distinct evaluation methods, and the effects correspondingly appear to be driven by distinct factors compared to the semantic knowledge experiments above; thus, we present the experiments and results in full detail in Appendix B.5.

## Related work

In-context learning: A variety of works have explored in-context learning (Brown et al., 2020; Dong et al., 2022; Lampinen et al., 2024a). Several works have studied patterns in the learning and generalization of ICL, either empirically (Chan et al., 2022a; Garg et al., 2022), mechanistically (Olsson et al., 2022), or theoretically (Xie et al., 2023; Zhang et al., 2024). Several recent works have found that even scaling to hundreds or thousands of in-context examples can improve LM performance on challenging datasets

Figure 4 | On the more richly structured semantic dataset, in-context learning still moderately outperforms finetuning. Furthermore, augmentation continues to show some benefit - even in rephrased questions about trained facts that do not involve reversal. However some generalization splits, such as the category-level holdout, remain very challenging. (Error bars are standard errors computed over task subsets featuring different types of the inferences in question, e.g. reversals of property relations vs. reversals of category inclusion relations.)

<!-- image -->

(Agarwal et al., 2024; Anil et al., 2024; Bertsch et al., 2024). Our work similarly relies on the ability of models to learn from many documents in context in order to generate the inferences for augmentation.

Out-of-context learning: Several other papers (e.g. Berglund et al., 2023; Meinke and Evans, 2023) have considered how models can generalize 'out-of-context'-that is, how they can use information that is not directly included in the prompt in a flexible way at test time. Our results may connect to these, e.g. in the patterns of partial generalization seen even with finetuning on the semantic structure dataset. However, we generally do not find reliable use of out-of-context information as in-context-i.e., in-context learning tends to outperform finetuning.

Data augmentation: A wide variety of works have explored how LLMs can be used to augment data for improved performance from small or narrow datasets, e.g. to improve cross-lingual generalization (e.g. Whitehouse et al., 2023). Ding et al. (2024) reviews some of this broader literature. There have also been targeted attempts to fix specific issues like the reversal curse with hardcoded augmentations (Golovneva et al., 2024). A closely related work by Akyürek et al. (2024) proposes 'deductive closure training' to improve coverage via prompting language models to generate deductive inferences from training documents. Padmanabhan et al. (2024) similarly propose generating continuations and then distilling them into the model. Several concurrent works Chen et al. (2024); Ruan et al. (2025) suggest that having LMs generate additional reasoning

directions, and using these to augment their training data, can improve performance on reasoning tasks. Our experiments show that in controlled settings, without the possibility of dataset contamination, similar approaches to augmenting small finetuning datasets can yield improved generalization-and relate the performance to that achieved through more basic finetuning and in-context learning.

Synthetic data: An equally broad literature has explored applications of synthetic data to improving LM performance; see Liu et al. (2024) for a recent survey. Earlier works considered hand-designed synthetic data targeted to improve generalization using domain knowledge of areas like linguistics or mathematics (e.g. Pratapa et al., 2018; Wu et al., 2021). More recent approaches have focused on generating the data directly from language models, either filtering with ground-truth scores (Zelikman et al., 2022) or simply through self-consistency (Huang et al., 2023; Wang et al., 2023). While a recent prominent article argued that training on synthetic data leads models to collapse (Shumailov et al., 2024), other works have noted that in settings where synthetic data is mixed with the original data performance continues to improve (Gerstgrasser et al., 2024). We correspondingly find that in our setting, rather than being harmful, synthesizing augmented data with the models consistently improves performance (even on rephrased information from the training split in the semantic structure dataset). These results contribute to the ongoing discussion of how incorporating synthetic data affects model performance.

## Discussion

In this paper, we have performed controlled experiments on how language models generalize various types of novel information from in-context learning and finetuning. Overall, we find that the models generalize better on average along several dimensions from in-context learning. Using in-context learning to augment the finetuning dataset can exploit the complementary benefits of both to yield better performance.

The distinct inductive biases of in-context learning and finetuning: Avariety of works have studied the inductive biases of in-context learning. One common theme has been to emphasize that in-context learning can approximate gradient descent (e.g. Von Oswald et al., 2023), in settings where that behavior is optimal. However, a variety of other works have found that the inductive biases of in-context learning can vary depending on factors such as dataset diversity (Raventós et al., 2024) or model scale (Wei et al., 2023). Several works have explicitly noted the distinct inductive biases of in-context learning and finetuning (Chan et al., 2022b; Shen et al., 2023). Our work contributes to this line of findings.

Accessible information and learning by thinking: Lombrozo (2024) highlights how 'learning by thinking' is a unifying theme across cognitive science and recent advances in AI-a system can improve performance through computation without further inputs. Lombrozo highlights that while superficially this may seem paradoxical-information cannot be createdthis further computation can increase the accessibility of information and thus improve performance in practice. This argument parallels the theoretical one made by Xu et al. (2020) on how computation can increase the accessibility of information. Our use of in-context inferences to improve finetuning performance beyond the original data follows this pattern. For example, the information about reversals and syllogisms is always hidden within the data, but finetuning on the in-context inferences make this information more explicit and thus more readily accessible at test time.

Train-time inference scaling: Recently, various works have begun to explore test-time inference scaling to improve performance (e.g. Guo et al., 2025; Jaech et al., 2024). These findings complement prior studies exploring how scaling training compute (e.g. via larger models or more data) improves performance (e.g. Hoffmann et al., 2022; Kaplan et al., 2020). Our results illustrate that scaling train-time compute through incontext inference methods can help to improve some aspects of model generalization.

Limitations: Our work suffers from several limitations. First, our main experiments rely on

nonsense words and implausible operations. Although these counterfactual tasks allow us to avoid the possibility of dataset contamination, they may interfere with the performance of the models to some extent. For example, preliminary experiments (Appx. B.2) suggest that the ICL performance of the models on the Reversal Curse dataset degrades if the names are replaced with nonsense. Thus, tasks with more plausible entities may see greater benefits from ICL. It's possible that finetuning makes the nonsense terms more familiar, and that this contributes to the gap between augmented finetuning and ICL. However, in that case we likely underestimate the gap between ICL and base finetuning (as the latter would effectively benefit from increased 'familiarity' of the nonsense entities). Second, we have not experimented with other language models, which would enhance the generality of our results. However, since the individual phenomena we build upon - e.g. the reversal curse when finetuning (Berglund et al., 2024) vs. the ability to do reversals in context (e.g. Lampinen et al., 2024b) - have been documented across multiple models, we believe it is reasonable to extrapolate our results to other settings. However, future work should study more carefully the differences in how models learn and generalize in these settings, including newer reasoning models (e.g. Guo et al., 2025).

Conclusions: We have explored the generalization of in-context learning and finetuning when LMs are exposed to various types of novel information structures. We find notable differencesoften with ICL generalizing better to certain types of inferences-and propose ways of achieving better performance by adding in-context inferences to finetuning data. We hope this work will contribute to the science of understanding learning and generalization in foundation models, and the practicalities of adapting them to downstream tasks.
<|endofpaper|>