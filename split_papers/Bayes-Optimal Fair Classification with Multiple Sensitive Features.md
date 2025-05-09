<|startofpaper|>
## Bayes-Optimal Fair Classification with Multiple Sensitive Features

Yi Yang ∗

Yinghui Huang †

Xiangyu Chang ‡

January, 2025

## Abstract

Existing theoretical work on Bayes-optimal fair classifiers usually considers a single (binary) sensitive feature. In practice, individuals are often defined by multiple sensitive features. In this paper, we characterize the Bayes-optimal fair classifier for multiple sensitive features under general approximate fairness measures, including mean difference and mean ratio . We show that these approximate measures for existing group fairness notions, including Demographic Parity, Equal Opportunity, Predictive Equality, and Accuracy Parity, are linear transformations of selection rates for specific groups defined by both labels and sensitive features. We then characterize that Bayes-optimal fair classifiers for multiple sensitive features become instance-dependent thresholding rules that rely on a weighted sum of these group membership probabilities. Our framework applies to both attribute-aware and attribute-blind settings and can accommodate composite fairness notions like Equalized Odds. Building on this, we propose two practical algorithms for Bayes-optimal fair classification via in-processing and post-processing. We show empirically that our methods compare favorably to existing methods.

## 1 Introduction

Machine learning (ML) models have become integral to decision-making processes in various highstakes fields, such as credit scoring and criminal justice. However, a growing concern has emerged regarding the fairness of these models, particularly with respect to outputs that may disadvantage certain social groups defined by sensitive features such as race, gender, or socio-economic status

∗ Department of Information Systems, Arizona State University; E-mail: Yi.Yang.10@asu.edu

† Department of Information Systems and Intelligent Business, School of Management, Xi'an Jiaotong University; E-mail: yinghui.huang@xjtu.edu.cn

‡ Department of Information Systems and Intelligent Business, School of Management, Xi'an Jiaotong University; E-mail: xiangyuchang@xjtu.edu.cn .

(Barocas et al., 2023). Therefore, addressing fairness issues in ML has garnered significant attention (Caton and Haas, 2024).

A considerable body of work has focused on fairness in classification settings, where specific groups may experience discrimination due to biased predictions. This has led to the formalization of several algorithmic fairness notions, such as Demographi Parity (Dwork et al., 2012; Corbett-Davies et al., 2017), Equal Opportunity (Hardt et al., 2016), and Accuracy Parity (Zafar et al., 2017a). These notions aim to equalize various quantities across different groups. While perfect fairness -ensuring exactly identical quantities across groups-may entirely eliminate discrimination, it often incurs significant efficiency loss. Thus, approximate fairness is frequently adopted as a more practical alternative, where fairness level is quantified and limited using approximate measures such as Mean Difference (Chai and Wang, 2022) and Mean Ratio (Menon and Williamson, 2018) derived from fairness notions. See Section 3 for their definitions.

Researchers have developed various fair ML algorithms to operationalize these fairness notions. They are typically categorized based on when bias mitigation occurs during the training process: pre-processing, in-processing, or post-processing (Caton and Haas, 2024). Pre-processing methods aim to reduce bias in the training data through techniques such as data cleaning or reweighting (Kamiran and Calders, 2012; Calmon et al., 2017) before applying classical ML algorithms, but fairness in the training data does not always guarantee fairness in the resulting models. In-processing methods modify the model training objective by adding fairness regularizers or incorporating the fairness constraints (Zemel et al., 2013; Agarwal et al., 2018; Zafar et al., 2017b; Yang et al., 2020; Zhao et al., 2020). Post-processing methods (Menon and Williamson, 2018; Gouic et al., 2020; Xian et al., 2023; Xian and Zhao, 2024; Chen et al., 2024) remap the model's predictions to satisfy fairness requirements.

Despite these significant advancements, foundational theoretical aspects of fair ML remain under-explored. One critical question concerns the characterization of Bayes-optimal classifiers for fairness-aware learning problems. A Bayes-optimal fair classifier minimizes classification risk while satisfying specific fairness constraints, serving as a theoretical benchmark or the 'best possible' classifier for a given fairness-aware problem. Although Menon and Williamson (2018) and Chzhen et al. (2019) characterized Bayes-optimal fair classifiers, their analyses are confined to scenarios with a single sensitive feature of binary values. This leaves more complex settings involving multiple sensitive features 1 , which are commonly encountered in practice, largely unaddressed. While several studies (Corbett-Davies et al., 2017; Schreuder and Chzhen, 2021; Zeng et al., 2024) have investigated the theoretical underpinnings of fair classification with multiple sensitive features, their works derive Bayes-optimal classifiers under the strict requirement of perfect fairness without addressing the

1 Alternatively, a multi-class sensitive feature. The connection between these two cases is discussed in Section 2.2.

Table 1: Our contributions in comparison with prior theoretical works for Bayes-optimal fair classifier with multiple sensitive features.

| References                       | Corbett- Davies et al. (2017)    | Schreuder and Chzhen (2021)      | Chen et al. (2024)               | Xian Zhao (2024)                 | and                              | Zeng et al. (2024)               | Our Work                         |
|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| Scope of Theoretical Framework   | Scope of Theoretical Framework   | Scope of Theoretical Framework   | Scope of Theoretical Framework   | Scope of Theoretical Framework   | Scope of Theoretical Framework   | Scope of Theoretical Framework   | Scope of Theoretical Framework   |
| Approximate Fairness (MD)        |                                  |                                  | ✓                                | ✓                                |                                  |                                  | ✓                                |
| Approximate Fairness (MR)        |                                  |                                  |                                  |                                  |                                  |                                  | ✓                                |
| Attribute-Blind Setting          |                                  |                                  | ✓                                | ✓                                |                                  | ✓                                | ✓                                |
| Fairness Metrics Considered      | Fairness Metrics Considered      | Fairness Metrics Considered      | Fairness Metrics Considered      | Fairness Metrics Considered      | Fairness Metrics Considered      | Fairness Metrics Considered      | Fairness Metrics Considered      |
| Demographic Parity               | ✓                                | ✓                                | ✓                                | ✓                                |                                  | ✓                                | ✓                                |
| Equal Opportunity                | ✓                                |                                  | ✓                                | ✓                                |                                  | ✓                                | ✓                                |
| Predictive Equality              | ✓                                |                                  |                                  |                                  |                                  | ✓                                | ✓                                |
| Accuracy Parity                  |                                  |                                  |                                  |                                  |                                  |                                  | ✓                                |
| Equalized Odds                   |                                  |                                  | ✓                                | ✓                                |                                  | ✓                                | ✓                                |
| Theoretically Optimal Algorithms | Theoretically Optimal Algorithms | Theoretically Optimal Algorithms | Theoretically Optimal Algorithms | Theoretically Optimal Algorithms | Theoretically Optimal Algorithms | Theoretically Optimal Algorithms | Theoretically Optimal Algorithms |
| In-processing                    |                                  |                                  |                                  |                                  |                                  | ✓                                | ✓                                |
| Post-processing                  | ✓                                | ✓                                | ✓                                | ✓                                |                                  | ✓                                | ✓                                |

practical challenges of approximate fairness. More recently, Chen et al. (2024) and Xian and Zhao (2024) extended the exploration of Bayes-optimal fair classifiers to approximate fairness settings with multiple sensitive features, proposing only post-processing algorithms for fair classification. However, their works are restricted to the mean difference measure and also fail to accommodate fairness notions like accuracy parity. For a summary and comparison with related work, see Table 1.

Therefore, it lacks a systematic approach for deriving the Bayes-optimal fair classifiers, especially with multiple sensitive features under general approximate fairness measures. To this end, we explore their form while also explicitly accommodating fairness notions such as accuracy parity.

We summarize our contributions as follows:

- · We demonstrate that general approximate fairness measures for existing group fairness notions are linear transformations of the selection rate achieved by a classifier over specific group memberships.
- · We characterize the form of Bayes-optimal fair classifiers for multiple sensitive features, generalizing the framework of Menon and Williamson (2018). Their work can be viewed as a

special case of our approach when restricted to a single (binary) sensitive feature.

- · Our characterization can accommodate the fairness notions such as accuracy parity, which has not been established before to the best of our knowledge.
- · Building on the theoretical results, we propose both in-processing and post-processing algorithms for recovering Bayes-optimal fair classifiers.

## 2 Background and Notation

## 2.1 Binary Classification

A binary classification problem is defined by a joint distribution D over input features X ∈ X and labels Y ∈ Y = 0 1 . The goal is to derive a measurable { , } randomized classifier parametrized by f : X → [0 , 1], which outputs a prediction ˆ Y f ∈ { 0 1 , } with a certain probability based on features X . Let Bern( p ) denote the Bernoulli distribution with success probability p ∈ [0 , 1], and let F denote the set of all such measurable functions f . Then, the randomized classifier is defined as in Definition 1.

Definition 1 (Randomized Classifier) A randomized classifier f ∈ F specifies, for any x ∈ X , the probability f ( x ) of predicting ˆ Y f = 1 given X = x , i.e., ˆ Y f | X = x ∼ Bern( f ( x )) .

Definition 1 illustrates that the randomized classifier predicts any instance x ∈ X to be positive with probability f ( x ). Typically, the quality of a classifier is evaluated using a statistical risk function R ( ; · D ) : F → R + . A canonical risk is the cost-sensitive risk (Menon and Williamson, 2018).

Definition 2 (Cost-Sensitive Risk) For a cost parameter c ∈ [0 , 1] and a classifier f , the costsensitive risk (of f ) is given by:

$$R _ { c s } ( f ; c ) & = ( 1 - c ) \cdot P ( \hat { Y } _ { f } = 0, Y = 1 ) \\ & + c \cdot P ( \hat { Y } _ { f } = 1, Y = 0 ).$$

The cost-sensitive risk allows for asymmetric penalization of false negatives and false positives, depending on the value of c . When c = 1 2 , it reduces to the conventional error rate.

Bayes-Optimal Classifiers: For a given problem, the Bayes-optimal classifier is theoretically the best method, achieving the lowest possible average risk. For the cost-sensitive risk with parameter c , a Bayes-optimal classifier is defined as any minimizer f ∗ ∈ argmin f ∈F R cs ( f ; c ) . Let η x ( ) := P Y ( = 1 | X = x ) be the posterior probability of the positive class given x , and ✶ [ · ] denote the indicator function (equal to 1 if the argument is true and 0 otherwise). Then, Elkan (2001) characterizes Bayes-optimal classifiers in terms of η x ( ) and c as follows:

Theorem 1 (Bayes-Optimal Classifiers (Elkan, 2001)) For a cost parameter c ∈ [0 , 1] , the Bayes-optimal classifier f ∗ ∈ F has the form

$$f ^ { * } ( x ) = 1 \left [ H ( x ) > 0 \right ] + \alpha \cdot 1 \left [ H ( x ) = 0 \right ],$$

for all x ∈ X , where H x ( ) = η x ( ) -c , and α ∈ [0 , 1] is an arbitrary parameter.

- (2) shows that the Bayes-optimal classifier operates as a thresholding rule on the posterior classprobability of an instance. It predicts instances as positive or negative based on the threshold defined by the cost parameter c .

## 2.2 Fairness-Aware Learning in Binary Classification

Fairness-aware learning extends the conventional binary classification problem by incorporating sensitive features in addition to the target feature Y . Specifically, we assume the presence of sensitive features A ∈ A (e.g., gender and race) with respect to which we aim to ensure fairness. We note that X may or may not include the sensitive features A in practical applications.

Group Notation with Multiple Sensitive Features : In real applications, individuals might be coded with multiple sensitive features. We consider K sensitive features, where each feature is denoted by A k ∈ A k for k ∈ [ K ]. 3 For example, A 1 might correspond to race, A 2 to gender, and so on. However, the presence of multiple sensitive features (e.g., race and gender simultaneously) can lead to non-equivalent definitions of group fairness (Yang et al., 2020):

- · Independent group fairness : Fairness is evaluated separately for each sensitive feature, leading to overlapping subgroups (i.e., each sensitive feature defines its own set of groups independently). 4
- · Intersectional group fairness : Fairness is enforced on all subgroups defined by intersections of sensitive features, resulting in non-overlapping groups associated with all possible combinations of sensitive features.

It is noteworthy that enforcing intersectional fairness inherently controls independent fairness, but the reverse does not always hold (Kearns et al., 2018). Thus, intersectional fairness is often considered ideal (Yang et al., 2020). Consequently, we focus on intersectional fairness here when addressing multiple sensitive features and extend our results to independent fairness in Section A.2 of the Appendix.

To implement intersectional fairness for multiple sensitive features, a new composite sensitive feature S is constructed to represent all possible intersectional combinations of the existing sensitive

2 Attribute-Blind Setting refers to the case where sensitive features cannot be used for prediction.

3 Here, A = A ×A ×···×A 1 2 K .

4 See Appendix A.1 for detailed explanations and examples.

features. Specifically, S ∈ S = { 1 , . . . , M } , where M = ∏ K k =1 |A | k , and |A | k denotes the number of possible values for the k -th sensitive feature. Thus, S defines M non-overlapping subgroups, each corresponding to a unique combination of sensitive feature values. Note that this approach is equivalent to treating S as a single sensitive feature with multiple categorical values, enabling our results to be directly applicable to that scenario.

For all m ∈ S , x ∈ X , and y ∈ Y , let P S,Y ( m,y ) := P S ( = m,Y = y ) denote the joint distribution of S and Y . Define p + := P Y ( = 1) and p -:= P Y ( = 0) as the marginal probabilities of the positive and negative classes. Let P S ( ) · represent the marginal distribution of S , while P S Y | = y ( ) and · P Y S | = m ( ) denote the conditional distributions of · S given Y = y and Y given S = m , respectively.

To address unfairness, various parity-based group fairness notions grounded in sensitive features have been proposed. Below are the key definitions considered in this paper.

Definition 3 (Demographic Parity (DP) (Dwork et al., 2012)) A classifier f satisfies demographic parity if its prediction ˆ Y f is independent of the sensitive feature S : P Y ( ˆ f = 1) = P Y ( ˆ f = 1 | S = m ) for all m ∈ [ M ] .

Definition 4 (Equal Opportunity (EO) (Hardt et al., 2016)) A classifier f satisfies equal opportunity if it achieves the same true positive rate across all groups: P Y ( ˆ f = 1 | Y = 1) = P Y ( ˆ f = 1 | S = m,Y = 1) for all m ∈ [ M ] .

Definition 5 (Predictive Equality (PE) (Corbett-Davies et al., 2017)) A classifier f satisfies predictive equality if it achieves the same false positive rate across all groups: P Y ( ˆ f = 1 | Y = 0) = P Y ( ˆ f = 1 | S = m,Y = 0) for all m ∈ [ M ] .

̸

Definition 6 (Accuracy Parity (AP) (Zafar et al., 2019)) A classifier f satisfies accuracy parity if it achieves the same error rate across all groups: P Y ( ˆ f = Y ) = P Y ( ˆ f = Y | S = m ) for all m ∈ [ M ] .

̸

In practice, achieving the above equalities (i.e., perfect fairness) often leads to a significant sacrifice in efficiency (i.e., increasing expected risk). Thus, implementing 'approximate' fairness is often more practical and preferable. Previous research typically quantifies fairness by measuring disparities in quantities that would be equalized under perfect fairness, focusing on optimizing risk while imposing constraints to limit these disparities (Zafar et al., 2017a).

## 3 General Approximate Fairness Measures

We focus on two general approximate fairness measures, mean difference and mean ratio , to quantify classifier disparity level. We begin by presenting the definitions of these measures and then

demonstrate they are linear transformations of a classifier's selection rates for specific groups.

## 3.1 Mean Difference

For a composite sensitive feature S ∈ { 1 , . . . , M } , the mean difference (MD) score (Calders and Verwer, 2010; Chai and Wang, 2022) quantifies the fairness of a classifier f by calculating the difference in a specified outcome between the overall population and the subgroup defined by S = m .

Definition 7 (Mean Difference) For all m = 1 , . . . , M , the mean difference measure for group m is defined as:

$$\text{MD} _ { m } ( f ) = P ( \mathcal { G } ( \hat { Y } _ { f } ) \ | \ Z = z ) - P ( \mathcal { G } ( \hat { Y } _ { f } ) \ | \ Z = z, S = m ),$$

where ˆ Y f is the prediction of f , and the components G · ( ) , Z , and z depend on the fairness notion being considered.

The flexibility in the choice of G · ( ), Z , and z allows Definition 7 to accommodate several commonly used group fairness notions, as shown in Table 2. Achieving perfect fairness indicates MD ( ) = 0 for all m f m . Usually, a limited level of disparity may be acceptable. To formalize this, we use the symmetrized version of the MD measure:

$$\text{MD} ( f ) = \max _ { m \in [ M ] } \max \left ( \text{MD} _ { m } ( f ), \text{MD} _ { m } ( 1 - f ) \right ) \leq \delta,$$

where δ is a pre-specified tolerance level for unfairness.

To simplify notation, we define E y,m = { Y = y, S = m } as the event where an individual has label Y = y and belongs to group m , with its probability denoted by P E ( y,m ) = P Y ( = y, S = m ). Then, Lemma 1 demonstrates that MD measures for these common group fairness notions are linear transformations of P Y ( ˆ f = 1 | E y,m ). All proofs are deferred to Section B in the Appendix.

Lemma 1 For any randomized classifier f , any δ ∈ [0 , 1] , and the group fairness notions in Table 2, MD( ) f ≤ δ ⇔ R MD m ( f ) ∈ -[ δ, δ ] for all m ∈ [ M ] , where

$$R _ { m } ^ { \text{MD} } ( f ) \coloneqq \sum _ { y \in \{ 0, 1 \} } \left \{ \left [ \sum _ { m ^ { \prime } = 1 } ^ { M } a _ { m ^ { \prime } } b _ { m ^ { \prime } } ^ { y } P ( \hat { Y } _ { f } = 1 \, | \, E _ { y, m ^ { \prime } } ) \right ] - b _ { m } ^ { y } P ( \hat { Y } _ { f } = 1 \, | \, E _ { y, m } ) + c _ { m } ^ { y } \right \}.$$

Here, the values of a m , b y m , and c y m depend on the fairness notion under consideration and are as defined in Table 2.

Table 2: Recovering existing group fairness criteria based on the choice of G · ( ), Z , and z for MD and MR measures. For the linear transformation parameter values, the MD and MR measures differ only in the term c y m for Accuracy Parity.

̸

| Notion   | G ( ˆ Y f )   | Z     | a m              | b y m                          | c y m (MD)                      | c y m (MR)                       |
|----------|---------------|-------|------------------|--------------------------------|---------------------------------|----------------------------------|
| DP       | { ˆ Y f = 1 } | U 5 U | P S ( m )        | P Y | S = m ( y )              | 0                               | 0                                |
| EO       | { ˆ Y f = 1 } | Y 1   | P S | Y =1 ( m ) | y                              | 0                               | 0                                |
| PE       | { ˆ Y f = 1 } | Y 0   | P S | Y =0 ( m ) | 1 - y                          | 0                               | 0                                |
| AP       | { ˆ Y f = Y } | U U   | P S ( m )        | (1 - 2 y ) · P Y | S = m ( y ) | (1 - y ) p + - yP Y | S = m (1) | ( y - 1) δp + + yP Y | S = m (1) |

## 3.2 Mean Ratio

Approximate fairness can also be assessed using the disparate impact (DI) factor (Feldman et al., 2015; Menon and Williamson, 2018), which is defined as the ratio of relevant probabilities. We refer to this as the mean ratio (MR) measure, as presented below.

Definition 8 (Mean Ratio) For all m = 1 , . . . , M , the mean ratio measure for group m is defined as:

$$\text{MR} _ { m } ( f ) = \frac { P ( \mathcal { G } ( \hat { Y } _ { f } ) \ | \ Z = z, S = m ) } { P ( \mathcal { G } ( \hat { Y } _ { f } ) \ | \ Z = z ) },$$

where ˆ Y f , G · ( ) , Z , and z are as defined in Definition 7.

Similarly, we consider the symmetrized version of the MR measure (Menon and Williamson, 2018):

$$\text{MR} ( f ) = \min _ { m \in [ M ] } \min \left ( \text{MR} _ { m } ( f ), \text{MR} _ { m } ( 1 - f ) \right ) \geq \delta,$$

where δ is a pre-specified tolerance level for unfairness. Then, Lemma 2 demonstrates that MR measures for the common group fairness notions also relate the linear transformations of P Y ( ˆ f = 1 | E y,m ).

Lemma 2 For any randomized classifier f , any δ ∈ [0 , 1] , and the group fairness notions in Table 2, MR( ) f ≥ δ ⇔ R MR m ( f ) ∈ [ δ -1 0] , for all m ∈ [ M ] , where

$$R _ { m } ^ { \text{MR} } ( f ) \coloneqq \sum _ { y \in \{ 0, 1 \} } \left \{ \left [ \delta \sum _ { m ^ { \prime } = 1 } ^ { M } a _ { m ^ { \prime } } b _ { m ^ { \prime } } ^ { y } P ( \hat { Y } _ { f } = 1 \, | \, E _ { y, m ^ { \prime } } ) \right ] - b _ { m } ^ { y } P ( \hat { Y } _ { f } = 1 \, | \, E _ { y, m } ) + c _ { m } ^ { y } \right \}.$$

Here, the values of a m , b y m , and c y m depend on the fairness notion under consideration and are as defined in Table 2.

## 4 Bayes-Optimal Fair Classifiers

Given approximate group fairness constraints as specified in (3) and (4), our goal is to compute a (randomized) fair classifier f ∗ B that minimizes the following objective:

- · For MD: min f ∈F { R cs ( f ; c ) : MD( f ) ≤ } δ .
- · For MR: min f ∈F { R cs ( f ; c ) : MR( f ) ≥ } δ .

As in the conventional binary classification, Bayes-optimal fair classifiers provide the theoretically best solution under the given fairness constraints. Note that the above constrained optimization problems can be further reduced to the following unconstrained problems according to the Lagrangian principle and Lemmas 1 and 2.

Lemma 3 Let D be any distribution, and consider an approximate fairness measure, either MD or MR. For any c ∈ [0 , 1] and δ ∈ [0 , 1] , there exists λ ∈ ❘ M such that:

- · For MD: min f ∈F { R cs ( f ; c ) : MD( f ) ≤ } δ = min f ∈F ( R cs ( f ; c ) -∑ M m =1 λ m · R MD m ( f ) ) .
- · For MR: min f ∈F { R cs ( f ; c ) : MR( f ) ≥ } δ = min f ∈F ( R cs ( f ; c ) -∑ M m =1 λ m · R MR m ( f ) ) .

Here, λ m is the m -th component of λ .

Lemma 3 shows that Bayes-optimal fair randomized classifiers can be derived by solving an unconstrained optimization problem with a fairness regularizer incorporated into the objective. The trade-off parameter vector λ controls the balance between cost-sensitive risk (efficiency) and fairness. In fact, each of its component λ m ∈ ❘ corresponds to the difference in Lagrange multipliers for the two bounds associated with group m , and it can take negative values.

With these foundations in place, we now present the form of Bayes-optimal fair classifiers for the MD and MR measures.

## 4.1 Mean Difference

We begin with the explicit form of the Bayes-optimal fair classifier for MD measure. Recall that η x ( ) := P Y ( = 1 | X = x ).

Theorem 2 (Bayes-Optimal Fair Classifier for MD Measure) For any c ∈ [0 , 1] and δ ∈ [0 , 1] , ∃ λ ∈ ❘ M such that the Bayes-optimal fair classifier f ∗ B ( x ) ∈ argmin f ∈F { R f ( ) : MD( f ) ≤ } δ has the form of

$$f _ { B } ^ { * } ( x ) = \mathbb { 1 } \left [ H _ { B } ^ { * } ( x ) > 0 \right ] + \alpha \cdot \mathbb { 1 } \left [ H _ { B } ^ { * } ( x ) = 0 \right ],$$

5 U refers to the complete set.

where

$$H _ { B } ^ { * } ( x ) = \eta ( x ) - c - \sum _ { m = 1 } ^ { M } \sum _ { y \in \{ 0, 1 \} } b _ { m } ^ { y } \left ( \lambda _ { m } - \Lambda _ { M } a _ { m } \right ) \gamma _ { m } ^ { y } ( x ).$$

Here, λ m is the m -th component of λ , Λ M = ∑ M i =1 λ m , γ y m ( x ) = P E ( y,m | X = ) x P E ( y,m ) , and α ∈ [0 , 1] is an arbitrary parameter. The values of a m and b y m depend on the fairness notion under consideration and are as shown in Table 2.

̸

In (5), setting λ = 0 results in the Bayes-optimal classifiers for the cost-sensitive risk, as described in (2). For λ = 0 , the optimal classifier f ∗ B ( x ) adjusts the λ = 0 solution by applying an instancedependent threshold correction. This correction is determined by the weighted sum of γ y m ( x )-the (normalized) probability that the individual x belongs to the group { Y = y, S = m } .

In the discussion above, we made no explicit assumption regarding whether the sensitive features are utilized during the prediction phase. Thus, the findings are applicable to the attribute-blind setting. If the sensitive features are available and allowed to be used for prediction 6 , the form of the Bayes-optimal fair classifier simplifies as follows:

Corollary 1 (Bayes-Optimal Fair Classifier for MD with S ) For any c ∈ [0 , 1] and δ ∈ [0 , 1] , ∃ λ ∈ ❘ M such that the Bayes-optimal fair classifier f ∗ B ( x, s ) ∈ argmin f ∈F { R f ( ) : MD( f ) ≤ } δ has the form of

$$f _ { B } ^ { * } ( x, s ) = 1 \left [ H _ { B } ^ { * } ( x, s ) > 0 \right ] + \alpha \cdot 1 \left [ H _ { B } ^ { * } ( x, s ) = 0 \right ],$$

where

$$H _ { B } ^ { * } ( x, s ) = \eta ( x, s ) - c - \sum _ { y \in \{ 0, 1 \} } b _ { s } ^ { y } \left ( \lambda _ { s } - \Lambda _ { M } a _ { s } \right ) \gamma _ { s } ^ { y } ( x, s ).$$

Here, η x, s ( ) = P Y ( = 1 | X = x, S = s ) , λ s is the s -th component of λ , Λ M = ∑ M i =1 λ m , and γ y s ( x, s ) = P Y ( = y X | = x,S = ) s P E ( y,s ) . α ∈ [0 , 1] is an arbitrary parameter. The values of a s and b y s depend on the selected fairness notion.

This result follows directly from Theorem 2, since for the data pair ( x, s ), we have P S ( = m X | = x, S = s ) = ✶ [ m = s ] and P S ( = m,Y | X = x, S = s ) = P Y ( | X = x, S = s ) ✶ [ m = s . ] Note that in this case, (6) can further reduce to applying a group-wise constant threshold to the class probabilities η x, s ( ) for each value of the sensitive feature. This simplification arises because γ y s ( x, s ) is a linear function of η x, s ( ) across all four fairness notions.

6 This refers to the Attribute-Aware Setting, i.e., A (and thus S ) are included in X . In what follows, we slightly abuse notation by separating A S ( ) from X , with X representing only the non-sensitive features.

## 4.2 Mean Ratio

We now turn to the optimal fair classifier for the MR measure. The result is analogous to Theorem 2, but it explicitly incorporates δ in the threshold correction.

Theorem 3 (Bayes-Optimal Fair Classifier for MR Measure) For any c ∈ [0 , 1] and δ ∈ [0 , 1] , ∃ λ ∈ ❘ M such that the Bayes-optimal fair classifier f ∗ B ( x ) ∈ argmin f ∈F { R f ( ) : MR( f ) ≥ } δ has the form of

$$f _ { B } ^ { * } ( x ) = 1 \left [ H _ { B } ^ { * } ( x ) > 0 \right ] + \alpha \cdot 1 \left [ H _ { B } ^ { * } ( x ) = 0 \right ],$$

where

$$H _ { B } ^ { * } ( x ) = \eta ( x ) - c - \sum _ { m = 1 } ^ { M } \sum _ { y \in \{ 0, 1 \} } b _ { m } ^ { y } \left ( \lambda _ { m } - \delta \Lambda _ { M } a _ { m } \right ) \gamma _ { m } ^ { y } ( x ).$$

Here, λ m is the m -th component of λ , Λ M = ∑ M i =1 λ m , γ y m ( x ) = P E ( y,m | X = ) x P E ( y,m ) , and α ∈ [0 , 1] is an arbitrary parameter. The values of a m and b y m depend on the fairness notion under consideration and are as shown in Table 2.

When S is available for prediction, the Bayes-optimal fair classifier for MR, similar to the MD case, takes a simplified form as detailed in Corollary 2.

Corollary 2 (Bayes-Optimal Fair Classifier for MR with S ) For any c ∈ [0 , 1] and δ ∈ [0 , 1] , ∃ λ ∈ ❘ M such that the Bayes-optimal fair classifier f ∗ B ( x, s ) ∈ argmin f ∈F { R f ( ) : MR( f ) ≥ } δ has the form of

$$f _ { B } ^ { * } ( x, s ) = 1 \left [ H _ { B } ^ { * } ( x, s ) > 0 \right ] + \alpha \cdot 1 \left [ H _ { B } ^ { * } ( x, s ) = 0 \right ],$$

where

$$H _ { B } ^ { * } ( x, s ) = \eta ( x, s ) - c - \sum _ { y \in \{ 0, 1 \} } b _ { s } ^ { y } \left ( \lambda _ { s } - \delta \Lambda _ { M } a _ { s } \right ) \gamma _ { s } ^ { y } ( x, s ).$$

Here, η x, s ( ) = P Y ( = 1 | X = x, S = s ) , λ s is the s -th component of λ , Λ M = ∑ M i =1 λ m , and γ y s ( x, s ) = P Y ( = y X | = x,S = ) s P E ( y,s ) . α ∈ [0 , 1] is an arbitrary parameter. The values of a s and b y s depend on the selected fairness notion.

As in the MD case, (8) can further reduce to applying a group-wise constant threshold to the class probabilities η x, s ( ) for each value of the sensitive feature.

Remark 1 In addition to the fairness notions discussed in Table 2, our results can be directly extended to composite fairness notions like Equalized Odds, as well as other composite fairness requirements combined from the notions in Table 2. See Section A.3 in the Appendix for more details.

## 5 Algorithms

Section 4 establishes that Bayes-optimal fair classifiers for the MD and MR measures are derived by applying instance-dependent threshold corrections to the unconstrained Bayes-optimal classifier. This insight facilitates the development of practical classifiers that can be trained on finite data using in-processing and post-processing techniques.

## 5.1 In-Processing-Based Bayes-Optimal Fair Classification

We introduce an in-processing method for Bayes-optimal fair classification. Theorems 2 and 3 show that Bayes-optimal fair classifiers apply instance-dependent threshold adjustments. Since cost-sensitive classification inherently accounts for such threshold adjustments, it serves as the basis of our approach.

Building on this, we propose a fair cost-sensitive classification framework. Specifically, we define a fair cost-sensitive risk function and then demonstrate that minimizing this risk yields the Bayes-optimal fair classifier.

## Theorem 4 (Fair Cost-sensitive Classifier is Bayes-Optimal) Let

$$c _ { y } ^ { \lambda } ( x ) = ( 1 - 2 y ) \left [ c + Q ^ { \lambda } ( x ) \right ] + y,$$

where y ∈ { 0 1 , } , and:

$$Q _ { \text{MD} } ^ { \lambda } ( x ) = \sum _ { m = 1 } ^ { M } \sum _ { y \in \{ 0, 1 \} } b _ { m } ^ { y } \left ( \lambda _ { m } - \Lambda _ { M } a _ { m } \right ) \gamma _ { m } ^ { y } ( x ),$$

$$Q _ { \text{MR} } ^ { \lambda } ( x ) = \sum _ { m = 1 } ^ { M } \sum _ { y \in \{ 0, 1 \} } ^ { \cdot \cdot \cdot } b _ { m } ^ { y } \left ( \lambda _ { m } - \delta \Lambda _ { M } a _ { m } \right ) \gamma _ { m } ^ { y } ( x ).$$

y y are as defined in Theorems 2.

Here, λ , λ m , Λ M , γ m ( x ) , and the values of a m and b m

Define the fair cost-sensitive risk of a classifier f as:

$$\Lambda _ { F C S } ( f ) = \sum _ { y \in \{ 0, 1 \} } \left \{ \begin{array} { c } \Lambda _ { \mathcal { X } } \\ \int _ { \mathcal { X } } c _ { y } ^ { \lambda } ( x ) \cdot \\ P ( \hat { Y } _ { f } = 1 - y, Y = y \ | \ X = x ) \, d P _ { X } ( x ) \right \}. \end{array}$$

Then, f In B ( x ) = argmin f ∈F R λ FCS ( f ) is a Bayes-optimal fair classifier.

Theorem 4 shows that a cost-sensitive classification with instance-dependent costs can yield a Bayes-optimal fair classifier. Building on this result, Algorithm 1 provides a summary of the proposed in-processing procedure. 7

7 We assume that S has already been constructed from A , and so have its observed values s i .

## Algorithm 1 Bayes-Optimal Fair Classification via In-Processing

Input: Cost parameter c , fairness tolerance level δ ≥ 0, dataset D = { x , s , y i i i } N i =1 , and trade-off parameter λ .

- 1: Estimate the group membership probabilities ˜ ( P S,Y | X = x ) using { ( x , s , y i i i ) } N i =1 , and calculate ˆ γ y m ( x ) = ˜ ( P E y,m | X = ) x ˜ ( P E y,m ) accordingly for all m ∈ [ M ] and y ∈ { 0 1 , } .
- 2: Estimate ˆ Q λ by plug-in estimation using ˆ γ y m and the expressions in (9) for MD or (10) for MR.
- 3: Denote ˆ c λ y = (1 -2 y ) [ c + ˆ Q λ ] + y for all y .
- 4: Use any cost-sensitive classification method to train ˆ f ∗ B ( x ) on { x , y i i } N i =1 by minimizing the empirical analogue of fair cost-sensitive risk in (11).

Output: ˆ ( f ∗ B x ).

Remark 2 On the line 1 of Algorithm 1, the group membership probabilities are estimated. This can be done directly by learning a multi-class predictor ˆ f S,Y : X → ∆ S×Y . Or one can break the problem down into learning two simple predictors, ˆ f Y : S × X → ∆ Y and ˆ f S : X → ∆ S , then combining them into ˆ f S,Y ( m,y ) = ˆ ( f S x ) m · ˆ f Y ( m,x ) y .

When S is available for prediction, the construction of the Bayes-optimal fair cost-sensitive classifier can be further simplified. See Corollary 7 in the Appendix for details.

## 5.2 Post-Processing-Based Bayes-Optimal Fair Classification

Recall that Theorems 2 and 3 show that the Bayes-optimal fair classifiers f ∗ B ( x ) modify the unconstrained Bayes-optimal classifier by applying an instance-dependent threshold correction. This correction depends on the (normalized) group membership probabilities of the individual x , given by γ y m ( x ). Motivated by this, we propose a post-processing algorithm that adopts a plugin approach for Bayes-optimal fair classification. Specifically, we construct a fair plugin classifier by separately estimating η x ( ) and γ y m ( x ), and then combining them according to (5) and (7).

Algorithm 2 outlines the proposed post-processing plug-in approach. The group membership probabilities can be estimated as discussed previously in Remark 2. When S is available for prediction, we can estimate ˆ( η x, s ) = ˜ ( P Y = 1 | X = x, S = ) using any method applied to the s dataset { ( x , s , y i i i ) } N i =1 instead of { ( x , y i i ) } N i =1 as described in Line 1 of Algorithm 2.

Remark 3 In Algorithms 1 and 2, the value of λ plays a critical role in balancing fairness and risk. It can be selected through various approaches. For example, a rough estimate can be obtained through grid search by ensuring that the achieved fairness level satisfies the pre-specified threshold (Menon and Williamson, 2018; Chen et al., 2024). For a more accurate determination, optimization

## Algorithm 2 Bayes-Optimal Fair Classification via Post-Processing

Input: Cost parameter c , fairness tolerance level δ ≥ 0, dataset D = { x , s , y i i i } N i =1 , and trade-off parameter λ .

- 1: Estimate ˆ( η x ) = ˜ ( P Y = 1 | X = x ) using any approach on { ( x , y i i ) } N i =1 .
- 2: Estimate the group membership probabilities ˜ ( P S,Y | X = x ) using { ( x , s , y i i i ) } N i =1 , and calculate ˆ γ y m ( x ) = ˜ ( P E y,m | X = ) x ˜ ( P E y,m ) accordingly for all m ∈ [ M ] and y ∈ { 0 1 , } .
- 3: Construct ˆ f ∗ B ( x ) by plugging the estimates ˆ( η x ) and ˆ γ y m ( x ) into the expression for the Bayesoptimal fair classifier: Use (5) for MD or (7) for MR.

Output: ˆ ( f ∗ B x ).

techniques based on the relationship between λ and the Lagrange multipliers can be employed, like solving a dual optimization problem (Xian and Zhao, 2024) or using iterative updates guided by the fairness-accuracy trade-off (Zeng et al., 2024).

## 6 Experiments

## 6.1 Set-up

Datasets. We consider two real-life classification benchmark datasets: (1) Adult (Becker and Kohavi, 1996): A UCI dataset where the task is to predict whether the income of an individual is over $50k per year; (2) COMPAS (Angwin et al., 2016): A dataset where the task is to predict the recidivism of criminals. Both datasets contain demographic features such as gender (binary) and race (binary). We use them to construct scenarios with a single sensitive feature as well as multiple sensitive features. Detailed dataset statistics and data processing procedures are provided in the Appendix C.

Models &amp; Hyper-parameters. We first train predictors on the training data to estimate the probabilities ˜ ( P S,Y | X = x ) and ˜ ( P Y | X = x ), which are then incorporated into the proposed algorithms. For the COMPAS dataset with a binary sensitive feature, we use logistic regression as the predictor, while for other settings and datasets, we use XGBoost. Following Menon and Williamson (2018) and Chen et al. (2024), the values of the hyperparameter λ ∈ -[ 1 1] are selected , via grid search in our algorithms. The cost parameter is fixed at c = 0 5. .

Baselines. We compare the proposed algorithms with established post-processing and inprocessing fair classification algorithms, including LinearPost (Xian et al., 2023) and Reduction (Agarwal et al., 2018). We also include a basic baseline classifier without fairness constraints for comparison. Specifically, we use logistic regression for the COMPAS dataset and XGBoost for the

Figure 1: Trade-offs between accuracy and fairness using MD. The prefix 'Multi-' represents the case of multiple sensitive features, and '(aware)' in the name of classifiers indicates the attribute-aware setting.

<!-- image -->

Adult dataset, allowing us to evaluate the impact of fairness interventions relative to standard classification models. For all baselines, we consider both the attribute-blind and attribute-aware settings.

Evaluation Metrics. For various values of λ , we report both the accuracy and fairness levels. The model accuracy is computed for the target label. All four fairness notions listed in Table 2 are considered. Approximate fairness is implemented using both the MD and MR measures, and the achieved fairness levels (i.e., values of MD( f ) and MR( f )) are reported accordingly. See Appendix C for more details on the experimental setup.

## 6.2 Results

Figure 1 shows the fairness-accuracy trade-offs for MD measure under DP and EO across two datasets. It considers cases where 'gender' or 'race' is the only sensitive feature and where both of them are considered sensitive. Each point corresponds to a specific value of the tuning parameter λ .

For DP and EO, compared to the fair baselines, our Algorithms achieve the more favorable fairness-accuracy trade-off in most cases, especially in the attribute-blind setting. Notably, for all fair methods, their performance gap between attribute-blind and attribute-aware settings is larger on COMPAS . This is likely due to its smaller dataset size, which may hinder accurate estimation of p Y ( | X )

and p S, Y ( | X ). Additionally, our in-processing method often outperforms our post-processing one in balancing fairness and accuracy, with a more significant advantage on the COMPAS dataset. Moreover, we evaluate our methods under other fairness notions (i.e., PE and AP) and assess our algorithms with respect to the MR measure. The complete results are provided in Appendix C. All results suggest that our algorithm effectively navigates the trade-off between fairness and accuracy.

## 7 Conclusion

This work analyzes the fair classification problem with multiple sensitive features. We characterize that Bayes-optimal fair classifiers for approximate fairness-under both the mean difference and mean ratio measures-can be represented by instance-dependent threshold corrections applied to the unconstrained Bayes-optimal classifier. The corrections are determined by a weighted sum of the probabilities that an individual belongs to specific groups. Our findings are applicable to both attribute-aware and attribute-blind settings and cover widely used fairness notions, including DP, EO, PE, and accuracy parity (AP). Notably, to the best of our knowledge, this is the first work to characterize the Bayes-optimal fair classifier under AP. Building on these insights, we proposed both in-processing and post-processing algorithms for learning Bayes-optimal fair classifiers from finite data. Empirical results show that our methods perform favorably compared to existing methods.
<|endofpaper|>