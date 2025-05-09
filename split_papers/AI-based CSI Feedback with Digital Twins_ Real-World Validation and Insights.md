<|startofpaper|>
## AI-based CSI Feedback with Digital Twins: Real-World Validation and Insights

Tzu-Hao Huang, Chao-Kai Wen, Fellow, IEEE , Shang-Ho Tsai, Senior Member, IEEE , and Trung Q. Duong, Fellow, IEEE

Abstract -Deep learning (DL) has shown great potential for enhancing channel state information (CSI) feedback in multipleinput multiple-output (MIMO) communication systems, a subject currently under study by the 3GPP standards body. Digital twins (DTs) have emerged as an effective means to generate sitespecific datasets for training DL-based CSI feedback models. However, most existing studies rely solely on simulations, leaving the effectiveness of DTs in reducing DL training costs yet to be validated through realistic experimental setups. This paper addresses this gap by establishing a real-world (RW) environment and corresponding virtual channels using ray tracing with replicated 3D models and accurate antenna properties. We evaluate whether models trained in DT environments can effectively operate in RW scenarios and quantify the benefits of online learning (OL) for performance enhancement. Results show that a dedicated DT remains essential even with OL to achieve satisfactory performance in RW scenarios.

matrices rather than full CSI. This method offers higher accuracy than traditional codebook schemes and is compatible with the 5G NR standard. More recent efforts further improve feedback efficiency by incorporating quantization-aware training [6] and entropy coding techniques [7].

## I. INTRODUCTION

E NHANCING spectrum efficiency through large-scale multiple-input multiple-output (MIMO) systems, which deploy numerous transmitting antennas at base stations (BSs), is a key strategy in 5G and emerging 6G communication systems [1]. Fully capitalizing on the benefits of large-scale MIMO requires accurate channel state information (CSI) at the BS, as CSI quality directly influences system performance [2]. In frequency division duplex (FDD) systems, downlink CSI is estimated at the user equipment (UE) and fed back to the BS. Current 5G New Radio (NR) systems, such as those employing Type II feedback [3], rely on codebookbased methods. However, as the number of antennas increases, codebook design becomes inefficient, leading to degraded CSI quality or excessive feedback overhead.

Recent deep learning (DL) advances have demonstrated strong potential for CSI feedback [4]. DL-based autoencoders have been applied to compress and reconstruct complete downlink CSI, while subsequent work [5] has introduced implicit feedback mechanisms that focus on feeding back precoding

- T.-H. Huang is with the Institute of Electrical Control Engineering, National Yang Ming Chiao Tung University, Hsinchu 300, Taiwan, Email: peter94135@gmail.com.
- C.-K. Wen is with the Institute of Communications Engineering, National Sun Yat-sen University, Kaohsiung 804, Taiwan, Email: chaokai.wen@mail.nsysu.edu.tw.
- S.-H. Tsai is with the Department of Electrical Engineering, National Yang Ming Chiao Tung University, Hsinchu 300, Taiwan, Email: shanghot@mail.nctu.edu.tw.
- T. Q. Duong is with the Faculty of Engineering and Applied Science, Memorial University, St. John's, NL A1C 5S7, Canada, and with the School of Electronics, Electrical Engineering and Computer Science, Queen's University Belfast, Belfast, U.K, Email: tduong@mun.ca.

However, training DL models requires large datasets that are challenging to obtain in real-world (RW) systems without performance degradation. Digital twins (DTs) have emerged as an efficient solution for generating site-specific datasets for DL model training [8]-[11]. Additionally, recent developments [12] have demonstrated that low-cost DT construction is feasible using publicly available data and open-source tools, further supporting the practicality of DT-based training. Nevertheless, discrepancies between DT simulations and RW environments often lead to performance degradation when models trained on DT data are deployed in practice. Integrating online learning (OL) techniques that leverage minimal RW data is essential to improve model performance [10]. To date, most research has relied on simulations, and a realistic experimental setup is needed to validate the effectiveness of DTs in reducing DL training costs.

This paper addresses this gap by establishing an RW environment and corresponding virtual channels using ray tracing with replicated 3D models and accurate antenna properties. We construct a comprehensive dataset that includes RW measurements and virtual channel data derived from these measurements. This dataset is used to train a DL-based autoencoder for implicit CSI feedback. To bridge the gap between virtual and real channels, OL is employed to finetune the DL models with a minimal amount of RW data. Through detailed experiments, we compare the communication performance of different feedback schemes and address the following questions: Can a model trained in a DT environment perform effectively in RW scenarios? How does OL improve performance?

## II. SYSTEM MODEL AND DESIGN WITH DT

## A. Communication System Model

We consider a single-user MIMO system with N t transmit antennas at the BS and N r receive antennas at the UE. The system employs an orthogonal frequency-division multiplexing (OFDM) waveform with N c subcarriers. The CSI is represented as H ∈ C N c × N r × N t . In the 5G New Radio (NR) system, the UE acquires the downlink CSI, computes N s eigenvectors (or spatial precoders), and feeds them back to the BS via uplink transmission. The s -th eigenvector, w s ∈ C N t ,

̂

Fig. 1. Neural network architecture and an overview of the efficient AI-driven CSI feedback training approach integrating DTs and OL.

<!-- image -->

normalized such that ∥ w s ∥ 2 = 1 , is obtained by eigenvalue decomposition:

$$\left ( \frac { 1 } { N _ { c } } \sum _ { n = 1 } ^ { N _ { c } } \mathbf H _ { n } ^ { H } \mathbf H _ { n } \right ) \mathbf w _ { s } = \lambda _ { s } \mathbf w _ { s }, \quad \quad ( 1 ) \quad \bullet \frac { \mathfrak { i } } { \mathfrak { i } } \mathfrak { i }$$

- · Antenna Properties ˜ A : Antenna effects, including gain, phase, type, placement, and polarization, are modeled to simulate transmitter and receiver behavior accurately.

where H n denotes the channel matrix for the n -th subcarrier ( n ∈ 1 , . . . , N c ), and λ s is the corresponding s -th largest eigenvalue. The precoding matrix is constructed as

$$W = [ w _ { 1 }, w _ { 2 }, \dots, w _ { N _ { s } } ] \in \mathbb { C } ^ { N _ { i } \times N _ { s } }. \quad \ \ ( 2 ) \ \ \text{tion}, t$$

The 5G NR system uses a codebook-based approach to feedback the precoding matrix. Specifically, the Type II codebook introduced in 3GPP Release-15 [3] employs DualCodebook technology based on the Type I codebook. This codebook supports up to two spatial streams and reconstructs the precoding matrix by combining up to four oversampled orthogonal discrete Fourier transform (DFT) beams. Although its performance approaches that of perfect feedback, the significant overhead reduces system efficiency. To address this challenge, we consider a DT-based approach to approximate RW scenarios.

## B. DT Framework

The RW channel, denoted by H , is influenced by three primary components: (i) Environmental Conditions E , including physical obstructions, atmospheric variations, reflections, and interference. (ii) Antenna Effects A , determined by design factors such as gain, radiation patterns, and polarization. (iii) Propagation Phenomena g ( ) · , which govern signal behaviors such as reflection, diffraction, and scattering. Thus, the RW channel is expressed as:

$$\mathcal { H } = g ( \mathcal { E }, \mathcal { A } ). \quad \quad \ \ ( 3 ) \quad \text{toer}$$

Accurately modeling E , A , and g ( ) · is challenging. Thus, we approximate H with a virtual channel ˜ H generated via advanced simulations:

- · 3D Model ˜ E : Detailed 3D models capture the positions, orientations, shapes, and materials of key components (BS, UE, reflectors, and scatterers), with material properties like conductivity and dielectric constants affecting signal propagation.
- · Ray Tracing ˜( ) g · : This deterministic method uses 3D geometry and material data to simulate propagation paths, accounting for transmission, reflection, scattering, and diffraction, and providing path gain, delay, and angular parameters.

By integrating all parameters from the ray tracing simulation, the virtual channel at the n -th subcarrier is modeled as:

$$\widetilde { \mathbf h } _ { n } = \sum _ { l = 1 } ^ { L _ { \mathbf h } } \mathbf G ( \theta _ { l } ^ { \text{AoD} }, \theta _ { l } ^ { \text{AoA} } ) e ^ { - j 2 \pi \frac { n } { N _ { c } } \tau _ { l } }, \quad \ \ ( 4 )$$

where L h is the total number of propagation paths, G ( ) · ∈ C N r × N t is the complex gain matrix determined by the angles of departure (AoD) and arrival (AoA), and τ l is the propagation delay. Both AoD and AoA include azimuth and elevation components.

An accurately constructed virtual channel ˜ H enables the approximation of the virtual precoding matrix ˜ W , closely matching its RW counterpart. However, DTs cannot fully replicate real channels due to limitations in modeling reflections, diffractions, antenna effects, and environmental factors such as interference and noise. To overcome these challenges, we propose a hybrid approach that integrates DL with DT data. As shown in Fig. 1, the model is initially trained on synthetic data from the DT and subsequently fine-tuned using a limited amount of RW data, thereby enhancing accuracy and robustness.

## C. CSI Feedback with DTs

To enable implicit feedback, we adopt the DL-based autoencoder EVCsiNet [5], which is designed for reduced computational complexity and latency. As illustrated in Fig. 1, neural networks for encoding and decoding are deployed at the UE and BS, respectively. The complex-valued input precoding matrix is preprocessed by separating its real and imaginary parts. The encoder and decoder, parameterized by Θ E and Θ D , are denoted as f e ( ; Θ · E ) and f d ( ; Θ · D ) , respectively. The overall autoencoder, f a ( ; Θ) · with Θ = (Θ E , Θ ) D , is expressed as:

$$f _ { \mathrm a } ( \mathbf W ; \Theta ) = f _ { \mathrm d } ( Q ( f _ { \mathrm e } ( \mathbf W ; \Theta _ { E } ) ) ; \Theta _ { D } ), \quad \ ( 5 )$$

where Q ( ) · denotes the quantizer. The encoder compresses the virtual precoding matrix W into a V -dimensional vector that is quantized into a codeword using B bits, while the decoder reconstructs W from the codeword.

The DL optimization objective using virtual channels ˜ H is given by

$$\dot { \Theta } = \left ( \dot { \Theta } _ { \mathrm E }, \dot { \Theta } _ { D } \right ) = \underset { \Theta } { \arg \min } \mathbb { E } _ { \widetilde { \mathcal { H } } } \left \{ L \left ( \widetilde { \mathbf W }, f _ { \mathrm a } ( \widetilde { \mathbf W } ; \Theta ) \right ) \right \}, \ ( 6 ) \quad \text{the RW}.$$

where L ( ) · is the mean square error (MSE) loss function, and E ˜ H {·} denotes the average over ˜ H . To bridge the gap between virtual and RW environments, OL is employed to fine-tune the model. Due to constraints such as battery life, computational load, and limited resources at the UE, OL is primarily conducted at the BS by fine-tuning only the decoder f d ( ) · . The fine-tuning objective is

$$\ddot { \Theta } _ { D } = \underset { \Theta _ { D } } { \arg \min } \mathbb { E } _ { \mathcal { H } } \left \{ L \left ( \mathbf W, f _ { \mathfrak a } ( \mathbf W ; \dot { \Theta } ) \right ) \right \}, \quad ( 7 ) \ \ C. \ F i a _ { \mathbf T \in \cdot }$$

where ˙ Θ is initialized from virtual channel training. Notably, this OL approach introduces minimal communication and computation overhead, as the encoder remains fixed and only the decoder is retrained at the BS.

## III. EXPERIMENTAL SETUP

## A. RW Scenario

RW measurements were conducted in an indoor corridor at National Sun Yat-sen University (NSYSU) as depicted in Fig. 2(a)-(c). The corridor measures approximately 19.7 m in length and 5.93 m in width, with a ceiling height of 2.8 m. The BS, an SGT100A RF vector signal generator with an 8-patch antenna (Fig. 2(f)), is positioned at the left-center of the area, while the receiver is an RTP084 digital oscilloscope with an 8-patch antenna (Fig. 2(g)). Data were collected at 78 marked points, with 200 measurements per point. The communication system follows the 5G NR standard [13] and operates at 3.8 GHz using an OFDM waveform with a 100 MHz channel bandwidth, 60 kHz subcarrier spacing, and 1,620 subcarriers. The BS transmits two spatial streams ( N s = 2 ), each delivering 8 64 . × 10 7 resource elements per second with 1024-QAM modulation. An expectation propagation-based (EP) MIMO detector is used to evaluate the throughput in this scenario.

## B. Virtual Channel in DT

The virtual channel ˜ H is generated using Wireless InSite ® ray-tracing simulations that replicate the RW environment with a detailed 3D model, accurately modeling reflective surfaces, BS/UE positions, and orientations. The UE movement area (red points in Fig. 2(a)) is discretized at 0.5-meter intervals, with each UE orientation randomly chosen from 100 possible directions. Antenna properties, measured in an anechoic chamber, ensure realistic simulation results. Based on the extracted propagation parameters, the virtual channels are computed as follows. Using the ray-traced propagation parameters, the channel between the BS and UEs is computed according to (4), and the corresponding virtual precoding matrix is derived via (1). The simulation waveform design matches that of

TABLE I SIMILARITY OF PROPAGATION PATHS BETWEEN THE DT AND THE RW.

| Position       |   A4 |   D4 |   F4 |    G |    H |
|----------------|------|------|------|------|------|
| Primary Path   | 0.98 | 1    | 0.98 | 0.95 | 0.98 |
| Secondary Path | 0.93 | 0.85 | 0.99 | 0.99 | 0.89 |

the RW scenario, thereby establishing the indoor DT dataset. To evaluate the robustness of the DL-based CSI feedback model, additional virtual channels are generated under various conditions, including different environmental and antenna configurations. An outdoor virtual scenario replicates the NSYSU campus (Fig. 2(d)-(e)), with the BS positioned 50 meters above a building and UEs distributed at 2-meter intervals across the campus square. This simulation yields the outdoor DT dataset.

## C. Fidelity of Virtual Channel

To assess the fidelity of the virtual channel, we compare the simulated AoA from the DT with RW AoAs measured in an indoor corridor at NSYSU (Fig. 2(a)-(b)). Because the standard antennas (Fig. 2(f)-(g)) are suboptimal for AoA extraction, a vertical dipole at the BS (Fig. 2(h)) and a 7-patch array with half-wavelength spacing at the receiver (Fig. 2(i)) were employed instead. The AoA is computed from the phase difference of the received plane wave. The DT AoA is represented by the 3D direction vector u ( ϕ, θ ) = [ cos( ϕ ) cos( θ , ) cos( ϕ ) sin( θ , ) sin( ϕ ) ] T , where θ and ϕ denote the azimuth and elevation angles, respectively. The estimated AoA is similarly represented with ( ˆ ϕ, θ ˆ ) , and the similarity between the two is measured via the inner product

$$\eta = \langle \mathbf u ( \phi, \theta ), \, \mathbf u ( \hat { \phi }, \hat { \theta } ) \rangle, \, - 1 \leq \eta \leq 1. \quad \quad ( 8 )$$

Five measurement points were selected (A4, D4, F4, G, and H; see Fig. 2(b)), with A4, D4, and F4 in strong signal areas and G and H in weaker regions. Table I lists the similarity values for the primary (strongest gain) and secondary (second strongest) paths, indicating high similarity between the DT and RW propagation paths.

Since the precoder W is the main input for EVCsiNet, we further compare the scenario similarity using the precoder cosine similarity. Fig. 3 shows that both the DT and RW datasets exhibit high similarity for positions aligned in the same direction relative to the BS. These findings, from both AoA and precoder perspectives, validate that the DT dataset effectively reflects RW conditions.

While the DT dataset closely approximates RW conditions, some discrepancies remain due to practical antenna effects. Consequently, OL is necessary. For OL, as in (7), RW precoding matrix data W is required. To balance performance and data collection costs, 30% of the RW measurement points in Fig. 2(b) were randomly selected. Furthermore, to maximize the information extracted from limited measurements, the 1,620 subcarriers were grouped into 60 subbands, each containing 27 consecutive subcarriers. This configuration enables each measurement point to generate 60 distinct precoding matrices for OL, thereby enhancing model adaptation with minimal data collection.

Fig. 2. (a) Simulated indoor corridor at NSYSU using Wireless InSite ® . (b) BS and UE RW measurement point locations. (c) Experimental measurement scenario. (d) Satellite image of the NSYSU campus. (e) Simulated outdoor scenario at the NSYSU campus using Wireless InSite ® . (f) 15 cm x 15 cm patch antenna at the BS for the CSI feedback experiment. (g) 7 cm x 15 cm patch antenna at the UE for the CSI feedback experiment. (h) 11.4 cm dipole antenna at the BS for the channel fidelity experiment. (i) 15.5 cm x 17 cm patch antenna at the UE for the channel fidelity experiment.

<!-- image -->

Fig. 3. Heatmaps of precoder cosine similarity in (a) DT and (b) RW datasets.

<!-- image -->

## IV. EVALUATIONS AND DISCUSSIONS

We evaluate three variants of EVCsiNet: the Indoor EVCsiNet , trained on the indoor DT dataset; the Outdoor EVCsiNet , trained on the outdoor DT dataset; and the CDL EVCsiNet , which serves as the baseline and is trained on the standardized cluster delay line (CDL) channel model specified in 3GPP TR 38.901. All variants adopt a compression ratio of 0.5, with 16-length codewords quantized to 2 bits per element, resulting in a 32-bit feedback overhead.

## A. Performance of the DT-Trained Model in RW Scenarios

Table II compares the similarity metric ( ρ ) for precoding matrix reconstruction using various CSI feedback methods (fourth column). Among all methods, the Type II method with 4 beams achieves the highest reconstruction performance ( ρ = 0 96 . ) but requires the most feedback bits (80 bits). Notably, the Indoor EVCsiNet with OL achieves reconstruction performance comparable to the Type II method with 3

TABLE II RECONSTRUCTION PERFORMANCE ACROSS FEEDBACK METHODS.

| Feedback Method        | Training Environment   |   Bits | Testing in RW   | Testing in Indoor DT   |
|------------------------|------------------------|--------|-----------------|------------------------|
| EVCsiNet (w/o / w/ OL) | Indoor DT Outdoor DT   |     32 | 0.67 / 0.82     | 0.93 / -               |
| EVCsiNet (w/o / w/ OL) |                        |     32 | 0.64 / 0.75     | 0.77 / 0.85            |
| EVCsiNet (w/o / w/ OL) | CDL                    |     32 | 0.44 / 0.74     | 0.67 / 0.77            |
| Type II Codebook       | N = 2 beams            |     41 | 0.72            | 0.70                   |
| Type II Codebook       | N = 3 beams            |     58 | 0.86            | 0.84                   |
| Type II Codebook       | N = 4 beams            |     80 | 0.96            | 0.96                   |

beams (i.e., ρ = 0 82 . vs. 0 86 . ) while significantly reducing the feedback overhead (32 bits vs. 58 bits).

Although directly applying the Indoor EVCsiNet to RW scenarios yields suboptimal performance, it still outperforms both the Outdoor and CDL variants. With OL, the Indoor EVCsiNet further improves and maintains a clear advantage over the other two. These results indicate that DT-based pretraining offers better initialization by reducing domain mismatch and significantly enhances the effectiveness of OL. In contrast, CDL-based pretraining followed by OL remains insufficient for achieving robust CSI reconstruction in RW conditions.

Fig. 4 compares the feedback overhead and throughput of the Indoor EVCsiNet with other CSI feedback methods in RW scenarios. Applying OL yields a 17 Mbps increase in throughput for the Indoor EVCsiNet, achieving performance comparable to the Type II method with 3 beams. Interestingly, the Indoor EVCsiNet outperforms the Type II method with 2 beams in terms of throughput despite exhibiting a lower similarity metric ( ρ ). This discrepancy is attributed to the unbalanced performance of the Type II method with 2 beams; with only 2 beams, the precoder corresponding to each of the largest eigenvalues is not accurately reconstructed, which hampers the BS's ability to transmit signals through the dominant eigenchannels effectively. In contrast, EVCsiNet reconstructs each precoder with balanced performance, avoiding this issue.

Fig. 4. RW throughput comparison across different feedback methods.

<!-- image -->

## B. Robustness of the DL Model

Table II shows that the Indoor EVCsiNet suffers performance degradation in RW scenarios due to mismatches in environment ( E ) and antenna effects ( A ). To assess model robustness and isolate the impact of A , we evaluate all models on the indoor DT dataset with fixed antenna settings. The last column reports the similarity metric ρ for each feedback method. The Indoor EVCsiNet achieves the best performance ( ρ = 0 93 . ), outperforming the Outdoor EVCsiNet ( ρ = 0 77 . ) and CDL EVCsiNet ( ρ = 0 67 . ), while using fewer feedback bits. It also approaches the 4-beam Type II codebook ( ρ = 0 96 . ), which requires significantly higher overhead. These results show that the Indoor EVCsiNet benefits from matched training conditions, whereas the CDL model struggles to generalize to the DT scenario.

Furthermore, incorporating 5% of the indoor DT dataset into OL with the Outdoor EVCsiNet improved its reconstruction similarity to 0.85, achieving performance comparable to the Type II method with 3 beams. Nevertheless, the Outdoor EVCsiNet underperformed relative to the Indoor EVCsiNet, underscoring the importance of matching environmental conditions during offline learning. While OL can mitigate environmental mismatches, it cannot fully compensate for them.

We next assess the impact of antenna effects ( A ) on Indoor EVCsiNet. Table III shows that UE antenna changes have minimal impact, while BS antenna variations significantly degrade performance, more than environmental changes, due to the precoder's sensitivity to BS directionality. This degradation occurs because the precoder highly depends on the BS antenna's directionality. While environmental changes prevent the DL model from operating under optimal conditions, any alteration in the BS antenna pattern directly affects its directional characteristics, leading to a more substantial performance decline.

Incorporating 5% of new BS antenna data into OL restores performance to ρ = 0 84 . , demonstrating that OL mitigates antenna variations. Although the detailed results are omitted for brevity, further evaluations reveal that performance remains

TABLE III ANTENNA PATTERN EFFECTS ON RECONSTRUCTION PERFORMANCE.

| Feedback Method   | Antenna Pattern           |    ρ |
|-------------------|---------------------------|------|
| EVCsiNet          | Original                  | 0.93 |
| EVCsiNet          | Change UE Pattern         | 0.94 |
| EVCsiNet          | Change BS Pattern         | 0.57 |
| EVCsiNet          | Change BS Pattern with OL | 0.84 |

stable under time-varying interference and user mobility, suggesting that these effects do not need to be explicitly modeled in the DT. Moreover, while EVCsiNet requires retraining under the corresponding DT as the number of BS antennas increases, the OL method adapts using only a small amount of RW data. Furthermore, increasing the feedback bits provides only marginal gains, as most of the precoding matrix can already be recovered at lower feedback rates, while OL enables robust CSI reconstruction under distribution mismatch.

## V. CONCLUSIONS

This paper investigated a DL-based CSI feedback model trained in a DT environment and tested in RW scenarios. The results indicate that integrating minimal RW data via OL is essential to mitigate performance degradation arising from discrepancies between DT and RW datasets. Furthermore, environmental conditions and BS antenna characteristics significantly impact DL model training, highlighting the importance of a dedicated DT even when OL is employed.
<|endofpaper|>