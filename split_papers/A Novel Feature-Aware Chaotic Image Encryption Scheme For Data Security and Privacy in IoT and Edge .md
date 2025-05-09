<|startofpaper|>
## A Novel Feature-Aware Chaotic Image Encryption Scheme For Data Security and Privacy in IoT and Edge Networks

Muhammad Shahbaz Khan , Ahmed Al-Dubai , Jawad Ahmad , Nikolaos Pitropakis ∗ ∗ † ∗ and Baraq Ghaleb ∗ ∗ School of Computing, Engineering and the Built Environment, Edinburgh Napier University, Edinburgh, UK. Emails: { muhammadshahbaz.khan, a.al-dubai, n.pitropakis, b.ghaleb } @napier.ac.uk † Cyber Security Center, Prince Mohammad Bin Fahd University, Al-Khobar, Saudi Arabia Email: jahmad@pmu.edu.sa

Abstract -The security of image data in the Internet of Things (IoT) and edge networks is crucial due to the increasing deployment of intelligent systems for real-time decision-making. Traditional encryption algorithms such as AES and RSA are computationally expensive for resource-constrained IoT devices and ineffective for large-volume image data, leading to inefficiencies in privacy-preserving distributed learning applications. To address these concerns, this paper proposes a novel Feature-Aware Chaotic Image Encryption scheme that integrates Feature-Aware Pixel Segmentation (FAPS) with Chaotic Chain Permutation and Confusion mechanisms to enhance security while maintaining efficiency. The proposed scheme consists of three stages: (1) FAPS, which extracts and reorganizes pixels based on high and low edge intensity features for correlation disruption; (2) Chaotic Chain Permutation, which employs a logistic chaotic map with SHA256-based dynamically updated keys for block-wise permutation; and (3) Chaotic chain Confusion, which utilises dynamically generated chaotic seed matrices for bitwise XOR operations. Extensive security and performance evaluations demonstrate that the proposed scheme significantly reduces pixel correlationalmost zero, achieves high entropy values close to 8, and resists differential cryptographic attacks. The optimum design of the proposed scheme makes it suitable for real-time deployment in resource-constrained environments.

and inefficient handling of large image data [8], [9]. Hence, lightweight and adaptive encryption algorithms are required to protect privacy while maintaining system scalability and efficiency.

Index Terms -IoT security, data privacy, image encryption, confusion, permutation

## I. INTRODUCTION

The rapid expansion of the Internet of Things (IoT) and edge computing has transformed various domains, including smart cities, industrial automation, and healthcare [1]-[3]. These systems rely on distributed machine learning (ML) and artificial intelligence (AI) to process real-time data and make intelligent decisions on resource-constrained devices. However, ensuring the security and privacy of image data in such decentralized environments remains a significant challenge [4]. Sensitive images, including medical scans, surveillance footage, and industrial monitoring data, are frequently transmitted and processed across heterogeneous networks, making them vulnerable to eavesdropping, unauthorized access, and adversarial attacks [5]-[7]. Traditional encryption techniques, such as AES and RSA, are not well-suited for IoT and edge-based AI applications due to their high computational complexity

Chaos-based image encryption has emerged as a promising solution for securing image data in distributed environments [10]-[12]. Chaotic systems possess properties such as sensitivity to initial conditions, pseudo-randomness, and ergodicity, making them well-suited for cryptographic applications [13][15]. Various chaotic maps, including the Logistic map [16], Henon map [17], and Lorenz system [18], have been employed in image encryption schemes. These methods leverage permutation and substitution processes driven by chaotic sequences to disrupt pixel correlation and enhance security. However, existing chaotic encryption schemes often fail to meet the privacy and efficiency requirements of IoT and edge networks. Many approaches rely on static chaotic parameters, which can lead to periodic behaviour and reduced security. Additionally, conventional chaotic permutations are often independent of the underlying image structure, making them computationally inefficient for real-time ML-based edge analytics.

To address the aforementioned challenges, this paper proposes a privacy-preserving image encryption framework for IoT and edge-based intelligent systems, integrating FeatureAware Pixel Segmentation (FAPS) with Chaotic Chain Permutation and Confusion mechanisms. By integrating these techniques, the proposed scheme enhances data protection and privacy. An overview of the proposed scheme is given in Fig. 1. The key contributions of this paper are as follows:

- · A novel Feature-Aware Pixel Segmentation (FAPS) technique that optimizes image encryption for AI-driven IoT and edge networks by reducing correlation in image data. It extracts and reorganizes pixels based on high and low edge intensity features for effective correlation disruption.
- · A Chaotic Chain Permutation method that employs a logistic chaotic map with SHA-256-based dynamic key generation, ensuring adaptive security and enhanced randomness.
- · A Chaotic Chain Confusion mechanism that utilises

Fig. 1: Overview of the Proposed Feature-Aware Encryption Scheme

<!-- image -->

dynamically generated chaotic seed matrices for bitwise XOR operations in the confusion stage, making the encryption scheme resilient to cryptographic attacks.

The rest of this paper is structured as follows: Section II presents details on the proposed encryption scheme, including feature-aware pixel segmentation, chaotic chain permutation, and chaotic chain confusion. Section IV provides security analysis and experimental results, while Section V presents a clear and concise conclusion.

## II. THE PROPOSED FEATURE-AWARE CHAOTIC IMAGE ENCRYPTION SCHEME

The proposed scheme consists of three stages: (1) FeatureAware Pixel Segmentation , which classifies pixels based on edge intensity to optimize encryption; (2) Chaotic Chain Permutation , which applies a dynamically updated logistic chaotic map for block-wise permutation; and (3) Chaotic Chain Confusion , which performs bitwise XOR with a dynamic chaotic seed matrices randomness. The complete block diagram of the proposed scheme entailing all three stages is given in Fig. 2 and are explained in the following subsections. In addition, a pseudo-code algorithm for the stepwise implementation of the proposed scheme is given in Algorithm 1.

## A. Stage 1: Feature-Aware Pixel Segmentation (FAPS)

This paper proposes a Feature-Aware Pixel Segmentation (FAPS) technique for preprocessing images before secure permutation. The method utilizes Sobel edge detection to segment pixels into high-variance and low-variance regions. The overview of variance classification for a 16 × 16 sample image is depicted in Fig. 3, whereas for a 256 × 256 Cameraman image, the process of edge detection with high and low variance region segmentation is depicted in Fig. 4.

Let I ( x, y ) be the greyscale image of size M × N , where each pixel has an intensity value in the range I ( x, y ) ∈ [0 , 255] . The proposed method follows these steps:

1) Sobel Edge Detection: The Sobel operator computes the gradient magnitude of each pixel to measure edge strength:

$$G _ { x } = I ( x, y ) * S _ { x }, \ G _ { y } = I ( x, y ) * S _ { y } \quad \ ( 1 ) \quad _ { \mathfrak { T u.. } }$$

where S x and S y are the horizontal and vertical Sobel kernels:

$$S _ { x } = \begin{bmatrix} \overset {. } { - 1 } & 0 & + 1 \\ - 2 & 0 & + 2 \\ - 1 & 0 & + 1 \end{bmatrix}, \ \ S _ { y } = \begin{bmatrix} + 1 & + 2 & + 1 \\ 0 & 0 & 0 \\ - 1 & - 2 & - 1 \end{bmatrix} \quad ( 2 )$$

The edge magnitude at each pixel is then computed as:

$$G _ { s o b e l } ( x, y ) = \sqrt { G _ { x } ^ { 2 } + G _ { y } ^ { 2 } }$$

The edge map is then normalized:

$$E ( x, y ) = \frac { G _ { s o b e l } ( x, y ) } { \max ( G _ { s o b e l } ) }$$

where E x,y ( ) ∈ [0 , 1] represents the normalized edge intensity.

2) High-Edge and Low-Edge Pixel Classification: In this step, a threshold T is defined, which is obtained using Otsu's method:

$$T = \arg \max _ { \tau } [ \sigma _ { B } ^ { 2 } ( \tau ) ]$$

where σ 2 B ( τ ) is the between-class variance for a given threshold τ . Using this threshold, we classify pixels into high-edge (HE) and low-edge (LE) regions:

$$P _ { H E } = \left \{ I ( x, y ) \ | \ E ( x, y ) > T \right \} \text{ \quad \ \ } ( 6 )$$

$$P _ { L E } = \{ I ( x, y ) \ | \ E ( x, y ) \leq T \} \quad \quad ( 7 )$$

where P HE contains textured and boundary regions, and P LE contains smooth regions.

3) Pixel Sorting and Grouping: To prepare for chaotic permutation, the pixels are reordered in a structured manner. High-edge pixels are sorted in descending order and placed in the upper half of the image:

$$P _ { H E } ^ { \prime } = \text{sort} ( P _ { H E }, \text{descend} )$$

On the other hand, the low-edge pixels are sorted in ascending order and placed in the lower half:

$$P _ { L E } ^ { \prime } = \text{sort} ( P _ { L E }, \text{ascend} )$$

The final pre-permutation image I ′ is defined as:

$$I ^ { \prime } = \begin{bmatrix} P ^ { \prime } _ { H E } \\ P ^ { \prime } _ { L E } \end{bmatrix}$$

Fig. 2: Complete block diagram of the proposed Feature-Aware Encryption Scheme

<!-- image -->

## B. Stage 2: Chaotic Chain Permutation

Once the image is preprocessed, a logistic map-based chaotic permutation is applied in a block-wise manner.

- 4) The process repeats for all blocks, ensuring that each block's permutation is influenced by the previous block's hash.
- 1) The image I ′ is divided into B × B non-overlapping blocks:

$$I ^ { \prime } & = \{ B _ { 1 }, B _ { 2 }, \dots, B _ { k } \}, \quad B _ { i } \in \mathbb { R } ^ { b \times b }, \quad k = \frac { M \times N } { B ^ { 2 } } \\ \cdot \quad \cdots \cdots \cdots \supset \cdots \cdots \cdots \cdots \cdots \infty$$

where each block B i has dimensions 32 × 32 .

- 2) The logistic chaotic map is used to generate an initial permutation key and is defined as:

$$X _ { n + 1 } = r X _ { n } ( 1 - X _ { n } ) \quad \quad ( 1 2 ) \, \frac { \ e s u } { \ h o c h }$$

where X n ∈ (0 , 1) is the state variable, and r is the chaotic control parameter. The initial key X 0 is chosen randomly within the chaotic range and is used to permute the first block.

- 3) A hash H 1 of the first permuted block B 1 is calculated using SHA-256. This hash is used to update the initial conditions and control parameters of the logistic map to generate a new permutation key for the 2nd block. This happens iteratively with each permuted block B i . Each clock B i is permuted using a new permutation key and its hash value H i is computed using SHA-256 for the next block.

$$H _ { i } = \text{SHA-2} 5 6 ( B _ { i } ^ { \prime } ) \text{ \quad \ \ } ( 1 3 )$$

The chaotic system parameters are then updated:

$$X _ { 0 } = \frac { H _ { i } } { 2 ^ { 2 5 6 } }, \ r = 3. 9 + 0. 1 \times \left ( \frac { H _ { i } \mod 1 0 0 } { 1 0 0 } \right ) \ ( 1 4 ) \quad \ \Big | _ {! }$$

- 5) The permuted blocks are combined to form the final encrypted image:

$$I _ { \text{perm} } = \left [ B _ { 1 } ^ { \prime } \ B _ { 2 } ^ { \prime } \ \dots \ B _ { k } ^ { \prime } \right ] \quad \ \ ( 1 5 )$$

## C. Stage 3: Chaotic Chain Confusion

After the permutation process, the image undergoes a blockwise confusion process using a logistic chaotic map and bitwise XOR operation to enhance security. This procedure ensures that each block is influenced by the previous block's hash, making the confusion process highly dependent on initial conditions.

- 1) The permuted image I perm is divided into 16 × 16 nonoverlapping blocks:

$$\delta _ { \text{al} } \quad & I _ { \text{perm} } = \{ B _ { 1 }, B _ { 2 }, \dots, B _ { m } \}, \quad B _ { i } \in \mathbb { R } ^ { 1 6 \times 1 6 }, \quad m = \frac { M \times N } { ( 1 6 ) } \\ \dots \quad & \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \cdot \dots \end{cases}$$

where each block B i has dimensions 16 × 16 .

- 2) A chaotic seed matrix S 1 of size 16 × 16 is generated using the logistic map:

$$X _ { n + 1 } = r X _ { n } ( 1 - X _ { n } ) \text{ \quad \ \ } ( 1 7 )$$

where X n ∈ (0 1) , and r is the chaotic control parameter. The initial seed matrix is given by:

$$S _ { 1 } ( i, j ) = \lfloor 2 5 6 X _ { n _ { i, j } } \rfloor, \ \ i, j = 1, 2, \dots, 1 6 \quad ( 1 8 )$$

where X n i,j are chaotic values mapped to integers in [0 , 255] .

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 3: Overview of the High Edge and Low Edge Pixel Classification in FAPS

<!-- image -->

<!-- image -->

(a)

(b)

Fig. 4: Different stages of the feature extraction process on Cameraman image.

<!-- image -->

<!-- image -->

- 3) The first block B 1 is confused using a bitwise XOR operation with the seed matrix:

$$C _ { 1 } = B _ { 1 } \oplus S _ { 1 } \quad \quad \ \ ( 1 9 ) \ \underset { \text{tier} } { \ m a }$$

where C 1 is the confused output block.

- 4) The confused block C 1 is hashed using SHA-256:

$$H _ { 1 } = \text{SHA-2} 5 6 ( C _ { 1 } ) \text{ \quad \ \ } ( 2 0 ) \text{ \ \ } -$$

This hash output is used to update the chaotic system parameters:

$$X _ { 0 } = \frac { H _ { 1 } } { 2 ^ { 2 5 6 } }, \ r = 3. 9 + 0. 1 \times \left ( \frac { H _ { 1 } \mod 1 0 0 } { 1 0 0 } \right ) \ ( 2 1 )$$

Using the updated parameters, a new chaotic seed matrix S 2 is generated:

$$S _ { 2 } ( i, j ) = \lfloor 2 5 6 X _ { n _ { i, j } } \rfloor \text{ \quad \ \ } ( 2 2 )$$

- 5) The process repeats iteratively for all blocks B i , where each confused block C i influences the next seed matrix generation:

$$C _ { i } = B _ { i } \oplus S _ { i }$$

$$H _ { i } = \text{SHA-2} 5 6 ( C _ { i } ) \text{ \quad \ \ } ( 2 4 )$$

$$X _ { 0 } = \frac { H _ { i } } { 2 ^ { 2 5 6 } }, \ \ r = 3. 9 + 0. 1 \times \left ( \frac { H _ { i } \mod 1 0 0 } { 1 0 0 } \right ) \ ( 2 5 )$$

where the updated values regenerate S i +1 for the next block.

- 6) After all blocks are processed, the final confused image I conf is obtained by combining all confused blocks:

$$I _ { \text{conf} } = \left [ C _ { 1 } \ \ C _ { 2 } \ \cdots \ \ C _ { m } \right ] \quad \ \ ( 2 6 )$$

## III. RESULTS AND SECURITY ANALYSIS

This section evaluates the performance of the proposed encryption scheme through entropy analysis, correlation analysis, and differential attacks or sensitivity analysis to minor changes in the plaintext image.

## A. Histogram Analysis

An effective encryption scheme should produce cipher images with a uniform histogram, ensuring resistance against frequency-based attacks. The histograms of the encrypted images as shown in Fig. 5 demonstrate a near-uniform distribution of pixel intensities, indicating that the encryption process effectively diffuses pixel values.

TABLE I: Correlation Evaluation

| Sr.   | Test Image   | Corr. Value   | Correlation Coefficients   | Correlation Coefficients   | Correlation Coefficients   |
|-------|--------------|---------------|----------------------------|----------------------------|----------------------------|
|       |              |               | Hor.                       | Ver.                       | Diag.                      |
| 1     | Cameraman    | 0.00012       | -0.0006                    | 0.0068                     | -0.0071                    |
| 2     | Baboon       | 0.00045       | 0.0028                     | 0.0029                     | 0.0021                     |
| 3     | Houses       | 0.00038       | -0.0049                    | -0.0036                    | 0.0053                     |

TABLE II: Information Entropy Evaluation

|   Sr. | Image     |   Plain Image |   Cipher Image |
|-------|-----------|---------------|----------------|
|     1 | Cameraman |         7.448 |          7.998 |
|     2 | Baboon    |         7.051 |          7.998 |
|     3 | Houses    |         7.011 |          7.998 |

## B. Correlation Analysis

Image encryption aims to eliminate pixel correlation to prevent statistical attacks. Table I shows the correlation coefficients for plain and cipher images. The plain images exhibit strong correlations due to natural redundancy, whereas the encrypted images achieve values close to zero in horizontal, vertical, and diagonal directions. Furthermore, Fig. 5 depicts effective spread out of all correlation coefficients depicting maximum correlation disruption. The correlation is found by:

$$r = \frac { \sum _ { i = 1 } ^ { N } ( x _ { i } - \mu _ { x } ) ( y _ { i } - \mu _ { y } ) } { \sqrt { \sum _ { i = 1 } ^ { N } ( x _ { i } - \mu _ { x } ) ^ { 2 } } } \sqrt { \sum _ { i = 1 } ^ { N } ( y _ { i } - \mu _ { y } ) ^ { 2 } } } \quad ( 2 7 )$$

where:

- · r is the correlation coefficient between adjacent pixels.
- · x i and y i are the intensity values of two adjacent pixels.
- · µ x and µ y are the mean intensity values of all pixels in the image.
- · N is the total number of pixel pairs considered for correlation computation.

## Algorithm 1 Implementation of the Proposed Image Encryption Scheme

- 1: Input: Grayscale image of size M × N
- 2: Output: Encrypted image
- 3: Step 1: Feature-Aware Pixel Segmentation
- 4: Apply Sobel edge detection to highlight texture and edges
- 5: Compute edge map and normalize values
- 6: Use Otsu's method to classify high-texture and low-texture pixels
- 7: Sort and group pixels based on feature classification
- 8: Reconstruct the segmented image
- 9: Step 2: Chaotic Chain Permutation
- 10: Divide image into non-overlapping 32 × 32 blocks
- 11: Initialize logistic chaotic map with an initial key
- 12: for each block do
- 13: Generate a unique permutation sequence using the chaotic map
- 14: Permute block pixels according to the sequence
- 15: Compute SHA-256 hash of the permuted block
- 16: Update chaotic map parameters using the hash output
- 17: end for
- 18: Reconstruct the permuted image
- 19: Step 3: Chaotic Chain Confusion
- 20: Divide image into non-overlapping 16 × 16 blocks
- 21: for each block do
- 22: Generate a dynamic chaotic seed matrix
- 23: Perform bitwise XOR operation between the block and seed matrix
- 24: Compute SHA-256 hash of the confused block
- 25: Update chaotic map parameters using the hash output
- 26: end for
- 27: Reconstruct the final encrypted image

## C. Entropy Analysis

Information entropy measures the randomness of an image, with an ideal entropy value for a perfectly encrypted image being close to 8. Table II presents the entropy values of plain images and their corresponding cipher images. The entropy of plain images is significantly lower due to redundant pixel structures, whereas the cipher images consistently achieve values near 7.998, indicating a highly unpredictable and secure encryption process.

$$H ( X ) = - \sum _ { i = 0 } ^ { 2 5 5 } P ( x _ { i } ) \log _ { 2 } P ( x _ { i } ) \text{ \quad \ \ } ( 2 8 )$$

where:

Fig. 5: Encryption results with histogram and correlation analysis

<!-- image -->

<!-- image -->

(a)

<!-- image -->

(d)

<!-- image -->

(g)

<!-- image -->

(b)

<!-- image -->

(e)

<!-- image -->

(h)

<!-- image -->

(c)

<!-- image -->

(f)

(i)

<!-- image -->

Fig. 6: Differential attack/Sensitivity analysis. (a-c) plain test images. (d-f) original cipher images. (g-i) cipher images of one-bit corrupted plain images. (j-l) Difference between original and corrupted cipher images.

<!-- image -->

<!-- image -->

<!-- image -->

- · H X ( ) is the Shannon entropy of the image.
- · P x ( i ) represents the probability of occurrence of the intensity level x i .
- · The summation runs over all possible intensity values from 0 to 255 for an 8-bit grayscale image.

## D. Differential Attack Resistance

A highly secure encryption algorithm should be sensitive to even the smallest modifications in the plaintext. To evaluate this, a one-bit difference test was conducted, where an image was encrypted twice: first in its original form and then with a single-bit change in the plain image. The absolute difference between the two resulting cipher images was computed to analyse the propagation effect of the minor modification. The results in Fig. 6 demonstrate that the difference between the two encrypted images is substantial, highlighting the avalanche effect of the proposed encryption scheme.

## IV. CONCLUSION

This paper presented a novel feature-aware chaotic image encryption scheme designed to enhance security and privacy in IoT and edge networks. The proposed approach integrated

Feature-Aware Pixel Segmentation, Chaotic Chain Permutation, and Chaotic Chain Confusion to effectively disrupt pixel correlation and improve resistance against statistical and differential attacks. Experimental results demonstrated that the scheme achieved near-ideal entropy values and significantly reduced correlation in encrypted images, ensuring strong security. Additionally, sensitivity analysis confirmed that the encryption process exhibited a high avalanche effect, making it resilient to differential attacks. The proposed method provided a lightweight yet robust encryption mechanism suitable for resource-constrained environments, thus contributing to secure image transmission and storage in intelligent distributed systems. Future work may explore hardware acceleration and adaptive chaotic models to further optimize performance and security.
<|endofpaper|>