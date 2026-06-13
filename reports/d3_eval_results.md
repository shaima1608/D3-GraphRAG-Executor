# D3 Evaluation Results

These are local RAGAS-style proxy metrics that run without external API keys.

| Metric | Value |
|---|---:|
| Queries | 20 |
| Faithfulness proxy | 0.8828 |
| Answer relevance proxy | 0.9045 |
| Citation coverage | 1.0 |
| Gold source hit | 0.95 |
| p95 latency seconds | 0.8285 |

## Metric definitions

- **Faithfulness proxy:** fraction of answer tokens supported by retrieved evidence.
- **Answer relevance proxy:** coverage of important query terms in answer and cited evidence.
- **Citation coverage:** fraction of answers with at least one citation.
- **Gold source hit:** whether the retrieved evidence includes the gold `Correct_PDF` source.
- **p95 latency:** 95th percentile runtime after warm-up.

## Per-question evidence check

| query                                                                                                                          | correct_pdf      |   gold_source_hit |   faithfulness_proxy |   answer_relevance_proxy |   latency_seconds |
|:-------------------------------------------------------------------------------------------------------------------------------|:-----------------|------------------:|---------------------:|-------------------------:|------------------:|
| What is the main AI-related objective of Efficient Estimation of Word Representations in Vector Space Tomas Mikolov?           | 1301.3781v3.pdf  |                 1 |               0.8636 |                   0.88   |            0.0713 |
| What is the main AI-related objective of Playing Atari with Deep Reinforcement Learning Volodymyr Mnih Koray Kavukcuoglu?      | 1312.5602v1.pdf  |                 1 |               0.9048 |                   0.9273 |            0.6278 |
| What is the main AI-related objective of ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION Diederik P. Kingma* University of?         | 1412.6980v9.pdf  |                 1 |               0.8696 |                   0.95   |            0.5754 |
| What is the main AI-related objective of MLlib: Machine Learning in Apache Spark Xiangrui Meng?                                | 15-237.pdf       |                 1 |               0.9221 |                   0.9111 |            0.4451 |
| What is the main AI-related objective of All electrical manipulation of magnetization dynamics in a ferromagnet by?            | 1508.07906v2.pdf |                 0 |               0.9286 |                   0.82   |            0.623  |
| What is the main AI-related objective of UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL?  | 1511.06434v2.pdf |                 1 |               0.918  |                   0.9556 |            0.8364 |
| What is the main AI-related objective of "Why Should I Trust You?" Explaining the Predictions of Any Classifier Marco Tulio?   | 1602.04938v3.pdf |                 1 |               0.8654 |                   0.8364 |            0.6775 |
| What is the main AI-related objective of Communication-Efficient Learning of Deep Networks from Decentralized Data H. Brendan? | 1602.05629v4.pdf |                 1 |               0.8108 |                   0.8    |            0.6335 |
| What is the main AI-related objective of Efficient and robust approximate nearest neighbor search using Hierarchical?          | 1603.09320v4.pdf |                 1 |               0.8571 |                   0.9111 |            0.4811 |
| What is the main AI-related objective of End to End Learning for Self-Driving Cars Mariusz Bojarski NVIDIA Corporation?        | 1604.07316v1.pdf |                 1 |               0.8667 |                   0.92   |            0.5182 |
| What is the main AI-related objective of Concrete Problems in AI Safety Dario Amodei Google Brain?                             | 1606.06565v2.pdf |                 1 |               0.9074 |                   0.9556 |            0.5372 |
| What is the main AI-related objective of Wide & Deep Learning for Recommender Systems Heng-Tze Cheng, Levent Koc, Jeremiah?    | 1606.07792v1.pdf |                 1 |               0.8909 |                   0.9333 |            0.4593 |
| What is the main AI-related objective of SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS Thomas N. Kipf?      | 1609.02907v4.pdf |                 1 |               0.8667 |                   0.9111 |            0.4283 |
| What is the main AI-related objective of Neural Collaborative Filtering Xiangnan He National University of?                    | 1708.05031v2.pdf |                 1 |               0.9079 |                   0.9    |            0.3931 |
| What is the main AI-related objective of Deep Reinforcement Learning for Conversational AI Mahipal Jadeja DA-IICT?             | 1709.05067v1.pdf |                 1 |               0.8696 |                   0.9111 |            0.3798 |
| What is the main AI-related objective of GRAPH ATTENTION NETWORKS Petar Velickovi c Department of Computer Science and?        | 1710.10903v3.pdf |                 1 |               0.9496 |                   0.82   |            0.4608 |
| What is the main AI-related objective of AI Safety Gridworlds Jan Leike DeepMind?                                              | 1711.09883v2.pdf |                 1 |               0.9444 |                   1      |            0.4627 |
| What is the main AI-related objective of Efficient Neural Architecture Search via Parameter Sharing Hieu Pham * 1 2 Melody Y.? | 1802.03268v2.pdf |                 1 |               0.6818 |                   0.9    |            0.5663 |
| What is the main AI-related objective of Horovod: fast and easy distributed deep learning in TensorFlow Alexander Sergeev?     | 1802.05799v3.pdf |                 1 |               0.8824 |                   0.9273 |            0.4272 |
| What is the main AI-related objective of Future of Humanity Institute?                                                         | 1802.07228v2.pdf |                 1 |               0.9481 |                   0.92   |            0.4356 |
