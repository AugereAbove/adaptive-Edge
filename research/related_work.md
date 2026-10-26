# Related work

This is a positioning scaffold, not a systematic review. Metadata was checked against linked primary arXiv records. These papers' results are not AdaptiveEdge results.

| Area | Verified reference | Relationship |
|---|---|---|
| Model routing | Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, Ion Stoica. *RouteLLM: Learning to Route LLMs with Preference Data* (2024). [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) | Learned preference routing versus transparent historical aggregation and local measurements. |
| Cascades | Lingjiao Chen, Matei Zaharia, James Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* (2023). [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) | Learned cascades versus selecting a configuration before generation. |
| Mixture-of-experts / conditional computation | Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017). [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | Internal expert routing differs from routing between complete model artifacts. |
| Speculative inference | Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via Speculative Decoding* (2022; revised 2023). [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | Token-level acceleration is complementary to request selection; AdaptiveEdge does not implement it. |

Before paper submission, expand coverage of LLM benchmarking and contamination, hardware-aware selection, multi-objective optimization, Pareto-efficient inference, dynamic neural network inference and early exits. Evaluate matched tasks/hardware before claiming superiority. The repository establishes an inspectable baseline and does not claim a novel routing algorithm.
