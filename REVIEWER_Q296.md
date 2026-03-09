# Reviewer 3 (Reviewer_Q296)

---

> Marginal gains compared to classical methods: While LLM2PRUNE consistently outperforms other learning-based baselines, its performance advantage over classical submodularity-based approaches under size constraints is often marginal, and in certain cases, it performs significantly worse in terms of the combined metric (as shown in Table 1). The authors justify this by emphasizing the "orders of magnitude" speedup their method provides. However, to strengthen the paper, a deeper error analysis is required. The authors should analyze the specific cases or graph topologies where LLM2PRUNE's solution quality degrades compared to the theoretical guarantees of classical methods, and discuss the limitations of the generated features in those scenarios.

The comment only considers the size-constraint setting. Under the **knapsack constraint** (Table 4), SS cannot be applied at all, and LLM2Prune significantly outperforms QuickPrune on MaxCov and IM: e.g., C = 0.9984 vs. 0.8447 (DBLP), 0.9982 vs. 0.9149 (YouTube), 0.9997 vs. 0.9049 (Skitter). These are not marginal differences.

Under the size constraint, QuickPrune and SS are competitive on solution quality but orders of magnitude slower (Figure 3), making them impractical on large graphs. LLM2Prune achieves competitive quality at a fraction of the runtime — that is the contribution.

Regarding the error analysis: the specific case where performance degrades is when the training proxy is structurally mismatched to the test graphs. When features are discovered on Erdős-Rényi (ER) graphs — which lack scale-free degree distribution and clustering — performance on real-world graphs drops significantly. The generated features in this case like Betweenness centrality, k-core number and PageRank are not very helpful. This is the operational boundary of LLM2Prune, and it fails under distribution shift. Within the supported structural regime (HK-like graphs), LLM2Prune does not exhibit systematic failure cases tied to real-world graphs.

---

> Lack of absolute runtime metrics in Figure 3: A central claim of the paper is that LLM2PRUNE is "orders of magnitude faster" than classical sequential pruning methods. However, Figure 3 presents runtime comparisons using a radar chart with a log scale (e.g., -1, 0, 1, 2, 3, 4) without defining the absolute units of measurement. This visual representation only illustrates relative rankings and obscures the actual computational cost. To substantiate the claims of scalability, the authors should provide a standard tabular format detailing the absolute runtime values (ideally breaking down the time spent on feature generation, classifier training, and actual inference) for all evaluated algorithms.

We agree that absolute runtimes should be reported clearly. Figure 6 (Appendix G.3) already provides a breakdown of LLM2Prune's runtime into feature extraction and GNN inference. For completeness, the table below reports absolute runtimes for Influence Maximization (size constraint). We will add this table to our revised paper.

Runtime in seconds — Influence Maximization (size constraint):

| Algorithm   | Facebook | Wiki   | Deezer | Slashdot | Twitter  | DBLP    | YouTube  | Skitter   |
|-------------|----------|--------|--------|----------|----------|---------|----------|-----------|
| LLM2Prune   | 0.194    | 0.203  | 0.235  | 0.257    | 0.314    | 0.458   | 1.118    | 2.420     |
| QuickPrune  | 1.860    | 41.960 | 1.100  | 226.000  | 3219.040 | 222.760 | 658.320  | 5109.000  |
| SS*         | 24.960   | 86.424 | 11.371 | 102.259  | 1242.836 | 118.537 | 3652.793 | 10433.744 |
| GCOMB-P     | 0.002    | 0.004  | 0.023  | 0.035    | 0.051    | 0.205   | 0.515    | 0.980     |
| LeNSE       | 31.453   | 36.476 | 42.164 | 44.193   | 81.610   | 49.286  | 238.906  | 1607.605  |

*As SS is extremely slow on large graphs, we increase the number of threads to speed up the pruning approach. For other approaches, we use the same number of threads and GPUs.

---

> Transferability of generated features from synthetic to real-world graphs: To mitigate the high computational cost of running beam search on every instance, the framework searches for features on a synthetic Holme-Kim random graph and subsequently applies these features to real-world datasets for downstream fine-tuning. This introduces a major transferability concern. While the Holme-Kim model resembles social networks, real-world networks possess highly heterogeneous topological structures that synthetic models may fail to capture perfectly. It would be better if the authors can provide insights or an ablation study explaining why features discovered in a simplified synthetic environment can robustly generalize to complex, diverse real-world topologies.

We chose Holme-Kim graphs as the synthetic proxy because they share key structural properties with real-world social, citation, and communication networks — scale-free degree distribution, high clustering, and community structure. Features that are informative on HK graphs tend to remain informative on real-world graphs precisely because these structural signals are shared.

This is empirically validated by Table 1: features discovered on HK graphs transfer to 8 diverse real-world graphs with no retraining, achieving strong performance across all problems. To understand the boundary of this transferability, we also ran feature discovery on Erdős-Rényi graphs — which lack scale-free structure and clustering — and observed significant performance degradation on the same real-world graphs. This confirms that transferability holds within the same structural family and breaks down across structurally different graph models. We will make this boundary explicit in the revised paper.

---

> Lack of analysis on the generated features in the main text: A primary contribution of this work is the automated discovery of features using the reasoning capabilities of LLMs. Yet, the main text leaves the reader wondering what specific features the LLM actually proposed. Are they simple, well-known metrics (e.g., node degree, betweenness centrality), common combinations, or genuinely novel graph descriptors? Although the authors note that the optimal features are listed in Appendix B, a qualitative analysis of these features should be moved to or summarized within the main text. Discussing the physical meaning and complexity of the highest-scoring features would significantly enhance the paper's insights and better validate the LLM's effectiveness as a "domain expert."

Thank you for this suggestion — we agree that a qualitative discussion of the discovered features belongs in the main text. Appendix B already lists the features per problem, and we will add a summary to Section 4 in the revised paper.

The features are interpretable and problem-specific. For MaxCov under size constraint, LLM2Prune identifies closed neighborhood size, a natural proxy for local coverage potential. For MaxCut, it discovers random cut expectation — a node's expected marginal contribution to the cut under a random partition — a non-trivial, problem-aware descriptor grounded in the problem objective. For IM under knapsack constraint, the selected features are degree-to-weight ratio, outgoing activation probability sum, and in-out degree difference — all reflecting the cost-effectiveness and propagation dynamics of each node. These illustrate LLM2Prune's ability to generate meaningful, problem-aware features without manual engineering.
