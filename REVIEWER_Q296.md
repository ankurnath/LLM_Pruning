# Reviewer 3 (Reviewer_Q296)

We thank the reviewer for their detailed and constructive feedback. We address each point below.

> Marginal gains compared to classical methods: While LLM2PRUNE consistently outperforms other learning-based baselines, its performance advantage over classical submodularity-based approaches under size constraints is often marginal, and in certain cases, it performs significantly worse in terms of the combined metric (as shown in Table 1). The authors justify this by emphasizing the "orders of magnitude" speedup their method provides. However, to strengthen the paper, a deeper error analysis is required. The authors should analyze the specific cases or graph topologies where LLM2PRUNE's solution quality degrades compared to the theoretical guarantees of classical methods, and discuss the limitations of the generated features in those scenarios.

We thank the reviewer for this careful observation. We acknowledge that LLM2Prune does exhibit degraded performance in specific cases: on the Facebook graph under MaxCut and MaxCover (gaps of ~14% and ~10% from SS respectively). We attribute this to a structural mismatch between the HK proxy used during beam search and Facebook's unusually dense community structure, which HK graphs do not fully capture. More generally, we observe that performance degrades when the synthetic proxy is structurally mismatched to the target graph. For instance, features such as betweenness centrality, k-core number, and PageRank discovered on Erdős-Rényi graphs do not transfer well to real-world networks. We acknowledge this as an operational boundary of LLM2Prune: the method is sensitive to distribution shift between the proxy and the target graph family, and we have discussed this limitation explicitly in the revised paper (see Limitations paragraph, Section 4, Empirical Evaluation).

Furthermore, under more complex knapsack constraints (Table 4), where SS cannot be applied at all, LLM2Prune substantially outperforms QuickPrune: e.g., C = 0.9984 vs. 0.8447 (DBLP), 0.9982 vs. 0.9149 (YouTube), 0.9997 vs. 0.9049 (Skitter). We believe these differences are substantial, and that the method's advantages are most pronounced precisely in the settings where classical approaches are inapplicable.

---

> Lack of absolute runtime metrics in Figure 3: A central claim of the paper is that LLM2PRUNE is "orders of magnitude faster" than classical sequential pruning methods. However, Figure 3 presents runtime comparisons using a radar chart with a log scale (e.g., -1, 0, 1, 2, 3, 4) without defining the absolute units of measurement. This visual representation only illustrates relative rankings and obscures the actual computational cost. To substantiate the claims of scalability, the authors should provide a standard tabular format detailing the absolute runtime values (ideally breaking down the time spent on feature generation, classifier training, and actual inference) for all evaluated algorithms.

We agree that absolute runtimes should be reported clearly. For completeness, the table below reports absolute inference time for Influence Maximization (size constraint). A breakdown by component has been included in the revised paper (see Runtime subsection, Appendix G.3, Additional Tables and Plots).

Runtime in seconds, Influence Maximization (size constraint):

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

Table 1 empirically shows that features discovered on HK graphs transfer to 8 diverse real-world graphs, achieving strong performance across all problems. We acknowledge, however, that transferability is not always perfect — as noted above, performance degrades on the Facebook graph, where the HK proxy does not fully capture its unusually dense community structure.

We argue that this is not a limitation unique to LLM2Prune, but rather a fundamental challenge shared by all learning-based approaches. Prior methods also propose features based on the problem structure alone, yet the same set of features does not necessarily perform well across all instances of that problem — a well-known difficulty analogous to the behaviour of heuristics for NP-hard problems, where no single algorithm dominates across all graph families. When significant distribution shift exists between training and test graphs, some form of sample data from the target distribution is unavoidable; there is no learning-based approach that can sidestep this requirement entirely.

That said, LLM2Prune performs well on 7 out of 8 datasets across all three problems under both size and knapsack constraints, and the one failure case (Facebook) is attributable to a specific and identifiable structural mismatch.

---

> Lack of analysis on the generated features in the main text: A primary contribution of this work is the automated discovery of features using the reasoning capabilities of LLMs. Yet, the main text leaves the reader wondering what specific features the LLM actually proposed. Are they simple, well-known metrics (e.g., node degree, betweenness centrality), common combinations, or genuinely novel graph descriptors? Although the authors note that the optimal features are listed in Appendix B, a qualitative analysis of these features should be moved to or summarized within the main text. Discussing the physical meaning and complexity of the highest-scoring features would significantly enhance the paper's insights and better validate the LLM's effectiveness as a "domain expert."

We agree that a qualitative discussion of the discovered features belongs in the main text, and we have added a summary to Section 4 in the revised paper (see Discovered Features paragraph, Section 4, Empirical Evaluation).

The features discovered by LLM2Prune are interpretable and problem-specific. For example, for MaxCut the method discovers random cut expectation, which captures a node's expected marginal contribution to the cut under a random partition. This is a non-trivial, problem-aware descriptor that is directly grounded in the problem objective. For IM under the knapsack constraint, the selected features are degree-to-weight ratio, outgoing activation probability sum, and in-out degree difference, all of which reflect the cost-effectiveness and propagation dynamics of each node. Together, these examples illustrate LLM2Prune's ability to generate meaningful, problem-aware features without any manual engineering. We will include these examples along with their discussion in Section 4 of the revised paper.
