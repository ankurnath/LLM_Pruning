# REVIEW.md — LLM2Prune Reviewer Action Items

---

## Reviewer 1 (Reviewer_NLT4)

> First, I encourage the authors to construct synthetic combinatorial optimization instances with an explicit, tunable notion of difficulty. For example, structural parameters such as graph density, modularity, constraint tightness, or objective ambiguity could be swept to create a difficulty ladder. Evaluating pruning ratio, objective degradation, and runtime as hardness increases would clarify whether performance degrades gracefully, whether sharp failure regimes exist, and whether beam-search feature discovery overfits to a narrow structural regime.

We ran controlled experiments on Holme-Kim (HK) graphs sweeping two difficulty axes —constraint tightness (budget k) and graph density (HK parameter m) — across three graph sizes (n ∈ {10,000, 50,000, 100,000}), using the model trained on HK with no retraining.
Results for Maximum Coverage are below.



Experiment 1 — Constraint Tightness (fixed HK graph, m=3, vary budget k):

| Budget | n=10,000 | | | n=50,000 | | | n=100,000 | | |
|--------|---------|---------|---------|---------|---------|---------|----------|----------|----------|
| | P_r | P_g | C | P_r | P_g | C | P_r | P_g | C |
| 10  | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 25  | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 50  | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 75  | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 100 | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 150 | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 200 | 0.9994 | 0.9500 | 0.9494 | 0.9999 | 0.9900 | 0.9899 | 1.0000 | 0.9950 | 0.9950 |

Experiment 2 — Graph Density (fixed budget=100, vary HK parameter m):

| m  | avg_deg | n=10,000 | | | n=50,000 | | | n=100,000 | | |
|----|---------|----------|---|---|----------|---|---|-----------|---|---|
|    |         | P_r | P_g | C | P_r | P_g | C | P_r | P_g | C |
| 2  | 4.0     | 1.0005 | 0.9500 | 0.9505 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 4  | 8.0     | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 6  | 12.0    | 1.0003 | 0.9500 | 0.9503 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 8  | 16.0    | 0.9999 | 0.9500 | 0.9499 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |
| 10 | 20.0    | 0.9996 | 0.9500 | 0.9497 | 1.0000 | 0.9900 | 0.9900 | 1.0000 | 0.9950 | 0.9950 |

Across all settings, P_r = 1.000 — LLM2Prune recovers the full heuristic objective value even under tight budgets (k=10) and high graph density (avg. degree 20). The combined metric C increases with n, since the candidate set becomes a smaller fraction of the graph as n grows (P_g improves from 0.95 at n=10k to 0.995 at n=100k), reflecting stronger pruning at scale. No sharp failure regime was observed across any axis. These results confirm that LLM2Prune generalises gracefully across a range of structural regimes and does not overfit to the specific graph size or density used during feature discovery.

---

> Could a non-LLM feature search (e.g., random search, evolutionary feature construction, or classical graph heuristics) achieve comparable pruning performance? This would help isolate the unique contribution of LLM reasoning.

Table 1 already includes classical non-LLM baselines (QuickPrune, SS) that use hand-crafted heuristics without any LLM involvement. LLM2Prune matches or outperforms these across all problems and datasets, demonstrating the value of LLM-guided feature discovery over expert-designed alternatives.

Second, to more directly isolate the LLM contribution, we ran a random feature baseline following [1] and [2]: we train the same GNN architecture on the random features, and evaluate on the same 8 real-world graphs. Results are shown below.
LLM2Prune substantially outperforms random search, particularly on Influence Maximization (IM) where random features yield C as low as 0.46 compared to LLM2Prune's consistent C > 0.85. Since the only difference between the two pipelines is the feature discovery method, this gap is entirely attributable to LLM reasoning.



Random feature baseline:

| Graph    | MC (P_r) | MC (P_g) | MC (C) | MaxCut (P_r) | MaxCut (P_g) | MaxCut (C) | IM (P_r) | IM (P_g) | IM (C) |
| -------- | -------- | -------- | ------ | ------------ | ------------ | ---------- | -------- | -------- | ------ |
| Facebook | 0.9974   | 0.7789   | 0.7768 | 1.0000       | 0.7651       | 0.7651     | 1.0091   | 0.4555   | 0.4597 |
| Wiki     | 1.0000   | 0.8633   | 0.8633 | 1.0000       | 0.8643       | 0.8643     | 0.9792   | 0.7660   | 0.7500 |
| Deezer   | 1.0000   | 0.7625   | 0.7625 | 1.0000       | 0.7539       | 0.7539     | 1.0056   | 0.4111   | 0.4134 |
| Slashdot | 1.0000   | 0.8567   | 0.8567 | 1.0000       | 0.8685       | 0.8685     | 0.9882   | 0.7487   | 0.7398 |
| Twitter  | 1.0000   | 0.8518   | 0.8518 | 1.0000       | 0.8522       | 0.8522     | 1.0096   | 0.6840   | 0.6906 |
| DBLP     | 1.0000   | 0.7899   | 0.7899 | 1.0000       | 0.7974       | 0.7974     | 1.0117   | 0.5147   | 0.5208 |
| YouTube  | 1.0000   | 0.8437   | 0.8437 | 1.0000       | 0.8555       | 0.8555     | 0.9494   | 0.6588   | 0.6255 |
| Skitter  | 1.0000   | 0.8625   | 0.8625 | 1.0000       | 0.6289       | 0.6289     | 0.9934   | 0.6053   | 0.6013 |

[1] Abboud, Ralph, et al. The surprising power of graph neural networks with random node initialization.
[2] Sato, Ryoma, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural networks.

---

> How sensitive is performance to prompt design? Minor paraphrasing or structural changes in prompts should be tested to determine whether results are stable or fragile.

Appendix G.5 compares GPT-5-nano and LLaMA-3.3-70B — two model families with different architectures, tokenizers, and pretraining data — and finds comparable pruning performance (Figure 7). Mizrahi et al. (2024) [3] show that prompt paraphrasing can cause significant rank changes across models; the fact that our two very different LLMs produce consistent results is therefore a meaningful signal that our pipeline is not exploiting prompt-specific phrasing artifacts.

Second, the beam search architecture provides a structural safeguard against prompt sensitivity. At each iteration, β=3 candidate feature sets are expanded in parallel. If one
expansion produces a poor feature set due to prompt variation or stochastic sampling, that branch is pruned and the remaining candidates continue. This is why beam search outperforms a simple feedback loop (Appendix F, Figure 5) — it tolerates bad expansions without cascading failure. Prompt fragility would require all β branches to fail simultaneously, which is unlikely given the diversity introduced by LLM sampling.

[3] Mizrahi et al., 2024, State of What Art? A Call for Multi-Prompt LLM Evaluation.

---

> What is the variance across LLM runs? Because feature generation is stochastic, repeated runs with different seeds should quantify stability in pruning quality and runtime.

Figure 5 (Appendix F) shows variance across pipeline variants with five different random seeds and Figure 3 shows the average runtime of LLM2Prune.

---

> Are generated features interpretable or redundant? Feature-importance analyses should reveal whether the model is leveraging meaningful graph structure or compensating for noisy, overlapping features.

Appendix B lists the discovered features per problem (e.g., closed neighborhood size for Maximum Cover, degree to weight ratio, outgoing activation probability sum, and in–out
degree difference for Influence Maximization under knapsack constraint). The generated features are easily interpretable and distinct. In fact, we observe LLM2Prune discards
many proposed features (some of them are indeed overlapping) in the early stages of feature space search and tries to keep the minimal feature set of features.

---

> What mechanisms prevent overfitting to the synthetic graphs used during feature discovery? Cross-family transfer experiments would help demonstrate genuine generalization.

The cross-family transfer is already demonstrated in Table 1: features discovered on synthetic Holme-Kim (HK) graphs are evaluated directly on 8 real-world graphs spanning social, citation, and communication networks — with no retraining. Strong performance across all of them confirms that the discovered features capture generalizable structural properties rather than HK-specific artifacts.

We deliberately chose HK graphs for feature discovery because they share key structural properties with real-world networks — scale-free degree distribution, clustering, and community structure — following established practice in graph algorithm benchmarking. This is precisely what prevents overfitting: the synthetic proxy is structurally similar to the target domain.

We also identify the operational boundary of this generalization. When feature discovery is run on Erdős-Rényi (ER) graphs — which lack scale-free structure and clustering — performance on real-world graphs degrades significantly. This confirms that LLM2Prune generalizes across graphs within the same structural family, but not across graphs that are fundamentally different in structure. We will make this boundary explicit in the revised paper.

---

> Can the beam-search pipeline recover from poor early feature proposals? Controlled perturbation experiments could test resilience to suboptimal initial branches.

This is directly addressed by the ablation in Appendix F (Figure 5), which compares beam search against a simple feedback loop. Beam search consistently achieves higher and more stable performance because it maintains β=3 candidate paths: when one branch produces a poor feature proposal early on, the other branches continue, and subsequent expansions from the surviving good branches can recover.

We note an important boundary: if all β branches simultaneously produce poor proposals — an unlikely but possible scenario — beam search degenerates into a simple feedback loop and inherits its failure to recover. Figure 5 empirically shows this degenerate regime does not occur in practice with β=3, as beam search consistently outperforms the feedback loop across all tested problems. The diversity introduced by parallel LLM sampling is sufficient to ensure at least one viable branch survives early iterations.


> Are there adversarial or degenerate graph constructions where pruning fails catastrophically? Identifying such cases would clarify the operational envelope of the method.

Yes, and we can identify it precisely. When feature discovery is performed on Erdős-Rényi (ER) graphs — which have homogeneous degree distributions and no clustering or community structure — the discovered features fail to transfer to real-world graphs such as Facebook or Twitter. Performance degrades significantly in this setting, as ER lacks the scale-free and clustered structure that makes HK a good proxy for real-world social networks.

This constitutes the degenerate case and clarifies the operational envelope of LLM2Prune: the method generalizes across graphs that share structural properties with the training proxy (scale-free degree distribution, clustering, community structure), but not across graphs that are fundamentally different in structure. We will report this result explicitly in the revised paper as a stated limitation, alongside the HK → real-world transfer results in Table 1 that demonstrate successful generalization within the supported regime.

---

> What fraction of performance gain is attributable to LLM-driven feature discovery versus downstream classifier learning? Ablations that freeze or randomize components would help disentangle these effects.

The random feature search baseline reported under Comment 2 directly answers this. Both the baseline and LLM2Prune use the identical GNN architecture, training procedure, and evaluation pipeline — the only difference is how features are generated. Since the GNN is identical in both cases, any performance difference is purely attributable to feature quality. Random features yield substantially lower C, particularly on Influence Maximization (C as low as 0.46 vs. LLM2Prune's C > 0.85 per Table 1), with a more modest gap on MC and MaxCut. Note that freezing GNN weights while varying features is not a valid ablation since the weights are trained on the features themselves. The GNN contributes the learning capacity, but it is the LLM that determines what structural signals are worth learning from.

---

## Reviewer 2 (Reviewer_4N39)

> To what extent did the semi-supervised learning improve the performance? Are the other learning-based approaches used for comparison fully supervised or semi-supervised?

The other learning-based baselines (GCOMB-P and LeNSE) are reinforcement learning methods using policy-gradient optimization — not supervised or semi-supervised classifiers. A direct comparison of supervision paradigms is therefore not applicable; the comparison is between a GNN-based pruning classifier and RL-based node-selection policies. LLM2Prune consistently outperforms both GCOMB-P and LeNSE across all problems and datasets (Table 1, Table 4), demonstrating that the semi-supervised GNN approach achieves both higher solution quality and faster inference than RL-based alternatives.

---

> The information of the input graph itself is not given to LLM. It does not help accelerate the search?


During feature discovery, we compensate for the lack of input graph information by carefully selecting a synthetic proxy (Holme-Kim graphs) that shares key structural properties with real-world target graphs — scale-free degree distribution, clustering, and community structure. The LLM reasons about the combinatorial optimization problem structure rather than the specific graph topology, which is why discovered features transfer robustly across diverse real-world graphs. Providing the actual graph statistics to the LLM during search is an interesting direction for future work, which could allow the search to tailor features to specific graph families more precisely.



> In runtime analysis (Fig 3), although learning based methods require the cost for preparing the training data, is it included in the reported time?

Yes, all reported runtimes include the full cost of preparing training data (feature extraction and GNN training). To make this explicit and to address a related comment from another reviewer, we have added a table reporting absolute runtimes for all evaluated algorithms on Influence Maximization (size constraint). Even including all preprocessing steps, LLM2Prune is 1–4 orders of magnitude faster than classical methods (QuickPrune, SS) on larger graphs.

---

> Explanations of the classifier should have been provided in more detail in the main methodology section. Because it does not appear until the experimental section, it is difficult for readers without prior background knowledge to understand the role of the classifier in the methods section.

We agree and will revise the methodology section to include a self-contained description of the classifier. 



> Providing algorithm of the entire procedure would help readers.

We agree and will add a formal algorithm box covering the complete pipeline in the revised methodology section.

---

> I do not fully understand what 'we instead perform ... each instance' in the paragraph just before Section 4.1 indicates. Does this mean that the classifier is trained by a Holme-Kim graph during the beam search (and M is evaluated on the original training graph?), and after finishing the beam search, the selected classifier is trained on the original training graph? In this case, the final M of the selected classifier appears to be different from the value calculated during the beam search, which seems somewhat odd. I misunderstand something?

The reviewer's understanding is correct. During beam search, both the GNN classifier and the metric M are evaluated entirely on synthetic HK graphs (a separate HK validation graph, not the target training graph). This is intentional: the purpose of beam search is to discover which feature set best characterizes high-value nodes in a structural family resembling real-world graphs, without using any target graph data.

After beam search completes and the best feature set is selected, the classifier is fine-tuned on the actual training graph. The fine-tuned classifier will indeed have a different M value from the one computed during search, since it is now evaluated on a different graph. However, this is not problematic — the beam search score serves only as a selection criterion among candidate feature sets, not as a prediction of final performance. The fine-tuning step adapts the classifier to the specific graph structure of the deployment domain, and the final performance is what is reported in Tables 1 and 4. We will clarify this two-stage process explicitly in the revised paper.

---

> I'm a bit confused about the definition of \cal{H}_d,i. Is it the set of the ancestors of (d,i)? If so, why not only \pi(d,i), but also the sum from \ell = 0 to d is required? (it seems some elements in the union are duplicated, in my current understanding).

Yes, H_{d,i} is the set of all ancestors of node (d,i) in the beam search tree — the full path from the root to the current node, including all intermediate feature sets. 
We include the full path (rather than only the immediate parent π(d,i)) because the LLM generates the next feature set conditioned on the entire history of what has been tried. With only the parent, the LLM lacks context about features tried and discarded at earlier depths, and may re-propose them. Providing the full sequence — including intermediate steps — gives the LLM richer context to avoid revisiting explored regions of the feature space. In practice, we observe that using the full path significantly reduces redundant proposals during search. We will clarify this design choice in the revised paper.

---

> The definitions of metric M should be clarified in the paper. According to the code, C on a validation graph is seemingly used, but not mentioned in the paper.

Thank you for this careful reading of the code. The reviewer is correct: during beam search, M is the combined metric C = P_r × P_g evaluated on a held-out synthetic HK validation graph (separate from the HK graph used for classifier training). P_r is the pruning ratio (fraction of heuristic objective value retained) and P_g is the graph reduction ratio (fraction of nodes pruned). Using a validation graph rather than the training graph prevents overfitting the feature selection to a single HK instance and ensures the selected features generalize across HK instances of the same structural family. We will add this explicit definition and clarification to the paper.

---

## Reviewer 3 (Reviewer_Q296)

> Marginal gains compared to classical methods: While LLM2PRUNE consistently outperforms other learning-based baselines, its performance advantage over classical submodularity-based approaches under size constraints is often marginal, and in certain cases, it performs significantly worse in terms of the combined metric (as shown in Table 1). The authors justify this by emphasizing the "orders of magnitude" speedup their method provides. However, to strengthen the paper, a deeper error analysis is required. The authors should analyze the specific cases or graph topologies where LLM2PRUNE's solution quality degrades compared to the theoretical guarantees of classical methods, and discuss the limitations of the generated features in those scenarios.

The comment only considers the size-constraint setting. Under the **knapsack constraint** (Table 4), SS cannot be applied at all, and LLM2Prune significantly outperforms QuickPrune on MaxCov and IM: e.g., C = 0.9984 vs. 0.8447 (DBLP), 0.9982 vs. 0.9149 (YouTube), 0.9997 vs. 0.9049 (Skitter). These are not marginal differences.

Under the size constraint, QuickPrune and SS are competitive on solution quality but orders of magnitude slower (Figure 3), making them impractical on large graphs. LLM2Prune achieves competitive quality at a fraction of the runtime — that is the contribution.

Regarding the error analysis: the specific case where performance degrades is when the training proxy is structurally mismatched to the test graphs. When features are discovered on Erdős-Rényi (ER) graphs — which lack scale-free degree distribution and clustering — performance on real-world graphs drops significantly. This is the operational boundary of LLM2Prune, and it is directly analogous to any ML model failing under distribution shift. Within the supported structural regime (HK-like graphs), LLM2Prune does not exhibit systematic failure cases tied to specific graph topologies.



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



> Transferability of generated features from synthetic to real-world graphs: To mitigate the high computational cost of running beam search on every instance, the framework searches for features on a synthetic Holme-Kim random graph and subsequently applies these features to real-world datasets for downstream fine-tuning. This introduces a major transferability concern. While the Holme-Kim model resembles social networks, real-world networks possess highly heterogeneous topological structures that synthetic models may fail to capture perfectly. It would be better if the authors can provide insights or an ablation study explaining why features discovered in a simplified synthetic environment can robustly generalize to complex, diverse real-world topologies.

This is standard ML generalisation: models transfer well within the same structural family, not across fundamentally different ones. HK graphs were chosen because they share scale-free degree distribution, high clustering, and community structure with real-world social networks — which is why features transfer robustly (Tables 1 and 4). When using Erdős-Rényi graphs instead — which lack these properties — performance degrades, as expected. We will make this boundary explicit in the revised paper.


> Lack of analysis on the generated features in the main text: A primary contribution of this work is the automated discovery of features using the reasoning capabilities of LLMs. Yet, the main text leaves the reader wondering what specific features the LLM actually proposed. Are they simple, well-known metrics (e.g., node degree, betweenness centrality), common combinations, or genuinely novel graph descriptors? Although the authors note that the optimal features are listed in Appendix B, a qualitative analysis of these features should be moved to or summarized within the main text. Discussing the physical meaning and complexity of the highest-scoring features would significantly enhance the paper's insights and better validate the LLM's effectiveness as a "domain expert."

Thank you for this suggestion — we agree that a qualitative discussion of the discovered features belongs in the main text. Appendix B already lists the features per problem, and we will add a summary to Section 4 in the revised paper.

The features are interpretable and problem-specific. For MaxCov under size constraint, LLM2Prune identifies closed neighborhood size, a natural proxy for local coverage potential. For MaxCut, it discovers random cut expectation — a node's expected marginal contribution to the cut under a random partition — a non-trivial, problem-aware descriptor grounded in the problem objective. For IM under knapsack constraint, the selected features are degree-to-weight ratio, outgoing activation probability sum, and in-out degree difference — all reflecting the cost-effectiveness and propagation dynamics of each node. These illustrate LLM2Prune's ability to generate meaningful, problem-aware features without manual engineering.