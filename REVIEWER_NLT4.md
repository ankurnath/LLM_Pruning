# Reviewer 1 (Reviewer_NLT4)

We thank the reviewer for their comments and feedback. We hope these answers address the major concerns, particularly the empirical contribution. Please let us know if further questions or clarifications are needed.

> First, I encourage the authors to construct synthetic combinatorial optimization instances with an explicit, tunable notion of difficulty. For example, structural parameters such as graph density, modularity, constraint tightness, or objective ambiguity could be swept to create a difficulty ladder. Evaluating pruning ratio, objective degradation, and runtime as hardness increases would clarify whether performance degrades gracefully, whether sharp failure regimes exist, and whethe2r beam-search feature discovery overfits to a narrow structural regime.

We run controlled experiments on Holme-Kim (HK) graphs sweeping two difficulty axes: constraint tightness (budget k) and graph density (HK parameter m), across three graph sizes (n ∈ {10,000, 50,000, 100,000}), using the model trained on HK with no retraining.
Results for Maximum Coverage under size constraint are below. Across all settings, P_r = 1.000; LLM2Prune recovers the full heuristic objective value even under tight budgets (k=10) and high graph density (avg. degree 20). The combined metric C increases with n, since the candidate set becomes a smaller fraction of the graph as n grows (P_g improves from 0.95 at n=10k to 0.995 at n=100k), reflecting stronger pruning at scale. No sharp failure regime was observed across any axis. These results confirm that LLM2Prune generalises gracefully across a range of structural regimes and does not overfit to the specific graph size or density used during feature discovery.

Experiment 1 — Constraint Tightness (fixed HK graph, m=3, vary budget k):

| Budget | n=10,000 | | | | n=50,000 | | | | n=100,000 | | | |
|--------|---------|---------|------|----------------|---------|---------|------|----------------|----------|----------|------|----------------|
| | P_r | P_g | C | Inference (ms) | P_r | P_g | C | Inference (ms) | P_r | P_g | C | Inference (ms) |
| 10  | 1.00 | 0.95 | 0.95 | 7.39 | 1.00 | 0.99 | 0.99 | 33.45 | 1.00 | 1.00 | 1.00 | 70.74 |
| 25  | 1.00 | 0.95 | 0.95 | 6.85 | 1.00 | 0.99 | 0.99 | 33.66 | 1.00 | 1.00 | 1.00 | 67.35 |
| 50  | 1.00 | 0.95 | 0.95 | 7.03 | 1.00 | 0.99 | 0.99 | 32.50 | 1.00 | 1.00 | 1.00 | 64.18 |
| 75  | 1.00 | 0.95 | 0.95 | 6.87 | 1.00 | 0.99 | 0.99 | 34.19 | 1.00 | 1.00 | 1.00 | 61.68 |
| 100 | 1.00 | 0.95 | 0.95 | 6.96 | 1.00 | 0.99 | 0.99 | 34.49 | 1.00 | 1.00 | 1.00 | 64.08 |
| 150 | 1.00 | 0.95 | 0.95 | 6.50 | 1.00 | 0.99 | 0.99 | 31.83 | 1.00 | 1.00 | 1.00 | 63.22 |
| 200 | 1.00 | 0.95 | 0.95 | 7.39 | 1.00 | 0.99 | 0.99 | 31.06 | 1.00 | 1.00 | 1.00 | 64.89 |

Experiment 2 — Graph Density (fixed budget=100, vary HK parameter m):

| m  | avg_deg | n=10,000 | | | | n=50,000 | | | | n=100,000 | | | |
|----|---------|---------|---------|------|----------------|---------|---------|------|----------------|----------|----------|------|----------------|
|    |         | P_r | P_g | C | Inference (ms) | P_r | P_g | C | Inference (ms) | P_r | P_g | C | Inference (ms) |
| 2  | 4.0  | 1.00 | 0.95 | 0.95 | 6.17  | 1.00 | 0.99 | 0.99 | 29.21  | 1.00 | 1.00 | 1.00 | 64.32  |
| 4  | 8.0  | 1.00 | 0.95 | 0.95 | 8.30  | 1.00 | 0.99 | 0.99 | 38.66  | 1.00 | 1.00 | 1.00 | 76.46  |
| 6  | 12.0 | 1.00 | 0.95 | 0.95 | 9.28  | 1.00 | 0.99 | 0.99 | 42.67  | 1.00 | 1.00 | 1.00 | 90.46  |
| 8  | 16.0 | 1.00 | 0.95 | 0.95 | 11.02 | 1.00 | 0.99 | 0.99 | 48.85  | 1.00 | 1.00 | 1.00 | 95.73  |
| 10 | 20.0 | 1.00 | 0.95 | 0.95 | 11.66 | 1.00 | 0.99 | 0.99 | 52.91  | 1.00 | 1.00 | 1.00 | 115.27 |


> Could a non-LLM feature search (e.g., random search, evolutionary feature construction, or classical graph heuristics) achieve comparable pruning performance? This would help isolate the unique contribution of LLM reasoning.

Table 1 includes classical non-LLM baselines (QuickPrune, SS) that use hand-crafted heuristics without any LLM involvement. LLM2Prune matches or outperforms these across all problems and datasets, demonstrating the value of LLM-guided feature discovery over expert-designed alternatives.

To more directly isolate the LLM contribution, we run a random feature baseline following [1] and [2]: we train the same GNN architecture on the random features and evaluate on the same 8 real-world graphs. Results are shown below. LLM2Prune substantially outperforms random search, particularly on Influence Maximization (IM) where random features yield C as low as 0.46 compared to LLM2Prune's consistent C > 0.85. Since the only difference between the two pipelines is the feature discovery method, this gap is entirely attributable to LLM reasoning.

Random feature baseline under size constraint experiments:

| Graph    | MaxCov (P_r) | MaxCov (P_g) | MaxCov (C) | MaxCut (P_r) | MaxCut (P_g) | MaxCut (C) | IM (P_r) | IM (P_g) | IM (C) |
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

We tested different prompt formulations and found that minor paraphrasing does not affect performance; models of this scale reliably follow task instructions regardless of surface-level phrasing variation. Moreover, every feature set the LLM proposes is evaluated by an objective, prompt-agnostic oracle: a GNN is trained on the proposed features and C is measured on a held-out synthetic validation graph. The final selected feature set is determined entirely by this oracle score. Appendix G.5 shows that GPT-5-nano and LLaMA-3.3-70B, with fundamentally different architectures and tokenizers, produce consistent results, which is a stronger test than paraphrasing within a single model.

---

> What is the variance across LLM runs? Because feature generation is stochastic, repeated runs with different seeds should quantify stability in pruning quality and runtime.

Figure 5 (Appendix F) directly reports variance across five random seeds for all pipeline variants. The beam search variant (red) shows consistently tight interquartile ranges and high medians across all six problem settings (Maximum Coverage, Maximum Cut, Influence Maximization, and their weighted variants), indicating stable performance regardless of the random seed. Figure 3 reports runtime of LLM2Prune.

---

> Are generated features interpretable or redundant? Feature-importance analyses should reveal whether the model is leveraging meaningful graph structure or compensating for noisy, overlapping features.

Appendix B lists the discovered features per problem (e.g., closed neighborhood size for Maximum Coverage, degree and random cut expectation for Maximum Cut, degree and average incoming activation probability for IM under size constraint). The generated features are easily interpretable and distinct. LLM2Prune discards many proposed features during the search, some of which are indeed overlapping, and converges to a minimal, non-redundant set. The feature-importance scores produced by the explainer guide this process explicitly: low-importance features are flagged and replaced in subsequent iterations.

---

> What mechanisms prevent overfitting to the synthetic graphs used during feature discovery? Cross-family transfer experiments would help demonstrate genuine generalization.

The cross-family transfer is demonstrated in Table 1: features discovered on synthetic Holme-Kim (HK) graphs are evaluated directly on 8 real-world graphs spanning social, citation, and communication networks. Strong performance across all of them confirms that the discovered features capture generalizable structural properties rather than HK-specific artifacts.
We deliberately chose HK graphs for feature discovery because they share key structural properties with real-world networks, namely scale-free degree distribution, clustering, and community structure, following established practice in graph algorithm benchmarking. This is precisely what prevents overfitting: the synthetic proxy is structurally similar to the target domain.
We also identify the operational boundary of this generalization. When feature discovery is run on Erdős-Rényi (ER) graphs, which lack scale-free structure and clustering, performance on real-world graphs degrades significantly. This confirms that LLM2Prune generalizes across graphs within the same structural family, but not across graphs that are fundamentally different in structure. We will make this boundary explicit in the revised paper.



---

> Can the beam-search pipeline recover from poor early feature proposals? Controlled perturbation experiments could test resilience to suboptimal initial branches.

We believe this is addressed by the ablation in Appendix F (Figure 5), which compares beam search against a simple feedback loop. If we start with suboptimal initial branches, this effectively reduces to running multiple suboptimal simple feedback loops. We note an important boundary: if all β branches simultaneously produce poor proposals, an unlikely but possible scenario, beam search degenerates into a simple feedback loop with poor initialization and can inherit its failure to recover. Figure 5 empirically shows this degenerate regime does not occur in practice with β=3, as beam search consistently outperforms the feedback loop across all tested problems. The diversity introduced by parallel LLM sampling is sufficient to ensure at least one viable branch survives early iterations.

---

> Are there adversarial or degenerate graph constructions where pruning fails catastrophically? Identifying such cases would clarify the operational envelope of the method.

When feature discovery is performed on Erdős-Rényi (ER) graphs, which have homogeneous degree distributions and no clustering or community structure, the discovered features fail to transfer to real-world graphs such as Facebook or Twitter. Performance degrades significantly in this setting, as ER lacks the scale-free and clustered structure that makes HK a good proxy for real-world social networks.
This constitutes the degenerate case and clarifies the operational envelope of LLM2Prune: the method generalizes across graphs that share structural properties with the training proxy (scale-free degree distribution, clustering, community structure), but not across graphs that are fundamentally different in structure. We will report this observation explicitly in the revised paper as a stated limitation, alongside the HK to real-world transfer results in Table 1 that demonstrate successful generalization within the supported regime.

---

> What fraction of performance gain is attributable to LLM-driven feature discovery versus downstream classifier learning? Ablations that freeze or randomize components would help disentangle these effects.

The random feature search baseline reported above directly answers this. Both the baseline and LLM2Prune use the identical GNN architecture, training procedure, and evaluation pipeline; the only difference is how features are generated. Since the GNN is identical in both cases, any performance difference is purely attributable to feature quality. Random features yield substantially lower C, particularly on Influence Maximization (C as low as 0.46 vs. LLM2Prune's C > 0.85 per Table 1), with a more modest gap on MaxCover and MaxCut. We note that freezing GNN weights while varying features is not a well-defined ablation in this setting, since GNN weights are optimized given input features — the weights encode patterns that are specific to the features they were trained on. Swapping features while keeping weights fixed would measure the incompatibility between mismatched weights and features rather than the intrinsic quality of the features themselves. The GNN contributes the learning capacity, but it is the LLM that determines what structural signals are worth learning from.
