# REVIEW.md — LLM2Prune Reviewer Action Items

    Comment: First, I encourage the authors to construct synthetic combinatorial optimization instances with an explicit, tunable notion of difficulty. For example, structural parameters such as graph density, modularity, constraint tightness, or objective ambiguity could be swept to create a difficulty ladder. Evaluating pruning ratio, objective degradation, and runtime as hardness increases would clarify whether performance degrades gracefully, whether sharp failure regimes exist, and whether beam-search feature discovery overfits to a narrow structural regime.

    Rebuttal:
    We ran controlled experiments on Holme-Kim (HK) graphs sweeping two difficulty axes —
    constraint tightness (budget k) and graph density (HK parameter m) — across three graph
    sizes (n ∈ {10,000, 50,000, 100,000}), using the model trained on HK with no retraining.
    Results for Maximum Coverage are below.

    Results (Maximum Coverage, HK model):

    Experiment 1 — Constraint Tightness (fixed HK graph, n=10,000, m=3, vary budget k):

    | n       | budget | P_r    | P_g    | C      |
    | ------- | ------ | ------ | ------ | ------ |
    | 10,000  | 10     | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 25     | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 50     | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 75     | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 100    | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 150    | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 200    | 0.9994 | 0.9500 | 0.9494 |
    | 50,000  | 10     | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 25     | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 50     | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 75     | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 100    | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 150    | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 200    | 0.9999 | 0.9900 | 0.9899 |
    | 100,000 | 10     | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 25     | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 50     | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 75     | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 100    | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 150    | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 200    | 1.0000 | 0.9950 | 0.9950 |

    Experiment 2 — Graph Density (fixed budget=100, vary HK parameter m):

    | n       | m  | avg_deg | P_r    | P_g    | C      |
    | ------- | -- | ------- | ------ | ------ | ------ |
    | 10,000  | 2  | 4.0     | 1.0005 | 0.9500 | 0.9505 |
    | 10,000  | 4  | 8.0     | 1.0000 | 0.9500 | 0.9500 |
    | 10,000  | 6  | 12.0    | 1.0003 | 0.9500 | 0.9503 |
    | 10,000  | 8  | 16.0    | 0.9999 | 0.9500 | 0.9499 |
    | 10,000  | 10 | 20.0    | 0.9996 | 0.9500 | 0.9497 |
    | 50,000  | 2  | 4.0     | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 4  | 8.0     | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 6  | 12.0    | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 8  | 16.0    | 1.0000 | 0.9900 | 0.9900 |
    | 50,000  | 10 | 20.0    | 1.0000 | 0.9900 | 0.9900 |
    | 100,000 | 2  | 4.0     | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 4  | 8.0     | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 6  | 12.0    | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 8  | 16.0    | 1.0000 | 0.9950 | 0.9950 |
    | 100,000 | 10 | 20.0    | 1.0000 | 0.9950 | 0.9950 |

    Across all settings, P_r = 1.000 — LLM2Prune recovers the full heuristic objective value
    even under tight budgets (k=10) and high graph density (avg. degree 20). The combined
    metric C increases with n, since the candidate set becomes a smaller fraction of the graph
    as n grows (P_g improves from 0.95 at n=10k to 0.995 at n=100k), reflecting stronger
    pruning at scale. No sharp failure regime was observed across any axis. These results
    confirm that LLM2Prune generalises gracefully across a range of structural regimes and
    does not overfit to the specific graph size or density used during feature discovery.


    Comment: Could a non-LLM feature search (e.g., random search, evolutionary feature construction, or classical graph heuristics) achieve comparable pruning performance? This would help isolate the unique contribution of LLM reasoning.

    Rebuttal:
    We address this in two ways.

    First, Table 1 already includes classical non-LLM baselines (QuickPrune, SS) that use
    hand-crafted heuristics without any LLM involvement. LLM2Prune matches or outperforms
    these across all problems and datasets, demonstrating the value of LLM-guided feature
    discovery over expert-designed alternatives.

    Second, to more directly isolate the LLM contribution, we ran a random feature 
    baseline following [1] and [2]: we  train the
    same GNN architecture on the random features, and evaluate on the same 8 real-world
    graphs. Results are shown below. LLM2Prune substantially outperforms random search,
    particularly on Influence Maximization (IM) where random features yield C as low as 0.46
    compared to LLM2Prune's consistent C > 0.85. Since the only difference between the two
    pipelines is the feature discovery method, this gap is entirely attributable to LLM reasoning.

    Random feature search baseline results:

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

    [1] Abboud, Ralph, et al. The surprising power of graph neural networks with random node initialization
    [2] Sato, Ryoma, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural networks.


    Comment: How sensitive is performance to prompt design? Minor paraphrasing or structural changes in prompts should be tested to determine whether results are stable or fragile.

    Rebuttal:
    We address prompt sensitivity at two levels.

    First, Appendix G.5 compares GPT-5-nano and LLaMA-3.3-70B — two model families with
    different architectures, tokenizers, and pretraining data — and finds comparable pruning
    performance (Figure 7). Mizrahi et al. (2024) show that prompt paraphrasing can cause
    significant rank changes across models; the fact that our two very different LLMs produce
    consistent results is therefore a meaningful signal that our pipeline is not exploiting
    prompt-specific phrasing artifacts.

    Second, the beam search architecture provides a structural safeguard against prompt
    sensitivity. At each iteration, β=3 candidate feature sets are expanded in parallel. If one
    expansion produces a poor feature set due to prompt variation or stochastic sampling, that
    branch is pruned and the remaining candidates continue. This is why beam search outperforms
    a simple feedback loop (Appendix F, Figure 5) — it tolerates bad expansions without
    cascading failure. Prompt fragility would require all β branches to fail simultaneously,
    which is unlikely given the diversity introduced by LLM sampling.

    [Mizrahi et al., 2024] State of What Art? A Call for Multi-Prompt LLM Evaluation.



    Comment: What is the variance across LLM runs? Because feature generation is stochastic, repeated runs with different seeds should quantify stability in pruning quality and runtime.

    Rebuttal:
    Figure 5 (Appendix F) shows variance across pipeline variants with five different random seeds and Figure 3 shows the average runtime of LLM2Prune.


    Comment: Are generated features interpretable or redundant? Feature-importance analyses should reveal whether the model is leveraging meaningful graph structure or compensating for noisy, overlapping features.

    Plan:
    Appendix B lists the discovered features per problem (e.g., closed neighborhood size for Maximum Cover,  degree to weight ratio, outgoing activation probability sum, and in–out degree
    difference for Influence Maximization under knapsack constraint). The generated features are easily interpretable and distinct. In fact, we observe LLM2Prune discards many proposed features (some of them are indeed overlapping) in the early stages of feature space search and tries to keep the minimal feature set of features.

    Comment: What mechanisms prevent overfitting to the synthetic graphs used during feature discovery? Cross-family transfer experiments would help demonstrate genuine generalization.

    Rebuttal:
    The cross-family transfer is already demonstrated in Table 1: features discovered on
    synthetic Holme-Kim (HK) graphs are evaluated directly on 8 real-world graphs spanning
    social, citation, and communication networks — with no retraining. Strong performance
    across all of them confirms that the discovered features capture generalizable structural
    properties rather than HK-specific artifacts.

    We deliberately chose HK graphs for feature discovery because they share key structural
    properties with real-world networks — scale-free degree distribution, clustering, and
    community structure — following established practice in graph algorithm benchmarking.
    This is precisely what prevents overfitting: the synthetic proxy is structurally similar
    to the target domain.

    We also identify the operational boundary of this generalization. When feature discovery
    is run on Erdős-Rényi (ER) graphs — which lack scale-free structure and clustering —
    performance on real-world graphs degrades significantly. This confirms that LLM2Prune
    generalizes across graphs within the same structural family, but not across graphs that
    are fundamentally different in structure. We will make this boundary explicit in the
    revised paper.

    Comment: Can the beam-search pipeline recover from poor early feature proposals? Controlled perturbation experiments could test resilience to suboptimal initial branches.

    Rebuttal:
    This is directly addressed by the ablation in Appendix F (Figure 5), which compares beam
    search against a simple feedback loop. Beam search consistently achieves higher and more
    stable performance because it maintains β=3 candidate paths: when one branch produces a
    poor feature proposal early on, the other branches continue, and subsequent expansions
    from the surviving good branches can recover.

    We note an important boundary: if all β branches simultaneously produce poor proposals —
    an unlikely but possible scenario — beam search degenerates into a simple feedback loop
    and inherits its failure to recover. Figure 5 empirically shows this degenerate regime
    does not occur in practice with β=3, as beam search consistently outperforms the feedback
    loop across all tested problems. The diversity introduced by parallel LLM sampling is
    sufficient to ensure at least one viable branch survives early iterations.

    Comment: Are there adversarial or degenerate graph constructions where pruning fails catastrophically? Identifying such cases would clarify the operational envelope of the method.

    Rebuttal:
    Yes, and we can identify it precisely. When feature discovery is performed on Erdős-Rényi
    (ER) graphs — which have homogeneous degree distributions and no clustering or community
    structure — the discovered features fail to transfer to real-world graphs such as Facebook
    or Twitter. Performance degrades significantly in this setting, as ER lacks the scale-free
    and clustered structure that makes HK a good proxy for real-world social networks.

    This constitutes the degenerate case and clarifies the operational envelope of LLM2Prune:
    the method generalizes across graphs that share structural properties with the training
    proxy (scale-free degree distribution, clustering, community structure), but not across
    graphs that are fundamentally different in structure. We will report this result explicitly
    in the revised paper as a stated limitation, alongside the HK → real-world transfer results
    in Table 1 that demonstrate successful generalization within the supported regime.

    Comment: What fraction of performance gain is attributable to LLM-driven feature discovery versus downstream classifier learning? Ablations that freeze or randomize components would help disentangle these effects.

    Rebuttal:
    The random feature search baseline reported under Comment 2 directly answers this. Both
    the baseline and LLM2Prune use the identical GNN architecture, training procedure, and
    evaluation pipeline — the only difference is how features are generated. Since the GNN
    is identical in both cases, any performance difference is purely attributable to feature
    quality. Random features yield substantially lower C, particularly on Influence
    Maximization (C as low as 0.46 vs. LLM2Prune's C > 0.85 per Table 1), with a more
    modest gap on MC and MaxCut. Note that freezing GNN weights while varying features is
    not a valid ablation since the weights are trained on the features themselves. The GNN
    contributes the learning capacity, but it is the LLM that determines what structural
    signals are worth learning from.










## Reviewer 1 (Reviewer_NLT4) 



---

## Reviewer 2 (Reviewer_4N39)



---

## Reviewer 3 (Reviewer_Q296) — Claims: NOT Validated



---

## Summary: What Actually Needs to Be Done

### Writing tasks (no new experiments needed):
- [ ] Move key feature analysis from Appendix B into main text (Section 4 or dedicated paragraph)
- [ ] Add absolute runtime table (supplement or replace Fig 3 radar chart)
- [ ] Expand Limitations section: discuss LLM failure modes and what happens with bad feature proposals
- [ ] Clarify that training data prep time is separate from inference time in runtime discussion
- [ ] Strengthen synthetic-to-real transfer explanation mechanistically

### New experiments needed:
- [ ] Report mean ± std over multiple independent runs (LLM variance)
- [ ] Add compute cost / wall-clock time for the full beam search feature discovery phase
- [ ] Add error analysis: for cases where we underperform classical methods, explain *why* (graph properties, problem structure)
- [ ] Add non-LLM feature baseline to ablation (e.g., random features, hand-crafted degree/clustering features) to isolate LLM value
- [ ] Synthetic difficulty experiment: vary Holme-Kim parameters or graph density and plot performance vs. difficulty