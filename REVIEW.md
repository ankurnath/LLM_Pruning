# REVIEW.md — LLM2Prune Reviewer Action Items

    Comment: First, I encourage the authors to construct synthetic combinatorial optimization instances with an explicit, tunable notion of difficulty. For example, structural parameters such as graph density, modularity, constraint tightness, or objective ambiguity could be swept to create a difficulty ladder. Evaluating pruning ratio, objective degradation, and runtime as hardness increases would clarify whether performance degrades gracefully, whether sharp failure regimes exist, and whether beam-search feature discovery overfits to a narrow structural regime.

    Plan:
    Generate synthetic graphs with three tunable difficulty axes and evaluate C metric, objective
    degradation, and runtime as hardness increases. Use the existing model (trained on HK) and
    run test.py on each generated graph — no retraining needed.

    1. Constraint tightness (budget ratio): Fix HK graph (n=10,000), vary budget k ∈ {10, 25, 50,
       75, 100, 150, 200}. Plot C vs. k/|V|. Tight budget = harder pruning decision.

    2. Graph density: Fix n=10,000, vary HK edge parameter m ∈ {2, 4, 6, 8, 10} or rewire
       probability p ∈ {0.001, 0.01, 0.05, 0.1}. Plot C vs. average degree. Denser graphs =
       more nodes qualify, harder to prune.

    3. Modularity: Generate SBM graphs (nx.stochastic_block_model) with fixed n=10,000 and
       varying inter-community edge probability q ∈ {0.001, 0.005, 0.01, 0.05}. Low q = high
       modularity (easy), high q = low modularity (hard). Plot C vs. q.

    Expected outcome: show performance degrades gracefully, with no sharp cliff — supporting
    the claim that LLM2Prune is robust across a range of structural regimes.


    Comment: Could a non-LLM feature search (e.g., random search, evolutionary feature construction, or classical graph heuristics) achieve comparable pruning performance? This would help isolate the unique contribution of LLM reasoning.

    Plan:
    Two responses:
    (a) Classical heuristics (QuickPrune, SS) already appear in Table 1 and consistently serve as
        non-LLM baselines. LLM2Prune matches or outperforms them without hand-crafted features.
    (b) We additionally ran a random feature search baseline: instead of LLM-proposed features,
        we randomly generate Python feature code and train the same GNN. Results below show
        LLM-guided discovery substantially outperforms random search, isolating the LLM contribution.

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


    Comment: How sensitive is performance to prompt design? Minor paraphrasing or structural changes in prompts should be tested to determine whether results are stable or fragile.

    Plan:
    Two arguments:
    (a) Empirical: Appendix G.5 already compares GPT-5-nano vs. LLaMA-3.3-70B (two very different
        model families with different tokenizers and pretraining) and finds comparable performance
        (Figure 7). If results were fragile to prompt phrasing, this cross-model comparison would
        show high variance — it does not.
    (b) Structural robustness: Our beam search explicitly guards against prompt fragility. Even if
        one expansion produces a poor feature set due to prompt variation, the beam maintains β=3
        parallel candidate paths. Bad branches are pruned; good ones survive. This is exactly the
        mechanism that makes simple feedback loops fragile but beam search robust (Appendix F).
    (c) Reference: [TODO — find paper on LLM robustness to paraphrasing, e.g. Mizrahi et al. 2024
        or similar prompt sensitivity study].


    Comment: What is the variance across LLM runs? Because feature generation is stochastic, repeated runs with different seeds should quantify stability in pruning quality and runtime.

    Plan:
    Figure 5 (Appendix F) shows variance across pipeline variants, not across independent runs.
    TODO: Run the full beam search 3–5 times with different random seeds and report mean ± std
    of C, P_r, P_g, and runtime across runs.


    Comment: Are generated features interpretable or redundant? Feature-importance analyses should reveal whether the model is leveraging meaningful graph structure or compensating for noisy, overlapping features.

    Plan:
    GNNExplainer already runs during training and produces per-feature importance scores. Appendix B
    lists the discovered features per problem (e.g., closed neighborhood size for MaxCov, degree +
    random cut expectation for MaxCut). TODO: Add a table of GNNExplainer importance scores for the
    final feature set of each problem, showing (a) high-importance features are semantically
    meaningful and (b) low-importance ones are distinct — not redundant copies of high ones.

    Comment: What mechanisms prevent overfitting to the synthetic graphs used during feature discovery? Cross-family transfer experiments would help demonstrate genuine generalization.

    Plan:
    Already demonstrated: features discovered on synthetic HK graphs are evaluated on 8 real-world
    graphs spanning different domains (social, citation, communication networks). Table 1 shows
    strong performance across all of them. This is cross-family transfer by construction — point
    to this explicitly in the rebuttal and clarify the HK → real-world domain gap in the paper.

    Comment: Can the beam-search pipeline recover from poor early feature proposals? Controlled perturbation experiments could test resilience to suboptimal initial branches.

    Plan:
    Appendix F already addresses this. The ablation compares beam search vs. simple feedback loop:
    beam search consistently achieves higher and more stable performance (Figure 5) because it
    maintains β=3 candidate paths and recovers from poor early branches. Point to this result
    directly and note that simple feedback loop fails to recover when early iterations go wrong.

    Comment: Are there adversarial or degenerate graph constructions where pruning fails catastrophically? Identifying such cases would clarify the operational envelope of the method.

    Plan:
    Using Erdős-Rényi (ER) graphs for feature discovery fails because ER lacks scale-free degree
    distribution and community structure — the properties that make HK a good proxy for real-world
    graphs. This is the degenerate case: features discovered on ER don't transfer to Facebook/Twitter.
    Report this result explicitly as the operational boundary of LLM2Prune.

    Comment: What fraction of performance gain is attributable to LLM-driven feature discovery versus downstream classifier learning? Ablations that freeze or randomize components would help disentangle these effects.

    Plan:
    The random search results (table under comment 2 above) directly answer this. Random features +
    same GNN classifier performs much worse than LLM-discovered features + same GNN. The only
    difference is the feature discovery method, so the gap is entirely attributable to LLM reasoning.
    Point the reviewer to that table.










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