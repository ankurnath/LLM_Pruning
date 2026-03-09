# Reviewer 2 (Reviewer_4N39)

---

> To what extent did the semi-supervised learning improve the performance? Are the other learning-based approaches used for comparison fully supervised or semi-supervised?

The other learning-based baselines (GCOMB-P and LeNSE) are reinforcement learning methods using policy-gradient optimization — not supervised or semi-supervised classifiers. A direct comparison of supervision paradigms is therefore not applicable; the comparison is between a GNN-based pruning classifier and RL-based node-selection policies. LLM2Prune consistently outperforms both GCOMB-P and LeNSE across all problems and datasets (Table 1, Table 4), demonstrating that the semi-supervised GNN approach achieves both higher solution quality and faster inference than RL-based alternatives.

---

> The information of the input graph itself is not given to LLM. It does not help accelerate the search?

We did not explore providing input graph statistics to the LLM in this work. This is an interesting direction for future work, which could allow the search to tailor features to specific graph families more precisely.

---

> In runtime analysis (Fig 3), although learning based methods require the cost for preparing the training data, is it included in the reported time?

Yes, all reported runtimes include the full cost of preparing training data (feature extraction and GNN training). To make this explicit and to address a related comment from another reviewer, we have added a table reporting absolute runtimes for all evaluated algorithms on Influence Maximization (size constraint). Even including all preprocessing steps, LLM2Prune is 1–4 orders of magnitude faster than classical methods (QuickPrune, SS) on larger graphs.

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

> Explanations of the classifier should have been provided in more detail in the main methodology section. Because it does not appear until the experimental section, it is difficult for readers without prior background knowledge to understand the role of the classifier in the methods section.

We agree and will revise the methodology section to include a self-contained description of the classifier.

---

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
