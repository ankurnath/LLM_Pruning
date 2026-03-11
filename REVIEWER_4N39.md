# Reviewer 2 (Reviewer_4N39)

Thank you for your thoughtful review and insightful comments. We greatly appreciate your time and effort in evaluating our work.

> To what extent did the semi-supervised learning improve the performance? Are the other learning-based approaches used for comparison fully supervised or semi-supervised?

In our experiments, semi-supervised learning improves performance by approximately 2–3% in the combined metric compared to a purely supervised training setup. The other learning-based approaches (GCOMB-P and LeNSE) are reinforcement learning methods optimized via policy gradient rather than supervised or semi-supervised classifiers, so they do not fall into either supervision paradigm. 

---

> The information of the input graph itself is not given to LLM. It does not help accelerate the search?

Incorporating graph-level statistics as additional context could potentially guide the search toward features tailored to particular graph families. However, this would also require deciding which statistics are most informative; an interesting direction for future work is to use the LLM itself to help identify or select relevant graph statistics for conditioning the search.

---

> In runtime analysis (Fig 3), although learning based methods require the cost for preparing the training data, is it included in the reported time?

All runtimes in Fig. 3 report inference time only. We focus on inference time because it is the operationally relevant cost for pruning: once the feature-space search is completed, the selected classifier is applied repeatedly to new graph instances, and this per-instance cost determines practical scalability.

The beam search used in LLM2Prune is a one-time cost that can be amortized across all future instances of the same graph family. However, we acknowledge that reporting inference time alone does not reflect the full training overhead.

To provide additional context, we report the fine-tuning time per dataset for Maximum Coverage below:

| Dataset  | Nodes   | Edges   | Finetune Time (s) |
|----------|---------|---------|-------------------|
| Wiki     | 4,891   | 30,228  | 11.7              |
| Facebook | 3,847   | 26,470  | 12.0              |
| DBLP     | 63,004  | 41,994  | 14.2              |
| Slashdot | 47,546  | 140,566 | 17.2              |
| Twitter  | 55,827  | 134,229 | 17.6              |
| Deezer   | 48,870  | 149,460 | 18.2              |
| Skitter  | 147,604 | 110,952 | 21.9              |
| YouTube  | 185,193 | 179,257 | 30.5              |

---

> Explanations of the classifier should have been provided in more detail in the main methodology section. Because it does not appear until the experimental section, it is difficult for readers without prior background knowledge to understand the role of the classifier in the methods section.

We agree and will revise the methodology section to include a self-contained description of the classifier.

---

> Providing algorithm of the entire procedure would help readers.

We agree and will add a formal algorithm box covering the complete pipeline in the revised methodology section.

---

> I do not fully understand what 'we instead perform ... each instance' in the paragraph just before Section 4.1 indicates. Does this mean that the classifier is trained by a Holme-Kim graph during the beam search (and M is evaluated on the original training graph?), and after finishing the beam search, the selected classifier is trained on the original training graph? In this case, the final M of the selected classifier appears to be different from the value calculated during the beam search, which seems somewhat odd. I misunderstand something?

During beam search, both the GNN classifier and the metric M are evaluated entirely on synthetic HK graphs. This is intentional: the purpose of beam search is to discover which feature set best characterizes high-value nodes in a structural family resembling real-world graphs, without using any target graph data.

After beam search completes and the best feature set is selected, the classifier is fine-tuned on the actual training graph for a few epochs. The fine-tuned classifier will indeed have a different M value from the one computed during search, since it is now evaluated on a different graph. However, this is not problematic, as the beam search score serves only as a selection criterion among candidate feature sets, not as a prediction of final performance. The fine-tuning step adapts the classifier to the specific graph structure of the deployment domain, and the final performance is what is reported in Tables 1 and 3. We will clarify this two-stage process explicitly in the revised paper.

---

> I'm a bit confused about the definition of \cal{H}_d,i. Is it the set of the ancestors of (d,i)? If so, why not only \pi(d,i), but also the sum from \ell = 0 to d is required? (it seems some elements in the union are duplicated, in my current understanding).

H_{d,i} is the full ordered path from the root to node (d,i) in the beam search tree — i.e., the sequence of feature sets tried at each depth along that path. Each element in H_{d,i} is the feature set chosen at a distinct depth along the path. The reason we condition on the full path rather than only π(d,i) is that the LLM generates the next candidate feature set based on the entire history of what has been tried and discarded. With only the immediate parent, the LLM lacks context about features explored and rejected at earlier depths, and may re-propose them. Providing the full sequence gives the LLM richer context to avoid revisiting explored regions of the feature space — and in practice we observe this reduces redundant proposals during search.

---

> The definitions of metric M should be clarified in the paper. According to the code, C on a validation graph is seemingly used, but not mentioned in the paper.

We will add an explicit definition of metric M and clarification to the paper.

---


