# LLM2Prune Revision Log

Changes made in response to TMLR reviewer promises. All new text is marked with `\textcolor{blue}{...}` in the source files.

---

## method.tex

1. **Metric M definition** (~line 14, node tuple description)
   - Added explicit definition: M is the combined metric C = P_r · P_g evaluated on a held-out synthetic Holme-Kim validation graph.
   - Addresses: Reviewer 4N39 ("The definitions of metric M should be clarified in the paper.")

2. **H_{d,i} full-path rationale** (after H_{d,i} clarification paragraph)
   - Added sentence explaining why the full path from root is used rather than just the immediate parent: prevents LLM from re-proposing features explored and discarded at earlier depths.
   - Addresses: Reviewer 4N39 ("I'm a bit confused about the definition of H_{d,i}...")

3. **Classifier description** (after semi-supervised learning paragraph, ~line 76)
   - Added paragraph describing the downstream classifier as a 2-layer GCN and the use of GNNExplainer to extract feature-importance vectors.
   - Addresses: Reviewer 4N39 ("Explanations of the classifier should have been provided in more detail in the main methodology section.")

4. **Two-stage deployment clarification** (after "Finally, when beam search finishes" paragraph, ~line 100)
   - Added paragraph clarifying that beam search runs on synthetic graphs matching the target distribution, M serves only as a selection criterion, and the classifier is subsequently fine-tuned on the actual target training graph.
   - Addresses: Reviewer 4N39 ("I do not fully understand what 'we instead perform ... each instance' indicates.")

5. **Algorithm box** (after two-stage paragraph, ~line 105)
   - Added formal Algorithm 1 (label: alg:llm2prune) covering the full pipeline: initialization, expansion, semi-supervised training, beam pruning, fine-tuning, and inference.
   - Addresses: Reviewer 4N39 ("Providing algorithm of the entire procedure would help readers.")

---

## experiments.tex

6. **Discovered Features paragraph** (after "Applications" paragraph, ~line 62)
   - Added qualitative discussion of features discovered by LLM2Prune, emphasizing objective-aware descriptors: random cut expectation (MaxCut), average incoming/outgoing activation probability (IM), coverage per cost and degree-to-weight ratio (knapsack). Points to Appendix B for full lists.
   - Addresses: Reviewer Q296 ("A qualitative analysis of these features should be moved to or summarized within the main text.")

7. **ER failure added to Limitations** (appended to Limitations paragraph, ~line 117)
   - Added sentence stating that LLM2Prune degrades under cross-family distribution shift, with Erdos-Renyi graphs as the concrete degenerate case.
   - Addresses: Reviewers NLT4 and Q296 ("We will make this boundary explicit in the revised paper.")

---

## appendix.tex

8. **Runtime subsection** (replaced existing figure with two tables)
   - Removed: figure showing runtime breakdown as a bar plot.
   - Added Table (label: tab:runtime_absolute): Absolute inference runtimes for all algorithms on Maximum Coverage under size constraint. Includes fairness note about equal resources (SS uses additional threads).
   - Added Table (label: tab:runtime_breakdown): LLM2Prune runtime breakdown into feature extraction and GNN inference per dataset. Data sourced from Maximum Coverage result pkl files.
   - Addresses: Reviewer Q296 ("Provide a standard tabular format detailing the absolute runtime values.") and Reviewer 4N39 ("Is inference time included in the reported time?")

---

## Skipped

- Synthetic difficulty experiment tables (Reviewer NLT4) — not needed per author decision.
- Random feature baseline table (Reviewer NLT4) — skipped per author decision.
- Fine-tuning time table (Reviewer 4N39) — skipped per author decision.
