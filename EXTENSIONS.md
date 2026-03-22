# LLM2Prune: Research Extensions

## 1. Multi-Dimensional Knapsack Problem (MKP)

### Problem Formulation
```
maximize:    sum(v_i * x_i)
subject to:  sum(w_di * x_i) <= C_d   for each dimension d
             x_i ∈ {0, 1}             for each item i
```

### Why MKP
- Natural extension of budget-constrained CO problems
- Multiple interacting capacity dimensions create complex structure that classical heuristics handle poorly
- LLM-discovered features could capture cross-dimensional interactions that hand-crafted features miss
- Clean experimental setup: Gurobi provides ground truth optimal solutions for training labels

### Proposed Pipeline
1. Use LLM-discovered features + RF/XGBoost to predict which items are likely in the optimal solution
2. Prune unlikely items from the instance
3. Hand reduced problem to Gurobi
4. Gurobi solves faster on the pruned instance

### Experimental Baselines
- Gurobi on full instance (optimality ceiling)
- Gurobi with time limit (practical baseline)
- LLM2Prune + Gurobi on pruned instance (proposed approach)

### Benchmark Instances
- OR-Library standard MKP benchmarks (widely used, easy to compare against published results)

---

## 2. LLM-Discovered Features for MKP

### Item-Level Features
- Value-to-weight ratio per dimension (v/w_d for each dimension d)
- Aggregate value-to-total-weight ratio (sum across dimensions)
- Max weight ratio across dimensions (bottleneck dimension)
- Weight variance across dimensions (measures how "balanced" an item is)
- Normalized value (value / max value)
- Dominance score: how many other items does this item dominate

### Budget/Capacity Features
- Weight fraction per dimension (w_d / C_d — how much of each capacity it consumes)
- Max capacity consumption across dimensions
- Slack consumption: after adding this item, how much slack remains per dimension
- Critical dimension indicator: which dimension becomes tightest if item is added

### Relational Features (between items)
- Compatibility score: how well does this item pair with others (joint capacity consumption)
- Conflict score: how many other items become infeasible if this item is selected
- Marginal value: expected value gain given current partial solution

### Global/Instance-Level Features
- Tightness ratio per dimension (sum of all weights / capacity)
- Efficiency spread: variance of value-to-weight ratios across all items
- Correlation between value and weight dimensions
- Number of binding constraints (near-tight dimensions)

---

## 3. Non-GNN Downstream Models

### Motivation
- MKP is naturally framed as **binary classification on items** — each item gets a feature vector, label = in solution or not
- RF/XGBoost naturally output item rankings via predicted probabilities — apply budget by taking top-k predictions
- If LLM features + RF/XGBoost ≈ LLM features + GNN, features are the key contributor (not model architecture)
- Cheaper to train, no graph construction needed, faster inference on large instances

### Models to Compare
| Model | Why |
|---|---|
| Random Forest | Built-in feature importance, interpretable |
| XGBoost/LightGBM | Strong gradient boosting baseline |
| Linear classifier | Tests if features are linearly separable |
| MLP | Non-graph neural baseline |
| GNN (existing) | Current approach |

### Key Finding Either Way
- RF/XGBoost competitive → features are what matter, GNN not necessary
- GNN significantly better → combination of LLM features + message passing is the key

---

## 4. Other Budget-Constrained CO Extensions

### Near-Term (minimal setup change)
- **Heterogeneous-cost Influence Maximization**: nodes have different seeding costs, budget limits total spend (direct generalization of existing IM work)
- **Budgeted Network Dismantling**: remove at most k nodes to maximally fragment a network
- **Budgeted Sensor Placement**: place k sensors to maximize coverage under cost constraints

### Longer-Term
- **Orienteering Problem**: traveler has time budget, visits nodes to collect rewards — combines routing and budget
- **Graph Interdiction**: attacker has budget to remove edges/nodes to disrupt network flow — adversarial setting
- **Multi-dimensional knapsack on graphs**: items have graph structure in addition to capacity constraints

---

## 5. Broader Framework Positioning

The extended work positions LLM2Prune as a **general-purpose feature engineering layer for learned CO solvers**, not a method tied to specific problems. The key message:

> Given any CO problem, LLM2Prune automatically discovers high-quality structural features that enable lightweight classifiers (RF, XGBoost, GNN) to prune the search space, making exact solvers like Gurobi tractable on large instances.
