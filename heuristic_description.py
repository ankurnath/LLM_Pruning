heuristic_description = {
    "Maximum Coverage": "Start with an empty set. At each step, compute the number of "
                        "new vertices each candidate would cover. "
                        "Select the vertex with the maximum gain and add it to the set. "
                        "Update the covered vertices. Repeat until k vertices are chosen.",

    "Maximum Cut": "Start with an empty set. At each step, for each candidate vertex, compute the increase in cut value "
    "if it were added to the set. "
    "Select the vertex that maximizes this increase and add it to the set. Repeat until k vertices are chosen.",

    "Influence Maximization": "Start with an empty seed set. A reverse reachable (RR) set is the set of nodes that can reach a randomly chosen node in a simulated diffusion process. Generate many RR sets from the graph. "
    "At each step, choose the vertex that covers the largest number of uncovered RR sets. "
    "Add it to the seed set and mark those RR sets as covered. Repeat until k vertices are chosen."
}
