# # Vertex Matching Problem
#
# ## Overview
# A k-matching in a graph is a set of k edges where no two edges share a common vertex.
# A perfect matching occurs when every vertex in the graph is matched. This example demonstrates:
# * Finding maximum matchings
# * Computing the matching polynomial
# * Visualizing matching configurations
#
# We'll explore these concepts using the Petersen graph.

using GenericTensorNetworks, Graphs

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# Define vertex layout for visualization
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ## Tensor Network Formulation
# Define the matching problem using tensor networks:
matching = Matching(graph)

# The problem consists of:
# 1. Matching constraints: No vertex can be matched more than once
# 2. Optimization objective: Maximize the number of matches
constraints(matching)

#

objectives(matching)

# Convert to tensor network representation:
problem = GenericTensorNetwork(matching)

# ## Mathematical Background
# For a graph $G=(V,E)$, we assign a boolean variable $s_{e} \in \{0,1\}$ to each edge $e$,
# where $1$ indicates the vertices are matched.
#
# The network uses two types of tensors:
#
# 1. Vertex Tensors: For vertex $v$ with incident edges $e_1,...,e_k$:
#    ```math
#    W_{s_{e₁},...,s_{eₖ}} = \begin{cases}
#        1 & \text{if } \sum_{i=1}^k s_{e_i} \leq 1\\
#        0 & \text{otherwise}
#    \end{cases}
#    ```
#    This ensures at most one incident edge is selected (at most one match per vertex)
#
# 2. Edge Tensors: For edge $e$:
#    ```math
#    B_{s_e} = \begin{cases}
#        1 & \text{if } s_e = 0\\
#        x & \text{if } s_e = 1
#    \end{cases}
#    ```
#    This assigns weight $x$ to matched edges and $1$ to unmatched edges

# ## Solution Analysis
# ### 1. Maximum Matching
# Find the size of the maximum matching:
max_matching = solve(problem, SizeMax())[]
read_size(max_matching)
# Note: A maximum matching size of 5 indicates a perfect matching exists
# (all vertices are paired)

# ### 2. Matching Polynomial
# The matching polynomial $M(G,x) = \sum_i c_i x^i$ counts matchings by size,
# where $c_i$ is the number of $i$-matchings in $G$
matching_poly = solve(problem, GraphPolynomial())[]
read_size_count(matching_poly)

# ### 3. Perfect Matching Visualization
# Find one perfect matching configuration:
match_config = solve(problem, SingleConfigMax())[]
size, config = read_size_config(match_config)

# Visualize the matching by highlighting matched edges in red:
show_graph(graph, locations; format=:svg, edge_colors=
    [isone(read_config(match_config)[i]) ? "red" : "black" for i=1:ne(graph)])
# Red edges indicate pairs of matched vertices

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
