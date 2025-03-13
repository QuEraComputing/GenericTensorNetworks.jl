# # Dominating Set Problem
#

# ## Overview
# A dominating set in graph theory is a subset D of vertices where every vertex in the graph
# either belongs to D or is adjacent to a vertex in D. The domination number γ(G) is the size
# of the smallest possible dominating set. This example demonstrates how to:
# * Find the domination number
# * Count and enumerate dominating sets
# * Compute the domination polynomial
# 
# We'll explore these concepts using the Petersen graph as our example.

using GenericTensorNetworks, Graphs

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# Define vertex layout for visualization
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ## Tensor Network Formulation
# We represent the dominating set problem using tensor networks:
dom = DominatingSet(graph)

# The problem has two components:
# 1. Domination constraints: Every vertex must be dominated
# 2. Optimization objective: Minimize the size of the dominating set
constraints(dom)
#
objectives(dom)

# Convert to tensor network representation with optimized contraction order:
problem = GenericTensorNetwork(dom; optimizer=TreeSA())

# ## Mathematical Background
# For a graph $G=(V,E)$, we assign a boolean variable $s_v \in \{0,1\}$ to each vertex $v$.
# The tensor network uses the following components:
#
# 1. Vertex Tensors: For vertex $v$ and its neighbors $N(v)$:
#    ```math
#    T(x_v)_{s_1,...,s_{|N(v)|},s_v} = 
#    \begin{cases}
#        0 & \text{if all } s \text{ values are } 0 \text{ (invalid configuration)} \\
#        1 & \text{if } s_v = 0 \text{ (vertex not in set)} \\
#        x_v^{w_v} & \text{otherwise (vertex in set)}
#    \end{cases}
#    ```
#
# Check the contraction complexity:
contraction_complexity(problem)

# ## Solution Analysis
# ### 1. Domination Polynomial
# The domination polynomial $D(G,x) = \sum_i d_i x^i$ counts dominating sets by size,
# where $d_i$ is the number of dominating sets of size $i$.
domination_polynomial = solve(problem, GraphPolynomial())[]

# ### 2. Minimum Dominating Set
# Find the domination number γ(G):
domination_number = solve(problem, SizeMin())[]

# Count minimum dominating sets:
counting_min_dominating_set = solve(problem, CountingMin())[]

# ### 3. Configuration Analysis
# Enumerate all minimum dominating sets:
min_configs = read_config(solve(problem, ConfigsMin())[])

# Verify solutions are valid:
all(c->is_dominating_set(graph, c), min_configs)

# Visualize all minimum dominating sets:
show_configs(graph, locations, reshape(collect(min_configs), 2, 5); padding_left=20)

# Note: For finding just one minimum dominating set, use SingleConfigMin property

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
