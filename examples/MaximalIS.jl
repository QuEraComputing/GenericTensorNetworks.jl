# # Maximal Independent Set Problem
#
# ## Overview
# A maximal independent set is an independent set that cannot be expanded by including more vertices.
# Unlike a maximum independent set, it's not necessarily the largest possible independent set.
# 
# This example demonstrates:
# * Finding maximal independent sets
# * Computing the independence polynomial
# * Finding minimum maximal independent sets
# * Comparing with traditional algorithms
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
# Define the maximal independent set problem:
maximalis = MaximalIS(graph)

# The problem consists of:
# 1. Independence constraints: No adjacent vertices can be selected
# 2. Maximality constraints: No more vertices can be added while maintaining independence
constraints(maximalis)
#
objectives(maximalis)

# Convert to tensor network representation:
problem = GenericTensorNetwork(maximalis)

# ## Mathematical Background
# For a graph $G=(V,E)$, we assign a boolean variable $s_v ∈ \{0,1\}$ to each vertex.
# For vertex $v$ with neighborhood $N(v)$, we define tensor:
#
# ```math
# T(x_v)_{s_1,...,s_{|N(v)|},s_v} = 
# \begin{cases}
# s_v x_v^{w_v} & \text{if all neighbors are 0 (v must be included for maximality)} \\
# 1-s_v & \text{if any neighbor is 1 (v must be excluded for independence)}
# \end{cases}
# ```
#
# Note: This tensor network is often more complex to contract than the regular
# independent set problem, as its complexity isn't directly tied to the graph's tree-width.
contraction_complexity(problem)

# ## Solution Analysis
# ### 1. Independence Polynomial
# The maximal independence polynomial $I_{\text{max}}(G,x) = \sum_i b_i x^i$ counts
# maximal independent sets by size, where $b_i$ is the number of sets of size $i$
maximal_indenpendence_polynomial = solve(problem, GraphPolynomial())[]

# ### 2. Independent Domination Number
# Find the size of the smallest maximal independent set:
independent_domination_number = solve(problem, SizeMin())[]

# Count minimum maximal independent sets:
counting_min_maximal_independent_set = solve(problem, CountingMin())[]

# ### 3. Configuration Analysis
# Find all maximal independent sets:
maximal_configs = read_config(solve(problem, ConfigsAll())[])

# Verify solutions are valid:
all(c->is_maximal_independent_set(graph, c), maximal_configs)

# Visualize all maximal independent sets:
show_configs(graph, locations, reshape(collect(maximal_configs), 3, 5); padding_left=20)

# ### 4. Comparison with Classical Algorithms
# Compare with Bron-Kerbosch algorithm on complement graph:
cliques = maximal_cliques(complement(graph))
# Note: For sparse graphs, our tensor network approach is typically faster
# and more memory efficient than Bron-Kerbosch.

# ### 5. Minimum Maximal Independent Sets
# Find all minimum maximal independent sets:
minimum_maximal_configs = read_config(solve(problem, ConfigsMin())[])

# Visualize minimum maximal independent sets:
show_configs(graph, locations, reshape(collect(minimum_maximal_configs), 2, 5); padding_left=20)

# Note: For finding just one minimum maximal independent set,
# use the SingleConfigMin property instead

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
