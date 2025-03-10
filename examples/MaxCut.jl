# # Maximum Cut Problem
#
# ## Overview
# A cut in graph theory divides vertices into two disjoint subsets. The size of a cut
# is measured by the number of edges (or sum of edge weights) that cross between the subsets.
# The Maximum Cut (MaxCut) problem seeks to find a cut with the largest possible size.
#
# Key concepts covered:
# * Finding maximum cuts
# * Computing cut polynomials
# * Visualizing cut configurations
#
# This example uses the Petersen graph to demonstrate these concepts.

using GenericTensorNetworks, Graphs

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# Define vertex layout for visualization
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ## Tensor Network Formulation
# Define the MaxCut problem using tensor networks:
maxcut = MaxCut(graph)

# The objective is to maximize the number of edges crossing the cut

objectives(maxcut)

# Convert to tensor network representation:
problem = GenericTensorNetwork(maxcut)

# ## Mathematical Background
# For a graph $G=(V,E)$, we assign a boolean variable $s_v ∈ \{0,1\}$ to each vertex,
# indicating which subset it belongs to. The tensor network uses:
#
# For edge $(i,j)$ with weight $w_{ij}$:
# ```math
# B(x_i, x_j, w_{ij}) = \begin{pmatrix}
#    1 & x_i^{w_{ij}} \\
#    x_j^{w_{ij}} & 1
# \end{pmatrix}
# ```
# * Contributes $x_i^{w_{ij}}$ or $x_j^{w_{ij}}$ when vertices are in different subsets
#
# The contraction complexity is $O(2^{tw(G)})$, where $tw(G)$ is the graph's tree-width.
# ## Solution Analysis
# ### 1. Maximum Cut Size
# Find the size of the maximum cut:
max_cut_size = solve(problem, SizeMax())[]

# ### 2. Cut Polynomial
# The cut polynomial $C(G,x) = \sum_i c_i x^i$ counts cuts by size,
# where $c_i/2$ is the number of cuts of size $i$
max_config = solve(problem, GraphPolynomial())[]

# ### 3. Maximum Cut Configuration
# Find one maximum cut solution:
max_vertex_config = read_config(solve(problem, SingleConfigMax())[])

# Verify the cut size matches our earlier computation:
max_cut_size_verify = cut_size(graph, max_vertex_config)

# Visualize the maximum cut:
show_graph(graph, locations; vertex_colors=[
    iszero(max_vertex_config[i]) ? "white" : "red" for i=1:nv(graph)], format=:svg)
# Note: Red and white vertices represent the two disjoint subsets of the cut


# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
