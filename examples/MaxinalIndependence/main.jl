# # Maximal independent set problem

# !!! note
#     This tutorial only covers the maximal independent set problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Introduction
using GraphTensorNetworks, Graphs

# Please check the docstring of [`MaximalIndependence`](@ref) for the definition of the maximal independence problem.
@doc MaximalIndependence

# In the following, we are going to defined an cutting problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# Then we define the maximal independent set problem as
problem = MaximalIndependence(graph);

# The tensor network structure is different from that of [`Independence`](@ref),
# Its tensor is defined on a vertex and its neighbourhood,
# and it makes the contraction of [`MaximalIndependence`](@ref) much harder.

# ## Solving properties

# ### Counting properties
# ##### maximal independence polynomial
max_config = solve(problem, GraphPolynomial())[]

# Since it only counts the maximal independent sets, the first several coefficients are 0.

# ### Configuration properties
# ##### finding all maximal independent set
max_edge_config = solve(problem, ConfigsAll())[]

# This result should be consistent with that given by the [Bron Kerbosch algorithm](https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm) on the complement of Petersen graph.
maximal_cliques = maximal_cliques(complement(graph))

# For sparse graphs, the generic tensor network approach is usually much faster and memory efficient.