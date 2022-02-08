# # Maximal independent set problem

# !!! note
#     This tutorial only covers the maximal independent set problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Problem definition
using GraphTensorNetworks, Graphs

# In graph theory, a [maximal independent set](https://en.wikipedia.org/wiki/Maximal_independent_set) is an independent set that is not a subset of any other independent set.
# It is different from maximum independent set because it does not require the set to have the max size.

# In the following, we are going to solve the maximal independent set problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# ## Tensor network representation
# For a vertex ``v\in V``, we define a boolean degree of freedom ``s_v\in\{0, 1\}``.
# We defined the restriction on its neighbourhood ``N[v]``:
# ```math
# T(x_v)_{s_1,s_2,\ldots,s_{|N(v)|},s_v} = \begin{cases}
#     s_vx_v & s_1=s_2=\ldots=s_{|N(v)|}=0,\\
#     1-s_v& \text{otherwise}.
# \end{cases}
# ```
# Intuitively, it means if all the neighbourhood vertices are not in ``I_{m}``, i.e., ``s_1=s_2=\ldots=s_{|N(v)|}=0``, then ``v`` should be in ``I_{m}`` and contribute a factor ``x_{v}``,
# otherwise, if any of the neighbourhood vertices is in ``I_{m}``, then ``v`` cannot be in ``I_{m}``.
# We construct the tensor network for the maximal independent set problem as
problem = MaximalIndependentSet(graph);

# Its contraction time space complexity of a [`MaximalIndependentSet`](@ref) instance is no longer determined by the tree-width of the original graph ``G``.
# It is often harder to contract this tensor network than to contract the one for regular independent set problem.

# ## Solving properties

# ### Counting properties
# ##### maximal independence polynomial
# The graph polynomial defined for the maximal independent set problem is
# ```math
# I_{\rm max}(G, x) = \sum_{k=0}^{\alpha(G)} b_k x^k,
# ```
# where ``b_k`` is the number of maximal independent sets of size ``k`` in graph ``G=(V, E)``.

max_config = solve(problem, GraphPolynomial())[]

# Since it only counts the maximal independent sets, the first several coefficients are 0.

# ### Configuration properties
# ##### finding all maximal independent set
max_edge_config = solve(problem, ConfigsAll())[]

# This result should be consistent with that given by the [Bron Kerbosch algorithm](https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm) on the complement of Petersen graph.
maximal_cliques = maximal_cliques(complement(graph))

# For sparse graphs, the generic tensor network approach is usually much faster and memory efficient.