# # Cutting problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition
# In graph theory, a [cut](https://en.wikipedia.org/wiki/Cut_(graph_theory)) is a partition of the vertices of a graph into two disjoint subsets.
# It is closely related to the [Spin-glass problem](@ref) in physics.
# Finding the maximum cut is NP-Hard, where a maximum cut is a cut whose size is at least the size of any other cut,
# where the size of a cut is the number of edges (or the sum of weights on edges) crossing the cut.

using GenericTensorNetworks, Graphs

# In the following, we are going to defined an cutting problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ## Generic tensor network representation
# We can define the cutting problem with the [`MaxCut`](@ref) type as
maxcut = MaxCut(graph)

# The tensor network representation of the cutting problem can be obtained by
problem = GenericTensorNetwork(maxcut)

# ### Theory (can skip)
#
# We associated a vertex ``v\in V`` with a boolean degree of freedom ``s_v\in\{0, 1\}``.
# Then the maximum cutting problem can be encoded to tensor networks by mapping an edge ``(i,j)\in E`` to an edge matrix labelled by ``s_i`` and ``s_j``
# ```math
# B(x_i, x_j, w_{ij}) = \left(\begin{matrix}
#     1 & x_{i}^{w_{ij}}\\
#     x_{j}^{w_{ij}} & 1
# \end{matrix}\right),
# ```
# where ``w_{ij}`` is a real number associated with edge ``(i, j)`` as the edge weight.
# If and only if the bipartition cuts on edge ``(i, j)``,
# this tensor contributes a factor ``x_{i}^{w_{ij}}`` or ``x_{j}^{w_{ij}}``.
# Similarly, one can assign weights to vertices, which corresponds to the onsite energy terms in the spin glass.
# The vertex tensor is
# ```math
# W(x_i, w_i) = \left(\begin{matrix}
#     1\\
#     x_{i}^{w_i}
# \end{matrix}\right),
# ```
# where ``w_i`` is a real number associated with vertex ``i`` as the vertex weight.

# Its contraction time space complexity is ``2^{{\rm tw}(G)}``, where ``{\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.

# ## Solving properties
# ### Maximum cut size ``\gamma(G)``
max_cut_size = solve(problem, SizeMax())[]

# ### Counting properties
# ##### graph polynomial
# The graph polynomial defined for the cutting problem is
# ```math
# C(G, x) = \sum_{k=0}^{\gamma(G)} c_k x^k,
# ```
# where ``\gamma(G)`` is the maximum cut size, 
# ``c_k/2`` is the number of cuts of size ``k`` in graph ``G=(V,E)``.
# Since the variable ``x`` is defined on edges,
# the coefficients of the polynomial is the number of configurations having different number of anti-parallel edges.
max_config = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding one max cut solution
max_vertex_config = solve(problem, SingleConfigMax())[].c.data

max_cut_size_verify = cut_size(graph, max_vertex_config)

# You should see a consistent result as above `max_cut_size`.

show_graph(graph, locations; vertex_colors=[
        iszero(max_vertex_config[i]) ? "white" : "red" for i=1:nv(graph)], format=:svg)

# where red vertices and white vertices are separated by the cut.
