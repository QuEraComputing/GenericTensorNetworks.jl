# # Vertex matching problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition
# A ``k``-matching in a graph ``G`` is a set of k edges, no two of which have a vertex in common.

using GenericTensorNetworks, Graphs

# In the following, we are going to defined a matching problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 2.0, i) for i=0:4]..., [rot15(0.0, 1.0, i) for i=0:4]...]

show_graph(graph; locs=locations)

# ## Generic tensor network representation
# We construct the tensor network for the matching problem by typing
problem = Matching(graph);
# ### Theory (can skip)
#
# Type [`Matching`](@ref) can be used for constructing the tensor network with optimized contraction order for a matching problem.
# We map an edge ``(u, v) \in E`` to a label ``\langle u, v\rangle \in \{0, 1\}`` in a tensor network,
# where 1 means two vertices of an edge are matched, 0 means otherwise.
# Then we define a tensor of rank ``d(v) = |N(v)|`` on vertex ``v`` such that,
# ```math
# W_{\langle v, n_1\rangle, \langle v, n_2 \rangle, \ldots, \langle v, n_{d(v)}\rangle} = \begin{cases}
#     1, & \sum_{i=1}^{d(v)} \langle v, n_i \rangle \leq 1,\\
#     0, & \text{otherwise},
# \end{cases}
# ```
# and a tensor of rank 1 on the bond
# ```math
# B_{\langle v, w\rangle} = \begin{cases}
# 1, & \langle v, w \rangle = 0 \\
# x, & \langle v, w \rangle = 1,
# \end{cases}
# ```
# where label ``\langle v, w \rangle`` is equivalent to ``\langle w,v\rangle``.
#

# ## Solving properties
# ### Maximum matching
# ### Configuration properties
max_matching = solve(problem, SizeMax())[]
# The largest number of matching is 5, which means we have a perfect matching (vertices are all paired).

# ##### matching polynomial
# The graph polynomial defined for the matching problem is known as the matching polynomial.
# Here, we adopt the first definition in the [wiki page](https://en.wikipedia.org/wiki/Matching_polynomial).
# ```math
# M(G, x) = \sum\limits_{k=1}^{|V|/2} c_k x^k,
# ```
# where ``k`` is the number of matches, and coefficients ``c_k`` are the corresponding counting.

matching_poly = solve(problem, GraphPolynomial())[]

# ## Configuration properties

# ##### one of the perfect matches
match_config = solve(problem, SingleConfigMax())[]

# Let us show the result by coloring the matched edges to red
show_graph(graph; locs=locations, edge_colors=
    [isone(match_config.c.data[i]) ? "red" : "black" for i=1:ne(graph)])

# where we use edges with red color to related pairs of matched vertices.
