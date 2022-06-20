# # Binary paint shop problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem Definition
# The [binary paint shop problem](http://m-hikari.com/ams/ams-2012/ams-93-96-2012/popovAMS93-96-2012-2.pdf) is defined as follows:
# we are given a ``2m`` length sequence containing ``m`` cars, where each car appears twice.
# Each car need to be colored red in one occurrence, and blue in the other.
# We need to choose which occurrence for each car to color with which color — the goal is to minimize the number of times we need to change the current color.

# In the following, we use a character to represent a car,
# and defined a binary paint shop problem as a string that each character appear exactly twice.

using GenericTensorNetworks, Graphs

sequence = collect("iadgbeadfcchghebif")

# We can visualize this paint shop problem as a graph
rot(a, b, θ) = cos(θ)*a + sin(θ)*b, cos(θ)*b - sin(θ)*a

locations = [rot(0.0, 2.0, -0.25π - 1.5*π*(i-0.5)/length(sequence)) for i=1:length(sequence)]

graph = path_graph(length(sequence))
for i=1:length(sequence) 
    j = findlast(==(sequence[i]), sequence)
    i != j && add_edge!(graph, i, j)
end

show_graph(graph; locs=locations, texts=string.(sequence), format=:svg, edge_colors=
    [sequence[e.src] == sequence[e.dst] ? "blue" : "black" for e in edges(graph)])

# Vertices connected by blue edges must have different colors,
# and the goal becomes a min-cut problem defined on black edges.

# ## Generic tensor network representation
# Let us construct the problem instance as bellow.
problem = PaintShop(sequence);

# ### Theory (can skip)
# Type [`PaintShop`](@ref) can be used for constructing the tensor network with optimized contraction order for solving a binary paint shop problem.
# To obtain its tensor network representation, we associating car ``c_i`` (the ``i``-th character in our example) with a degree of freedom ``s_{c_i} \in \{0, 1\}``,
# where we use ``0`` to denote the first appearance of a car is colored red and ``1`` to denote the first appearance of a car is colored blue.
# For each black edges ``(i, i+1)``, we define an edge tensor labeled by ``(s_{c_i}, s_{c_{i+1}})`` as follows:
# If both cars on this edge are their first or second appearance
# ```math
# B^{\rm parallel} = \left(\begin{matrix}
# x & 1 \\
# 1 & x \\
# \end{matrix}\right),
# ```
#
# otherwise,
# ```math
# B^{\rm anti-parallel} = \left(\begin{matrix}
# 1 & x \\
# x & 1 \\
# \end{matrix}\right).
# ```
# It can be understood as, when both cars are their first appearance,
# they tend to have the same configuration so that the color is not changed.
# Otherwise, they tend to have different configuration to keep the color unchanged.

# ### Counting properties
# ##### graph polynomial
# The graph polynomial defined for the maximal independent set problem is
# ```math
# I_{\rm max}(G, x) = \sum_{k=0}^{\alpha(G)} b_k x^k,
# ```
# where ``b_k`` is the number of maximal independent sets of size ``k`` in graph ``G=(V, E)``.

max_config = solve(problem, GraphPolynomial())[]

# Since it only counts the maximal independent sets, the first several coefficients are 0.

# ### Counting properties
# ##### graph polynomial
# The graph polynomial of the binary paint shop problem in our convention is defined as
# ```math
# P(G, x) = \sum_{k=0}^{\delta(G)} p_k x^k 
# ```
# where ``p_k`` is the number of possible coloring with number of color changes ``2m-1-k``.
paint_polynomial = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding best solutions
best_configs = solve(problem, ConfigsMin())[]

# One can see to identical bitstrings corresponding two different vertex configurations, they are related to bit-flip symmetry.

painting1 = paint_shop_coloring_from_config(problem, best_configs.c.data[1])

show_graph(graph; locs=locations, format=:svg, texts=string.(sequence),
    edge_colors=[sequence[e.src] == sequence[e.dst] ? "blue" : "black" for e in edges(graph)],
    vertex_colors=[isone(c) ? "red" : "black" for c in painting1], vertex_text_color="white")

# Since we have different choices of initial color, the number of best solution is 2.

# The following function will check the solution and return you the number of color switches
num_paint_shop_color_switch(sequence, painting1)
