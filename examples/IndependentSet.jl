# # Independent set problem

# ## Problem definition
# In graph theory, an [independent set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)) is a set of vertices in a graph, no two of which are adjacent.
#
# In the following, we are going to solve the solution space properties of the independent set problem on the Petersen graph. To start, let us define a Petersen graph instance.

using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the [`show_graph`](@ref) function
## set the vertex locations manually instead of using the default spring layout
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 2.0, i) for i=0:4]..., [rot15(0.0, 1.0, i) for i=0:4]...]

show_graph(graph; locs=locations)

# The graphical display is available in the following editors
# * a [VSCode](https://github.com/julia-vscode/julia-vscode) editor,
# * a [Jupyter](https://github.com/JunoLab/Juno.jl) notebook,
# * or a [Pluto](https://github.com/fonsp/Pluto.jl) notebook,

# ## Generic tensor network representation
# The generic tensor network representation of the independent set problem can be constructed with [`IndependentSet`](@ref).
problem = IndependentSet(graph; optimizer=TreeSA());

# Here, the key word argument `optimizer` specifies the tensor network contraction order optimizer as a local search based optimizer [`TreeSA`](@ref).
# The resulting contraction order optimized tensor network is contained in the `code` field of `problem`.
#
# ### Theory (can skip)
# Let ``G=(V, E)`` be a graph with each vertex $v\in V$ associated with a weight ``w_v``.
# To reduce the independent set problem on it to a tensor network contraction, we first map a vertex ``v\in V`` to a label ``s_v \in \{0, 1\}`` of dimension ``2``, where we use ``0`` (``1``) to denote a vertex absent (present) in the set.
# For each vertex ``v``, we defined a parameterized rank-one tensor indexed by ``s_v`` as
# ```math
# W(x_v^{w_v}) = \left(\begin{matrix}
#     1 \\
#     x_v^{w_v}
#     \end{matrix}\right)
# ```
# where ``x_v`` is a variable associated with ``v``.
# Similarly, for each edge ``(u, v) \in E``, we define a matrix ``B`` indexed by ``s_u`` and ``s_v`` as
# ```math
# B = \left(\begin{matrix}
# 1  & 1\\
# 1 & 0
# \end{matrix}\right).
# ```
# Ideally, an optimal contraction order has a space complexity ``2^{{\rm tw}(G)}``,
# where ``{\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G`` (or `graph` in the code).
# We can check the time, space and read-write complexities by typing

timespacereadwrite_complexity(problem)

# The three return values are `log2` values of the the number of element-wise multiplication operations, the number elements in the largest tensor during contraction and the number of tensor element read-write operations.
# For more information about how to improve the contraction order, please check the [Performance Tips](@ref).

# ## Solution space properties

# ### Maximum independent set size ``\alpha(G)``
# We can compute solution space properties with the [`solve`](@ref) function, which takes two positional arguments, the problem instance and the wanted property.
maximum_independent_set_size = solve(problem, SizeMax())[]

# Here [`SizeMax`](@ref) means finding the solution with maximum set size.
# The return value has [`Tropical`](@ref) type. We can get its content by typing
maximum_independent_set_size.n

# ### Counting properties
# ##### Count all solutions and best several solutions
# We can count all independent sets with the [`CountingAll`](@ref) property.
count_all_independent_sets = solve(problem, CountingAll())[]

# The return value has type `Float64`.

# We can count the maximum independent sets with [`CountingMax`](@ref).
count_maximum_independent_sets = solve(problem, CountingMax())[]

# The return value has type [`CountingTropical`](@ref), which contains two fields.
# They are `n` being the maximum independent set size and `c` being the number of the maximum independent sets.

count_maximum_independent_sets.c

# Similarly, we can count independent sets of sizes ``\alpha(G)`` and ``\alpha(G)-1`` by feeding an integer positional argument to [`CountingMax`](@ref).
count_max2_independent_sets = solve(problem, CountingMax(2))[]

# The return value has type [`TruncatedPoly`](@ref), which contains two fields.
# They are `maxorder` being the maximum independent set size and `coeffs` being the number of independent sets having sizes ``\alpha(G)-1`` and ``\alpha(G)``.

count_max2_independent_sets.coeffs

# ##### Find the graph polynomial
# We can count the number of independent sets at any size, which is equivalent to finding the coefficients of an independence polynomial that defined as
# ```math
# I(G, x) = \sum_{k=0}^{\alpha(G)} a_k x^k,
# ```
# where ``\alpha(G)`` is the maximum independent set size, 
# ``a_k`` is the number of independent sets of size ``k``.
# The total number of independent sets is thus equal to ``I(G, 1)``.
# There are 3 methods to compute a graph polynomial, `:finitefield`, `:fft` and `:polynomial`.
# These methods are introduced in the docstring of [`GraphPolynomial`](@ref).
independence_polynomial = solve(problem, GraphPolynomial(; method=:finitefield))[]

# The return type is [`Polynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomial-2).
independence_polynomial.coeffs

# ### Configuration properties
# ##### Find one best solution
# We can use the bounded or unbounded [`SingleConfigMax`](@ref) to find one of the solutions with largest size.
# The unbounded (default) version uses a joint type of [`CountingTropical`](@ref) and [`ConfigSampler`](@ref) in computation,
# where `CountingTropical` finds the maximum size and `ConfigSampler` samples one of the best solutions.
# The bounded version uses the binary gradient back-propagation (see our paper) to compute the gradients.
# It requires caching intermediate states, but is often faster (on CPU) because it can use [`TropicalGEMM`](https://github.com/TensorBFS/TropicalGEMM.jl) (see [Performance Tips](@ref)).
max_config = solve(problem, SingleConfigMax(; bounded=false))[]

# The return value has type [`CountingTropical`](@ref) with its counting field having [`ConfigSampler`](@ref) type. The `data` field of [`ConfigSampler`](@ref) is a bit string that corresponds to the solution
single_solution = max_config.c.data

# This bit string should be read from left to right, with the i-th bit being 1 (0) to indicate the i-th vertex is present (absent) in the set.
# We can visualize this MIS with the following function.
show_graph(graph; locs=locations, vertex_colors=
    [iszero(single_solution[i]) ? "white" : "red" for i=1:nv(graph)])

# ##### Enumerate all solutions and best several solutions
# We can use bounded or unbounded [`ConfigsMax`](@ref) to find all solutions with largest-K set sizes.
# In most cases, the bounded (default) version is preferred because it can reduce the memory usage significantly.
all_max_configs = solve(problem, ConfigsMax(; bounded=true))[]

# The return value has type [`CountingTropical`](@ref), while its counting field having type [`ConfigEnumerator`](@ref). The `data` field of a [`ConfigEnumerator`](@ref) instance contains a vector of bit strings.

all_max_configs.c.data

# These solutions can be visualized with the [`show_gallery`](@ref) function.
show_gallery(graph, (1, length(all_max_configs.c)); locs=locations, vertex_configs=all_max_configs.c);

# We can use [`ConfigsAll`](@ref) to enumerate all sets satisfying the independence constraint.
all_independent_sets = solve(problem, ConfigsAll())[]

# The return value has type [`ConfigEnumerator`](@ref).

# ##### Sample solutions
# It is often difficult to store all configurations in a vector.
# A more clever way to store the data is using the sum product tree format.
all_independent_sets_tree = solve(problem, ConfigsAll(; tree_storage=true))[]

# The return value has the [`SumProductTree`](@ref) type. Its length corresponds to the number of configurations.
length(all_independent_sets_tree)

# We can use `Base.collect` function to create a [`ConfigEnumerator`](@ref) or use [`generate_samples`](@ref) to generate samples from it.

collect(all_independent_sets_tree)

generate_samples(all_independent_sets_tree, 10)
