# # Independent Set Problem
#
# ## Overview
# This example demonstrates how to solve the Independent Set problem using tensor networks.
# An independent set is a set of vertices in a graph where no two vertices are adjacent.
# We'll explore this problem using the Petersen graph as our example.
#
# ## Problem definition
# In graph theory, an [independent set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)) is a set of vertices in a graph, no two of which are adjacent.
#
# In the following, we are going to solve the solution space properties of the independent set problem on the Petersen graph. To start, let us define a Petersen graph instance.

using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the [`show_graph`](@ref) function
## set the vertex locations manually instead of using the default spring layout
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# The graphical display is available in the following editors
# * a [VSCode](https://github.com/julia-vscode/julia-vscode) editor,
# * a [Jupyter](https://github.com/JunoLab/Juno.jl) notebook,
# * or a [Pluto](https://github.com/fonsp/Pluto.jl) notebook,

# ## Tensor Network Formulation
# We represent the independent set problem using a tensor network approach.
# This allows us to efficiently compute various properties of the solution space.

iset = IndependentSet(graph)

# The problem has two main components:
# 1. Independence constraints: Ensure no adjacent vertices are selected
# 2. Optimization objective: Maximize the size of the independent set

constraints(iset)

#

objectives(iset)

# The tensor network representation of the independent set problem can be obtained by
problem = GenericTensorNetwork(iset; optimizer=TreeSA())
# where the key word argument `optimizer` specifies the tensor network contraction order optimizer as a local search based optimizer [`TreeSA`](@ref).

# Here, the key word argument `optimizer` specifies the tensor network contraction order optimizer as a local search based optimizer [`TreeSA`](@ref).
# The resulting contraction order optimized tensor network is contained in the `code` field of `problem`.
#
# ### Mathematical Background
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

contraction_complexity(problem)

# For more information about how to improve the contraction order, please check the [Performance Tips](@ref).

# ## Solution Space Analysis
#
# ### 1. Maximum Independent Set Size ($α(G)$)
# First, we compute the size of the largest independent set:
maximum_independent_set_size = solve(problem, SizeMax())[]
read_size(maximum_independent_set_size)

# ### 2. Counting Solutions
# We can analyze the solution space in several ways:
#
# #### a. Total Count
# Count all possible independent sets:
count_all_independent_sets = solve(problem, CountingAll())[]

# #### b. Maximum Solutions
# Count independent sets of maximum size:
count_maximum_independent_sets = solve(problem, CountingMax())[]
read_size_count(count_maximum_independent_sets)

# ## Configuration Analysis
#
# ### 1. Finding Optimal Solutions
# We can find a single optimal solution using SingleConfigMax:
max_config = solve(problem, SingleConfigMax(; bounded=false))[]
single_solution = read_config(max_config)

# Visualize the maximum independent set:
show_graph(graph, locations; format=:svg, vertex_colors=
    [iszero(single_solution[i]) ? "white" : "red" for i=1:nv(graph)])

# ### 2. Solution Enumeration
# We can enumerate all optimal solutions or generate samples:

# a. Find all maximum independent sets:
all_max_configs = solve(problem, ConfigsMax(; bounded=true))[]
_, configs_vector = read_size_config(all_max_configs)

# b. Store all independent sets efficiently using a tree structure:
all_independent_sets_tree = solve(problem, ConfigsAll(; tree_storage=true))[]

# Generate a sample of 10 random solutions:
generate_samples(all_independent_sets_tree, 10)
