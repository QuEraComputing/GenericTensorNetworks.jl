# # Independent set problem

# ## Problem definition
# In graph theory, an [independent set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)) is a set of vertices in a graph, no two of which are adjacent.
#
# In the following, we are going to solve the solution space properties of the independent set problem on the Petersen graph. To start, let us define a Petersen graph instance.

using GraphTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the [`show_graph`](@ref) function
## set the vertex locations manually instead of using the default spring layout
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# The graphical display is available in the following editors
# * a [VSCode](https://github.com/julia-vscode/julia-vscode) editor,
# * a [Jupyter](https://github.com/JunoLab/Juno.jl) notebook,
# * or a [Pluto](https://github.com/fonsp/Pluto.jl) notebook,

# ## Generic tensor network representation
# ### Theory (can skip)
#
# To reduce the independent set problem on ``G=(V, E)`` to a tensor network contraction, we first map a vertex ``v\in V`` to a label ``s_v \in \{0, 1\}`` of dimension ``2``, where we use ``0`` (``1``) to denote a vertex absent (present) in the set.
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

# ### Define a problem instance
# The generic tensor network representation of the independent set problem can be constructed with [`IndependentSet`](@ref).
problem = IndependentSet(graph; optimizer=TreeSA());

# Here, the key word argument `optimizer` specifies the tensor network contraction order optimizer as a local search based optimizer [`TreeSA`](@ref).
# The resulting contraction order optimized tensor network is contained in the `code` field of `problem`.
# Ideally, an optimal contraction order has a space complexity ``2^{{\rm tw}(G)}``,
# where ``{\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G`` (or `graph` in the code).
# We can check the time, space and read-write complexities by typing

timespacereadwrite_complexity(problem)

# The three return values are `log2` values of the the number of element-wise multiplication operations, the number elements in the largest tensor during contraction and the number of tensor element read-write operations.
# For more information about how to improve the contraction order, please check the [Performance Tips](@ref).

# ## Solving properties

# ### Maximum independent set size ``\alpha(G)``
# We can compute solution space properties with the [`solve`](@ref) function, which takes two positional arguments, the problem instance and the wanted property.
maximum_independent_set_size = solve(problem, SizeMax())[]

# Here [`SizeMax`](@ref) means finding the solution with maximum set size.
# The return value has [`Tropical`](@ref) type. We can get its content by typing
maximum_independent_set_size.n

# ### Counting properties
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

# ##### graph polynomial
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

# We can use bounded or unbounded [`ConfigsMax`](@ref) to find all solutions with largest-K set sizes.
# In most cases, the bounded (default) version is prefered because it can reduce the memory usage significantly.
all_max_configs = solve(problem, ConfigsMax(; bounded=true))[]

# The return value has type [`CountingTropical`](@ref), while its counting field having type [`ConfigEnumerator`](@ref). The `data` field of a [`ConfigEnumerator`](@ref) instance contains a vector of bit strings.

all_max_configs.c.data

# Let us visualize the solutions with the visualization package [`Compose`](https://github.com/GiovineItalia/Compose.jl).
using Compose

m = length(all_max_configs.c)

imgs = ntuple(k->show_graph(graph;
                    locs=locations, scale=0.25,
                    vertex_colors=[iszero(all_max_configs.c[k][i]) ? "white" : "red"
                    for i=1:nv(graph)]), m);

Compose.set_default_graphic_size(18cm, 4cm)

Compose.compose(context(),
     ntuple(k->(context((k-1)/m, 0.0, 1.2/m, 1.0), imgs[k]), m)...)

# We can use [`ConfigsAll`](@ref) to enumerate all sets satisfying the independence constraint.
all_independent_sets = solve(problem, ConfigsAll())[]

# The return value has type [`ConfigEnumerator`](@ref).

# It is often difficult to store all configurations in a vector.
# A more clever way to store the data is using the sum product tree format.
all_independent_sets_tree = solve(problem, ConfigsAll(; tree_storage=true))[]

# The return value has the [`SumProductTree`](@ref) type. Its length corresponds to the number of configurations.
length(all_independent_sets_tree)

# We can use `Base.collect` function to create a [`ConfigEnumerator`](@ref) or use [`generate_samples`](@ref) to generate samples from it.

collect(all_independent_sets_tree)

generate_samples(all_independent_sets_tree, 10)

# ## Save and load configurations
# We can use [`save_configs`](@ref) and [`load_configs`](@ref) to save and read a [`ConfigEnumerator`](@ref) instance to the disk.
filename = tempname()

save_configs(filename, all_independent_sets; format=:binary)

loaded_sets = load_configs(filename; format=:binary, bitlength=10)

# !!! note
#     When loading data, one needs to provide the `bitlength` if the data is saved in binary format.
#     Because the bitstring length is not stored.
#
# For the [`SumProductTree`](@ref) type, one can use [`save_sumproduct`](@ref) and [`load_sumproduct`](@ref) to save and load serialized data.

save_sumproduct(filename, all_independent_sets_tree)

loaded_sets_tree = load_sumproduct(filename)

# ##### Loading configurations from python
# To loading configurations from file in the `:binary` format in python.
# We suggest using the following script to unpack the data correctly.
# ```python
# import numpy as np
#
# def loadfile(filename:str, bitlength:int):
#     C = int(np.ceil(bitlength / 64))
#     arr = np.fromfile(filename, dtype="uint8")
#     # Some axes should be transformed from big endian to little endian
#     res = np.unpackbits(arr).reshape(-1, C, 8, 8)[:,::-1,::-1,:]
#     res = res.reshape(-1, C*64)[:, :(64*C-bitlength)-1:-1]
#     print("number of configurations = %d"%(len(res)))
#     return res  # in big endian format
#
# res = loadfile(filename, 10)
# ```

# !!! note
#     Check section [Maximal independent set problem](@ref) for solution space properties related the maximal independent sets. That example also contains using cases of finding solution space properties related to minimum sizes:
#     * [`SizeMin`](@ref) for finding minimum several set sizes,
#     * [`CountingMin`](@ref) for counting minimum several set sizes,
#     * [`SingleConfigMin`](@ref) for finding one solution with minimum several sizes,
#     * [`ConfigsMin`](@ref) for enumerating solutions with minimum several sizes,


# ## Weighted Graphs
# [`IndependentSet`](@ref) accepts `weights` as a key word argument.
# The following code computes the weighted MIS problem.
problem = IndependentSet(graph; weights=collect(1:10))

max_config_weighted = solve(problem, SingleConfigMax())[]

show_graph(graph; locs=locations, vertex_colors=
          [iszero(max_config_weighted.c.data[i]) ? "white" : "red" for i=1:nv(graph)])

# For weighted MIS problem, a property that many people care about is the "energy spectrum", or the largest weights.
# We just feed a positional argument in the [`SizeMax`](@ref) constructor as the number of largest weights.
spectrum = solve(problem, SizeMax(10))[]

# The return value has type [`ExtendedTropical`](@ref), which contains one field `orders`. The `orders` is a vector of [`Tropical`](@ref) numbers.
spectrum.orders

# We can get weighted independent sets with maximum 5 sizes.
max5_configs = solve(problem, SingleConfigMax(5))[]

# The return value also has type [`ExtendedTropical`](@ref), but this time the element type of `orders` has been changed to [`CountingTropical`](@ref)`{Float64,`[`ConfigSampler`](@ref)`}`.
max5_configs.orders

# Let us visually check these graphs
imgs_max5 = ntuple(k->show_graph(graph;
                    locs=locations, scale=0.25,
                    vertex_colors=[iszero(max5_configs.orders[k].c.data[i]) ? "white" : "red"
                    for i=1:nv(graph)]), 5);

Compose.set_default_graphic_size(18cm, 4cm)

Compose.compose(context(),
     ntuple(k->(context((k-1)/5, 0.0, 1.2/5, 1.0), imgs_max5[k]), 5)...)

# ## Open vertices and MIS tensor analysis
# The following code computes the MIS tropical tensor (reference to be added) with open vertices 1, 2 and 3.
problem = IndependentSet(graph; openvertices=[1,2,3])

mis_tropical_tensor = solve(problem, SizeMax())

# The MIS tropical tensor shows the MIS size under different configuration of open vertices.
# It is useful in MIS tropical tensor analysis.
# We can compatify (reference to be added) this MIS-Tropical tensor by typing

mis_compactify!(mis_tropical_tensor)

# It will eliminate some entries having no contribution to the MIS size when embeding this local graph into a larger one.
