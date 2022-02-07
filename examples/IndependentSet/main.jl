# # Independent set problem

# ## Introduction
using GraphTensorNetworks, Graphs

# Please check the docstring of [`Independence`](@ref) for the definition of independence problem.
@doc Independence

# In the following, we are going to defined an independent set problem for the Petersen graph.

graph = Graphs.smallgraph(:petersen)

# We can visualize this graph using the following function
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations)

# Let us contruct the problem instance with optimized tensor network contraction order as bellow.
problem = Independence(graph; optimizer=TreeSA(sc_weight=1.0, ntrials=10,
                         βs=0.01:0.1:15.0, niters=20, rw_weight=0.2),
                         simplifier=MergeGreedy());


# ## Solving properties

# ### Maximum independent set size ``\alpha(G)``
maximum_independent_set_size = solve(problem, SizeMax())

# ### Counting properties
# ##### counting all independent sets
count_all_independent_sets = solve(problem, CountingAll())

# ##### counting independent sets with sizes ``\alpha(G)`` and ``\alpha(G)-1``
count_max2_independent_sets = solve(problem, CountingMax(2))

# ##### independence polynomial
# For the definition of independence polynomial, please check the docstring of [`Independence`](@ref) or this [wiki page](https://mathworld.wolfram.com/IndependencePolynomial.html).
# There are 3 methods to compute a graph polynomial, `:finitefield`, `:fft` and `:polynomial`.
# These methods are introduced in the docstring of [`GraphPolynomial`](@ref).
independence_polynomial = solve(problem, GraphPolynomial(; method=:finitefield))

# ### Configuration properties
# ##### finding one maximum independent set (MIS)
# There are two approaches to find one of the best solution.
# The unbounded (default) version uses [`ConfigSampler`](@ref) to sample one of the best solutions directly.
# The bounded version uses the binary gradient back-propagation (see our paper) to compute the gradients.
# It requires caching intermediate states, but is often faster on CPU because it can use [`TropicalGEMM`](https://github.com/TensorBFS/TropicalGEMM.jl).
max_config = solve(problem, SingleConfigMax(; bounded=false))[]

# The return value contains a bit string, and one should read this bit string from left to right.
# Having value 1 at i-th bit means vertex ``i`` is in the maximum independent set.
# One can visualize this MIS with the following function.
show_graph(graph; locs=locations, colors=[iszero(max_config.c.data[i]) ? "white" : "red"
                                 for i=1:nv(graph)])

# ##### enumeration of all MISs
# There are two approaches to enumerate all best-K solutions.
# The bounded (default) version is always prefered because it can significantly use the memory usage.
all_max_configs = solve(problem, ConfigsMax(1; bounded=true))[]

using Compose

m = length(all_max_configs.c)

imgs = ntuple(k->(context((k-1)/m, 0.0, 1.2/m, 1.0), show_graph(graph;
                            locs=locations, scale=0.25,
                            colors=[iszero(all_max_configs.c[k][i]) ? "white" : "red"
                                 for i=1:nv(graph)])), m)

Compose.set_default_graphic_size(18cm, 4cm); Compose.compose(context(), imgs...)

# ##### enumeration of all IS configurations
all_independent_sets = solve(problem, ConfigsAll())[]