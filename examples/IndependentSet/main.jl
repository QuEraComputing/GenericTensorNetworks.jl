# # Independent set problem

# ## Problem definition

# Please check the docstring of [`Independence`](@ref).

# ## Solving properties

using GraphTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

problem = Independence(graph; optimizer=TreeSA(sc_weight=1.0, ntrials=10,
                         βs=0.01:0.1:15.0, niters=20, rw_weight=0.2));

# maximum independent set size
maximum_independent_set_size = solve(problem, SizeMax())

# all independent sets
count_all_independent_sets = solve(problem, CountingAll())

# counting independent sets of max two sizes
count_max2_independent_sets = solve(problem, CountingMax(2))

# compute the independence polynomial
independence_polynomial = solve(problem, GraphPolynomial(; method=:finitefield))

# find one MIS
max_config = solve(problem, SingleConfigMax(; bounded=false))

# enumerate all MISs
all_max_configs = solve(problem, ConfigsMax(; bounded=true))

# enumerate all IS configurations
all_independent_sets = solve(problem, ConfigsAll())