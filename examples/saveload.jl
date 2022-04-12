# # Save and load solutions
# Let us use the maximum independent set problem on Petersen graph as an example.
# The following code enumerates all independent sets.

using GenericTensorNetworks, Graphs

problem = IndependentSet(Graphs.smallgraph(:petersen))

all_independent_sets = solve(problem, ConfigsAll())[]

# The return value has type [`ConfigEnumerator`](@ref).
# We can use [`save_configs`](@ref) and [`load_configs`](@ref) to save and read a [`ConfigEnumerator`](@ref) instance to the disk.
filename = tempname()

save_configs(filename, all_independent_sets; format=:binary)

loaded_sets = load_configs(filename; format=:binary, bitlength=10)

# !!! note
#     When loading the data in the binary format, bit string length information `bitlength` is required.
#
# For the [`SumProductTree`](@ref) type output, we can use [`save_sumproduct`](@ref) and [`load_sumproduct`](@ref) to save and load serialized data.

all_independent_sets_tree = solve(problem, ConfigsAll(; tree_storage=true))[]

save_sumproduct(filename, all_independent_sets_tree)

loaded_sets_tree = load_sumproduct(filename)

# ## Loading solutions to python
# The following python script loads and unpacks the solutions as a numpy array from a `:binary` format file.
# ```python
# import numpy as np
#
# def loadfile(filename:str, bitlength:int):
#     C = int(np.ceil(bitlength / 64))
#     arr = np.fromfile(filename, dtype="uint8")
#     # Some axes should be transformed from big endian to little endian
#     res = np.unpackbits(arr).reshape(-1, C, 8, 8)[:,::-1,::-1,:]
#     res = res.reshape(-1, C*64)[:, :(64*C-bitlength)-1:-1]
#     print("number of solutions = %d"%(len(res)))
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



