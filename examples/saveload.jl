# # Saving and Loading Solutions
#
# ## Overview
# When working with large solution spaces, it's often necessary to save results to disk
# for later analysis or to share with other tools. This example demonstrates how to:
#
# * Save and load configuration enumerators
# * Save and load sum-product trees
# * Export solutions for use in Python
#
# We'll use the Maximum Independent Set problem on the Petersen graph as our example.

using GenericTensorNetworks, Graphs

# Create our problem instance
problem = GenericTensorNetwork(IndependentSet(Graphs.smallgraph(:petersen)))

# ## Saving Configuration Enumerators
# First, let's enumerate all independent sets
all_independent_sets = solve(problem, ConfigsAll())[]

# The result is a `ConfigEnumerator` instance containing all valid configurations.
# We can save this to disk and load it later:

# Create a temporary file for demonstration
filename = tempname()

# Save configurations in binary format (most efficient)
save_configs(filename, all_independent_sets; format=:binary)

# Load configurations from the saved file
loaded_sets = load_configs(filename; format=:binary, bitlength=10)

# !!! note
#     When loading data in binary format, you must specify the `bitlength` parameter,
#     which represents the number of bits used for each configuration.
#     In this example, the Petersen graph has 10 vertices, so we use `bitlength=10`.

# ## Saving Sum-Product Trees
# For larger solution spaces, the `SumProductTree` format is more memory-efficient.
# It stores solutions in a compressed tree structure:

# Generate solutions using tree storage
all_independent_sets_tree = solve(problem, ConfigsAll(; tree_storage=true))[]

# Save the sum-product tree
save_sumproduct(filename, all_independent_sets_tree)

# Load the sum-product tree
loaded_sets_tree = load_sumproduct(filename)

# ## Interoperability with Python
# The binary format can be loaded in Python for further analysis.
# Here's a Python function to load and unpack the solutions as a NumPy array:

# ```python
# import numpy as np
#
# def loadfile(filename: str, bitlength: int):
#     """
#     Load binary configuration data saved by GenericTensorNetworks.jl
#     
#     Parameters:
#     -----------
#     filename : str
#         Path to the binary file
#     bitlength : int
#         Number of bits per configuration (typically number of vertices)
#         
#     Returns:
#     --------
#     numpy.ndarray
#         2D array where each row is a configuration
#     """
#     C = int(np.ceil(bitlength / 64))
#     arr = np.fromfile(filename, dtype="uint8")
#     # Transform from big endian to little endian
#     res = np.unpackbits(arr).reshape(-1, C, 8, 8)[:,::-1,::-1,:]
#     res = res.reshape(-1, C*64)[:, :(64*C-bitlength)-1:-1]
#     print(f"Number of solutions: {len(res)}")
#     return res  # in big endian format
#
# # Example usage:
# solutions = loadfile(filename, 10)
# ```

# ## Additional Resources
# For more examples of working with solution spaces:
#
# * See the [Maximal Independent Set Problem](@ref) section for examples of:
#   * Using `SizeMin` to find minimum set sizes
#   * Using `CountingMin` to count minimum set sizes
#   * Using `SingleConfigMin` to find one minimum solution
#   * Using `ConfigsMin` to enumerate all minimum solutions



