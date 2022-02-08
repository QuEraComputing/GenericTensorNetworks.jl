# # Binary paint shop problem

# !!! note
#     This tutorial only covers the binary paint shop problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Introduction
using GraphTensorNetworks, Graphs

# Please check the docstring of [`PaintShop`](@ref) for the definition of the binary paint shop problem.
@doc PaintShop

# In the following, we are going to defined a binary paint shop problem for the following string

sequence = "abaccb"