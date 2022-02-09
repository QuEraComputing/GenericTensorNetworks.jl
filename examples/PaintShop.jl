# # Binary paint shop problem

# !!! note
#     This tutorial only covers the binary paint shop problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Problme Definition
# The [binary paint shop problem](http://m-hikari.com/ams/ams-2012/ams-93-96-2012/popovAMS93-96-2012-2.pdf).

# In the following, we are going to defined a binary paint shop problem for the following string

using GraphTensorNetworks, Graphs

sequence = "abaccb"