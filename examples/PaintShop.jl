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

sequence = collect("iadgbeadfcchghebfi")

# We can visualize this graph using the following function
rot(a, b, θ) = cos(θ*π)*a + sin(θ*π)*b, cos(θ*π)*b - sin(θ*π)*a

locations = [rot(0.0, 1.0, 2π*i/length(sequence)) for i=1:length(sequence)]

graph = let
    g = line_graph(length(sequence))
    for i=1:length(sequence) 
        j = findlast(==(sequence[i]), sequence)
        i != j && add_edge!(g, i, j)
    end
    g
end

show_graph(graph; locs=locations)

gp = PaintShop(sequence)
