# # Binary paint shop problem

# !!! note
#     This tutorial only covers the binary paint shop problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Problme Definition
# The [binary paint shop problem](http://m-hikari.com/ams/ams-2012/ams-93-96-2012/popovAMS93-96-2012-2.pdf) is defined as follows:
# we are given a ``2m`` length sequence containing ``m`` cars, where each car appears twice.
# Each car need to be colored red in one occurrence, and blue in the other.
# We need to choose which occurrence for each car to color with which color — the goal is to minimize the number of times we need to change the current color.

# In the following, we use a character to represent a car,
# and defined a binary paint shop problem as a string that each character appear exactly twice.

using GraphTensorNetworks, Graphs

sequence = collect("iadgbeadfcchghebif")

# We can visualize this paint shop problem as a graph
rot(a, b, θ) = cos(θ)*a + sin(θ)*b, cos(θ)*b - sin(θ)*a

locations = [rot(0.0, 1.0, -0.25π - 1.5*π*(i-0.5)/length(sequence)) for i=1:length(sequence)]

graph = line_graph(length(sequence))
for i=1:length(sequence) 
    j = findlast(==(sequence[i]), sequence)
    i != j && add_edge!(graph, i, j)
end

show_graph(graph; locs=locations, texts=string.(sequence), edge_colors=[sequence[e.src] == sequence[e.dst] ? "blue" : "black" for e in edges(graph)])

# ## Tensor network representation
# Its tensor network representation is obtained by mapping a pair of cars into a boolean variable,
# where we use 0 to denote the first car is in red and 1 to denote the second car is in red.
# The paint shop problem is converted to finding the minimum energy of a spin glass problem.

chars = unique(sequence)

mapped_graph = SimpleGraph(length(chars))
weights = Dict{Tuple{Int,Int},Int}()
for i=2:length(sequence)
    a, b = sequence[i-1], sequence[i]
    l, m = findfirst(==(a), chars), findfirst(==(b), chars)
    add_edge!(mapped_graph, l, m)
    edge = minmax(l, m)
    # both are the first appearence of a car, or both are the second appearence of a car
    # prefer to have the same boolean value: s_a * s_b
    if (i-1 == findfirst(==(a), sequence)) == (i == findfirst(==(b), sequence))
        weights[edge] = get(weights, edge, 0) + 1
    else
        weights[edge] = get(weights, edge, 0) - 1
    end
end

weight_vector = [weights[minmax(e.src, e.dst)] for e in edges(mapped_graph)]
weight_color_map = Dict(1=>"red", -1=>"cyan", 2=>"green", 0=>"purple")
show_graph(mapped_graph; locs=GraphTensorNetworks.spring_layout(mapped_graph), texts=string.(chars), edge_colors=[weight_color_map[w] for w in weight_vector])

# Vertices connected by blue edges must have different colors,
# and the goal becomes a min-cut problem defined on black edges.

gp = PaintShop(sequence)
