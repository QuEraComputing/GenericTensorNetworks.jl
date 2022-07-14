"""
    show_einsum(ein::AbstractEinsum;
        tensor_locs=nothing,
        label_locs=nothing,  # dict
        spring::Bool=true,
        optimal_distance=1.0,

        tensor_size=0.3,
        tensor_color="black",
        tensor_text_color="white",
        annotate_tensors=false,

        label_size=0.15,
        label_color="black",
        open_label_color="red",
        annotate_labels=true,
        kwargs...
        )

Positional arguments
-----------------------------
* `ein` is an Einsum contraction code (provided by package `OMEinsum`).

Keyword arguments
-----------------------------
* `tensor_locs` is a vector of tuples for specifying the vertex locations.
* `label_locs` is a vector of tuples for specifying the vertex locations.
* `spring` is switch to use spring method to optimize the location.
* `optimal_distance` is a optimal distance parameter for `spring` optimizer.

* `tensor_colors` is a vector of strings to specify the colors of tensor nodes.
* `tensor_color` is a string to specify the default color of tensor nodes.
* `tensor_size` is a real number to specify the size of tensor nodes.
* `tensor_text_color` is a color strings to specify tensor text color.
* `annotate_tensors` is a boolean switch for annotate different tensors by integers.

* `label_size` is a real number to specify label text node size.
* `label_color` is a color strings to specify label text color.
* `open_label_color` is a color strings to specify open label text color.
* `annotate_labels` is a boolean switch for annotate different labels.

* `format` is the output format, which can be `:svg`, `:png` or `:pdf`.
* `filename` is a string as the output filename.
 
$(LuxorGraphPlot.CONFIGHELP)
"""
function show_einsum(ein::AbstractEinsum;
        label_size=0.15,
        label_color="black",
        open_label_color="red",
        tensor_size=0.3,
        tensor_colors=nothing,
        tensor_color="black",
        tensor_text_color="white",
        annotate_labels=true,
        annotate_tensors=false,
        tensor_locs=nothing,
        label_locs=nothing,  # dict
        spring::Bool=true,
        optimal_distance=1.0,
        kwargs...
        )
    ixs = getixsv(ein)
    iy = getiyv(ein)
    labels = uniquelabels(ein)
    m = length(labels)
    n = length(ixs)
    labelmap = Dict(zip(labels, 1:m))
    colors = [["transparent" for l in labels]..., [LuxorGraphPlot._get(tensor_colors, i, tensor_color) for i=1:n]...]
    sizes = [fill(label_size, m)..., fill(tensor_size, n)...]
    texts = [[annotate_labels ? "$(labels[i])" : "" for i=1:m]..., [annotate_tensors ? "$i" : "" for i=1:n]...]
    vertex_text_colors = [[l ∈ iy ? open_label_color : label_color for l in labels]..., [tensor_text_color for ix in ixs]...]
    graph = SimpleGraph(m+n)
    for (j, ix) in enumerate(ixs)
        for l in ix
            add_edge!(graph, j+m, labelmap[l])
        end
    end
    if label_locs === nothing && tensor_locs === nothing
        locs = LuxorGraphPlot.autolocs(graph, nothing, spring, optimal_distance, trues(nv(graph)))
    elseif label_locs === nothing
        # infer label locs from tensor locs
        label_locs = [(lst = [iloc for (iloc,ix) in zip(tensor_locs, ixs) if l ∈ ix]; reduce((x,y)->x .+ y, lst) ./ length(lst)) for l in labels]
        locs = [label_locs..., tensor_locs...]
    elseif tensor_locs === nothing
        # infer tensor locs from label locs
        tensor_locs = [length(ix) == 0 ? (optimal_distance*randn(), optimal_distance*randn()) : reduce((x,y)-> x .+ y, [label_locs[l] for l in ix]) ./ length(ix) for ix in ixs]
        locs = [label_locs..., tensor_locs...]
    else
        locs = [label_locs..., tensor_locs...]
    end
    show_graph(locs, [(e.src, e.dst) for e in edges(graph)]; texts, vertex_colors=colors,
        vertex_text_colors,
        vertex_sizes=sizes, vertex_line_width=0, kwargs...)
end
