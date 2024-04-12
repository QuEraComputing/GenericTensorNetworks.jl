"""
    show_einsum(ein::AbstractEinsum;
        tensor_locs=nothing,
        label_locs=nothing,  # dict
        spring::Bool=true,
        optimal_distance=25.0,

        tensor_size=15,
        tensor_color="black",
        tensor_text_color="white",
        annotate_tensors=false,

        label_size=7,
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

* `tensor_color` is a string to specify the color of tensor nodes.
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
        label_size=7,
        label_color="black",
        open_label_color="red",
        tensor_size=15,
        tensor_color="black",
        tensor_text_color="white",
        annotate_labels=true,
        annotate_tensors=false,
        tensor_locs=nothing,
        label_locs=nothing,  # dict
        layout::Symbol=:auto,
        optimal_distance=25.0,
        kwargs...
        )
    ixs = getixsv(ein)
    iy = getiyv(ein)
    labels = uniquelabels(ein)
    m = length(labels)
    n = length(ixs)
    labelmap = Dict(zip(labels, 1:m))
    colors = [["transparent" for l in labels]..., fill(tensor_color, n)...]
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
        locs = LuxorGraphPlot.render_locs(graph, LuxorGraphPlot.Layout(layout; optimal_distance, spring_mask = trues(nv(graph))))
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
    show_graph(GraphViz(; locs, edges=[(e.src, e.dst) for e in edges(graph)], texts, vertex_colors=colors,
        vertex_text_colors,
        vertex_sizes=sizes); config=LuxorGraphPlot.GraphDisplayConfig(vertex_line_width=0, kwargs...))
end

"""
    show_configs(gp::GraphProblem, locs, configs::AbstractMatrix; kwargs...)
    show_configs(graph::SimpleGraph, locs, configs::AbstractMatrix; nflavor=2, kwargs...)

Show a gallery of configurations on a graph.
"""
function show_configs(gp::GraphProblem, locs, configs::AbstractMatrix; kwargs...)
    show_configs(gp.graph, locs, configs; nflavor=nflavor(gp), kwargs...)
end
function show_configs(graph::SimpleGraph, locs, configs::AbstractMatrix;
        nflavor::Int=2,
        kwargs...)
    cmap = range(colorant"white", stop=colorant"red", length=nflavor)
    locs = render_locs(graph, locs)
    graphs = map(configs) do cfg
        @assert all(0 .<= cfg .<= nflavor-1)
        GraphViz(graph; locs, vertex_colors=cmap[cfg .+ 1])
    end
    show_gallery(graphs; kwargs...)
end