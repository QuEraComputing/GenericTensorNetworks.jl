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
        open_label_color="black",
        annotate_labels=true,
        kwargs...
        )

Positional arguments
-----------------------------
* `ein` is an Einsum contraction code (provided by package `OMEinsum`).

Keyword arguments
-----------------------------
* `locs` is a tuple of `tensor_locs` (vector) and `label_locs` (dict).
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
        open_label_color="black",
        tensor_size=15,
        tensor_color="black",
        tensor_text_color="white",
        annotate_labels=true,
        annotate_tensors=false,
        layout = SpringLayout(),
        locs = nothing,  # dict
        filename = nothing,
        format=:svg,
        padding_left=10.0,
        padding_right=10.0,
        padding_top=10.0,
        padding_bottom=10.0,
        config=LuxorGraphPlot.GraphDisplayConfig(; vertex_line_width=0.0),
        tensor_texts = nothing,
        )
    layout = deepcopy(layout)
    ixs = getixsv(ein)
    iy = getiyv(ein)
    labels = uniquelabels(ein)
    m = length(labels)
    n = length(ixs)
    labelmap = Dict(zip(labels, 1:m))
    colors = [["transparent" for l in labels]..., fill(tensor_color, n)...]
    sizes = [fill(label_size, m)..., fill(tensor_size, n)...]
    if tensor_texts === nothing
        tensor_texts = [annotate_tensors ? "$i" : "" for i=1:n]
    end
    texts = [[annotate_labels ? "$(labels[i])" : "" for i=1:m]..., tensor_texts...]
    vertex_text_colors = [[l ∈ iy ? open_label_color : label_color for l in labels]..., [tensor_text_color for ix in ixs]...]
    graph = SimpleGraph(m+n)
    for (j, ix) in enumerate(ixs)
        for l in ix
            add_edge!(graph, j+m, labelmap[l])
        end
    end
    if locs === nothing
        locs = getfield.(LuxorGraphPlot.render_locs(graph, layout), :data)
    else
        tensor_locs, label_locs = locs
        @assert tensor_locs isa AbstractVector && label_locs isa AbstractDict "locs should be a tuple of `tensor_locs` (vector) and `label_locs` (dict)"
        @assert length(tensor_locs) == n "the length of tensor_locs should be $n"
        @assert length(label_locs) == m "the length of label_locs should be $m"
        locs = [[label_locs[l] for l in labels]..., tensor_locs...]
    end
    show_graph(GraphViz(; locs, edges=[(e.src, e.dst) for e in edges(graph)], texts, vertex_colors=colors,
        vertex_text_colors,
        vertex_sizes=sizes);
        filename, format,
        padding_left, padding_right, padding_top, padding_bottom, config
        )
end

"""
    show_configs(gp::ConstraintSatisfactionProblem, locs, configs::AbstractMatrix; kwargs...)
    show_configs(graph::SimpleGraph, locs, configs::AbstractMatrix; num_flavors=2, kwargs...)

Show a gallery of configurations on a graph.
"""
function show_configs(gp::ConstraintSatisfactionProblem, locs, configs::AbstractMatrix; kwargs...)
    show_configs(gp.graph, locs, configs; num_flavors=num_flavors(gp), kwargs...)
end
function show_configs(graph::SimpleGraph, locs, configs::AbstractMatrix;
        num_flavors::Int=2,
        kwargs...)
    cmap = range(colorant"white", stop=colorant"red", length=num_flavors)
    graphs = map(configs) do cfg
        @assert all(0 .<= cfg .<= num_flavors-1)
        GraphViz(graph, locs; vertex_colors=cmap[cfg .+ 1])
    end
    show_gallery(graphs; kwargs...)
end

"""
    show_landscape(is_neighbor, configurations::TruncatedPoly;
        layer_distance=200,
        config=GraphDisplayConfig(; edge_color="gray", vertex_stroke_color="transparent", vertex_size=5),
        layout_method=:spring,
        optimal_distance=30.0,
        colors=fill("green", K),
        kwargs...)

Show the energy landscape of configurations.

### Arguments
- `is_neighbor`: a function to determine if two configurations are neighbors.
- `configurations`: a TruncatedPoly object, which is the default output of the [`solve`](@ref) function with [`ConfigsMax`](@ref) property as the argument.

### Keyword arguments
- `layer_distance`: the distance between layers.
- `config`: a `LuxorGraphPlot.GraphDisplayConfig` object.
- `layout_method`: the layout method, either `:spring`, `:stress` or `:spectral`
- `optimal_distance`: the optimal distance for the layout.
- `colors`: a vector of colors for each layer.
- `kwargs...`: other keyword arguments passed to `show_graph`.
"""
function show_landscape(is_neighbor, configurations::TruncatedPoly{K};
            layer_distance=200,
            config=GraphDisplayConfig(; edge_color="gray", vertex_stroke_color="transparent", vertex_size=5),
            layout_method = :spring,
            optimal_distance = 30.0,
            colors=fill("green", K),
        kwargs...) where K
    @assert length(colors) == K "colors should have length $K"
    nv = 0
    kv = read_size_config(configurations)
    zlocs = Float64[]
    vertex_colors = String[]
    for (c, (k, v)) in zip(colors, kv)
        nv += length(v)
        append!(zlocs, fill(k*layer_distance, length(v)))
        append!(vertex_colors, fill(c, length(v)))
    end
    graph = SimpleGraph(nv)
    configs = vcat(getindex.(kv, 2)...)
    for (i, c) in enumerate(configs)
        for (j, d) in enumerate(configs)
            if is_neighbor(c, d)
                add_edge!(graph, i, j)
            end
        end
    end
    if layout_method == :spring
        layout=LayeredSpringLayout(; zlocs, optimal_distance)
    elseif layout_method == :stress
        layout=LayeredStressLayout(; zlocs, optimal_distance)
    elseif layout_method == :spectral
        layout=LuxorGraphPlot.Layered(SpectralLayout(), zlocs, 0.2)
    else
        error("layout_method should be :spring, :stress or :spectral")
    end
    show_graph(graph, layout; config, vertex_colors, kwargs...)
end