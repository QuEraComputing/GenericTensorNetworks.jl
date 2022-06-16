module ShowGraph
using Luxor, Graphs
export show_graph, show_einsum, show_gallery, spring_layout!

const CONFIGHELP = """
Extra keyword arguments
-------------------------------
* general
    * `pad::Float64 = 1.0`, the padding space
    * `unit::Float64 = 60`, the unit distance as the number of pixels
    * `offsetx::Float64 = 0.0`, the origin of x axis
    * `offsety::Float64 = 0.0`, the origin of y axis
    * `xspan::Float64 = 1.0`, the width of the graph/image
    * `yspan::Float64 = 1.0`, the height of the graph/image
    * `fontsize::Float64 = 12`, the font size
* vertex
    * `vertex_text_color::String = "black"`, the default text color
    * `vertex_stroke_color = "black"`, the default stroke color for vertices
    * `vertex_fill_color = "transparent"`, the default default fill color for vertices
    * `vertex_size::Float64 = 0.15`, the default vertex size
    * `vertex_line_width::Float64 = 1`, the default vertex stroke line width
* edge
    * `edge_color::String = "black"`, the default edge color
    * `edge_line_width::Float64 = 1`, the default line width
"""

function autoconfig(locations; pad, kwargs...)
    n = length(locations)
    if n >= 1
        # compute the size and the margin
        xmin = minimum(x->x[1], locations)
        ymin = minimum(x->x[2], locations)
        xmax = maximum(x->x[1], locations)
        ymax = maximum(x->x[2], locations)
        xspan = xmax - xmin
        yspan = ymax - ymin
        offsetx = -xmin + pad
        offsety = -ymin + pad
    else
        xspan = 0.0
        yspan = 0.0
        offsetx = 0.0
        offsety = 0.0
    end
    return GraphDisplayConfig(; pad, offsetx, offsety, xspan, yspan, kwargs...)
end

"""
    show_graph(graph::SimpleGraph;
        locs=nothing,
        spring::Bool=true,
        optimal_distance=1.0,
        spring_mask=trues(nv(graph)),

        vertex_colors=nothing,
        vertex_sizes=nothing,
        vertex_stroke_colors=nothing,
        vertex_text_colors=nothing,
        edge_colors=nothing,
        texts = nothing,
        format=:png,
        filename=nothing,
        kwargs...)

Show a graph in VSCode, Pluto or Jupyter notebook, or save it to a file.

Positional arguments
-----------------------------
* `graph` is a graph instance.

Keyword arguments
-----------------------------
* `locs` is a vector of tuples for specifying the vertex locations.
* `spring` is switch to use spring method to optimize the location.
* `optimal_distance` is a optimal distance parameter for `spring` optimizer.
* `spring_mask` specfies which location is optimizable for `spring` optimizer.

* `vertex_colors` is a vector of color strings for specifying vertex fill colors.
* `vertex_sizes` is a vector of real numbers for specifying vertex sizes.
* `vertex_stroke_colors` is a vector of color strings for specifying vertex stroke colors.
* `vertex_text_colors` is a vector of color strings for specifying vertex text colors.
* `edge_colors` is a vector of color strings for specifying edge colors.
* `texts` is a vector of strings for labeling vertices.
* `format` is the output format, which can be `:svg`, `:png` or `:pdf`.
* `filename` is a string as the output filename.

$CONFIGHELP

Example
------------------------------
```jldoctest
julia> using Graphs, GenericTensorNetworks

julia> show_graph(smallgraph(:petersen); format=:png, vertex_colors=rand(["blue", "red"], 10));
```
"""
function show_graph(locations, edges;
        vertex_colors=nothing,
        vertex_sizes=nothing,
        vertex_stroke_colors=nothing,
        vertex_text_colors=nothing,
        edge_colors=nothing,
        texts = nothing,
        format=:png, filename=nothing,
        pad=1.0,
        kwargs...)
    if length(locations) == 0
        _draw(()->nothing, 100, 100; format, filename)
    else
        config = autoconfig(locations; pad, kwargs...)
        Dx, Dy = (config.xspan+2*config.pad)*config.unit, (config.yspan+2*config.pad)*config.unit
        _draw(Dx, Dy; format, filename) do
            _show_graph(map(loc->(loc[1]+config.offsetx, loc[2]+config.offsety), locations), edges,
            vertex_colors, vertex_stroke_colors, vertex_text_colors, vertex_sizes, edge_colors, texts, config)
        end
    end
end

# NOTE: the final positions are in range [-5, 5]
function show_graph(graph::SimpleGraph;
        locs=nothing,
        spring::Bool=true,
        optimal_distance=1.0,
        spring_mask=trues(nv(graph)),
        kwargs...)
    locs = autolocs(graph, locs, spring, optimal_distance, spring_mask)
    show_graph(locs, [(e.src, e.dst) for e in edges(graph)]; kwargs...)
end

function autolocs(graph, locs, spring, optimal_distance, spring_mask)
    if spring
        locs_x = locs === nothing ? [2*rand()-1.0 for i=1:nv(graph)] : getindex.(locs, 1)
        locs_y = locs === nothing ? [2*rand()-1.0 for i=1:nv(graph)] : getindex.(locs, 2)
        spring_layout!(graph;
                    C=optimal_distance,
                    locs_x,
                    locs_y,
                    mask=spring_mask   # mask for which to relocate
                    )
        collect(zip(locs_x, locs_y))
    else
        locs
    end
end

function _draw(f, Dx, Dy; format, filename)
    if filename === nothing
        if format == :pdf
            _format = tempname()*".pdf"
        else
            _format = format
        end
    else
        _format = filename
    end
    Luxor.Drawing(round(Int,Dx), round(Int,Dy), _format)
    Luxor.origin(0, 0)
    f()
    Luxor.finish()
    Luxor.preview()
end
Base.@kwdef struct GraphDisplayConfig
    # line, vertex and text
    pad::Float64 = 1.0
    unit::Int = 60   # how many pixels as unit?
    offsetx::Float64 = 0.0  # the zero of x axis
    offsety::Float64 = 0.0  # the zero of y axis
    xspan::Float64 = 1.0
    yspan::Float64 = 1.0
    fontsize::Float64 = 12

    # vertex
    vertex_text_color::String = "black"
    vertex_stroke_color = "black"
    vertex_fill_color = "transparent"
    vertex_size::Float64 = 0.15
    vertex_line_width::Float64 = 1  # in pt
    # edge
    edge_color::String = "black"
    edge_line_width::Float64 = 1  # in pt
end

function _show_graph(locs, edges, vertex_colors, vertex_stroke_colors, vertex_text_colors, vertex_sizes, edge_colors, texts, config)
    # edges
    for (k, (i, j)) in enumerate(edges)
        ri = _get(vertex_sizes, i, config.vertex_size)
        rj = _get(vertex_sizes, j, config.vertex_size)
        draw_edge(locs[i], locs[j]; color=_get(edge_colors,k,config.edge_color),
            line_width=config.edge_line_width, ri, rj, unit=config.unit)
    end
    # vertices
    for (i, vertex) in enumerate(locs)
        draw_vertex(vertex...; fill_color=_get(vertex_colors, i, config.vertex_fill_color),
            stroke_color=_get(vertex_stroke_colors, i, config.vertex_stroke_color),
            r=_get(vertex_sizes, i, config.vertex_size), line_width=config.vertex_line_width, unit=config.unit)
        draw_text(vertex..., _get(texts, i, "$i"); fontsize=config.fontsize,
            color=_get(vertex_text_colors, i, config.vertex_text_color), unit=config.unit)
    end
end
_get(::Nothing, i::Int, default) = default
_get(x, i::Int, default) = x[i]

function draw_text(x, y, text; fontsize, color, unit)
    Luxor.fontsize(fontsize)
    setcolor(color)
    Luxor.text(text, Point(unit*x, unit*y), valign=:middle, halign=:center)
end
function draw_edge(i, j; color, line_width, ri, rj, unit)
    setcolor(color)
    setline(line_width)
    a, b = Point(i...), Point(j...)
    nints, ip1, ip2 =  intersectionlinecircle(a, b, a, ri)
    a_ = ip1
    nints, ip1, ip2 =  intersectionlinecircle(a, b, b, rj)
    b_ = ip2
    line(a_ * unit, b_ * unit, :stroke)
end
function draw_vertex(x, y; stroke_color, fill_color, line_width, r, unit)
    setcolor(fill_color)
    circle(unit*Point(x, y), unit*r, :fill)
    setline(line_width)
    setcolor(stroke_color)
    circle(Point(unit*x, unit*y), unit*r, :stroke)
end

"""
Spring layout for graph plotting, returns a vector of vertex locations.

!!! note
    This function is copied from [`GraphPlot.jl`](https://github.com/JuliaGraphs/GraphPlot.jl),
    where you can find more information about his function.
"""
function spring_layout!(g::AbstractGraph;
                       locs_x=2*rand(nv(g)).-1.0,
                       locs_y=2*rand(nv(g)).-1.0,
                       C=2.0,   # the optimal vertex distance
                       MAXITER=100,
                       INITTEMP=2.0,
                       mask::AbstractVector{Bool}=trues(length(locs_x))   # mask for which to relocate
                       )

    nvg = nv(g)
    adj_matrix = adjacency_matrix(g)

    # The optimal distance bewteen vertices
    k = C * sqrt(4.0 / nvg)
    k² = k * k

    # Store forces and apply at end of iteration all at once
    force_x = zeros(nvg)
    force_y = zeros(nvg)

    # Iterate MAXITER times
    @inbounds for iter = 1:MAXITER
        # Calculate forces
        for i = 1:nvg
            force_vec_x = 0.0
            force_vec_y = 0.0
            for j = 1:nvg
                i == j && continue
                d_x = locs_x[j] - locs_x[i]
                d_y = locs_y[j] - locs_y[i]
                dist²  = (d_x * d_x) + (d_y * d_y)
                dist = sqrt(dist²)

                if !( iszero(adj_matrix[i,j]) && iszero(adj_matrix[j,i]) )
                    # Attractive + repulsive force
                    # F_d = dist² / k - k² / dist # original FR algorithm
                    F_d = dist / k - k² / dist²
                else
                    # Just repulsive
                    # F_d = -k² / dist  # original FR algorithm
                    F_d = -k² / dist²
                end
                force_vec_x += F_d*d_x
                force_vec_y += F_d*d_y
            end
            force_x[i] = force_vec_x
            force_y[i] = force_vec_y
        end
        # Cool down
        temp = INITTEMP / iter
        # Now apply them, but limit to temperature
        for i = 1:nvg
            mask[i] || continue
            fx = force_x[i]
            fy = force_y[i]
            force_mag  = sqrt((fx * fx) + (fy * fy))
            scale      = min(force_mag, temp) / force_mag
            locs_x[i] += force_x[i] * scale
            locs_y[i] += force_y[i] * scale
        end
    end

    locs_x, locs_y
end

"""
    show_gallery(graph::SimpleGraph, grid::Tuple{Int,Int};
        locs=nothing,
        spring::Bool=true,
        optimal_distance=1.0,
        spring_mask=trues(nv(graph)),

        vertex_configs=nothing,
        edge_configs=nothing,

        vertex_sizes=nothing,
        vertex_stroke_colors=nothing,
        vertex_text_colors=nothing,
        texts=nothing,
        format=:png,
        filename=nothing,
        kwargs...)

Show a gallery of graphs for multiple vertex configurations or edge configurations in VSCode, Pluto or Jupyter notebook, or save it to a file.

Positional arguments
-----------------------------
* `graph` is a graph instance.
* `grid` is the grid layout of the gallery, e.g. input value `(2, 3)` means a grid layout with 2 rows and 3 columns.

Keyword arguments
-----------------------------
* `locs` is a vector of tuples for specifying the vertex locations.
* `spring` is switch to use spring method to optimize the location.
* `optimal_distance` is a optimal distance parameter for `spring` optimizer.
* `spring_mask` specfies which location is optimizable for `spring` optimizer.

* `vertex_configs` is an iterator of bit strings for specifying vertex configurations, e.g. a [`ConfigEnumerator`](@ref) instance. It will be rendered as vertex colors.
* `edge_configs` is an iterator of bit strings for specifying edge configurations. It will be rendered as edge colors.

* `vertex_sizes` is a vector of real numbers for specifying vertex sizes.
* `vertex_stroke_colors` is a vector of color strings for specifying vertex stroke colors.
* `vertex_text_colors` is a vector of color strings for specifying vertex text colors.
* `texts` is a vector of strings for labeling vertices.
* `format` is the output format, which can be `:svg`, `:png` or `:pdf`.
* `filename` is a string as the output filename.

$CONFIGHELP

Example
-------------------------------
```jldoctest
julia> using Graphs, GenericTensorNetworks

julia> show_gallery(smallgraph(:petersen), (2, 3); format=:png, vertex_configs=[rand(Bool, 10) for k=1:6]);
```
"""
function show_gallery(graph::SimpleGraph, grid::Tuple{Int,Int};
        locs=nothing,
        spring::Bool=true,
        optimal_distance=1.0,
        spring_mask=trues(nv(graph)),
        kwargs...)
    locs = autolocs(graph, locs, spring, optimal_distance, spring_mask)
    show_gallery(locs, [(e.src, e.dst) for e in edges(graph)], grid; kwargs...)
end
function show_gallery(locs, edges, grid::Tuple{Int,Int};
        vertex_configs=nothing,
        edge_configs=nothing,
        vertex_sizes=nothing,
        vertex_stroke_colors=nothing,
        vertex_text_colors=nothing,
        texts=nothing,
        format=:png,
        filename=nothing,
        pad=1.0,
        kwargs...)
    config = autoconfig(locs; pad, kwargs...)
    m, n = grid
    nv, ne = length(locs), length(edges)
    dx = (config.xspan+2*config.pad)*config.unit
    dy = (config.yspan+2*config.pad)*config.unit
    Dx, Dy = dx*n, dy*m
    locs = map(loc->(loc[1]+config.offsetx, loc[2]+config.offsety), locs)
    _draw(Dx, Dy; format, filename) do
        for i=1:m
            for j=1:n
                origin((j-1)*dx, (i-1)*dy)
                # set colors
                k = (i-1) * n + j
                vertex_colors = if vertex_configs isa Nothing
                    fill("white", nv)
                else
                    k > length(vertex_configs) && break
                    [iszero(vertex_configs[k][i]) ? config.vertex_fill_color : "red" for i=1:nv]
                end
                edge_colors = if edge_configs isa Nothing
                    fill("black", ne)
                else
                    k > length(edge_configs) && break
                    [iszero(edge_configs[k][i]) ? config.edge_color : "red" for i=1:ne]
                end
                _show_graph(locs, edges, vertex_colors, vertex_stroke_colors, vertex_text_colors,
                vertex_sizes, edge_colors, texts, config)
            end
        end
    end
end

end

using .ShowGraph

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
 
$(ShowGraph.CONFIGHELP)
"""
function show_einsum(ein::AbstractEinsum;
        label_size=0.15,
        label_color="black",
        open_label_color="red",
        tensor_size=0.3,
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
        locs = ShowGraph.autolocs(graph, nothing, spring, optimal_distance, trues(nv(graph)))
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