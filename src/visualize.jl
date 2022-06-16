using Luxor

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
    show_graph(graph;
        locs=spring_layout(graph),
        vertex_colors=["black", "black", ...],
        edge_colors=["black", "black", ...],
        texts=["1", "2", ...],
        format=:png,
        io=nothing,
        pad=1.0,
        kwargs...
        )

Show a graph featured with vertex and edge information.

Positional arguments
-----------------------------
* `graph` is a graph instance.

Keyword arguments
-----------------------------
* `locs` is a vector of tuples for specifying the vertex locations.
* `vertex_colors` is a vector of color strings for specifying vertex configurations, e.g. a [`ConfigEnumerator`](@ref) instance.
* `edge_colors` is a vector of color strings for specifying edge configurations.
* `texts` is a vector of strings for labeling vertices.
* `format` is the output format, which can be `:svg`, `:png` or `:pdf`..
* `io` can be `nothing` for the direct output, or a filename to saving to a file. For direct output, you will need a VSCode editor, an Atom editor, a Pluto notebook or a Jupyter notebook to display the image.

Extra keyword arguments
-------------------------------
* line, vertex and text
    * `pad::Float64` = 1.0
* vertex
    * `vertex_text_color::String` = "black"
    * `vertex_stroke_color` = "black"
    * `vertex_fill_color` = "white"
* edge
    * `edge_color::String` = "black"

Example
------------------------------
```jldoctest
julia> using Graphs, GenericTensorNetworks

julia> show_graph(smallgraph(:petersen); format=:png, io=tempname(), vertex_colors=rand(["blue", "red"], 10));
```
"""
function show_graph(locations, edges;
        vertex_colors=nothing,
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
            _show_graph(map(loc->(loc[1]+config.offsetx, loc[2]+config.offsety), locations), edges, vertex_colors, edge_colors, texts, config)
        end
    end
end

# NOTE: the final positions are in range [-5, 5]
function show_graph(graph::SimpleGraph; C=5.0, locs=map(x->x .* C, spring_layout(graph)), kwargs...)
    show_graph(locs, [(e.src, e.dst) for e in edges(graph)]; kwargs...)
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
    Luxor.Drawing(Dx, Dy, _format)
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

    # vertex
    vertex_text_color::String = "black"
    vertex_stroke_color = "black"
    vertex_fill_color = "white"
    vertex_size::Float64 = 0.15
    vertex_fontsize::Float64 = 12
    vertex_line_width::Float64 = 1  # in pt
    # edge
    edge_color::String = "black"
    edge_line_width::Float64 = 1  # in pt
end

function _show_graph(locs, edges, vertex_colors, edge_colors, texts, config)
    if vertex_colors !== nothing
        @assert length(locs) == length(vertex_colors)
    else
        vertex_colors = fill(config.vertex_fill_color, length(locs))
    end
    if edge_colors !== nothing
        @assert length(edges) == length(edge_colors)
    else
        edge_colors = fill(config.edge_color, length(edges))
    end
    if texts !== nothing
        @assert length(locs) == length(texts)
    end
    for (i, vertex) in enumerate(locs)
        draw_vertex(vertex...; fill_color=vertex_colors[i], stroke_color=config.vertex_stroke_color, r=config.vertex_size, line_width=config.vertex_line_width, unit=config.unit)
        if config.vertex_text_color !== "transparent"
            draw_text(vertex..., texts === nothing ? "$i" : texts[i]; fontsize=config.vertex_fontsize, color=config.vertex_text_color, unit=config.unit)
        end
    end
    for (k, (i, j)) in enumerate(edges)
        draw_edge(locs[i], locs[j]; color=edge_colors[k], line_width=config.edge_line_width, r=config.vertex_size, unit=config.unit)
    end
end
function draw_text(x, y, text; fontsize, color, unit)
    Luxor.fontsize(fontsize)
    setcolor(color)
    Luxor.text(text, Point(unit*x, unit*y), valign=:middle, halign=:center)
end
function draw_edge(i, j; color, line_width, r, unit)
    setcolor(color)
    setline(line_width)
    a, b = Point(i...), Point(j...)
    nints, ip1, ip2 =  intersectionlinecircle(a, b, a, r)
    a_ = ip1
    nints, ip1, ip2 =  intersectionlinecircle(a, b, b, r)
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
function spring_layout(g::AbstractGraph,
                       locs_x=2*rand(nv(g)).-1.0,
                       locs_y=2*rand(nv(g)).-1.0;
                       C=2.0,
                       MAXITER=100,
                       INITTEMP=2.0)

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
            fx = force_x[i]
            fy = force_y[i]
            force_mag  = sqrt((fx * fx) + (fy * fy))
            scale      = min(force_mag, temp) / force_mag
            locs_x[i] += force_x[i] * scale
            locs_y[i] += force_y[i] * scale
        end
    end

    # Scale to unit square
    min_x, max_x = minimum(locs_x), maximum(locs_x)
    min_y, max_y = minimum(locs_y), maximum(locs_y)
    function scaler(z, a, b)
        2.0*((z - a)/(b - a)) - 1.0
    end
    map!(z -> scaler(z, min_x, max_x), locs_x, locs_x)
    map!(z -> scaler(z, min_y, max_y), locs_y, locs_y)

    return collect(zip(locs_x, locs_y))
end

"""
    show_gallery(graph::SimpleGraph, grid::Tuple{Int,Int};
        locs=spring_layout(graph), 
        vertex_configs=nothing,
        edge_configs=nothing,
        texts=["1", "2", ...],
        format=:png,
        io=nothing,
        kwargs...)

Show a gallery of graphs for multiple vertex configurations or edge configurations.

Positional arguments
-----------------------------
* `graph` is a graph instance.
* `grid` is the grid layout of the gallery, e.g. input value `(2, 3)` means a grid layout with 2 rows and 3 columns.

Keyword arguments
-----------------------------
* `locs` is a vector of tuples for specifying the vertex locations.
* `vertex_configs` is an iterator of bit strings for specifying vertex configurations, e.g. a [`ConfigEnumerator`](@ref) instance.
* `edge_configs` is an iterator of bit strings for specifying edge configurations.
* `texts` is a vector of strings for labeling vertices.
* `format` is the output format, which can be `:svg`, `:pdf` or `:png`.
* `io` can be `nothing` for the direct output, or a filename to saving to a file. For direct output, you will need a VSCode editor, an Atom editor, a Pluto notebook or a Jupyter notebook to display the image.

Extra keyword arguments
-------------------------------
* line, vertex and text
    * `pad::Float64` = 1.0
* vertex
    * `vertex_text_color::String` = "black"
    * `vertex_stroke_color` = "black"
    * `vertex_fill_color` = "white"
* edge
    * `edge_color::String` = "black"

Example
-------------------------------
```jldoctest
julia> using Graphs, GenericTensorNetworks

julia> show_gallery(smallgraph(:petersen), (2, 3); format=:png, io=tempname(), vertex_configs=[rand(Bool, 10) for k=1:6]);
```
"""
function show_gallery(graph::SimpleGraph, grid::Tuple{Int,Int}; C=5.0, locs=map(x->x .* C, spring_layout(graph)), kwargs...)
    show_gallery(locs, [(e.src, e.dst) for e in edges(graph)], grid; kwargs...)
end
function show_gallery(locs, edges, grid::Tuple{Int,Int};
        vertex_configs=nothing,
        edge_configs=nothing,
        texts=nothing,
        format=:png, filename=nothing,
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
                _show_graph(locs, edges, vertex_colors, edge_colors, texts, config)
            end
        end
    end
end