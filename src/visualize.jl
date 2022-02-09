using Compose, Viznet, Cairo

struct Rescaler{T}
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    pad::T
end

getscale(r::Rescaler) = min(1/(r.xmax-r.xmin+2*r.pad), 1/(r.ymax-r.ymin+2*r.pad))

function config_plotting(sites)
    n = length(sites)
    if n <= 1
        return (1.0, 0.5, 0.4, 1.0)
    end
    shortest_distance = Inf
    for i=1:n
        for j=i+1:n
            shortest_distance = min(sqrt(sum(abs2, sites[i] .- sites[j])), shortest_distance)
        end
    end

    rescaler = get_rescaler(sites, 0.0)
    xpad = (rescaler.xmax - rescaler.xmin) * 0.2 + shortest_distance
    ypad = (rescaler.ymax - rescaler.ymin) * 0.2 + shortest_distance
    pad = max(xpad, ypad)
    scale = shortest_distance
    return (pad=pad, scale=scale)
end

function (r::Rescaler{T})(x; dims=(1,2)) where T
    xmin, ymin, xmax, ymax, pad = r.xmin, r.ymin, r.xmax, r.ymax, r.pad
    scale = getscale(r)
    if dims == (1,2)
        return (x[1]-xmin+pad, ymax+pad-x[2]) .* scale
    elseif dims == 1
        return (x - xmin + pad) * scale
    elseif dims == 2
        return (ymax + pad - x) * scale
    else
        throw(ArgumentError("dims should be (1,2), 1 or 2."))
    end
end

function get_rescaler(locs::AbstractVector{<:Tuple}, pad)
    xmin = minimum(x->x[1], locs)
    ymin = minimum(x->x[2], locs)
    xmax = maximum(x->x[1], locs)
    ymax = maximum(x->x[2], locs)
    return Rescaler(promote(xmin, xmax, ymin, ymax, pad)...)
end

default_vertex_style(scale, stroke_color, fill_color) = compose(context(), Viznet.nodestyle(:default, r=0.15cm*scale), Compose.stroke(stroke_color), fill(fill_color), linewidth(0.3mm*scale))
default_text_style(scale, color) = Viznet.textstyle(:default, fontsize(4pt*scale), fill(color))
default_edge_style(scale, color) = Viznet.bondstyle(:default, Compose.stroke(color), linewidth(0.3mm*scale))

"""
    show_graph(locations, edges;
        vertex_colors=["black", "black", ...],
        edge_colors=["black", "black", ...],
        texts=["1", "2", ...],
        format=SVG,
        edge_color="black",
        )

Plots vertices at `locations` with vertex colors specified by `vertex_colors` and texts specified by `texts`.
You will need a `VSCode`, `Pluto` notebook or `Jupyter` notebook to show the image.
If you want to write this image to the disk without displaying it in a frontend, please try

```julia
julia> open("test.png", "w") do f
            viz_atoms(f, generate_sites(SquareLattice(), 5, 5))
       end
```

The `format` keyword argument can also be `Compose.SVG` or `Compose.PDF`.
"""
function show_graph(locations, edges;
        vertex_colors=nothing,
        edge_colors=nothing,
        texts = nothing,
        format=SVG, io=nothing,
        kwargs...)
    if length(locations) == 0
        dx, dy = 12cm, 12cm
        img = Compose.compose(context())
    else
        img, (dx, dy) = viz_atoms(locations, edges; vertex_colors=vertex_colors, edge_colors=edge_colors, texts=texts, config=GraphDisplayConfig(; config_plotting(locations)..., kwargs...))
    end
    if io === nothing
        Compose.set_default_graphic_size(dx, dy)
        return img
    else
        return format(io, dx, dy)(img)
    end
end
function show_graph(graph::SimpleGraph; locs=spring_layout(graph), kwargs...)
    show_graph(locs, [(e.src, e.dst) for e in edges(graph)]; kwargs...)
end

function fit_image(rescaler::Rescaler, image_size, imgs...)
    X = rescaler.xmax - rescaler.xmin + 2*rescaler.pad
    Y = rescaler.ymax - rescaler.ymin + 2*rescaler.pad
    img_rescale = image_size/max(X, Y)*cm
    if Y < X
        return Compose.compose(context(0, 0, 1.0, X/Y), imgs...), (X*img_rescale, Y*img_rescale)
    else
        return Compose.compose(context(0, 0, Y/X, 1.0), imgs...), (X*img_rescale, Y*img_rescale)
    end
end

# Returns a 2-tuple of (image::Context, size)
function viz_atoms(locs, edges; vertex_colors, edge_colors, texts, config)
    rescaler = get_rescaler(locs, config.pad)
    img = _viz_atoms(rescaler.(locs), edges, vertex_colors, edge_colors, texts, config, getscale(rescaler))
    return fit_image(rescaler, config.image_size, img)
end

Base.@kwdef struct GraphDisplayConfig
    # line, vertex and text
    scale::Float64 = 1.0
    pad::Float64 = 1.5

    # vertex
    vertex_text_color::String = "black"
    vertex_stroke_color = "black"
    vertex_fill_color = "white"
    # bond
    edge_color::String = "black"
    # image size in cm
    image_size::Float64 = 12
end

function _viz_atoms(locs, edges, vertex_colors, edge_colors, texts, config, rescale)
    rescale = rescale * config.image_size * config.scale * 1.6
    if vertex_colors !== nothing
        @assert length(locs) == length(vertex_colors)
        vertex_styles = [default_vertex_style(rescale, config.vertex_stroke_color, color) for color in vertex_colors]
    else
        vertex_styles = fill(default_vertex_style(rescale, config.vertex_stroke_color, config.vertex_fill_color), length(locs))
    end
    if edge_colors !== nothing
        @assert length(edges) == length(edge_colors)
        edge_styles = [default_edge_style(rescale, color) for color in edge_colors]
    else
        edge_styles = fill(default_edge_style(rescale, config.edge_color), length(edges))
    end
    if texts !== nothing
        @assert length(locs) == length(texts)
    end
    text_style = default_text_style(rescale, config.vertex_text_color)
    img1 = Viznet.canvas() do
        for (i, vertex) in enumerate(locs)
            vertex_styles[i] >> vertex
            if config.vertex_text_color !== "transparent"
                text_style >> (vertex, texts === nothing ? "$i" : texts[i])
            end
        end
        for (k, (i, j)) in enumerate(edges)
            edge_styles[k] >> (locs[i], locs[j])
        end
    end
    Compose.compose(context(), img1)
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