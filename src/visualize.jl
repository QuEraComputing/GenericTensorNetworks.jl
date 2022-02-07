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

default_node_style(scale, stroke_color, fill_color) = compose(context(), Viznet.nodestyle(:default, r=0.15cm*scale), Compose.stroke(stroke_color), fill(fill_color), linewidth(0.3mm*scale))
default_text_style(scale, color) = Viznet.textstyle(:default, fontsize(4pt*scale), fill(color))
default_bond_style(scale, color) = Viznet.bondstyle(:default, Compose.stroke(color), linewidth(0.3mm*scale))

"""
    show_graph(locations, edges;
        colors=["black", "black", ...],
        texts=["1", "2", ...],
        format=SVG,
        bond_color="black",
        )

Plots vertices at `locations` with colors specified by `colors` and texts specified by `texts`.
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
        colors=nothing,
        texts = nothing,
        format=SVG, io=nothing,
        kwargs...)
    if length(locations) == 0
        dx, dy = 12cm, 12cm
        img = Compose.compose(context())
    else
        img, (dx, dy) = viz_atoms(locations, edges; colors=colors, texts=texts, config=GraphDisplayConfig(; config_plotting(locations)..., kwargs...))
    end
    if io === nothing
        Compose.set_default_graphic_size(dx, dy)
        return img
    else
        return format(io, dx, dy)(img)
    end
end
function show_graph(graph::SimpleGraph; locs, kwargs...)
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
function viz_atoms(locs, edges; colors, texts, config)
    rescaler = get_rescaler(locs, config.pad)
    img = _viz_atoms(rescaler.(locs), edges, colors, texts, config, getscale(rescaler))
    return fit_image(rescaler, config.image_size, img)
end

Base.@kwdef struct GraphDisplayConfig
    # line, node and text
    scale::Float64 = 1.0
    pad::Float64 = 1.5

    # node
    node_text_color::String = "black"
    node_stroke_color = "black"
    node_fill_color = "white"
    # bond
    bond_color::String = "black"
    # image size in cm
    image_size::Float64 = 12
end

function _viz_atoms(locs, edges, colors, texts, config, rescale)
    rescale = rescale * config.image_size * config.scale * 1.6
    if colors !== nothing
        @assert length(locs) == length(colors)
        node_styles = [default_node_style(rescale, config.node_stroke_color, color) for color in colors]
    else
        node_styles = fill(default_node_style(rescale, config.node_stroke_color, config.node_fill_color), length(locs))
    end
    if texts !== nothing
        @assert length(locs) == length(texts)
    end
    edge_style = default_bond_style(rescale, config.bond_color)
    text_style = default_text_style(rescale, config.node_text_color)
    img1 = Viznet.canvas() do
        for (i, node) in enumerate(locs)
            node_styles[i] >> node
            if config.node_text_color !== "transparent"
                text_style >> (node, texts === nothing ? "$i" : texts[i])
            end
        end
        for (i, j) in edges
            edge_style >> (locs[i], locs[j])
        end
    end
    Compose.compose(context(), img1)
end