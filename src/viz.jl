using Viznet
export vizeinsum, vizconfig
using Compose

function vizconfig(g::SimpleGraph; locs, config=zeros(Int, length(locs)), unit=1.0, graphsize=12cm, radius=0.03)
    vizconfig([string(v)=>locs[v] for v in LightGraphs.vertices(g)], [(e.src, e.dst) for e in edges(g)]; config=config, unit=unit, graphsize=graphsize, radius=radius)
end

function vizconfig(nodes, edges; config=zeros(Int, length(nodes)), unit=1.0, graphsize=12cm, radius=0.03)
	tb = textstyle(:default, fill("white"), fontsize(10pt*unit))
	nb = nodestyle(:circle, fill("black"), r=radius*unit)
	nb2 = nodestyle(:circle, fill("red"),r=radius*unit)
	eb = bondstyle(:default, linewidth(0.4mm*unit))
	img = canvas() do
		for (i, (t, p)) in enumerate(nodes)
			(config[i]==1 ? nb2 : nb) >> (p...,)
			tb >> ((p...,), t)
		end
		for (i,j) in edges
			eb >> ((nodes[i].second...,), (nodes[j].second...,))
		end
	end
    XMIN = minimum(x->x.second[1], nodes)
    YMIN = minimum(x->x.second[2], nodes)
    XMAX = maximum(x->x.second[1], nodes)
    YMAX = maximum(x->x.second[2], nodes)
    zoom_into(img, XMIN, XMAX, YMIN, YMAX; graphsize=graphsize)
end

function zoom_into(img, XMIN, XMAX, YMIN, YMAX; graphsize, rescale=1.0)
    sx = 0.8/(XMAX-XMIN) * rescale
    sy = 0.8/(YMAX-YMIN) * rescale
	Compose.set_default_graphic_size(graphsize*sy/sx, graphsize)
    Compose.compose(context(-((XMIN+XMAX)/2*sx-0.5), -((YMIN+YMAX)/2*sy-0.5), sx, sy), img)
end

function vizeinsum(nodes, edges; config=zeros(Int, length(nodes)), unit=1.0, graphsize=12cm, textcolor="black", textoffset=(0.0, 0.0), rescale=1.0)
    XMIN = minimum(x->x.second[1], nodes)
    YMIN = minimum(x->x.second[2], nodes)
    XMAX = maximum(x->x.second[1], nodes)
    YMAX = maximum(x->x.second[2], nodes)
	tb = textstyle(:default, fill(textcolor), fontsize(3pt*unit))
	nb = nodestyle(:circle, fill("black"), linewidth(0); r=0.005*unit)
	bt = nodestyle(:square, fill("black"), linewidth(0); r=0.010*unit)
	bt1 = nodestyle(:circle, fill("transparent"), stroke("black"), linewidth(0.08*unit); r=0.009*unit)
	nb2 = nodestyle(:circle, fill("red"), linewidth(0); r=0.005*unit)
	eb = bondstyle(:default, linewidth(0.08*unit))
	img = canvas() do
		for (i, (t, p)) in enumerate(nodes)
			(config[i]==1 ? nb2 : nb) >> p
			tb >> (p .+ textoffset, string(t))
		end
		for e in edges
			if length(e) >= 2
				center = mapreduce(x->nodes[x].second, (x,y)->(x .+ y), e) ./ length(e)
				for vi in e
					eb >> (nodes[vi].second, center)
				end
				bt >> center
			else
				bt1 >> nodes[e[1]].second
			end
		end
	end
    zoom_into(img, XMIN, XMAX, YMIN, YMAX; graphsize=graphsize, rescale=rescale)
end

function vizeinsum(::EinCode{ixs, iy}, locs::AbstractVector{<:Pair}; kwargs...) where {ixs, iy}
	vizeinsum(ixs, iy, Dict(locs); kwargs...)
end
function vizeinsum(code::NestedEinsum, locs::AbstractVector{<:Pair}; kwargs...)
	vizeinsum(OMEinsum.flatten(code), locs; kwargs...)
end
function vizeinsum(ixs::Tuple, iy::Tuple, locs::Dict; kwargs...)
	legs = unique!([Iterators.flatten(ixs)..., iy...])
	nodes = [l=>locs[l] for l in legs]
	edges = [map(i->findfirst(==(i), legs), ix) for ix in ixs]
	vizeinsum(nodes, edges; config=[l ∈ iy for l in legs], kwargs...)
end
