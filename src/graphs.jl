using LightGraphs, OMEinsumContractionOrders
export random_regular_graph, diagonal_coupled_graph, isindependentset
export square_lattice_graph, unitdisk_graph

isindependentset(g, v) = !any(e->v[e.src] == 1 && v[e.dst] == 1, edges(g))

function square_lattice_graph(mask::AbstractMatrix{Bool})
    locs = [(i, j) for i=1:size(mask, 1), j=1:size(mask, 2) if mask[i,j]]
    unitdisk_graph(locs, 1.1)
end

function random_diagonal_coupled_graph(m::Int, n::Int, ρ::Real)
    diagonal_coupled_graph(rand(m, n) .< ρ)
end

function diagonal_coupled_graph(mask::AbstractMatrix{Bool})
    locs = [(i, j) for i=1:size(mask, 1), j=1:size(mask, 2) if mask[i,j]]
    unitdisk_graph(locs, 1.5)
end

function unitdisk_graph(locs::Vector, unit::Real)
    n = length(locs)
    g = SimpleGraph(n)
    for i=1:n, j=i+1:n
        if sum(abs2, locs[i] .- locs[j]) < unit ^ 2
            add_edge!(g, i, j)
        end
    end
    return g
end