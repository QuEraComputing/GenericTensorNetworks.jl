### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 77211b40-3e1b-46dc-b803-07c0ba8cbf60
using Pkg; Pkg.activate()

# ╔═╡ ababcad9-415c-4a1e-97a5-322a15e98c80
using Revise, GenericTensorNetworks, Graphs, PlutoUI

# ╔═╡ 9d955aa6-f8dc-4a16-a883-60d3ddf61636
graph = Graphs.smallgraph(:petersen)

# ╔═╡ 83b69249-53cd-4f88-953d-caba7d4a4b25
# set the vertex locations manually instead of using the default spring layout
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

# ╔═╡ 1520b6a8-79cb-49c2-8ea8-a84bb2120158
locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.5, i) for i=0:4]...]

# ╔═╡ 3bd9bf83-2dd2-4800-bc56-dd41985a82a0
show_graph(graph; locs=locations, scale=0.6)

# ╔═╡ 8138f684-cdbd-4d43-aff3-f1d80ae090e1
problem = IndependentSet(graph; optimizer=TreeSA());

# ╔═╡ 23744442-aeb1-4bdf-b87f-69c27853f2a7
max_config = solve(problem, SingleConfigMax())[]

# ╔═╡ 204d0632-32b1-416d-903d-8d253f65d44f
single_solution = max_config.c.data

# ╔═╡ d5446d05-a80c-4d37-8549-f440b7067d6e
show_graph(graph; locs=locations, vertex_colors=
    [iszero(single_solution[i]) ? "white" : "red" for i=1:nv(graph)], scale=0.6)

# ╔═╡ 5d71602e-fd6e-4a63-9cbe-f359538bf20e
show_graph(graph; locs=locations, vertex_colors=
    [iszero(x) ? "white" : "red" for x in [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]], scale=0.6)

# ╔═╡ 847bcb04-a706-4d45-92b9-8ef754520b5a
wrong_solution = [true, false, true, false, false, true, false, false, false, true]

# ╔═╡ 5ebb1b2b-9fe7-44b5-bb0d-10889950fe56
show_graph(graph; locs=locations, vertex_colors=
    [iszero(wrong_solution[i]) ? "white" : "red" for i=1:nv(graph)], scale=0.6)

# ╔═╡ 90f10354-a77f-43c3-9033-1e175886fc2f
poly = solve(problem, GraphPolynomial())[]

# ╔═╡ 91fcc8db-b11b-4c1e-bec2-85507536c86b
max_configs2 = solve(problem, ConfigsMax(5))[]

# ╔═╡ 5e952998-a7d3-44a6-bcdb-b7cd6a0b7f74
@bind iconfig NumberField(0:4)

# ╔═╡ 64f35113-5590-4e0d-99ff-6350083c77a1
show_gallery(graph, (ceil(Int, length(max_configs2.coeffs[iconfig+1])/5), 5); locs=locations, vertex_configs=max_configs2.coeffs[iconfig+1], image_size=3.5, scale=0.6)

# ╔═╡ Cell order:
# ╠═77211b40-3e1b-46dc-b803-07c0ba8cbf60
# ╠═ababcad9-415c-4a1e-97a5-322a15e98c80
# ╠═9d955aa6-f8dc-4a16-a883-60d3ddf61636
# ╠═83b69249-53cd-4f88-953d-caba7d4a4b25
# ╠═1520b6a8-79cb-49c2-8ea8-a84bb2120158
# ╠═3bd9bf83-2dd2-4800-bc56-dd41985a82a0
# ╠═8138f684-cdbd-4d43-aff3-f1d80ae090e1
# ╠═23744442-aeb1-4bdf-b87f-69c27853f2a7
# ╠═204d0632-32b1-416d-903d-8d253f65d44f
# ╠═d5446d05-a80c-4d37-8549-f440b7067d6e
# ╠═5d71602e-fd6e-4a63-9cbe-f359538bf20e
# ╠═847bcb04-a706-4d45-92b9-8ef754520b5a
# ╠═5ebb1b2b-9fe7-44b5-bb0d-10889950fe56
# ╠═90f10354-a77f-43c3-9033-1e175886fc2f
# ╠═91fcc8db-b11b-4c1e-bec2-85507536c86b
# ╟─5e952998-a7d3-44a6-bcdb-b7cd6a0b7f74
# ╠═64f35113-5590-4e0d-99ff-6350083c77a1
