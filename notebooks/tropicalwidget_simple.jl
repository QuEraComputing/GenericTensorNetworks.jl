### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 53ce616e-8055-11eb-0caf-e968c041ff4f
begin
	using Revise
	using Pkg
	Pkg.activate(dirname(pwd()))
	using LightGraphs # graph support
	using PlutoUI
	using Viznet, Compose
	using TropicalNumbers, OMEinsum, GraphTensorNetworks#, TropicalGEMM
end

# ╔═╡ 100e3eb1-9d5c-4ae0-9a1b-d279927443fc
md"## We have a unit disk gadget"

# ╔═╡ efdc5d3e-58e0-410d-b15a-2363e9c9f156
locs = let
	a = 0.12
	ymid = xmid = 0.5
	X = 0.33
	Y = 0.17
	D = 0.15
	y = [ymid-Y, ymid-Y+D, ymid-a/2, ymid+a/2, ymid+Y-D, ymid+Y]
	x = [xmid-X, xmid-X+D, xmid-1.5a, xmid-a/2, xmid+a/2, xmid+1.5a, xmid+X-D, xmid+X]
	xmin, xmax, ymin, ymax = x[1], x[end], y[1], y[end]
	[(xmid, y[1]), (xmin, ymid), (xmid, ymax), (xmax, ymid),
		(x[3], y[3]), (x[4], y[3]),
		(x[5], y[3]), (x[6], y[3]), (x[3], y[4]),
		(x[4], y[4]), (x[5], y[4]), (x[6], y[4])]
end

# ╔═╡ 9d0ba8b7-5a34-418d-97cf-863f376f8453
graph = unitdisk_graph(locs, 0.23) # SimpleGraph

# ╔═╡ 78ee8772-83e4-40fb-8151-0123370481d9
vizconfig(graph; locs=locs, config=rand(Bool, 12), graphsize=8cm)

# ╔═╡ e068f0b8-7b2c-49b2-94d1-83dead406326
gp = Independence(graph; outputs=(1,2,3,4))

# ╔═╡ 90c88c7f-e0c7-4845-a8b9-7f7562bd256f
solve(gp, :config_single)

# ╔═╡ f657f321-255e-44f2-a1d5-9093fa8eca28
md"## Obtaining configurations"

# ╔═╡ fdbe425d-5539-4a38-8f8f-aa98fb20ce64
@doc run_task

# ╔═╡ adb6a76a-18a2-46a8-8ad6-352cac1d2efc
res_configs = solve(gp, :config_all);

# ╔═╡ 6ae40df1-f576-44f0-9d48-561dab3cb899
md"""
 $(@bind s1 CheckBox()) ``s_1``
$(@bind s2 CheckBox()) ``s_2``
$(@bind s3 CheckBox()) ``s_3``
$(@bind s4 CheckBox()) ``s_4``
"""

# ╔═╡ 21aa560c-d2f0-467b-a0bf-7b77024aeeaa
let
	n = length(res_configs[s1+1,s2+1,s3+1,s4+1].c.data)
	md"$(@bind index Slider(1:n; show_value=true, default=n))"
end

# ╔═╡ caa3bc03-f6ff-4ebe-9608-518b8924adae
vizconfig(graph; locs=locs,config=res_configs[s1+1,s2+1,s3+1,s4+1].c.data[index], graphsize=8cm)

# ╔═╡ Cell order:
# ╠═53ce616e-8055-11eb-0caf-e968c041ff4f
# ╟─100e3eb1-9d5c-4ae0-9a1b-d279927443fc
# ╠═efdc5d3e-58e0-410d-b15a-2363e9c9f156
# ╠═9d0ba8b7-5a34-418d-97cf-863f376f8453
# ╠═78ee8772-83e4-40fb-8151-0123370481d9
# ╠═e068f0b8-7b2c-49b2-94d1-83dead406326
# ╠═90c88c7f-e0c7-4845-a8b9-7f7562bd256f
# ╟─f657f321-255e-44f2-a1d5-9093fa8eca28
# ╟─fdbe425d-5539-4a38-8f8f-aa98fb20ce64
# ╠═adb6a76a-18a2-46a8-8ad6-352cac1d2efc
# ╟─6ae40df1-f576-44f0-9d48-561dab3cb899
# ╟─21aa560c-d2f0-467b-a0bf-7b77024aeeaa
# ╠═caa3bc03-f6ff-4ebe-9608-518b8924adae
