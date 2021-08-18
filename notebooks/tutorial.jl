### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 54d71646-ffad-11eb-3866-5f1c0bc5d0bf
begin
	using Pkg
	Pkg.activate(dirname(pwd()))
	using Revise, GraphTensorNetworks
end

# ╔═╡ 83fc0a3c-ef51-4d5c-9c46-baa645f80fd2
locs = [rand(2) for i=1:70];

# ╔═╡ 1522938d-d60b-4133-8625-5a09615a7b26
g = unitdisk_graph(locs, 0.2)

# ╔═╡ 05ac5610-4a1d-4433-ba0d-1d13fd8a6733
vizconfig(g; locs=locs, unit=0.5)

# ╔═╡ b6e916e2-7d43-4085-a3c8-5c57069e8384
gp = Independence(g; optmethod=:auto);

# ╔═╡ 03840394-3a5b-4c24-8b52-15f7e400f900
timespace_complexity(gp)

# ╔═╡ 11c71de9-caa9-40c6-84cb-a0f29d873805
results = solutions(gp, CountingTropical{Float64}; all=false)

# ╔═╡ edfc5600-9a4a-4657-b165-8eb3e8540edf
vizconfig(g; locs=locs, unit=0.5, config=results[].c.data)

# ╔═╡ 3666f629-7a87-4b50-aaf1-cc128d598aaf
md"## Different Algebras"

# ╔═╡ 51f0f187-d14d-48ab-ab30-96e71d796959
md"There are commutative semiring algebras defined in the package. For each type, we have implemented two operations `+` and `*` and two special elements `zero` and `one` on it."

# ╔═╡ a5b63d08-2917-4e9f-9c5c-b5f17181d2aa
md"### Polynomials
`Polynomial` type is defined in package [Polynomials.jl](https://github.com/JuliaMath/Polynomials.jl)
"

# ╔═╡ 0d3b2752-c800-43f3-b241-9c96084f1fc9
xp, yp = Polynomial([1.0, 2.0, 3.0]), Polynomial([2.0, 1.0])

# ╔═╡ e779a111-ba70-4350-88bc-2b94d11741c6
xp + yp

# ╔═╡ 677895f4-be68-4565-97ca-6f4e31c56819
xp * yp

# ╔═╡ e1bc2d30-919a-4675-a674-c82c534c4ccf
zero(Polynomial{Float64,:x})

# ╔═╡ c560a001-8fbb-4fbb-94fc-c68cf5e67c1e
one(Polynomial{Float64,:x})

# ╔═╡ ae3c308d-40cc-4cbe-ac37-f0f2cd0eecbe
md"### Polynomial truncated to largest two orders"

# ╔═╡ 69779c4f-b939-4d0c-8548-1dc31578949c
xp2, yp2 = Max2Poly(2.0, 3.0, 2.0), Max2Poly(2.0, 1.0, 1.0)

# ╔═╡ b7f7453f-445a-40e3-9639-1c93e5b257ff
zero(Max2Poly{Float64,Float64})

# ╔═╡ 9da24765-b182-4f60-a318-d7a808e2872a
one(Max2Poly{Float64,Float64})

# ╔═╡ 731012e5-257f-44e1-8888-16a6cbe50279
zero(Max2Poly{Float64,Float64})

# ╔═╡ 2761c355-2b10-4437-8496-a0c260bbac2c
md"### Tropical algebra"

# ╔═╡ 7c584a2c-ab29-4e04-971b-06aeeb61fd0f
xt, yt = Tropical(2.0), Tropical(3.0)

# ╔═╡ e5b166f4-7307-4712-8790-0eb914ff839c
xt + yt  # same as regular `max`

# ╔═╡ 50dadd8b-e73f-461b-baa9-8e1c5052e7fd
xt * yt  # same as regular `+`

# ╔═╡ ff6ffae0-dfd4-44ba-bcfc-02435ef4221a
zero(Tropical{Float64})

# ╔═╡ ab5b237c-b4aa-4cc9-91ca-b51a3d3d9d0e
one(Tropical{Float64})

# ╔═╡ ce303530-bbe8-43fc-add0-2481f78a2e9f
md"### Tropical algebra with counting"

# ╔═╡ aefff7b7-f372-4eba-b9c9-effbb547665e
xct, yct = CountingTropical(2.0, 2.0), CountingTropical(3.0, 5.0)

# ╔═╡ 585d7ccb-5f23-4dc0-a482-076721f0be27
xct + yct

# ╔═╡ 82a2eee5-451e-4679-a11a-74a690c13817
xct * yct

# ╔═╡ b26358f3-d752-4df3-9a50-526a91615796
zero(CountingTropical{Float64,Float64})

# ╔═╡ be5b52c5-787f-4521-b14a-25f54b439d79
one(CountingTropical{Float64,Float64})

# ╔═╡ 539503f4-68f5-4f83-918f-d21163e4df6f
md"### Configuration and Set algebra"

# ╔═╡ a77d1d41-3964-41cb-ad1a-71db411537ee
md"Let us first define bit strings"

# ╔═╡ 6fd32061-0a0c-46b7-9bcb-00736d372f8c
bs1, bs2, bs3 = StaticBitVector([1,0,1,1,0]), StaticBitVector([0,0,0,1,1]), StaticBitVector([1,1,1,1,0])

# ╔═╡ 2a800fa6-4967-4d01-aac1-548f852e7cad
cs1, cs2 = ConfigSampler(bs1), ConfigSampler(bs2)

# ╔═╡ 74203e2b-b231-4a45-8d74-a3ee3299cad8
cs1 + cs2

# ╔═╡ 6392edb7-c86c-40e5-8dc1-cd3cbbf436c2
cs1 * cs2

# ╔═╡ 97b8e445-99b5-488e-9437-76618cf6321e
zero(cs1)

# ╔═╡ 07f2dff4-d9de-49a2-b8f4-e29b43df5e23
one(cs2)

# ╔═╡ 31f866f8-9ad0-439d-8c08-29a27efd3948
ce1, ce2 = ConfigEnumerator([bs1, bs2]), ConfigEnumerator([bs3])

# ╔═╡ ee28ad87-479e-49e0-a534-6bfca71dfb96
ce1 + ce2

# ╔═╡ f783b71b-5b7e-457a-be66-015f0e602ea2
ce1 * ce2

# ╔═╡ 554cebaf-bbe8-4a02-905b-e4b508e0c064
zero(typeof(ce1))

# ╔═╡ 12619536-8ea4-465b-b60d-3230bfb7b570
one(typeof(ce2))

# ╔═╡ Cell order:
# ╠═54d71646-ffad-11eb-3866-5f1c0bc5d0bf
# ╠═83fc0a3c-ef51-4d5c-9c46-baa645f80fd2
# ╠═1522938d-d60b-4133-8625-5a09615a7b26
# ╠═05ac5610-4a1d-4433-ba0d-1d13fd8a6733
# ╠═b6e916e2-7d43-4085-a3c8-5c57069e8384
# ╠═03840394-3a5b-4c24-8b52-15f7e400f900
# ╠═11c71de9-caa9-40c6-84cb-a0f29d873805
# ╠═edfc5600-9a4a-4657-b165-8eb3e8540edf
# ╟─3666f629-7a87-4b50-aaf1-cc128d598aaf
# ╟─51f0f187-d14d-48ab-ab30-96e71d796959
# ╟─a5b63d08-2917-4e9f-9c5c-b5f17181d2aa
# ╠═0d3b2752-c800-43f3-b241-9c96084f1fc9
# ╠═e779a111-ba70-4350-88bc-2b94d11741c6
# ╠═677895f4-be68-4565-97ca-6f4e31c56819
# ╠═e1bc2d30-919a-4675-a674-c82c534c4ccf
# ╠═c560a001-8fbb-4fbb-94fc-c68cf5e67c1e
# ╟─ae3c308d-40cc-4cbe-ac37-f0f2cd0eecbe
# ╠═69779c4f-b939-4d0c-8548-1dc31578949c
# ╠═b7f7453f-445a-40e3-9639-1c93e5b257ff
# ╠═9da24765-b182-4f60-a318-d7a808e2872a
# ╠═731012e5-257f-44e1-8888-16a6cbe50279
# ╟─2761c355-2b10-4437-8496-a0c260bbac2c
# ╠═7c584a2c-ab29-4e04-971b-06aeeb61fd0f
# ╠═e5b166f4-7307-4712-8790-0eb914ff839c
# ╠═50dadd8b-e73f-461b-baa9-8e1c5052e7fd
# ╠═ff6ffae0-dfd4-44ba-bcfc-02435ef4221a
# ╠═ab5b237c-b4aa-4cc9-91ca-b51a3d3d9d0e
# ╟─ce303530-bbe8-43fc-add0-2481f78a2e9f
# ╠═aefff7b7-f372-4eba-b9c9-effbb547665e
# ╠═585d7ccb-5f23-4dc0-a482-076721f0be27
# ╠═82a2eee5-451e-4679-a11a-74a690c13817
# ╠═b26358f3-d752-4df3-9a50-526a91615796
# ╠═be5b52c5-787f-4521-b14a-25f54b439d79
# ╟─539503f4-68f5-4f83-918f-d21163e4df6f
# ╟─a77d1d41-3964-41cb-ad1a-71db411537ee
# ╠═6fd32061-0a0c-46b7-9bcb-00736d372f8c
# ╠═2a800fa6-4967-4d01-aac1-548f852e7cad
# ╠═74203e2b-b231-4a45-8d74-a3ee3299cad8
# ╠═6392edb7-c86c-40e5-8dc1-cd3cbbf436c2
# ╠═97b8e445-99b5-488e-9437-76618cf6321e
# ╠═07f2dff4-d9de-49a2-b8f4-e29b43df5e23
# ╠═31f866f8-9ad0-439d-8c08-29a27efd3948
# ╠═ee28ad87-479e-49e0-a534-6bfca71dfb96
# ╠═f783b71b-5b7e-457a-be66-015f0e602ea2
# ╠═554cebaf-bbe8-4a02-905b-e4b508e0c064
# ╠═12619536-8ea4-465b-b60d-3230bfb7b570
