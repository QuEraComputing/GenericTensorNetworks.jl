function loadjson_graph(dict)
    nv = dict["nv"]
    edges = dict["edges"]
    graph = SimpleGraph(nv)
    for (i, j) in edges
        add_edge!(graph, i, j)
    end
    return graph
end

function loadjson_optimizer(dict)
    method = dict["method"]
    dict = dict["method.$method"]  # extract the related method
    if method == "TreeSA"
        return TreeSA(;
            sc_target=dict["sc_target"],
            sc_weight=dict["sc_weight"],
            rw_weight=dict["rw_weight"],
            ntrials=dict["ntrials"],
            niters=dict["niters"],
            nslices=dict["nslices"],
            βs=dict["betas"]
        )
    elseif method == "GreedyMethod"
        return GreedyMethod(;
            nrepeat=dict["nrepeat"]
        )
    else
        error("")
    end
end

function tojson(data)
    if data isa Tropical
        return Dict("size"=>data.n)
    elseif data isa ExtendedTropical{K,<:Tropical} where K
        return [Dict("size"=>x.n) for x in data.orders]
    elseif data isa Real
        return Dict("count"=>data)
    elseif data isa CountingTropical{T, <:Real} where T
        return Dict("size"=>data.n, "count"=>data.c)
    elseif data isa TruncatedPoly{K, <:Real} where {K}
        K = length(data.coeffs)
        return [Dict("size"=>size, "count"=>count) for (size, count) in zip(data.maxorder-K+1:data.maxorder, data.coeffs)]
    elseif data isa Polynomial
        return [Dict("size"=>i-1, "count"=>c) for (i, c) in enumerate(data.coeffs)]
    elseif data isa CountingTropical{T, <:ConfigSampler} where T
        return Dict("size"=>data.n, "config"=>[data.c.data...])
    elseif data isa ExtendedTropical{K, <:CountingTropical{T, <:ConfigSampler}} where {K,T}
        return [Dict(
                    "size"=>di.n,
                    "config"=>[di.c.data...]
                    ) for di in data.orders
                ]
    elseif data isa CountingTropical{T, <:ConfigEnumerator} where T
        return Dict("size"=>data.n, "configs"=>[[x...] for x in data.c])
    elseif data isa TruncatedPoly{K, <:ConfigEnumerator} where K
        sizes = collect(data.maxorder-length(data.coeffs)+1:data.maxorder)
        configs = [data.coeffs...]
        return [Dict("size"=>size, "configs"=>[[x...] for x in config]) for (size, config) in zip(sizes, configs)]
    elseif data isa ConfigEnumerator
        return Dict("configs"=>[[x...] for x in data])
    elseif data isa CountingTropical{T, <:SumProductTree} where T
        return Dict("size"=>data.n, "configs"=>[[x...] for x in collect(data.c)])
    elseif data isa TruncatedPoly{K, <:SumProductTree} where K
        sizes = collect(data.maxorder-length(data.coeffs)+1:data.maxorder)
        configs = collect.([data.coeffs...])
        return [Dict("size"=>size, "configs"=>[[x...] for x in config]) for (size, config) in zip(sizes, configs)]
    elseif data isa SumProductTree
        return Dict("configs"=>[[x...] for x in collect(data)])
    else
        error("unknown data type: `$(typeof(data))`.")
    end
end

function application(dict)
    dict = verify(application_verifier, dict)
    api = dict["api"]
    if api == "solve"
        return app_solve(dict["api.solve"])
    elseif api == "graph"
        return app_graph(dict["api.graph"])
    elseif api == "opteinsum"
        return app_opteinsum(dict["api.opteinsum"])
    elseif api == "help"
        Dict("help"=>string(application_verifier))
    end
end

function app_opteinsum(dict)
    code = GenericTensorNetworks.OMEinsum.DynamicEinCode(dict["inputs"], dict["output"])
    optimizer = loadjson_optimizer(dict["optimizer"])
    simp = get(dict, "simplifier", "")  # optional
    simplifier = if simp == "MergeGreedy"
        MergeGreedy()
    elseif simp == "MergeVectors"
        MergeVectors()
    else
        nothing
    end
    code = optimize_code(code, dict["sizes"], optimizer, simplifier)
    tc, sc, rw = timespacereadwrite_complexity(code, dict["sizes"])
    @info("time, space, RW complexitys are: $tc, $sc, $rw")
    d = _todict(code)
    d["time-complexity"] = tc
    d["space-complexity"] = sc
    d["readwrite-complexity"] = rw
    return d
end

function app_solve(dict)
    graph = loadjson_graph(dict["graph"])
    PROB = parseproblem(dict["problem"])

    if haskey(dict, "tensornetwork")   # with tensor network
        code = _fromdict(dict["tensornetwork"])
        instance = PROB(code,
                graph,  # dummy one
                get(dict, "weights", NoWeight()),  # weights is optional
                Dict{GenericTensorNetworks.OMEinsum.labeltype(code), Int}(dict["fixedvertices"])
            )
    else
        optimizer = loadjson_optimizer(dict["optimizer"])

        instance = PROB(graph;
                weights=get(dict, "weights", NoWeight()),  # weights is optional
                openvertices=dict["openvertices"],
                fixedvertices=dict["fixedvertices"],
                optimizer,
                simplifier=MergeGreedy()
        )
    end
    tc, sc, rw = timespacereadwrite_complexity(instance)
    @info("time, space, RW complexitys are: $tc, $sc, $rw")

    # special property
    if dict["property"] == "tensornetwork"
        d = _todict(instance.code)
        d["time-complexity"] = tc
        d["space-complexity"] = sc
        d["readwrite-complexity"] = rw
        return d
    end

    property = parseproperty(dict["property"])
    cudadevice = dict["cudadevice"]
    cudadevice >= 0 && CUDA.device!(cudadevice)
    return tojson.(solve(instance, property; usecuda=cudadevice>=0))
end

const problem_dict = Dict{String,Any}(
    "IndependentSet" => IndependentSet,
    "MaximalIS" => MaximalIS,
    "DominatingSet" => DominatingSet,
    "MaxCut" => MaxCut,
    "Coloring{3}" => Coloring{3},
    "Matching" => Matching,
)
parseproblem(s::String) = problem_dict[s]

const property_dict = Dict{String,Any}(
    "SizeMax" => SizeMax,
    "SizeMin" => SizeMin,
    "SingleConfigMax" => SingleConfigMax,
    "SingleConfigMin" => SingleConfigMin,
    "ConfigsMax" => ConfigsMax,
    "ConfigsMin" => ConfigsMin,
    "ConfigsAll" => ConfigsAll,
    "GraphPolynomial" => GraphPolynomial,
    "CountingMax" => CountingMax,
    "CountingMin" => CountingMin,
    "CountingAll" => CountingAll,
)
function parseproperty(property::String)
    m1 = match(r"(\w+)(\d+)", property)
    if m1 === nothing
        m2 = match(r"(\w+)Tree", property)
        if m2 === nothing
            return property_dict[property]()
        else
            return property_dict[m2[1]](; tree_storage=true)
        end
    else
        m2 = match(r"(\w+)Tree", m1[1])
        K = parse(Int, m1[2])
        if m2 === nothing
            return property_dict[m1[1]](K)
        else
            return property_dict[m2[1]](K; tree_storage=true)
        end
    end
end

function app_graph(dict)
    type = dict["type"]
    cfg = dict["type.$type"]
    g = if type == "kings"
        Random.seed!(cfg["seed"])
        random_diagonal_coupled_graph(cfg["m"], cfg["n"], cfg["filling"])
    elseif type == "square"
        Random.seed!(cfg["seed"])
        random_square_lattice_graph(cfg["m"], cfg["n"], cfg["filling"])
    elseif type == "smallgraph"
        Random.seed!(cfg["seed"])
        smallgraph(Symbol(cfg["name"]))
    elseif type == "regular"
        random_regular_graph(cfg["n"], cfg["d"], seed=cfg["seed"])
    else
        error("unkown graph type: $type")
    end
    return Dict("nv"=>nv(g), "edges"=>[[e.src, e.dst] for e in edges(g)])
end

function _fromdict(dict)
    lt = string(dict["label-type"])
    LT = if lt == "Char"
        Char
    elseif lt ∈ ("Int64", "Int", "Int32")
        Int
    else
        error("label type `$lt` not known.")
    end
    ne = OMEinsumContractionOrders.fromdict(LT, dict["tree"])
    if haskey(dict, "slices")
        return SlicedEinsum(Slicing(_convert.(LT, dict["slices"])), ne)
    else
        return ne
    end
end

function _todict(ne::SlicedEinsum)
    dict = OMEinsumContractionOrders._todict(ne.eins)
    dict["slices"] = ne.slicing.legs
    return dict
end
function _todict(ne::NestedEinsum)
    return OMEinsumContractionOrders._todict(ne)
end
