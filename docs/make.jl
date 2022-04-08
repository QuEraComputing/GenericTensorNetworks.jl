using Pkg
using GraphTensorNetworks
using GraphTensorNetworks: TropicalNumbers, Polynomials, Mods, OMEinsum, OMEinsumContractionOrders
using Documenter
using DocThemeIndigo
using Literate

for each in readdir(pkgdir(GraphTensorNetworks, "examples"))
    input_file = pkgdir(GraphTensorNetworks, "examples", each)
    endswith(input_file, ".jl") || continue
    @info "building" input_file
    output_dir = pkgdir(GraphTensorNetworks, "docs", "src", "tutorials")
    @info "executing" input_file
    Literate.markdown(input_file, output_dir; name=each[1:end-3], execute=false)
end

indigo = DocThemeIndigo.install(GraphTensorNetworks)
DocMeta.setdocmeta!(GraphTensorNetworks, :DocTestSetup, :(using GraphTensorNetworks); recursive=true)

makedocs(;
    modules=[GraphTensorNetworks, TropicalNumbers, Polynomials, Mods, OMEinsum, OMEinsumContractionOrders],
    authors="Jinguo Liu",
    repo="https://github.com/QuEraComputing/GraphTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GraphTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://QuEraComputing.github.io/GraphTensorNetworks.jl",
        assets=String[indigo],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Independent set problem" => "tutorials/IndependentSet.md",
            "Maximal independent set problem" => "tutorials/MaximalIS.md",
            "Cutting problem" => "tutorials/MaxCut.md",
            "Matching problem" => "tutorials/Matching.md",
            "Binary paint shop problem" => "tutorials/PaintShop.md",
            "Coloring problem" => "tutorials/Coloring.md",
            "Dominating set problem" => "tutorials/DominatingSet.md",
            "Satisfiability problem" => "tutorials/Satisfiability.md",
            "Other problems" => "tutorials/Others.md",
        ],
        "Performance Tips" => "performancetips.md",
        "References" => "ref.md",
    ],
    doctest=false,
)

deploydocs(;
    repo="github.com/QuEraComputing/GraphTensorNetworks.jl",
)
