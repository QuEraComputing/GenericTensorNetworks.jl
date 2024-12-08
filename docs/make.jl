using Pkg
using GenericTensorNetworks
using GenericTensorNetworks: TropicalNumbers, Polynomials, OMEinsum, OMEinsum.OMEinsumContractionOrders, LuxorGraphPlot, ProblemReductions
using Documenter
using DocThemeIndigo
using Literate

for each in readdir(pkgdir(GenericTensorNetworks, "examples"))
    input_file = pkgdir(GenericTensorNetworks, "examples", each)
    endswith(input_file, ".jl") || continue
    @info "building" input_file
    output_dir = pkgdir(GenericTensorNetworks, "docs", "src", "generated")
    Literate.markdown(input_file, output_dir; name=each[1:end-3], execute=false)
end

indigo = DocThemeIndigo.install(GenericTensorNetworks)
DocMeta.setdocmeta!(GenericTensorNetworks, :DocTestSetup, :(using GenericTensorNetworks); recursive=true)

makedocs(;
    modules=[GenericTensorNetworks, ProblemReductions, TropicalNumbers, OMEinsum, OMEinsumContractionOrders, LuxorGraphPlot],
    authors="Jinguo Liu",
    repo="https://github.com/QuEraComputing/GenericTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GenericTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://QuEraComputing.github.io/GenericTensorNetworks.jl",
        assets=String[indigo],
    ),
    pages=[
        "Home" => "index.md",
        "Problems" => [
            "Independent set problem" => "generated/IndependentSet.md",
            "Maximal independent set problem" => "generated/MaximalIS.md",
            "Spin glass problem" => "generated/SpinGlass.md",
            "Cutting problem" => "generated/MaxCut.md",
            "Vertex matching problem" => "generated/Matching.md",
            "Binary paint shop problem" => "generated/PaintShop.md",
            "Coloring problem" => "generated/Coloring.md",
            "Dominating set problem" => "generated/DominatingSet.md",
            "Satisfiability problem" => "generated/Satisfiability.md",
            "Set covering problem" => "generated/SetCovering.md",
            "Set packing problem" => "generated/SetPacking.md",
        ],
        "Topics" => [
            "Gist" => "gist.md",
            "Save and load solutions" => "generated/saveload.md",
            "Sum product tree representation" => "sumproduct.md",
            "Weighted problems" => "generated/weighted.md",
            "Open and fixed degrees of freedom" => "generated/open.md"
        ],
        "Performance Tips" => "performancetips.md",
        "References" => "ref.md",
    ],
    doctest=false,
    warnonly = :missing_docs,
)

deploydocs(;
    repo="github.com/QuEraComputing/GenericTensorNetworks.jl",
)
