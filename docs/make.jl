using Pkg
using GenericTensorNetworks
using GenericTensorNetworks: TropicalNumbers, Polynomials, Mods, OMEinsum, OMEinsum.OMEinsumContractionOrders, LuxorGraphPlot
using Documenter
using DocThemeIndigo
using PlutoStaticHTML
using Literate

for each in readdir(pkgdir(GenericTensorNetworks, "examples"))
    input_file = pkgdir(GenericTensorNetworks, "examples", each)
    endswith(input_file, ".jl") || continue
    @info "building" input_file
    output_dir = pkgdir(GenericTensorNetworks, "docs", "src", "generated")
    Literate.markdown(input_file, output_dir; name=each[1:end-3], execute=false)
end

let
    """Run all Pluto notebooks (".jl" files) in `notebook_dir` and write outputs to HTML files."""
    notebook_dir = joinpath(pkgdir(GenericTensorNetworks), "notebooks")
    target_dir = joinpath(pkgdir(GenericTensorNetworks), "docs", "src", "notebooks")
    cp(notebook_dir, target_dir; force=true)
    @info "Building tutorials"
    # Evaluate notebooks in the same process to avoid having to recompile from scratch each time.
    # This is similar to how Documenter and Franklin evaluate code.
    # Note that things like method overrides and other global changes may leak between notebooks!
    use_distributed = true
    output_format = documenter_output
    bopts = BuildOptions(target_dir; use_distributed, output_format)
    build_notebooks(bopts)
    return nothing
end

indigo = DocThemeIndigo.install(GenericTensorNetworks)
DocMeta.setdocmeta!(GenericTensorNetworks, :DocTestSetup, :(using GenericTensorNetworks); recursive=true)

makedocs(;
    modules=[GenericTensorNetworks, TropicalNumbers, Mods, OMEinsum, OMEinsumContractionOrders, LuxorGraphPlot],
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
            #"Other problems" => "generated/Others.md",
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
)

deploydocs(;
    repo="github.com/QuEraComputing/GenericTensorNetworks.jl",
)
