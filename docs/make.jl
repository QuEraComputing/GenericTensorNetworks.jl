using Pkg
using GenericTensorNetworks
using GenericTensorNetworks: TropicalNumbers, Polynomials, Mods, OMEinsum, OMEinsumContractionOrders
using Documenter
using DocThemeIndigo
using PlutoStaticHTML
using Literate

for each in readdir(pkgdir(GenericTensorNetworks, "examples"))
    input_file = pkgdir(GenericTensorNetworks, "examples", each)
    endswith(input_file, ".jl") || continue
    @info "building" input_file
    output_dir = pkgdir(GenericTensorNetworks, "docs", "src", "tutorials")
    @info "executing" input_file
    Literate.markdown(input_file, output_dir; name=each[1:end-3], execute=false)
end

"""Run all Pluto notebooks (".jl" files) in `notebook_dir` and write outputs to HTML files."""
function build()
    notebook_dir = joinpath(pkgdir(GenericTensorNetworks), "notebooks")
    target_dir = joinpath(pkgdir(GenericTensorNetworks), "docs", "src", "notebooks")
    cp(notebook_dir, target_dir)
    println("Building tutorials")
    # Evaluate notebooks in the same process to avoid having to recompile from scratch each time.
    # This is similar to how Documenter and Franklin evaluate code.
    # Note that things like method overrides and other global changes may leak between notebooks!
    use_distributed = false
    output_format = documenter_output
    bopts = BuildOptions(target_dir; use_distributed, output_format)
    build_notebooks(bopts)
    return nothing
end

indigo = DocThemeIndigo.install(GenericTensorNetworks)
DocMeta.setdocmeta!(GenericTensorNetworks, :DocTestSetup, :(using GenericTensorNetworks); recursive=true)

makedocs(;
    modules=[GenericTensorNetworks, TropicalNumbers, Mods, OMEinsum, OMEinsumContractionOrders],
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
            "Independent set problem" => "tutorials/IndependentSet.md",
            "Maximal independent set problem" => "tutorials/MaximalIS.md",
            "Cutting problem" => "tutorials/MaxCut.md",
            "Vertex Matching problem" => "tutorials/Matching.md",
            "Binary paint shop problem" => "tutorials/PaintShop.md",
            "Coloring problem" => "tutorials/Coloring.md",
            "Dominating set problem" => "tutorials/DominatingSet.md",
            "Satisfiability problem" => "tutorials/Satisfiability.md",
            "Set covering problem" => "tutorials/SetCovering.md",
            "Set packing problem" => "tutorials/SetPacking.md",
            #"Other problems" => "tutorials/Others.md",
        ],
        "Topics" => [
            "Make extensions" => "extension.md",
            "Save and load solutions" => "tutorials/saveload.md",
            "Sum product tree representation" => "sumproduct.md",
            "Weighted problems" => "tutorials/weighted.md",
            "Open degree of freedoms" => "tutorials/open.md"
        ],
        "Performance Tips" => "performancetips.md",
        "References" => "ref.md",
    ],
    doctest=false,
)

deploydocs(;
    repo="github.com/QuEraComputing/GenericTensorNetworks.jl",
)
