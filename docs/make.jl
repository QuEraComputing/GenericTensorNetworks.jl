using Pkg
using GraphTensorNetworks
using Documenter
using DocThemeIndigo
using Literate

for each in readdir(pkgdir(GraphTensorNetworks, "examples"))
    project_dir = pkgdir(GraphTensorNetworks, "examples", each)
    isdir(project_dir) || continue
    @info "building" project_dir
    input_file = pkgdir(GraphTensorNetworks, "examples", each, "main.jl")
    output_dir = pkgdir(GraphTensorNetworks, "docs", "src", "tutorials")
    @info "executing" input_file
    Literate.markdown(input_file, output_dir; name=each, execute=true)
end

indigo = DocThemeIndigo.install(GraphTensorNetworks)
DocMeta.setdocmeta!(GraphTensorNetworks, :DocTestSetup, :(using GraphTensorNetworks); recursive=true)

makedocs(;
    modules=[GraphTensorNetworks],
    authors="QuEra Computing Inc.",
    repo="https://github.com/Happy-Diode/GraphTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GraphTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Happy-Diode.github.io/GraphTensorNetworks.jl",
        assets=String[indigo],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Adiabatic Evolution" => "tutorials/MaxCut.md",
        ],
        "References" => "ref.md",
    ],
)

deploydocs(;
    repo="github.com/Happy-Diode/GraphTensorNetworks.jl",
)
