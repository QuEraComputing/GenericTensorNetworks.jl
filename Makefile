JL = julia --project

default: init test

init:
	$(JL) -e 'using Pkg; Pkg.precompile(); Pkg.activate("docs"); Pkg.develop(path=".")'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.activate("docs"); Pkg.update()'

test:
	$(JL) -e 'using Pkg; Pkg.test("GenericTensorNetworks")'

coverage:
	$(JL) -e 'using Pkg; Pkg.test("GenericTensorNetworks"; coverage=true)'

serve:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs(;skip_dirs=["docs/build", "docs/src/assets", "docs/src/generated"], literate_dir="examples")'

clean:
	rm -rf docs/build
	find . -name "*.cov" -type f -print0 | xargs -0 /bin/rm -f

.PHONY: init test coverage serve clean init-docs update update-docs