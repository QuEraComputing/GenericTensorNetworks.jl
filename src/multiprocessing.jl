module SimpleMultiprocessing
using Distributed
export multiprocess_run

function do_work(f, jobs, results) # define work function everywhere
    while true
        job = take!(jobs)
        @info "running argument $job on device $(Distributed.myid())"
        res = f(job)
        put!(results, res)
    end
end

"""
    multiprocess_run(func, inputs::AbstractVector)

Execute function `func` on `inputs` with multiple processing.

Example
---------------------------
Suppose we have a file `run.jl` with the following contents
```julia
using GenericTensorNetworks.SimpleMultiprocessing

results = multiprocess_run(x->x^2, randn(8))
```

In an terminal, you may run the script with 4 processes by typing
```bash
\$ julia -p4 run.jl
      From worker 2:	[ Info: running argument -0.17544008350172655 on device 2
      From worker 5:	[ Info: running argument 0.34578117779452555 on device 5
      From worker 3:	[ Info: running argument 2.0312551239727705 on device 3
      From worker 4:	[ Info: running argument -0.7319353419291961 on device 4
      From worker 2:	[ Info: running argument 0.013132180639054629 on device 2
      From worker 3:	[ Info: running argument 0.9960101782201602 on device 3
      From worker 4:	[ Info: running argument -0.5613942832743966 on device 4
      From worker 5:	[ Info: running argument 0.39460402723831134 on device 5
```
"""
function multiprocess_run(func, inputs::AbstractVector{T}) where T
    n = length(inputs)
    jobs = RemoteChannel(()->Channel{T}(n));
    results = RemoteChannel(()->Channel{Any}(n));
    for i in 1:n
        put!(jobs, inputs[i])
    end
    for p in workers() # start tasks on the workers to process requests in parallel
        remote_do(do_work, p, func, jobs, results)
    end
    return Any[take!(results) for i=1:n]
end

end