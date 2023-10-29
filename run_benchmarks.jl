using DataFrames
import CSV

# for julia, the version of a benchmark is the name of the .jl file
# for C, the version is the name of the corresponding rule in the Makefile



  
get_benchmarks() = readdir(joinpath(@__DIR__, "benchmarks"))

# assume bm is valid
get_julia_versions(bm) = readdir(joinpath(@__DIR__, "benchmarks", bm)) |> filter(endswith(".jl")) .|> x->x[1:end-3]
get_c_versions(bm) = get_rules(open(io->read(io, String), joinpath(@__DIR__, "benchmarks", bm, "Makefile")))

function has_cuda(bm) # determine somehow whether benchmarks use cuda or not
    # TODO
end

julia_has_bm(bm, ver) = bm ∈ get_benchmarks() && ver ∈ get_julia_versions(bm)

function c_has_bm(bm, ver)
    if !(bm ∈ get_benchmarks()) return false end
    makefile = ""
    try
        makefile = open(io->read(io, String), joinpath(@__DIR__, "benchmarks", bm, "Makefile"))
    catch e
        @warn "Does $bm have a Makefile?" maxlog=1
        display(e)
        return false
    end
    rules = get_rules(makefile)
    return ver ∈ rules
end

get_rules(makefile) = [m.captures[1] for m in eachmatch(r"(\w+):", makefile)]


function run(benchmark)
    results = empty_df()

    versions = get_julia_versions(benchmark)
    print("julia: ")
    for version in versions
        print(version, " ")
        append!(results, run_julia_bm(benchmark, version))
    end
    println()

    versions = get_c_versions(benchmark)
    print("C: ")
    for version in versions
        print(version, " ")
        append!(results, run_c_bm(benchmark, version))
    end
    println()

    return results
end
 
function run(benchmark, version)
    results = empty_df()
    if julia_has_bm(benchmark, version)
        append!(results, run_julia_bm(benchmark, version))
    end
    if c_has_bm(benchmark, version)
        append!(results, run_c_bm(benchmark, version))
    end
    return results
end

function run_julia_bm(bm, ver)
    file = joinpath(@__DIR__, "benchmarks", bm, "$(ver).jl")
    t = include(file)
    # return [bm, ver, "julia", false, t.median_ms, t.median_lb_ms, t.median_ub_ms]
    return DataFrame(benchmark=bm, version=ver, language="julia", cuda=false, median=t.median_ms, median_lb=t.median_lb_ms, median_ub=t.median_ub_ms, nr_runs=t.nr_runs)
end

function run_c_bm(bm, ver)
    path = joinpath(@__DIR__, "benchmarks", bm)
    cd(path)
    makefile_path = joinpath(@__DIR__, "benchmarks", bm, "Makefile")
    out = read(`make --silent -f $(makefile_path) $(ver)`, String)
    t = out |> Meta.parse |> eval
    # return [bm, ver, "C", false, t.median_ms, t.median_lb_ms, t.median_ub_ms]
    return DataFrame(benchmark=bm, version=ver, language="C", cuda=false, median=t.median_ms, median_lb=t.median_lb_ms, median_ub=t.median_ub_ms, nr_runs=t.nr_runs)
end

empty_df() = DataFrame(
    [[],          [],        [],         [],     [],       [],          [],          []], 
    ["benchmark", "version", "language", "cuda", "median", "median_lb", "median_ub", "nr_runs"]
)


# can pass name of benchmark to run as first argument, and version as second argument :o
# if only benchmark is specified, it will run all versions
# if nothing is specified, it will run all benchmarks
function main(args)
    results = empty_df()

    if length(args) >= 1
        if args[1] ∈ get_benchmarks()
            benchmarks = [args[1]]
        else
            println("benchmark does not exist?")
            benchmarks = []
        end
        if length(args) == 2 # if both benchmark and version are specified
            append!(results, run(args[1], args[2]))
            
            benchmarks = []
        end
    else
        benchmarks = get_benchmarks()
    end

    for bm in benchmarks
        println("benching ", bm, ": ")
        append!(results, run(bm))
        println()
    end

    display(results)

    cd(@__DIR__)
    CSV.write("./results.csv", results)
end

main(ARGS)