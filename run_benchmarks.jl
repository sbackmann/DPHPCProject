using DataFrames
import CSV

# for julia, the version of a benchmark is the name of the .jl file
# for C, the version is the name of the corresponding rule in the Makefile

# TODO
# add automatic cuda/gpu detection, to determine which versions use gpu
# also run the python versions somehow somewhere maybe ?
# command line arguments to control which presets and which languages to run


  
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

get_rules(makefile) = [m.captures[1] for m in eachmatch(r"\n([^\W_]+):", makefile)]


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

function result_dataframe(bm, ver, lang, gpu, t)
    DataFrame(benchmark=bm, version=ver, language=lang, gpu=gpu, 
              median=t.median_ms, median_lb=t.median_lb_ms, median_ub=t.median_ub_ms, 
              nr_runs=t.nr_runs, preset=t.preset)
end

function run_julia_bm(bm, ver)
    file = joinpath(@__DIR__, "benchmarks", bm, "$(ver).jl")
    include(file) # timing results are stored in RESULTS
    results = empty_df()
    for t in RESULTS
        append!(results, result_dataframe(bm, ver, "julia", false, t))
    end
    return results
end

function run_c_bm(bm, ver)
    path = joinpath(@__DIR__, "benchmarks", bm)
    cd(path)
    out = read(`make --silent $(ver)`, String)
    raw_results = [m.captures[1] for m in eachmatch(r"dphpcresult(\(.+?\))", out)]
    results = empty_df()
    for result in raw_results
        t = result |> Meta.parse |> eval
        append!(results, result_dataframe(bm, ver, "C", false, t))
    end
    return results
end

empty_df() = DataFrame(                          # time measurements in ms
    [[],          [],        [],         [],     [],       [],          [],          [],       []], 
    ["benchmark", "version", "language", "gpu", "median", "median_lb", "median_ub", "nr_runs", "preset"]
)


# can pass name of benchmark to run as first argument, and version as second argument :o
# if only benchmark is specified, it will run all versions
# if nothing is specified, it will run all benchmarks
function main(args)
    results = empty_df()

    try

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

    catch e
        cd(@__DIR__)
        display(results)
        CSV.write("./results.csv", results)
        rethrow()
    end

    display(results)

    cd(@__DIR__)
    CSV.write("./results.csv", results)
end



main(ARGS)


