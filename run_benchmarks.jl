using DataFrames
import CSV

# for julia, the version of a benchmark is the name of the .jl file
# for C, the version is the name of the corresponding rule in the Makefile

# TODO
# add automatic cuda/gpu detection, to determine which versions use gpu
# also run the python versions -> clear the database, run all the versions needed, get results from the database

function uses_gpu(bm, ver, lang) # determine somehow whether benchmarks use cuda or not
    # TODO
    # look for nvcc in the rule in the makefile
    # look for import/using CUDA in the .jl file
    return false
end
  
get_benchmarks() = readdir(joinpath(@__DIR__, "benchmarks"))

# assume bm is valid
get_julia_versions(bm) = readdir(joinpath(@__DIR__, "benchmarks", bm)) |> filter(endswith(".jl")) .|> x->x[1:end-3]
get_c_versions(bm) = get_rules(open(io->read(io, String), joinpath(@__DIR__, "benchmarks", bm, "Makefile")))



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


function run(benchmark, languages)
    results = empty_df()

    if :julia ∈ languages
        versions = get_julia_versions(benchmark)
        print("julia: ")
        for version in versions
            print(version, " ")
            append!(results, run_julia_bm(benchmark, version))
        end
        println()
    end

    if :C ∈ languages
        versions = get_c_versions(benchmark)
        print("C: ")
        for version in versions
            print(version, " ")
            append!(results, run_c_bm(benchmark, version))
        end
        println()
    end

    return results
end

function run!(results, f, args...)
    try
        append!(results, f(args...))
    catch e
        display(e)
    end
end
 
function run(benchmark, version, languages)
    results = empty_df()
    if :julia ∈ languages && julia_has_bm(benchmark, version)
        run!(results, run_julia_bm, benchmark, version)
    end
    if :C ∈ languages && c_has_bm(benchmark, version)
        run!(results, run_c_bm, benchmark, version)
    end
    return results
end

function result_dataframe(bm, ver, lang, t)
    DataFrame(benchmark=bm, language=lang, version=ver, preset=t.preset, gpu=uses_gpu(bm, ver, lang), 
              median=t.median_ms, median_lb=t.median_lb_ms, median_ub=t.median_ub_ms, nr_runs=t.nr_runs)
end

function run_julia_bm(bm, ver)
    file = joinpath(@__DIR__, "benchmarks", bm, "$(ver).jl")
    include(file) # timing results are stored in RESULTS
    results = empty_df()
    for t in RESULTS
        append!(results, result_dataframe(bm, ver, "julia", t))
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
        append!(results, result_dataframe(bm, ver, "C", t))
    end
    return results
end

function make_presets_header(presets)
    n = length(presets)
    s = """
    int nr_presets_to_run = $n;
    char* presets_to_run[] = {$(string(presets)[2:end-1])};
    """
    path = joinpath(@__DIR__, "timing", "presets.h")
    file = open(path, "w")
    write(file, s)
    close(file)
end

empty_df() = DataFrame(                                   # time measurements in ms
    [[],          [],         [],        [],       [],    [],       [],          [],          []       ], 
    ["benchmark", "language", "version", "preset", "gpu", "median", "median_lb", "median_ub", "nr_runs"]
)

function get_presets(args)
    id = findfirst(startswith("-p"), args)
    if isnothing(id)
        return ["missing", "S", "M", "L", "paper"]
    end
    ps = lowercase(args[id])[3:end]
    presets = ["missing"] # versions where no preset is specified are always run
    if contains(ps, "s") push!(presets, "S") end
    if contains(ps, "m") push!(presets, "M") end
    if contains(ps, "l") push!(presets, "L") end
    if contains(ps, "p") push!(presets, "paper") end
    return presets
end

function get_languages(args)
    id = findfirst(startswith("-l"), args)
    if isnothing(id)
        return [:julia, :C] # default no python
    end
    ls = lowercase(args[id])[3:end]
    langs = []
    if contains(ls, "j") push!(langs, :julia)  end
    if contains(ls, "c") push!(langs, :C)      end
    if contains(ls, "p") push!(langs, :python) end
    return langs
end

function get_benchmarks(args)
    bms = get_benchmarks()
    id = findfirst(startswith("-b"), args)
    if !isnothing(id)
        arg = args[id]
        bm = arg[3:end]
        if bm ∈ bms
            return [bm]
        else 
            println("do not understand $arg, the benchmark $bm doesnt exist")
            return []
        end
    end
    id = findfirst(!startswith("-"), args)
    if !isnothing(id)
        bm = args[id]
        if bm ∈ bms
            return [bm]
        else 
            println("the benchmark $bm doesnt exist (benchmark has to come before the version)")
            return []
        end
    end
    return bms
end

function get_version(args)
    bm_id = findfirst(!startswith("-"), args)
    v_id = findlast(!startswith("-"), args)
    if !isnothing(v_id) && bm_id != v_id
        return args[v_id]
    end
    id = findfirst(startswith("-v"), args)
    if !isnothing(id)
        arg = args[id]
        return arg[3:end]
    end
    return nothing
end

# can pass name of benchmark to run as first argument, and version as second argument :o
# if only benchmark is specified, it will run all versions
# if nothing is specified, it will run all benchmarks
function main(args)
    results = empty_df()

    global PRESETS_TO_RUN = get_presets(args)
    make_presets_header(PRESETS_TO_RUN)

    languages = get_languages(args)
    benchmarks = get_benchmarks(args)
    version = get_version(args)

    try

        if !isnothing(version)
            append!(results, run(benchmarks[1], version, languages))
        else
            for bm in benchmarks
                println("benching ", bm, ": ")
                append!(results, run(bm, languages))
                println()
            end
        end

    catch e
        cd(@__DIR__)
        rethrow()
    end

    display(results)

    cd(@__DIR__)
    CSV.write("./results.csv", results)
end



main(ARGS)


