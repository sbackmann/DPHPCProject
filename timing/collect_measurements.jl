using DataFrames
import CSV

include("NPBenchManager.jl")

# for julia, the version of a benchmark is the name of the .jl file
# for C, the version is the name of the corresponding rule in the Makefile
INITIAL_WD = pwd()
ROOT = joinpath(@__DIR__, "..")

function uses_gpu(bm, ver, lang) # determine somehow whether benchmarks use cuda or not
    if lang == "julia"
        keywords = ["using CUDA", "import CUDA", "@cuda", "CuArray"]
        path = joinpath(ROOT, "benchmarks", bm, "$(ver).jl")
        file = read(open(path, "r"), String)
        for k in keywords
            if contains(file, k) return true end
        end
    elseif lang == "C"
        keywords = ["nvcc", ".cu"]
        path = joinpath(ROOT, "benchmarks", bm, "Makefile")
        makefile = read(open(path, "r"), String)
        m = match(Regex("$(ver):\\N*\\n((?:\\h+\\S\\N+\\n)+)"), makefile)
        s = m.captures[1]
        for k in keywords
            if contains(s, k) return true end
        end
    end
    return false
end

get_benchmarks() = readdir(joinpath(ROOT, "benchmarks"))

# assume bm is valid
get_julia_versions(bm) = readdir(joinpath(ROOT, "benchmarks", bm)) |> filter(x -> endswith(x, ".jl") && !startswith(x, "_")) .|> x->x[1:end-3]
get_c_versions(bm) = get_rules(open(io->read(io, String), joinpath(ROOT, "benchmarks", bm, "Makefile")))

get_rules(makefile) = [m.captures[1] for m in eachmatch(r"\n([^_\s]\w*):", makefile)]


julia_has_bm(bm, ver) = bm ∈ get_benchmarks() && ver ∈ get_julia_versions(bm)

function c_has_bm(bm, ver)
    if !(bm ∈ get_benchmarks()) return false end
    makefile = ""
    try
        makefile = open(io->read(io, String), joinpath(ROOT, "benchmarks", bm, "Makefile"))
    catch e
        @warn "Does $bm have a Makefile?" maxlog=1
        display(e)
        return false
    end
    rules = get_rules(makefile)
    return ver ∈ rules
end




function run(benchmark, languages)
    results = empty_df()

    if :julia ∈ languages
        versions = get_julia_versions(benchmark)
        print("julia: ")
        for version in versions
            print(version, " ")
            run!(results, run_julia_bm, benchmark, version)
        end
        println()
    end

    if :C ∈ languages
        versions = get_c_versions(benchmark)
        print("C: ")
        for version in versions
            print(version, " ")
            run!(results, run_c_bm, benchmark, version)
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
        Base.show_backtrace(stdout, backtrace())
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
    DataFrame(
        benchmark=NPBenchManager.get_short_name(bm), 
        language=lang, 
        version=ver, 
        preset=t.preset, 
        gpu=uses_gpu(bm, ver, lang), 
        median=t.median_ms, 
        median_lb=t.median_lb_ms, 
        median_ub=t.median_ub_ms, 
        nr_runs=t.nr_runs
    )
end

function run_julia_bm(bm, ver)
    global RESULTS
    RESULTS = []
    file = joinpath(ROOT, "benchmarks", bm, "$(ver).jl")
    include(file) # timing results are stored in RESULTS
    results = empty_df()
    for t in RESULTS
        append!(results, result_dataframe(bm, ver, "julia", t))
    end
    return results
end

function run_c_bm(bm, ver)
    path = joinpath(ROOT, "benchmarks", bm)
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
    const char* presets_to_run[] = {$(string(presets)[2:end-1])};
    """
    path = joinpath(ROOT, "timing", "presets.h")
    file = open(path, "w")
    write(file, s)
    close(file)
end

empty_df() = DataFrame(                                   # time measurements in ms
    [[],          [],         [],        [],       [],    [],       [],          [],          []       ], 
    ["benchmark", "language", "version", "preset", "gpu", "median", "median_lb", "median_ub", "nr_runs"]
)


function merge(results, old_results)
    keep_old = antijoin(old_results, results, on=[:benchmark, :language, :version, :preset, :gpu], matchmissing=:equal)
    return vcat(keep_old, results)
end

# restore the natural order of things
function reset()
    global PRESETS_TO_RUN = ["missing", "S"] # for when running a julia file with Ctrl+Enter...
    make_presets_header(PRESETS_TO_RUN)
    cd(INITIAL_WD)
end


function collect_measurements(benchmarks::Vector{String}, languages::Vector{Symbol}, 
                              presets::Vector{String},    version::Union{Nothing, String})
    results = empty_df()

    global PROFILING = false
    global PROFILING_GPU = false
    global PRESETS_TO_RUN = presets
    make_presets_header(PRESETS_TO_RUN)

    if :python ∈ languages
        NPBenchManager.init_db() # delete old python measurements
    end

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
        reset()
        rethrow()
    end

    if :python ∈ languages
        NPBenchManager.run(benchmarks, PRESETS_TO_RUN) # run all python benchmarks
        append!(results, NPBenchManager.get_results())
    end

    reset()

    display(results)

    results_file_path = joinpath(ROOT, "results.csv")

    old_results = try 
        CSV.read(results_file_path, DataFrame)
    catch e
        empty_df()
    end
    results = merge(results, old_results)
    CSV.write(results_file_path, results)
end

    



