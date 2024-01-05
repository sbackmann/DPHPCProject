
module NPBenchManager

# some functionality to help integrate NPBench into our timing infrastructure

using DataStructures: OrderedDict
import JSON3
import SQLite
using DataFrames

# Database, where NPBench stores its results
DB_path = joinpath(@__DIR__, "..", "npbench.db") 

python_versions = ["cupy", "dace_cpu", "dace_gpu", "numba", "numpy", "pythran"]

function init_db()
    try
        rm(DB_path)
    catch e end
    db = SQLite.DB(DB_path)
    SQLite.DBInterface.execute(db, 
    """CREATE TABLE results (
        id integer PRIMARY KEY,
        timestamp integer NOT NULL,
        benchmark text NOT NULL,
        kind text,
        domain text,
        dwarf text,
        preset text NOT NULL,
        mode text NOT NULL,
        framework text NOT NULL,
        version text NOT NULL,
        details text,
        validated integer,
        time real
    )""")
end

function get_raw_results()
    db = SQLite.DB(DB_path)
    DataFrame(SQLite.DBInterface.execute(db, "SELECT id, timestamp, benchmark, preset, framework, details, time FROM results"))
end

is_gpu(version) = version ∈ ["dace_gpu", "cupy"]

function get_results()
    df = get_raw_results()
    groups = groupby(df, [:timestamp, :details])
    out_df =  DataFrame(                                   # time measurements in ms
        [[],          [],         [],        [],       [],    [],       [],          [],          []       ], 
        ["benchmark", "language", "version", "preset", "gpu", "median", "median_lb", "median_ub", "nr_runs"]
    )
    for g in groups
        s = sort(g, :time) # each group 10 measurements of one benchmark version
        @assert size(s, 1) == 10
        median_lb, median, median_ub = s[[2, 6, 9], :time] .* 1000 # convert seconds to milliseconds
        benchmark = s[1, :benchmark]
        version = s[1, :framework] .* (s[1, :details] != "default" ? " ($(s[1, :details]))" : "")
        preset = s[1, :preset]
        gpu = is_gpu(s[1, :framework])
        append!(out_df,
            DataFrame(benchmark=benchmark, language="python", version=version, preset=preset, gpu=gpu, median=median, median_lb=median_lb, median_ub=median_ub, nr_runs=10)
        )
    end
    out_df
end

function Base.run(bms::Vector{String}, presets::Vector{String})
    cd(@__DIR__); cd("..")
    for bm in bms, preset in presets, version in python_versions
        if preset == "missing" continue end
        try
            run(`python3 NPBench/run_benchmark.py -b $(bm) -f $(version) -p $(preset)`)
        catch e
        end
    end
end

function get_bench_info(bm::String)
    bench_info_path = joinpath(@__DIR__, "..", "NPBench", "bench_info")
    info = read(joinpath(bench_info_path, "$bm.json"), String) |> JSON3.read
    return info.benchmark
end

get_short_name(bm::String) = get_bench_info(bm).short_name

function get_parameters(bm::String)
    dict = get_bench_info(bm).parameters |> OrderedDict
    ordered = OrderedDict()
    ordered["S"]     = dict[:S]     |> OrderedDict
    ordered["M"]     = dict[:M]     |> OrderedDict
    ordered["L"]     = dict[:L]     |> OrderedDict
    ordered["paper"] = dict[:paper] |> OrderedDict
    ordered
end

function set_parameters(bm::String, preset::String, params::Tuple)
    old = get_parameters(bm)
    if haskey(old, preset)
        old_params = old[preset]
        for (i, (k,v)) in enumerate(old_params)
            old_params[k] = params[i]
        end
    end
    new = old
    all_info = get_bench_info(bm)
    all_info["parameters"] = new
    info = Dict("benchmark" => all_info)
    bench_info_path = joinpath(@__DIR__, "..", "NPBench", "bench_info", "$bm.json")

    open(bench_info_path, "w") do io
        JSON3.pretty(io, info)
    end
    make_parameter_header(bm)
end

function reset_parameters(bm::String) # reset to defaults
    bench_info_path = joinpath(@__DIR__, "..", "NPBench", "bench_info",     "$bm.json")
    old_info_path   = joinpath(@__DIR__, "..", "NPBench", "bench_info_old", "$bm.json")
    write(bench_info_path, read(old_info_path, String))
    make_parameter_header(bm)
end

function make_parameter_header(bm::String)
    params = get_parameters(bm)
    n = length(params["S"])
    ps = Dict()
    for preset in keys(params)
        s = ""
        for (i, (k, v)) in enumerate(params[preset])
            s *= "_params[$(i-1)] = $v;\n\t\t"
        end
        ps[preset] = s
    end

    header = """
#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = $(n);
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = (int*) realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        $(ps["S"])return _params;
    } else if (strcmp(preset, "M") == 0) {
        $(ps["M"])return _params;
    } else if (strcmp(preset, "L") == 0) {
        $(ps["L"])return _params;
    } else if (strcmp(preset, "paper") == 0) {
        $(ps["paper"])return _params;
    }
    return NULL;
}

#endif 
"""
    path = joinpath(@__DIR__, "..", "benchmarks", bm, "_parameters.h")
    write(path, header)
end
 
end