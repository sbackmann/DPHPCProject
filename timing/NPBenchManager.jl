
module NPBenchManager

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
    groups = groupby(df, :timestamp)
    out_df =  DataFrame(                                   # time measurements in ms
        [[],          [],         [],        [],       [],    [],       [],          [],          []       ], 
        ["benchmark", "language", "version", "preset", "gpu", "median", "median_lb", "median_ub", "nr_runs"]
    )
    for g in groups
        s = sort(g, :time) # each group 10 measurements of one benchmark version
        median_lb, median, median_ub = s[[2, 6, 9], :time] .* 1000 # convert seconds to milliseconds
        benchmark = s[1, :benchmark]
        version = s[1, :framework]
        preset = s[1, :preset]
        gpu = is_gpu(version)
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

end