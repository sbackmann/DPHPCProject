# this is now old, might not work anymore, just for future reference if need to get linecounts or something

# import SQLite, CSV
# using DataFrames

# db = SQLite.DB(joinpath(@__DIR__, "../NPBench/npbench.db"))
# DataFrame(SQLite.DBInterface.execute(db, "SELECT * FROM results"))
# results = DataFrame(SQLite.DBInterface.execute(db, "SELECT id, time, timestamp, benchmark, framework, preset, details FROM results"))
# lcounts = DataFrame(SQLite.DBInterface.execute(db, "SELECT id, count, benchmark, framework, details FROM lcounts"))

# CSV.write(joinpath(@__DIR__, "npbench_results.csv"), results)
# CSV.write(joinpath(@__DIR__, "npbench_linecounts.csv"), lcounts)