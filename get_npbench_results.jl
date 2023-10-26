import SQLite, CSV
using DataFrames

db = SQLite.DB("NPBench/npbench.db")
results = DataFrame(SQLite.DBInterface.execute(db, "SELECT id, time, timestamp, benchmark, framework, preset, details FROM results"))
lcounts = DataFrame(SQLite.DBInterface.execute(db, "SELECT id, count, benchmark, framework, details FROM lcounts"))

CSV.write("npbench_results.csv", results)
CSV.write("npbench_linecounts.csv", lcounts)