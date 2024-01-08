include("timing/NPBenchManager.jl")

NPBenchManager.set_parameters("covariance", "S", (50, 60))
run(`julia run_benchmarks.jl covariance`)

NPBenchManager.set_parameters("doitgen", "S", (30, 30, 64))
run(`julia run_benchmarks.jl doitgen`)

NPBenchManager.set_parameters("floyd_warshall", "S", (200,))
run(`julia run_benchmarks.jl floyd_warshall`)

NPBenchManager.set_parameters("gemm", "S", (20,30,40))
run(`julia run_benchmarks.jl gemm`)

NPBenchManager.set_parameters("jacobi_2d", "S", (20,60))
run(`julia run_benchmarks.jl jacobi_2d`)

NPBenchManager.set_parameters("lu", "S", (30,))
run(`julia run_benchmarks.jl lu`)

NPBenchManager.set_parameters("syrk", "S", (30,50))
run(`julia run_benchmarks.jl syrk`)

NPBenchManager.set_parameters("trisolv", "S", (50,))
run(`julia run_benchmarks.jl trisolv`)

run(`julia run_benchmarks.jl syrk -lp`)

for bm in ["covariance", "doitgen", "floyd_warshall", "gemm", "jacobi_2d", "lu", "syrk", "trisolv"]
    NPBenchManager.reset_parameters(bm)
end