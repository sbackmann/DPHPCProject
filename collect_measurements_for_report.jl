include("timing/NPBenchManager.jl")

NPBenchManager.set_parameters("covariance", "paper", (2000, 2400))
NPBenchManager.set_parameters("doitgen",    "paper", (250, 280, 320))
NPBenchManager.set_parameters("floyd_warshall", "paper", (1200,))
NPBenchManager.set_parameters("gemm",       "paper", (2000,2200,2400))
NPBenchManager.set_parameters("jacobi_2d",  "paper", (300,1600))
NPBenchManager.set_parameters("lu",         "paper", (500,)) # very slow on gpu, but very fast on cpu
NPBenchManager.set_parameters("syrk",       "paper", (1600,2000)) # very slow for python
NPBenchManager.set_parameters("trisolv",    "paper", (14000,))

run(`julia run_benchmarks.jl -pp -ljpc`)

for bm in ["covariance", "doitgen", "floyd_warshall", "gemm", "jacobi_2d", "lu", "syrk", "trisolv"]
    NPBenchManager.reset_parameters(bm)
end