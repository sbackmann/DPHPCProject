import numpy as np
import os
import pickle

ASSERT = False

dev_benchmarks = {
    "S": 200,
    #"M": 400,
    #"L": 800,
    #"paper": 2800
}

def main():
    for benchmark in dev_benchmarks:
        n = dev_benchmarks[benchmark]
        print(f"Running on set {benchmark}")

        graph = initialize(n)
        res = kernel(n, graph)
        if ASSERT and benchmark == "S":
            #create_testfile(res, benchmark)
            assert_correctness(res, benchmark)


def initialize(n, datatype=np.int32):
    def f(i, j):
        loc = ((i + j) % 13 == 0) | ((i + j) % 7 == 0) | ((i + j) % 11 == 0)
        return np.where(loc, i * j % 7 + 1, 999)
    
    return np.fromfunction(lambda i, j: f(i, j),  (n, n), dtype=datatype)



def kernel(n, graph):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

    return graph

def create_testfile(graph, prefix):
    with open(f"benchmarks/floyd-warshall/test_cases/{prefix}.pickle", "wb") as f:
        pickle.dump(graph, f)


def assert_correctness(graph, prefix):
    with open(f"benchmarks/floyd-warshall/test_cases/{prefix}.pickle", "rb") as f:
        graph_test = pickle.load(f)
    assert np.array_equal(graph, graph_test), "Result is incorrect"

if __name__ == "__main__":
    main()