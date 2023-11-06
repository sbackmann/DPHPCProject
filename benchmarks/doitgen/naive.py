import numpy as np
import os
import pickle

ASSERT = True

dev_benchmarks = {
    "S": (60, 60, 128),
    #"M": (110, 125, 256),
    #"L": (220, 250, 512),
    #"paper": (220, 250, 270)
}

def main():
    for benchmark in dev_benchmarks:
        n_r, n_q, n_p = dev_benchmarks[benchmark]
        print(f"Running on set {benchmark}")

        A, C4, sum = initialize(n_r, n_q, n_p)
        res = kernel(n_r, n_q, n_p, A, C4, sum)
        if ASSERT and benchmark == "S":
            assert_correctness(res, benchmark)


def initialize(n_r, n_q, n_p, datatype=np.float64):
    A = np.fromfunction(lambda r, q, p: ((r * q + p) % n_p) / n_p, (n_r, n_q, n_p), dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % n_p) / n_p, (n_p, n_p), dtype=datatype)
    sum = np.zeros(n_p, dtype=datatype)
    return A, C4, sum

def kernel(n_r, n_q, n_p, A, C4, sum):
    for r in range(n_r):
        for q in range(n_q):
            for p in range(n_p):
                sum[p] = 0.0
                for s in range(n_p):
                    sum[p] += A[r][q][s] * C4[s][p]
            for p in range(n_p):
                A[r][q][p] = sum[p]

    return A

def create_testfile(A, prefix):
    with open(f"benchmarks/doitgen/test_cases/{prefix}.pickle", "wb") as f:
        pickle.dump(A, f)


def assert_correctness(A, prefix):
    with open(f"benchmarks/doitgen/test_cases/{prefix}.pickle", "rb") as f:
        A_test = pickle.load(f)
    assert np.array_equal(A, A_test), "Result is incorrect"

if __name__ == "__main__":
    main()