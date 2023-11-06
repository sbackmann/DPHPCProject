import numpy as np
import os

input_m = 2
input_n = 2

dev_benchmarks = {
    # "S": (2, 2),
    "M": (5, 5),
    "L": (5, 7),
    # "L": (10, 20),
}

def main():
    for benchmark in dev_benchmarks:
        M, N = dev_benchmarks[benchmark]
        print(f"Running on set M={M}, N={N}")

        data = initialize(M, N, datatype=np.float64)
        print("Data")
        print(data)
        print("Cov")
        cov = kernel(M, np.float64(N), data)
        print(cov)
        print("End")


def initialize(M, N, datatype=np.float64):
    return np.fromfunction(lambda i, j: (i * j) / M, (N, M), dtype=datatype)

def kernel(M, float_n, data):
    mean = np.mean(data, axis=0)
    print("mean", mean)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


if __name__ == "__main__":
    main()