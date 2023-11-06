import numpy as np
import os

dev_benchmarks = {
    "S": (4),
}
        #     "S": { "N": 2000 },
        #     "M": { "N": 5000 },
        #     "L": { "N": 14000 },
        #     "paper": { "N": 16000 }
        # },

def main():
    for benchmark in dev_benchmarks:
        N = dev_benchmarks[benchmark]
        print(f"Running on set N={N}")

        L, x, b = initialize(N, datatype=np.float64)
        print("Data")
        print(f"L: \n{L}")
        print(f"x: \n{x}")
        print(f"b: \n{b}")
        
        x = kernel(L, x, b)
        print(x)
        is_correct(L, x, b)

        print("End")


def initialize(N, datatype=np.float64):
    L = np.fromfunction(lambda i, j: (i + N - j + 1) * 2 / N, (N, N),
                        dtype=datatype)

    x = np.full((N, ), -999, dtype=datatype)
    b = np.fromfunction(lambda i: i, (N, ), dtype=datatype)

    return L, x, b


def kernel(L, x, b):
    for i in range(x.shape[0]):
        exp1 = L[i, :i] 
        exp2 = x[:i]
        dp = L[i, :i] @ x[:i]
        print(f"dp: {dp}")

        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    
    return x

def is_correct(L, x, b):
    N = L.shape[0]
    for i in range(N):
        for j in range(N):
            if i < j:
                L[i, j] = 0

    expected_solution = np.linalg.solve(L, b)
    print(expected_solution)

    # Compare the result with the expected solution
    if np.allclose(x, expected_solution):
        print("Test passed: The solution matches the expected result.")
        return True
    else:
        print("Test failed: The solution does not match the expected result.")
        return False


if __name__ == "__main__":
    main()