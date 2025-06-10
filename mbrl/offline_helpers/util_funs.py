


import numpy as np


def dynamic_time_warp(a, b, cost_fun):
    n, m = len(a), len(b)
    dtw = np.zeros((n + 1, m + 1))
    dtw[0, :] = np.inf
    dtw[:, 0] = np.inf
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_fun(a[i - 1], b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]



if __name__ == "__main__":
    # Example usage
    a = np.array([1, 2, 3])
    b = np.array([1, 3, 4, 4, 4])
    
    def cost_function(x, y):
        return abs(x - y)
    
    distance = dynamic_time_warp(a, b, cost_function)
    print(f"Dynamic Time Warping Distance: {distance}")