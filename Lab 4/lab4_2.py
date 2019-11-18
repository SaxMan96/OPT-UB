import numpy as np

from lab4_1 import loop

if __name__ == '__main__':
    np.random.seed(0)
    test_starting_points = np.random.uniform(-3, 3, (20, 3))
    for test_starting_point in test_starting_points:
        print(test_starting_point[:2], end="\t")
        z = loop(test_starting_point, "im not used here")
        print(z[:2], end='\t')
        print(np.round(np.exp(3 * z[0]) + np.exp(-4 * z[1]), 3))
