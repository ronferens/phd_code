import numpy as np


def xor(x: np.array) -> int:
    w = np.array([2, 1])
    u = np.array([[0.5, -0.5], [-1, 1]])
    b1 = np.array([0, 0])
    b2 = -1

    h = np.zeros(2)
    for i in range(2):
        # h(x) = max(Ux + b1, 0)
        h[i] = np.maximum(u[i, 0]*x[0] + u[i, 1]*x[1] + b1[i], 0)
    # f(x) = wh(x) + b2
    f = w[0]*h[0] + w[1]*h[1] + b2
    # Returning the result of sing(f)
    return 1 if np.sign(f) >= 0 else 0


# Printing the function's output for all cases:
print(f'XOR(0, 0): {xor(np.array([0, 0]))}')
print(f'XOR(1, 0): {xor(np.array([1, 0]))}')
print(f'XOR(0, 1): {xor(np.array([0, 1]))}')
print(f'XOR(1, 1): {xor(np.array([1, 1]))}')