import numpy as np


def _integrate(func, a, b, n=100):
    """
    Approximates the integral of a function using the trapezoidal rule.

    Args:
        func (function): Function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of trapezoids.

    Returns:
        float: Approximate value of the integral.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.vectorize(func)(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
