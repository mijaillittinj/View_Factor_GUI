import numpy as np
from scipy.integrate import nquad

def f(h, r, d, p, b):
    # Helper: arctan term
    def arctan_term():
        return np.arctan(r / np.sqrt(d**2 - r**2))

    # First term (double integral)
    def integrand1(u, v):
        return (
            np.log((0.5*p*np.cos(u) - r*np.cos(v))**2 +
                   (d - r*np.sin(v))**2 +
                   (0.5*p*np.sin(u) + b + 0.5*p)**2)
            * 0.5 * p * r * np.sin(u) * np.sin(v)
        )

    v1_a = arctan_term()
    v1_b = np.pi - arctan_term()
    I1, _ = nquad(integrand1, [[0, 2*np.pi], [v1_a, v1_b]])

    # Second term (double integral)
    def integrand2(u, v):
        theta = arctan_term()
        return (
            np.log((0.5*p*np.cos(u) + r*np.cos(theta))**2 +
                   (d - r*np.sin(theta))**2 +
                   (0.5*p*np.sin(u) + b + 0.5*p - h*v)**2)
            * 0.5 * p * h * np.cos(u)
        )

    I2, _ = nquad(integrand2, [[0, 2*np.pi], [0, 1]])

    # Third term (double integral)
    def integrand3(u, v):
        return (
            np.log((0.5*p*np.cos(u) - r*np.cos(v))**2 +
                   (d - r*np.sin(v))**2 +
                   (0.5*p*np.sin(u) + b + 0.5*p - h)**2)
            * 0.5 * p * r * np.sin(u) * np.sin(v)
        )

    v3_a = np.pi - arctan_term()
    v3_b = arctan_term()
    I3, _ = nquad(integrand3, [[0, 2*np.pi], [v3_a, v3_b]])

    # Fourth term (double integral, subtracted)
    def integrand4(u, v):
        theta = arctan_term()
        return (
            np.log((0.5*p*np.cos(u) - r*np.cos(theta))**2 +
                   (d - r*np.sin(theta))**2 +
                   (0.5*p*np.sin(u) + b + 0.5*p - h + h*v)**2)
            * 0.5 * p * h * np.cos(u)
        )

    I4, _ = nquad(integrand4, [[0, 2*np.pi], [0, 1]])

    norm = 1.0 / (2 * np.pi * 2 * np.pi * (p/2)**2)
    return norm * (I1 + I2 + I3 - I4)