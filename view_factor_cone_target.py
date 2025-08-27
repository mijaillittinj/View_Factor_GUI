import numpy as np
from scipy.integrate import nquad

def f(h, r, d, x, p, b):
    # Helper for repeated expression
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
            np.log((0.5*p*np.cos(u) -
                    x - (r*np.cos(theta) - x)*v)**2 +
                   (d - r*np.sin(theta)*v)**2 +
                   (0.5*p*np.sin(u) + b + 0.5*p - h + h*v)**2)
            * 0.5 * p *
            ((r*np.cos(theta) - x)*np.sin(u) + h*np.cos(u))
        )

    I2, _ = nquad(integrand2, [[0, 2*np.pi], [0, 1]])

    # Third term (double integral)
    def integrand3(u, v):
        theta = arctan_term()
        return (
            np.log((0.5*p*np.cos(u) +
                    r*np.cos(theta) - (x + r*np.cos(theta))*v)**2 +
                   (d - r*np.sin(theta) + r*np.sin(theta)*v)**2 +
                   (0.5*p*np.sin(u) + b + 0.5*p - h*v)**2)
            * 0.5 * p * (h*np.cos(u) - (x + r*np.cos(theta))*np.sin(u))
        )

    I3, _ = nquad(integrand3, [[0, 2*np.pi], [0, 1]])

    norm = 1.0 / (2*np.pi * 2*np.pi * (p/2)**2)
    return norm * (I1 - I2 + I3)

# p = 0.01 # Radius of the differential element target
# b = 30 # Height of the differential element target


# h = 50 # Height of the cone
# r = 20 # Radius of the cone
# d = 40 # Distance between the cone and the target
# x = -5 # x-coordinate of the height of the cone
# print(f(h, r, d, x, p, b))