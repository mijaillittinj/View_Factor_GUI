"""
View Factor Calculator Module

This module contains all the view factor calculation methods for different geometries.
"""

import numpy as np
from scipy.integrate import nquad
from view_factor_cone_target import f as cone_view_factor_analytical
from view_factor_cylinder_target import f as cylinder_view_factor_analytical


class ViewFactorCalculator:
    """Main class for calculating view factors between radiometer and flame geometries."""
    
    @staticmethod
    def monte_carlo(d_r, distance, d_f, h_f, z_radiometer=0, shape="Cylinder", n_rays=10000, animate=False):
        """
        Calculate view factor using Monte Carlo ray tracing method.
        
        Args:
            d_r (float): Radiometer diameter
            distance (float): Distance between radiometer and flame
            d_f (float): Flame diameter
            h_f (float): Flame height
            z_radiometer (float): Z-coordinate of radiometer center
            shape (str): Flame shape ("Cylinder" or "Cone")
            n_rays (int): Number of rays for Monte Carlo simulation
            animate (bool): Whether to return ray data for animation
            
        Returns:
            tuple: (view_factor, ray_data) where ray_data is None if animate=False
        """
        hits = 0
        pts_src, pts_dst = [], []
        
        for i in range(n_rays):
            # Generate random point on radiometer surface
            r = d_r/2 * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            y0 = r * np.cos(theta)
            z0 = z_radiometer + r * np.sin(theta)
            x0 = 0
            
            # Generate random ray direction
            phi = 2 * np.pi * np.random.rand()
            cos_theta = np.random.rand()
            sin_theta = np.sqrt(1 - cos_theta**2)
            dx = cos_theta
            dy = sin_theta * np.cos(phi)
            dz = sin_theta * np.sin(phi)
            
            if shape == "Cylinder":
                hit = ViewFactorCalculator._ray_cylinder_intersection(
                    x0, y0, z0, dx, dy, dz, distance, d_f, h_f
                )
            elif shape == "Cone":
                hit = ViewFactorCalculator._ray_cone_intersection(
                    x0, y0, z0, dx, dy, dz, distance, d_f, h_f
                )
            else:
                continue
                
            if hit:
                hits += 1
                if animate and i < 200:
                    pts_src.append([x0, y0, z0])
                    pts_dst.append(hit)
        
        view_factor = hits / n_rays if n_rays > 0 else 0.0
        ray_data = (np.array(pts_src), np.array(pts_dst)) if animate else None
        return view_factor, ray_data
    
    @staticmethod
    def _ray_cylinder_intersection(x0, y0, z0, dx, dy, dz, distance, d_f, h_f):
        """Calculate intersection of ray with cylinder."""
        a = dx**2 + dy**2
        b = 2 * ((x0 - distance) * dx + y0 * dy)
        c = (x0 - distance)**2 + y0**2 - (d_f/2)**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0 or abs(a) < 1e-12:
            return None
            
        sqrtD = np.sqrt(discriminant)
        t1 = (-b - sqrtD) / (2*a)
        t2 = (-b + sqrtD) / (2*a)
        ts = [t for t in [t1, t2] if t > 1e-8]
        
        for t in ts:
            x_hit = x0 + t*dx
            y_hit = y0 + t*dy
            z_hit = z0 + t*dz
            if 0 < z_hit < h_f and x_hit > 0:
                return [x_hit, y_hit, z_hit]
        return None
    
    @staticmethod
    def _ray_cone_intersection(x0, y0, z0, dx, dy, dz, distance, d_f, h_f):
        """Calculate intersection of ray with cone."""
        def cone_eq(t):
            x = x0 + t*dx
            y = y0 + t*dy
            z = z0 + t*dz
            radius = (d_f/2) * (1 - z/h_f)
            return np.sqrt((x - distance)**2 + y**2) - radius
            
        t_min, t_max = 1e-6, 10*distance + 10*h_f
        t_vals = np.linspace(t_min, t_max, 100)
        prev_val = cone_eq(t_min)
        
        for t_val in t_vals[1:]:
            cur_val = cone_eq(t_val)
            if prev_val * cur_val < 0:
                # Root finding
                t0, t1 = t_val - (t_max/100), t_val
                for _ in range(10):
                    tm = 0.5 * (t0 + t1)
                    if cone_eq(tm) * cone_eq(t0) < 0:
                        t1 = tm
                    else:
                        t0 = tm
                t_hit = 0.5 * (t0 + t1)
                
                x_hit = x0 + t_hit*dx
                y_hit = y0 + t_hit*dy
                z_hit = z0 + t_hit*dz
                
                if (0 < z_hit < h_f and x_hit > 0 and 
                    ((x_hit - distance)**2 + y_hit**2) <= ((d_f/2)*(1 - z_hit/h_f))**2 + 1e-12):
                    return [x_hit, y_hit, z_hit]
                break
            prev_val = cur_val
        return None
    
    @staticmethod
    def analytical_cone(h_f, d_f, distance, cone_x, d_r, z_radiometer):
        """
        Calculate view factor for cone using analytical method.
        
        Args:
            h_f (float): Cone height
            d_f (float): Cone diameter
            distance (float): Distance between cone and target
            cone_x (float): X-coordinate of cone height (for inclination)
            d_r (float): Target diameter (radiometer)
            z_radiometer (float): Target height
            
        Returns:
            tuple: (view_factor, view_angle_degrees, half_flame_vf)
        """
        h = h_f  # cone height
        r = d_f/2  # cone radius
        d = distance  # distance
        x = cone_x  # inclination
        p = d_r/2  # target radius
        b = z_radiometer  # target height
        
        # Full view factor using imported function
        vf = cone_view_factor_analytical(h, r, d, x, p, b)
        view_angle_rad = 2 * np.arctan(r / np.sqrt(d**2 - r**2))
        view_angle_deg = np.degrees(view_angle_rad)
        
        # Half flame view factor using modified function
        half_vf = ViewFactorCalculator._cone_view_factor_with_arctan(h, r, d, x, p, b, arctan_value=0.0)
        
        return vf, view_angle_deg, half_vf
    
    @staticmethod
    def analytical_cylinder(h_f, d_f, distance, d_r, z_radiometer):
        """
        Calculate view factor for cylinder using analytical method.
        
        Args:
            h_f (float): Cylinder height
            d_f (float): Cylinder diameter
            distance (float): Distance between cylinder and target
            d_r (float): Target diameter (radiometer)
            z_radiometer (float): Target height
            
        Returns:
            tuple: (view_factor, view_angle_degrees, half_flame_vf)
        """
        h = h_f  # cylinder height
        r = d_f/2  # cylinder radius
        d = distance  # distance
        p = d_r/2  # target radius
        b = z_radiometer  # target height
        
        # Full view factor using imported function
        vf = cylinder_view_factor_analytical(h, r, d, p, b)
        view_angle_rad = 2 * np.arctan(r / np.sqrt(d**2 - r**2))
        view_angle_deg = np.degrees(view_angle_rad)
        
        # Half flame view factor using modified function
        half_vf = ViewFactorCalculator._cylinder_view_factor_with_arctan(h, r, d, p, b, arctan_value=0.0)
        
        return vf, view_angle_deg, half_vf
    
    @staticmethod
    def _cone_view_factor_with_arctan(h, r, d, x, p, b, arctan_value=None):
        """
        Calculate cone view factor with custom arctan value.
        
        Args:
            h, r, d, x, p, b: Standard parameters
            arctan_value: Custom arctan value (None for normal calculation, 0 for half flame)
        """
        # Helper for repeated expression
        def arctan_term():
            if arctan_value is not None:
                return arctan_value
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
    
    @staticmethod
    def _cylinder_view_factor_with_arctan(h, r, d, p, b, arctan_value=None):
        """
        Calculate cylinder view factor with custom arctan value.
        
        Args:
            h, r, d, p, b: Standard parameters
            arctan_value: Custom arctan value (None for normal calculation, 0 for half flame)
        """
        # Helper: arctan term
        def arctan_term():
            if arctan_value is not None:
                return arctan_value
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
    
    @staticmethod
    def calculate_view_angle(d_f, distance):
        """
        Calculate the view angle subtended by a circular target.
        
        Args:
            d_f (float): Target diameter
            distance (float): Distance to target
            
        Returns:
            float: View angle in degrees
        """
        r = d_f/2
        if distance <= r:
            return 180.0  # Target completely fills the view
        
        view_angle_rad = 2 * np.arctan(r / np.sqrt(distance**2 - r**2))
        return np.degrees(view_angle_rad) 