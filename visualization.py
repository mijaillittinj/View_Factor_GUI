"""
Visualization Module

This module contains all plotting and visualization functions for the view factor calculator.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class GeometryVisualizer:
    """Class for creating 3D and 2D visualizations of the geometry."""
    
    @staticmethod
    def create_3d_plot(d_r, distance, d_f, h_f, shape, z_radiometer=0, cone_x=0):
        """
        Create a 3D visualization of the radiometer-flame geometry.
        
        Args:
            d_r (float): Radiometer diameter
            distance (float): Distance between radiometer and flame
            d_f (float): Flame diameter
            h_f (float): Flame height
            shape (str): Flame shape ("Cylinder" or "Cone")
            z_radiometer (float): Z-coordinate of radiometer center
            cone_x (float): Cone inclination parameter
            
        Returns:
            matplotlib.figure.Figure: The 3D plot figure
        """
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw radiometer
        GeometryVisualizer._draw_radiometer(ax, d_r, z_radiometer)
        
        # Draw flame geometry
        if shape == "Cylinder":
            GeometryVisualizer._draw_cylinder(ax, distance, d_f, h_f)
        elif shape == "Cone":
            GeometryVisualizer._draw_cone(ax, distance, d_f, h_f, cone_x)
        
        # Set plot properties
        GeometryVisualizer._set_3d_plot_properties(ax, d_r, distance, d_f, h_f, z_radiometer)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_top_view(d_r, distance, d_f, h_f, shape, cone_x=0):
        """
        Create a top-down 2D visualization of the geometry.
        
        Args:
            d_r (float): Radiometer diameter
            distance (float): Distance between radiometer and flame
            d_f (float): Flame diameter
            h_f (float): Flame height
            shape (str): Flame shape ("Cylinder" or "Cone")
            cone_x (float): Cone inclination parameter
            
        Returns:
            matplotlib.figure.Figure: The top view figure
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Draw radiometer
        GeometryVisualizer._draw_radiometer_2d(ax, d_r)
        
        # Draw flame
        GeometryVisualizer._draw_flame_2d(ax, distance, d_f, shape, cone_x)
        
        # Draw view angle for cone and cylinder
        if shape in ["Cone", "Cylinder"]:
            GeometryVisualizer._draw_view_angle(ax, d_f, distance)
        
        # Set plot properties
        GeometryVisualizer._set_2d_plot_properties(ax, d_r, distance, d_f)
        
        plt.legend()
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _draw_radiometer(ax, d_r, z_radiometer):
        """Draw the radiometer disk in 3D."""
        phi = np.linspace(0, 2*np.pi, 60)
        y_r = (d_r/2) * np.cos(phi)
        z_r = z_radiometer + (d_r/2) * np.sin(phi)
        x_r = np.zeros_like(phi)
        verts = [list(zip(x_r, y_r, z_r))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor='dodgerblue', alpha=0.8, edgecolor='k'))
    
    @staticmethod
    def _draw_radiometer_2d(ax, d_r):
        """Draw the radiometer line in 2D."""
        rad_line = plt.Line2D([0, 0], [-d_r/2, d_r/2], color='b', lw=3, label='Radiometer')
        ax.add_line(rad_line)
    
    @staticmethod
    def _draw_cylinder(ax, distance, d_f, h_f):
        """Draw a cylinder in 3D."""
        z = np.linspace(0, h_f, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cyl = distance + (d_f/2) * np.cos(theta_grid)
        y_cyl = (d_f/2) * np.sin(theta_grid)
        z_cyl = z_grid
        ax.plot_surface(x_cyl, y_cyl, z_cyl, color='tomato', alpha=0.4, edgecolor='r', linewidth=0.1)
        
        # Draw base and top circles
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(distance + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta), 
                np.zeros_like(theta), color='r', linewidth=2)
        ax.plot(distance + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta), 
                np.ones_like(theta)*h_f, color='r', linewidth=2)
    
    @staticmethod
    def _draw_cone(ax, distance, d_f, h_f, cone_x):
        """Draw a cone in 3D."""
        z_cone = np.linspace(0, h_f, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        theta_grid, z_grid = np.meshgrid(theta, z_cone)
        r_grid = (d_f/2) * (1 - z_grid/h_f)
        
        # Apply inclination
        x_shift = cone_x * (z_grid / h_f)
        x_cone = distance + r_grid * np.cos(theta_grid) + x_shift
        y_cone = r_grid * np.sin(theta_grid)
        z_cone_surface = z_grid
        
        ax.plot_surface(x_cone, y_cone, z_cone_surface, color='orange', alpha=0.4, 
                       edgecolor='darkorange', linewidth=0.1)
        
        # Draw base circle
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(distance + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta), 
                np.zeros_like(theta), color='darkorange', linewidth=2)
        
        # Draw tip marker
        ax.scatter(distance + cone_x, 0, h_f, color='orange', s=40, marker='o')
        
        # Draw edge lines
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x0 = distance + (d_f/2)*np.cos(angle)
            y0 = (d_f/2)*np.sin(angle)
            ax.plot([x0, distance + cone_x], [y0, 0], [0, h_f], 
                   color='darkorange', linewidth=1.2, alpha=0.8)
    
    @staticmethod
    def _draw_flame_2d(ax, distance, d_f, shape, cone_x):
        """Draw the flame in 2D."""
        flame_color = 'r' if shape == "Cylinder" else 'orange'
        
        if shape == "Cone" and cone_x != 0:
            # For inclined cone, show base and tip
            flame_circle = plt.Circle((distance, 0), d_f/2, color=flame_color, 
                                    fill=False, label=f'{shape} (base)')
            ax.add_patch(flame_circle)
            ax.scatter(distance + cone_x, 0, color=flame_color, s=50, marker='o', 
                      label=f'{shape} (tip)')
        else:
            flame_circle = plt.Circle((distance, 0), d_f/2, color=flame_color, 
                                    fill=False, label=shape)
            ax.add_patch(flame_circle)
    
    @staticmethod
    def _draw_view_angle(ax, d_f, distance):
        """Draw the view angle for cone geometry."""
        r = d_f/2
        d = distance
        
        if d > r:  # Only draw if radiometer is outside the cone
            view_angle_rad = 2 * np.arctan(r / np.sqrt(d**2 - r**2))
            half_angle = view_angle_rad / 2
            
            # Draw boundary lines
            line_length = 1.5 * distance
            angle1 = half_angle
            angle2 = -half_angle
            
            x1 = line_length * np.cos(angle1)
            y1 = line_length * np.sin(angle1)
            x2 = line_length * np.cos(angle2)
            y2 = line_length * np.sin(angle2)
            
            ax.plot([0, x1], [0, y1], 'g--', linewidth=1.5, alpha=0.7, label='View angle')
            ax.plot([0, x2], [0, y2], 'g--', linewidth=1.5, alpha=0.7)
            
            # Draw arc
            arc_angles = np.linspace(angle2, angle1, 50)
            arc_x = 0.3 * distance * np.cos(arc_angles)
            arc_y = 0.3 * distance * np.sin(arc_angles)
            ax.plot(arc_x, arc_y, 'g-', linewidth=2, alpha=0.8)
            
            # Add angle label
            label_x = 0.4 * distance * np.cos(0)
            label_y = 0.4 * distance * np.sin(0)
            view_angle_deg = np.degrees(view_angle_rad)
            ax.text(label_x, label_y, f'{view_angle_deg:.1f}°', 
                   fontsize=10, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    @staticmethod
    def _set_3d_plot_properties(ax, d_r, distance, d_f, h_f, z_radiometer):
        """Set properties for 3D plot."""
        Lx = distance + d_f
        Ly = max(d_r, d_f)
        Lz = max(h_f, z_radiometer + d_r/2)
        
        ax.set_box_aspect([Lx, 2*Ly, Lz])
        ax.set_xlim(-d_r, distance + d_f)
        ax.set_ylim(-Ly, Ly)
        ax.set_zlim(0, Lz)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Schematic')
        ax.grid(False)
        ax.view_init(elev=18, azim=115)
    
    @staticmethod
    def _set_2d_plot_properties(ax, d_r, distance, d_f):
        """Set properties for 2D plot."""
        ax.set_xlim(-d_r, distance + d_f)
        ax.set_ylim(-max(d_r, d_f), max(d_r, d_f))
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Top View') 