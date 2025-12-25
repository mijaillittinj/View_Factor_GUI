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
        Geometry convention (after fix):
          - Flame is centered at x = 0
          - Radiometer is centered at x = -distance
        """
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        x_radiometer = distance
        x_flame = 0.0

        # Draw radiometer
        GeometryVisualizer._draw_radiometer(ax, d_r, x_radiometer, z_radiometer)

        # Draw flame geometry (at origin)
        if shape == "Cylinder":
            GeometryVisualizer._draw_cylinder(ax, x_flame, d_f, h_f)
        elif shape == "Cone":
            GeometryVisualizer._draw_cone(ax, x_flame, d_f, h_f, cone_x)

        # Set plot properties
        GeometryVisualizer._set_3d_plot_properties(
            ax, d_r, distance, d_f, h_f, z_radiometer
        )

        plt.tight_layout()
        return fig

    @staticmethod
    def create_top_view(d_r, distance, d_f, h_f, shape, cone_x=0):
        """
        Geometry convention (after fix):
          - Flame is centered at x = 0
          - Radiometer is centered at x = -distance
        """
        fig, ax = plt.subplots(figsize=(4, 4))

        x_radiometer = distance
        x_flame = 0.0

        # Draw radiometer
        GeometryVisualizer._draw_radiometer_2d(ax, d_r, x_radiometer)

        # Draw flame (at origin)
        GeometryVisualizer._draw_flame_2d(ax, x_flame, d_f, shape, cone_x)

        # Draw view angle from radiometer to flame
        if shape in ["Cone", "Cylinder"]:
            GeometryVisualizer._draw_view_angle(ax, d_f, x_radiometer, x_flame)

        # Set plot properties
        GeometryVisualizer._set_2d_plot_properties(ax, d_r, distance, d_f)

        plt.legend()
        plt.tight_layout()
        return fig

    # -------------------- DRAWING --------------------

    @staticmethod
    def _draw_radiometer(ax, d_r, x_radiometer, z_radiometer):
        """Draw the radiometer disk in 3D, centered at x = x_radiometer."""
        phi = np.linspace(0, 2*np.pi, 60)
        y_r = (d_r/2) * np.cos(phi)
        z_r = z_radiometer + (d_r/2) * np.sin(phi)
        x_r = np.full_like(phi, x_radiometer)
        verts = [list(zip(x_r, y_r, z_r))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor='dodgerblue',
                                             alpha=0.8, edgecolor='k'))

    @staticmethod
    def _draw_radiometer_2d(ax, d_r, x_radiometer):
        """Draw the radiometer line in 2D at x = x_radiometer."""
        rad_line = plt.Line2D([x_radiometer, x_radiometer],
                              [-d_r/2, d_r/2],
                              color='b', lw=3, label='Radiometer')
        ax.add_line(rad_line)

    @staticmethod
    def _draw_cylinder(ax, x_center, d_f, h_f):
        """Draw a cylinder in 3D centered at x = x_center (flame at origin)."""
        z = np.linspace(0, h_f, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x_cyl = x_center + (d_f/2) * np.cos(theta_grid)
        y_cyl = (d_f/2) * np.sin(theta_grid)
        z_cyl = z_grid

        ax.plot_surface(x_cyl, y_cyl, z_cyl, color='tomato',
                        alpha=0.4, edgecolor='r', linewidth=0.1)

        # base and top circles
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(x_center + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta),
                np.zeros_like(theta), color='r', linewidth=2)
        ax.plot(x_center + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta),
                np.ones_like(theta)*h_f, color='r', linewidth=2)

    @staticmethod
    def _draw_cone(ax, x_center, d_f, h_f, cone_x):
        """Draw a cone in 3D with base centered at x = x_center (flame at origin)."""
        z_cone = np.linspace(0, h_f, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        theta_grid, z_grid = np.meshgrid(theta, z_cone)
        r_grid = (d_f/2) * (1 - z_grid/h_f)

        # Apply inclination (tip shifts by cone_x along +x)
        x_shift = cone_x * (z_grid / h_f)
        x_cone = x_center + r_grid * np.cos(theta_grid) + x_shift
        y_cone = r_grid * np.sin(theta_grid)
        z_cone_surface = z_grid

        ax.plot_surface(x_cone, y_cone, z_cone_surface,
                        color='orange', alpha=0.4,
                        edgecolor='darkorange', linewidth=0.1)

        # base circle (at z=0)
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(x_center + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta),
                np.zeros_like(theta), color='darkorange', linewidth=2)

        # tip marker at (x_center + cone_x, 0, h_f)
        ax.scatter(x_center + cone_x, 0, h_f, color='orange', s=40, marker='o')

        # edge lines
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x0 = x_center + (d_f/2)*np.cos(angle)
            y0 = (d_f/2)*np.sin(angle)
            ax.plot([x0, x_center + cone_x], [y0, 0], [0, h_f],
                    color='darkorange', linewidth=1.2, alpha=0.8)

    @staticmethod
    def _draw_flame_2d(ax, x_center, d_f, shape, cone_x):
        """Draw the flame in 2D centered at x = x_center (origin)."""
        flame_color = 'r' if shape == "Cylinder" else 'orange'

        if shape == "Cone" and cone_x != 0:
            flame_circle = plt.Circle((x_center, 0), d_f/2,
                                      color=flame_color, fill=False,
                                      label=f'{shape} (base)')
            ax.add_patch(flame_circle)
            ax.scatter(x_center + cone_x, 0, color=flame_color, s=50,
                       marker='o', label=f'{shape} (tip)')
        else:
            flame_circle = plt.Circle((x_center, 0), d_f/2,
                                      color=flame_color, fill=False,
                                      label=shape)
            ax.add_patch(flame_circle)

    @staticmethod
    def _draw_view_angle(ax, d_f, x_radiometer, x_flame=0.0):
        """
        Draw the view angle as seen from the radiometer (x_radiometer) toward the flame center (x_flame).
        The angle bisector points from radiometer -> flame.
        """
        r = d_f / 2
        d = abs(x_flame - x_radiometer)

        if d <= r:
            return  # radiometer "inside" projection -> no real external tangent angle

        # Half-angle of tangents from radiometer to circle of radius r at distance d
        view_angle_rad = 2 * np.arctan(r / np.sqrt(d**2 - r**2))
        half_angle = view_angle_rad / 2

        # Base direction: +x if flame is to the right, pi if flame is to the left
        base_dir = 0.0 if (x_flame > x_radiometer) else np.pi

        # Draw boundary rays
        line_length = 1.5 * d
        a1 = base_dir + half_angle
        a2 = base_dir - half_angle

        x1 = x_radiometer + line_length * np.cos(a1)
        y1 = line_length * np.sin(a1)
        x2 = x_radiometer + line_length * np.cos(a2)
        y2 = line_length * np.sin(a2)

        ax.plot([x_radiometer, x1], [0, y1], 'g--', linewidth=1.5, alpha=0.7, label='View angle')
        ax.plot([x_radiometer, x2], [0, y2], 'g--', linewidth=1.5, alpha=0.7)

        # Arc (centered at radiometer)
        arc_angles = np.linspace(a2, a1, 60)
        arc_x = x_radiometer + 0.3 * d * np.cos(arc_angles)
        arc_y = 0.3 * d * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, 'g-', linewidth=2, alpha=0.8)

        # Label near the bisector
        view_angle_deg = np.degrees(view_angle_rad)
        lx = x_radiometer + 0.45 * d * np.cos(base_dir)
        ly = 0.0
        ax.text(lx, ly, f'{view_angle_deg:.1f}°',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))


    # -------------------- AXES / LIMITS --------------------

    @staticmethod
    def _set_3d_plot_properties(ax, d_r, distance, d_f, h_f, z_radiometer):
        """
        With flame at x=0 and radiometer at x=-distance, choose symmetric-ish bounds.
        """
        xmin = -d_f
        xmax = distance + d_r

        Ly = max(d_r, d_f)
        Lz = max(h_f, z_radiometer + d_r/2)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-Ly, Ly)
        ax.set_zlim(0, Lz)

        Lx = xmax - xmin
        ax.set_box_aspect([Lx, 2*Ly, Lz])

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Schematic')
        ax.grid(False)
        ax.view_init(elev=18, azim=115)

    @staticmethod
    def _set_2d_plot_properties(ax, d_r, distance, d_f):
        """
        flame at x=0, radiometer at x=-distance
        """
        xmin = -d_f
        xmax = distance + d_r

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-max(d_r, d_f), max(d_r, d_f))
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Top View')
