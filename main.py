import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive features
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import io
import FreeSimpleGUI as sg
from view_factor_cone_target import f as cone_view_factor_analytical

# Set Arial as the default font for matplotlib
plt.rcParams['font.family'] = 'Arial'

sg.theme('Default1')
def montecarlo_view_factor(d_r, distance, d_f, h_f, z_radiometer=0, shape="Cylinder", n_rays=10000, animate=False):
    hits = 0
    pts_src, pts_dst = [], []
    for i in range(n_rays):
        r = d_r/2 * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        y0 = r * np.cos(theta)
        z0 = z_radiometer + r * np.sin(theta)
        x0 = 0
        phi = 2 * np.pi * np.random.rand()
        cos_theta = np.random.rand()
        sin_theta = np.sqrt(1 - cos_theta**2)
        dx = cos_theta
        dy = sin_theta * np.cos(phi)
        dz = sin_theta * np.sin(phi)
        if shape == "Cylinder":
            a = dx**2 + dy**2
            b = 2 * ((x0 - distance) * dx + y0 * dy)
            c = (x0 - distance)**2 + y0**2 - (d_f/2)**2
            discriminant = b**2 - 4*a*c
            if discriminant < 0 or abs(a) < 1e-12:
                continue
            sqrtD = np.sqrt(discriminant)
            t1 = (-b - sqrtD) / (2*a)
            t2 = (-b + sqrtD) / (2*a)
            ts = [t for t in [t1, t2] if t > 1e-8]
            for t in ts:
                x_hit = x0 + t*dx
                y_hit = y0 + t*dy
                z_hit = z0 + t*dz
                if 0 < z_hit < h_f and x_hit > 0:
                    hits += 1
                    if animate and i < 200:
                        pts_src.append([x0, y0, z0])
                        pts_dst.append([x_hit, y_hit, z_hit])
                    break
        elif shape == "Cone":
            def cone_eq(t):
                x = x0 + t*dx
                y = y0 + t*dy
                z = z0 + t*dz
                radius = (d_f/2) * (1 - z/h_f)
                return np.sqrt((x - distance)**2 + y**2) - radius
            t_min, t_max = 1e-6, 10*distance + 10*h_f
            found = False
            t_hit = None
            t_vals = np.linspace(t_min, t_max, 100)
            prev_val = cone_eq(t_min)
            for t_val in t_vals[1:]:
                cur_val = cone_eq(t_val)
                if prev_val * cur_val < 0:
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
                    if 0 < z_hit < h_f and x_hit > 0 and (((x_hit - distance)**2 + y_hit**2) <= ((d_f/2)*(1 - z_hit/h_f))**2 + 1e-12):
                        hits += 1
                        if animate and i < 200:
                            pts_src.append([x0, y0, z0])
                            pts_dst.append([x_hit, y_hit, z_hit])
                        found = True
                    break
                prev_val = cur_val
    view_factor = hits / n_rays if n_rays > 0 else 0.0
    return view_factor, (np.array(pts_src), np.array(pts_dst)) if animate else None

def draw_3d_placeholder(d_r, distance, d_f, h_f, shape, z_radiometer=0, cone_x=0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    


    # Radiometer disk in yz-plane at x=0, centered at z_radiometer
    phi = np.linspace(0, 2*np.pi, 60)
    y_r = (d_r/2) * np.cos(phi)
    z_r = z_radiometer + (d_r/2) * np.sin(phi)
    x_r = np.zeros_like(phi)
    verts = [list(zip(x_r, y_r, z_r))]
    ax.add_collection3d(Poly3DCollection(verts, facecolor='dodgerblue', alpha=0.8, edgecolor='k'))

    if shape == "Cylinder":
        z = np.linspace(0, h_f, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cyl = distance + (d_f/2) * np.cos(theta_grid)
        y_cyl = (d_f/2) * np.sin(theta_grid)
        z_cyl = z_grid
        ax.plot_surface(x_cyl, y_cyl, z_cyl, color='tomato', alpha=0.4, edgecolor='r', linewidth=0.1)
        # Cylinder base and top circles
        ax.plot(distance + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta), np.zeros_like(theta), color='r', linewidth=2)
        ax.plot(distance + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta), np.ones_like(theta)*h_f, color='r', linewidth=2)
        # ax.scatter(distance, 0, h_f/2, color='k', s=30)
    elif shape == "Cone":
        z_cone = np.linspace(0, h_f, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        theta_grid, z_grid = np.meshgrid(theta, z_cone)
        r_grid = (d_f/2) * (1 - z_grid/h_f)
        # Apply inclination: shift x-coordinate based on height
        x_shift = cone_x * (z_grid / h_f)
        x_cone = distance + r_grid * np.cos(theta_grid) + x_shift
        y_cone = r_grid * np.sin(theta_grid)
        z_cone_surface = z_grid
        ax.plot_surface(x_cone, y_cone, z_cone_surface, color='orange', alpha=0.4, edgecolor='darkorange', linewidth=0.1)
        # Base circle
        ax.plot(distance + (d_f/2)*np.cos(theta), (d_f/2)*np.sin(theta), np.zeros_like(theta), color='darkorange', linewidth=2)
        # Tip marker
        ax.scatter(distance + cone_x, 0, h_f, color='orange', s=40, marker='o')
        # Edge lines for cone
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x0 = distance + (d_f/2)*np.cos(angle)
            y0 = (d_f/2)*np.sin(angle)
            ax.plot([x0, distance + cone_x], [y0, 0], [0, h_f], color='darkorange', linewidth=1.2, alpha=0.8)

    Lx = distance + d_f
    Ly = max(d_r, d_f)
    Lz = max(h_f, z_radiometer + d_r/2)
    ax.set_box_aspect([Lx, 2*Ly, Lz])
    ax.set_xlim(-d_r, distance + d_f)
    ax.set_ylim(-Ly, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel('X [m]', fontname='Arial')
    ax.set_ylabel('Y [m]', fontname='Arial')
    ax.set_zlabel('Z [m]', fontname='Arial')
    ax.set_title('3D Schematic', fontname='Arial')
    ax.grid(False)
    ax.view_init(elev=18, azim=115)
    plt.tight_layout()
    return fig
def draw_topview_placeholder(d_r, distance, d_f, h_f, shape, cone_x=0):
    fig, ax = plt.subplots(figsize=(4,4))
    rad_line = plt.Line2D([0, 0], [-d_r/2, d_r/2], color='b', lw=3, label='Radiometer')
    ax.add_line(rad_line)
    flame_color = 'r' if shape == "Cylinder" else 'orange'
    if shape == "Cone" and cone_x != 0:
        # For inclined cone, show the base and tip positions
        flame_circle = plt.Circle((distance, 0), d_f/2, color=flame_color, fill=False, label=f'{shape} (base)')
        ax.add_patch(flame_circle)
        # Show tip position
        ax.scatter(distance + cone_x, 0, color=flame_color, s=50, marker='o', label=f'{shape} (tip)')
    else:
        flame_circle = plt.Circle((distance, 0), d_f/2, color=flame_color, fill=False, label=shape)
        ax.add_patch(flame_circle)
    
    # Draw view angle for cone shapes
    if shape == "Cone":
        r = d_f/2  # cone radius
        d = distance  # distance between radiometer and cone center
        
        # Calculate view angle
        if d > r:  # Only draw if radiometer is outside the cone
            view_angle_rad = 2 * np.arctan(r / np.sqrt(d**2 - r**2))
            half_angle = view_angle_rad / 2
            
            # Draw lines at the calculated view angle
            line_length = 1.5 * distance  # Make lines extend beyond the cone
            
            # Upper boundary line
            angle1 = half_angle
            x1 = line_length * np.cos(angle1)
            y1 = line_length * np.sin(angle1)
            ax.plot([0, x1], [0, y1], 'g--', linewidth=1.5, alpha=0.7, label='View angle')
            
            # Lower boundary line
            angle2 = -half_angle
            x2 = line_length * np.cos(angle2)
            y2 = line_length * np.sin(angle2)
            ax.plot([0, x2], [0, y2], 'g--', linewidth=1.5, alpha=0.7)
            
            # Draw arc showing the view angle
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
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8), fontname='Arial')
    ax.set_xlim(-d_r, distance + d_f)
    ax.set_ylim(-max(d_r, d_f), max(d_r, d_f))
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]', fontname='Arial')
    ax.set_ylabel('Y [m]', fontname='Arial')
    ax.set_title('Top View', fontname='Arial')
    plt.legend(prop={'family': 'Arial'})
    plt.tight_layout()
    return fig

label_width = 23
input_width = 10
col1 = [
    [sg.Text("Flame Shape:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Combo(['Cylinder', 'Cone'], key='shape', default_value='Cylinder', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Radiometer diameter [m]:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('0.05', key='d_r', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Radiometer height [m]:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('0', key='z_radiometer', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Distance (radiometer–flame) [m]:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('0.20', key='distance', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Flame diameter [m]:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('0.10', key='d_f', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Flame height [m]:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('0.15', key='h_f', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Cone inclination x [m]:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('0', key='cone_x', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Calculation Method:", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Combo(['Monte Carlo', 'Analytical (Cone only)'], key='method', default_value='Monte Carlo', size=(input_width, 1), font=("Arial", 14))],
    [sg.Text("Rays (Monte Carlo):", size=(label_width, 1), justification='r', font=("Arial", 14)),
     sg.Input('10000', key='n_rays', size=(input_width, 1), font=("Arial", 14))],
    [sg.Push(), sg.Button('Calculate', key='calc', font=("Arial", 14)), sg.Push()],
]

# Plots side-by-side in a row!
plots_row = [
    sg.Canvas(key='canvas3d', size=(400, 400)),
    sg.Canvas(key='canvastop', size=(400, 400)),
]

col2 = [
    plots_row,
    [sg.Text('View factor:', size=(12,1), font=("Arial", 14)), sg.Text('', key='vf', font=("Arial", 16, "bold"))]
]

layout = [
    [sg.Column(col1, vertical_alignment='top'), sg.VSeperator(), sg.Column(col2, vertical_alignment='top')]
]

window = sg.Window('Radiometer–Flame View Factor Calculator', layout, finalize=True, font=("Arial", 14))

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'calc':
        try:
            d_r = float(values['d_r'])
            z_radiometer = float(values['z_radiometer'])
            distance = float(values['distance'])
            d_f = float(values['d_f'])
            h_f = float(values['h_f'])
            cone_x = float(values['cone_x'])
            shape = values['shape']
            method = values['method']
            n_rays = int(float(values['n_rays']))
        except Exception as e:
            sg.popup_error(f"Invalid input: {e}", font=("Arial", 14))
            continue
        # Create figures
        fig3d = draw_3d_placeholder(d_r, distance, d_f, h_f, shape, z_radiometer, cone_x)
        figtop = draw_topview_placeholder(d_r, distance, d_f, h_f, shape, cone_x)
        
        # Draw on canvases with navigation tools
        canvas3d = window['canvas3d'].Widget
        for widget in canvas3d.winfo_children():
            widget.destroy()
        
        canvas3d_fig = FigureCanvasTkAgg(fig3d, canvas3d)
        canvas3d_fig.draw()
        canvas3d_fig.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        # Add navigation toolbar for 3D plot
        toolbar3d = NavigationToolbar2Tk(canvas3d_fig, canvas3d)
        toolbar3d.update()
        
        canvas_top = window['canvastop'].Widget
        for widget in canvas_top.winfo_children():
            widget.destroy()
        
        canvas_top_fig = FigureCanvasTkAgg(figtop, canvas_top)
        canvas_top_fig.draw()
        canvas_top_fig.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        # Add navigation toolbar for top view plot
        toolbar_top = NavigationToolbar2Tk(canvas_top_fig, canvas_top)
        toolbar_top.update()
        if method == 'Monte Carlo':
            vf, _ = montecarlo_view_factor(
                d_r, distance, d_f, h_f, z_radiometer=z_radiometer, shape=shape, n_rays=n_rays, animate=False
            )
            std = (vf * (1 - vf) / n_rays) ** 0.5 if n_rays > 0 else 0
            window['vf'].update(f"{vf:.4f} ± {std:.4f}")
        elif method == 'Analytical (Cone only)':
            if shape != 'Cone':
                sg.popup_error("Analytical calculation is only available for Cone shape.", font=("Arial", 14))
                continue
            
            # Validate parameters for analytical calculation
            if d_f/2 >= distance:
                sg.popup_error("Analytical calculation requires cone radius to be less than distance.", font=("Arial", 14))
                continue
                
            try:
                # Convert parameters to match the analytical function
                # The analytical function expects: f(h, r, d, x, p, b)
                # where: h=cone height, r=cone radius, d=distance, x=x-coordinate of cone height (for inclination), p=target radius, b=target height
                h = h_f  # cone height
                r = d_f/2  # cone radius (half of diameter)
                d = distance  # distance between cone and target
                x = cone_x  # x-coordinate of cone height (0 for vertical cone, non-zero for inclined)
                p = d_r  # target radius (half of radiometer diameter)
                b = z_radiometer  # target height (z-coordinate of radiometer)
                
                vf = cone_view_factor_analytical(h, r, d, x, p, b)
                # Calculate view angle: 2 * arctan(r / sqrt(d^2 - r^2))
                view_angle_rad = 2 * np.arctan(r / np.sqrt(d**2 - r**2))
                view_angle_deg = np.degrees(view_angle_rad)
                window['vf'].update(f"{vf:.4f} (View angle: {view_angle_deg:.1f}°)")
            except Exception as e:
                sg.popup_error(f"Analytical calculation failed: {e}", font=("Arial", 14))
                continue
window.close()