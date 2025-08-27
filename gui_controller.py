"""
GUI Controller Module

This module handles the PySimpleGUI interface and event processing for the view factor calculator.
"""

import FreeSimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from view_factor_calculator import ViewFactorCalculator
from visualization import GeometryVisualizer


class ViewFactorGUI:
    """Main GUI controller for the view factor calculator."""
    
    def __init__(self):
        """Initialize the GUI."""
        sg.theme('Default1')
        self.window = None
        self.setup_layout()
    
    def setup_layout(self):
        """Create the GUI layout."""
        label_width = 23
        input_width = 10
        
        # Input column
        col1 = [
            [sg.Text("Flame Shape:", size=(label_width, 1), justification='r'),
             sg.Combo(['Cylinder', 'Cone'], key='shape', default_value='Cylinder', size=(input_width, 1), enable_events=True)],
            [sg.Text("Radiometer diameter [m]:", size=(label_width, 1), justification='r'),
             sg.Input('0.05', key='d_r', size=(input_width, 1))],
            [sg.Text("Radiometer height [m]:", size=(label_width, 1), justification='r'),
             sg.Input('0', key='z_radiometer', size=(input_width, 1))],
            [sg.Text("Distance (radiometer–flame) [m]:", size=(label_width, 1), justification='r'),
             sg.Input('0.20', key='distance', size=(input_width, 1))],
            [sg.Text("Flame diameter [m]:", size=(label_width, 1), justification='r'),
             sg.Input('0.10', key='d_f', size=(input_width, 1))],
            [sg.Text("Flame height [m]:", size=(label_width, 1), justification='r'),
             sg.Input('0.15', key='h_f', size=(input_width, 1))],
            [sg.Column([
                [sg.Text("Cone inclination x [m]:", size=(label_width, 1), justification='r'),
                 sg.Input('0', key='cone_x', size=(input_width, 1))]
            ], key='cone_x_col', visible=False)],
            [sg.Text("Calculation Method:", size=(label_width, 1), justification='r'),
             sg.Combo(['Analytical'], key='method', default_value='Analytical', size=(input_width, 1))],
            [sg.Push(), sg.Button('Calculate', key='calc'), sg.Push()],
        ]
        
        # Plots column
        plots_row = [
            sg.Canvas(key='canvas3d', size=(400, 400)),
            sg.Canvas(key='canvastop', size=(400, 400)),
        ]
        
        col2 = [
            plots_row,
            [sg.Text('View factor:', size=(12,1)), sg.Text('', key='vf', font=("Arial", 16, "bold"))]
        ]
        
        layout = [
            [sg.Column(col1, vertical_alignment='top'), sg.VSeperator(), sg.Column(col2, vertical_alignment='top')]
        ]
        
        self.window = sg.Window('Radiometer–Flame View Factor Calculator', layout, 
                               finalize=True, font=("Arial", 14))
        
        # Initialize cone inclination visibility based on default shape
        self.handle_shape_change('Cylinder')  # Default shape is Cylinder, so hide cone inclination
    
    def run(self):
        """Run the GUI event loop."""
        while True:
            event, values = self.window.read()
            
            if event in (sg.WIN_CLOSED, 'Exit'):
                break
                
            if event == 'calc':
                self.handle_calculation(values)
            
            # Handle shape change to show/hide cone inclination
            if event == 'shape':
                self.handle_shape_change(values['shape'])
        
        self.window.close()
    
    def handle_calculation(self, values):
        """Handle the calculation button event."""
        try:
            # Parse input values
            params = self.parse_inputs(values)
            
            # Update method options based on shape
            self.update_method_options(params['shape'])
            
            # Create visualizations
            self.update_plots(params)
            
            # Perform calculation
            self.perform_calculation(params)
            
        except Exception as e:
            sg.popup_error(f"Error: {e}")
    
    def parse_inputs(self, values):
        """Parse and validate input values."""
        try:
            return {
                'd_r': float(values['d_r']),
                'z_radiometer': float(values['z_radiometer']),
                'distance': float(values['distance']),
                'd_f': float(values['d_f']),
                'h_f': float(values['h_f']),
                'cone_x': float(values['cone_x']),
                'shape': values['shape'],
                'method': values['method']
            }
        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")
    
    def update_method_options(self, shape):
        """Update calculation method options based on selected shape."""
        # Only analytical method is available
        methods = ['Analytical']
        self.window['method'].update(values=methods, value=methods[0])
    
    def handle_shape_change(self, shape):
        """Handle shape selection change to show/hide relevant inputs."""
        if shape == "Cone":
            self.window['cone_x_col'].update(visible=True)
        elif shape == "Cylinder":
            self.window['cone_x_col'].update(visible=False)
    
    def update_plots(self, params):
        """Update the 3D and 2D plots."""
        # Create figures
        fig3d = GeometryVisualizer.create_3d_plot(
            params['d_r'], params['distance'], params['d_f'], params['h_f'],
            params['shape'], params['z_radiometer'], params['cone_x']
        )
        figtop = GeometryVisualizer.create_top_view(
            params['d_r'], params['distance'], params['d_f'], params['h_f'],
            params['shape'], params['cone_x']
        )
        
        # Update 3D plot
        self.update_canvas('canvas3d', fig3d)
        
        # Update top view plot
        self.update_canvas('canvastop', figtop)
    
    def update_canvas(self, canvas_key, fig):
        """Update a matplotlib canvas with a new figure."""
        canvas = self.window[canvas_key].Widget
        
        # Clear previous content
        for widget in canvas.winfo_children():
            widget.destroy()
        
        # Create new canvas
        canvas_fig = FigureCanvasTkAgg(fig, canvas)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas_fig, canvas)
        toolbar.update()
    
    def perform_calculation(self, params):
        """Perform the view factor calculation based on selected method."""
        shape = params['shape']
        
        # Use analytical method based on shape
        if shape == 'Cone':
            self.calculate_analytical_cone(params)
        elif shape == 'Cylinder':
            self.calculate_analytical_cylinder(params)
        else:
            sg.popup_error(f"Unsupported shape: {shape}")
    

    
    def calculate_analytical_cone(self, params):
        """Calculate view factor for cone using analytical method."""
        # Validate parameters
        if params['d_f']/2 >= params['distance']:
            sg.popup_error("Analytical calculation requires cone radius to be less than distance.")
            return
        
        try:
            vf, view_angle, half_vf = ViewFactorCalculator.analytical_cone(
                params['h_f'], params['d_f'], params['distance'], 
                params['cone_x'], params['d_r'], params['z_radiometer']
            )
            result_text = f"Full: {vf:.4f} | Half: {half_vf:.4f} (View angle: {view_angle:.1f}°)"
            self.window['vf'].update(result_text)
        except Exception as e:
            sg.popup_error(f"Analytical calculation failed: {e}")
    
    def calculate_analytical_cylinder(self, params):
        """Calculate view factor for cylinder using analytical method."""
        try:
            vf, view_angle, half_vf = ViewFactorCalculator.analytical_cylinder(
                params['h_f'], params['d_f'], params['distance'], 
                params['d_r'], params['z_radiometer']
            )
            result_text = f"Full: {vf:.4f} | Half: {half_vf:.4f} (View angle: {view_angle:.1f}°)"
            self.window['vf'].update(result_text)
        except Exception as e:
            sg.popup_error(f"Analytical calculation failed: {e}")


def main():
    """Main function to run the GUI."""
    gui = ViewFactorGUI()
    gui.run()


if __name__ == "__main__":
    main() 