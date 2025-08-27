# Radiometer–Flame View Factor Calculator

A GUI application for calculating view factors between a radiometer and flame geometries using both Monte Carlo simulation and analytical methods.

## Features

- **Multiple Flame Shapes**: Support for Cylinder and Cone geometries
- **Analytical Calculation Methods**:
  - **Cone**: Exact analytical solution for cone geometry
  - **Cylinder**: Exact analytical solution for cylinder geometry
- **3D Visualization**: Interactive 3D plots showing the geometry
- **Top View**: 2D top-down view for better spatial understanding
- **Parameter Control**: Adjustable dimensions for radiometer and flame
- **Interactive Plots**: Pan, zoom, and rotate capabilities

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
# Original version
python main.py

# Refactored version (recommended)
python main_refactored.py
```

## Usage

1. **Select Flame Shape**: Choose between Cylinder or Cone
2. **Set Parameters**:
   - Radiometer diameter and height
   - Distance between radiometer and flame
   - Flame diameter and height
3. **Calculate**: Click the Calculate button to compute the view factor using the appropriate analytical method

## Parameters

- **Radiometer diameter**: Diameter of the radiometer disk
- **Radiometer height**: Z-coordinate of the radiometer center
- **Distance**: Separation between radiometer and flame
- **Flame diameter**: Base diameter of the flame
- **Flame height**: Height of the flame geometry

## Calculation Methods

### Analytical Methods
- **Cone**: Uses exact mathematical solution from `view_factor_cone_target.py`
- **Cylinder**: Uses exact mathematical solution from `view_factor_cylinder_target.py`
- No uncertainty (deterministic)
- Fast computation
- Requires appropriate geometric constraints

## Project Structure

### Original Structure
- `main.py`: Single file containing all functionality (331 lines)

### Refactored Structure (Recommended)
- `main_refactored.py`: Clean entry point
- `gui_controller.py`: GUI interface and event handling
- `view_factor_calculator.py`: All calculation logic
- `visualization.py`: Plotting and visualization functions
- `view_factor_cone_target.py`: Analytical cone calculations
- `view_factor_cylinder_target.py`: Analytical cylinder calculations

### Benefits of Refactored Structure
- **Separation of Concerns**: Each module has a single responsibility
- **Maintainability**: Easier to modify and extend
- **Testability**: Individual components can be tested separately
- **Reusability**: Calculation and visualization modules can be used independently
- **Readability**: Cleaner, more organized code

## Output

The view factor is displayed as a decimal between 0 and 1, representing the fraction of radiation leaving the radiometer that reaches the flame.

For both cone and cylinder calculations, the view angle is also shown (e.g., "0.1234 (View angle: 45.2°)"). 