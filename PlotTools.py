import matplotlib.pyplot as plt
plt.style.use('scientific2')

import matplotlib as mpl
def create_subplot(N, M, axis_width=2, axis_height=2, margin=0.5, spacing=1, 
                   colorbar_subplots=None, colorbar_positions=None, colorbar_sizes=None,
                   colorbar_spacing=0.1, is_3d=None, show=True):
    """
    Create a subplot of N rows and M columns with optional 3D subplots and colorbars for specific subplots.

    Parameters:
    N (int): Number of rows.
    M (int): Number of columns.
    axis_width (float): Width of each axis in inches. Default is 2.
    axis_height (float): Height of each axis in inches. Default is 2.
    margin (float): Margin around each axis in inches. Default is 0.5.
    spacing (float): Spacing between axes in inches. Default is 1.
    colorbar_subplots (list of tuples): Indices of the subplots (row, column) to add the colorbars to. Default is None.
    colorbar_positions (list of str): Positions of the colorbars ('right', 'top', 'left', 'bottom'). Default is None.
    colorbar_sizes (list of float): Sizes of the colorbars in inches. Default is None.
    colorbar_spacing (float): Spacing between the axis and the colorbar in inches. Default is 0.1.
    is_3d (list of tuples): Indices of the subplots (row, column) to make 3D. Default is None.
    """
    # Validate colorbar inputs
    if colorbar_subplots and colorbar_positions and colorbar_sizes:
        if not (len(colorbar_subplots) == len(colorbar_positions) == len(colorbar_sizes)):
            raise ValueError("colorbar_subplots, colorbar_positions, and colorbar_sizes must have the same length")
    else:  # Ensure all or none
        colorbar_subplots, colorbar_positions, colorbar_sizes = [], [], []

    # Initialize is_3d if not specified
    if is_3d is None:
        is_3d = []

    # Calculate the total figure size
    fig_width = M * axis_width + (M + 1) * margin + (M - 1) * spacing
    fig_height = N * axis_height + (N + 1) * margin + (N - 1) * spacing

    # Create the figure with the calculated size
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

    # Create axes for each subplot
    axes = []
    for i in range(N):
        for j in range(M):
            left = (margin + j * (axis_width + spacing)) / fig_width
            bottom = 1 - (margin + (i + 1) * axis_height + i * spacing) / fig_height
            width = axis_width / fig_width
            height = axis_height / fig_height
            if (i, j) in is_3d:
                ax = fig.add_axes([left, bottom, width, height], projection='3d')
            else:
                ax = fig.add_axes([left, bottom, width, height])
            axes.append(ax)

    # Initialize an empty list for colorbar axes to return
    cax_list = []

    # Add colorbars to the specified subplots
    for cb_subplot, cb_position, cb_size in zip(colorbar_subplots, colorbar_positions, colorbar_sizes):
        row, col = cb_subplot
        # Validate the index
        if 0 <= row < N and 0 <= col < M:
            # Calculate position for the new colorbar axes
            base_ax = axes[row * M + col]
            x0, y0, width, height = base_ax.get_position().bounds
            if cb_position == 'right':
                cax = fig.add_axes([x0 + width + colorbar_spacing / fig_width, y0, cb_size / fig_width, height])
            elif cb_position == 'left':
                cax = fig.add_axes([x0 - (colorbar_spacing + cb_size) / fig_width, y0, cb_size / fig_width, height])
            elif cb_position == 'top':
                cax = fig.add_axes([x0, y0 + height + colorbar_spacing / fig_height, width, cb_size / fig_height])
            elif cb_position == 'bottom':
                cax = fig.add_axes([x0, y0 - (colorbar_spacing + cb_size) / fig_height, width, cb_size / fig_height])
            cax_list.append(cax)
        else:
            print(f"Invalid colorbar_subplot: {cb_subplot}. No colorbar added.")
            cax_list.append(None)

    return fig, axes, cax_list