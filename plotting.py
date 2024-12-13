import matplotlib.pyplot as plt
import numpy as np
import re

def plot_pressure_distribution(Cp_upper, Cp_lower, x_c_upper, x_c_lower, angle=0):
	"""
	Plot a pressure distribution graph.

	Parameters:
	Cp_upper (list or array): Coefficients of pressure on the upper surface.
	Cp_lower (list or array): Coefficients of pressure on the lower surface.
	x_c_upper (list or array): x/c values for the upper surface.
	x_c_lower (list or array): x/c values for the lower surface.
	angle (float): Angle of attack in degrees, optional.
	"""
	plt.figure(figsize=(10, 6))

	# Plotting the pressure distribution
	plt.plot(x_c_upper, Cp_upper, label='Upper Surface', marker='o', color='blue', linestyle='-')
	plt.plot(x_c_lower, Cp_lower, label='Lower Surface', marker='o', color='red', linestyle='-')

	# Invert y-axis as it's a pressure distribution graph
	plt.gca().invert_yaxis()

	# Adding labels, title, and grid
	plt.xlabel('x/c', fontsize=12)
	plt.ylabel('$C_p$', fontsize=12)
	plt.title(f'Pressure Distribution over an Airfoil (Alpha = {angle}\u00b0)', fontsize=14)
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.axhline(0, color='black', linewidth=0.8, linestyle='-')  # Highlight the zero line for clarity

	# Adding a legend
	plt.legend(fontsize=12)

	# Set the x-axis limits to be between 0 and 1
	plt.xlim(0, 1)

	# Adjust the y-axis limits to show the full range of Cp values
	y_min = min(min(Cp_upper), min(Cp_lower)) - 0.1
	y_max = max(max(Cp_upper), max(Cp_lower)) + 0.1
	plt.ylim(y_max, y_min)  # Explicitly invert the y-axis range

	# Show the plot
	plt.tight_layout()
	plt.show()

# Example usage:
# Cp_upper = [-1.2, -1.0, -0.8, -0.5, -0.3]
# Cp_lower = [0.2, 0.3, 0.5, 0.6, 0.7]
# x_c_upper = [0.0, 0.25, 0.5, 0.75, 1.0]
# x_c_lower = [0.0, 0.25, 0.5, 0.75, 1.0]
# plot_pressure_distribution(Cp_upper, Cp_lower, x_c_upper, x_c_lower, angle=5)

import random

def get_bounds(a):
	multiples = [0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025]

	# Compute target value
	target = (np.max(a) - np.min(a)) / 6.5

	# Find the closest multiple
	closest_multiple = min(multiples, key=lambda x: abs(x - target))
	return np.ceil(np.max(a) / closest_multiple) * closest_multiple, np.floor(np.min(a) / closest_multiple) * closest_multiple, closest_multiple

def PlotCls(Cns, angles, name, name_x= r'$\alpha [deg]$', title_comment=''):

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(angles, Cns, label= re.sub(r'\[.*?\]', '', name)
, color='blue')  # First graph (blue)
	ax.scatter(angles, Cns, color='blue')

	if name_x != r'$\alpha [deg]$':
		bounds = get_bounds(angles)
		ax.set_xticks(np.arange(bounds[1], bounds[0], bounds[2]))
		ax.set_xticks(np.arange(bounds[1], bounds[0], bounds[2]/5), minor=True)
	else:
		ax.set_xticks(np.arange(-7.5, 17.5, 2.5))
		ax.set_xticks(np.arange(-7.5, 17.5, 0.5), minor=True)
		
	bounds = get_bounds(Cns)
	ax.set_yticks(np.arange(bounds[1], bounds[0], bounds[2]))
	ax.set_yticks(np.arange(bounds[1], bounds[0], bounds[2]/5), minor=True)
	# And a corresponding grid
	ax.grid(which='both')

	# Or if you want different settings for the grids:
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	# Add labels and title
	plt.xlabel(name_x)
	plt.ylabel(name)
	title = '' + title_comment
	plt.title(title)
	# plt.grid()

	# Show legend
	plt.legend()

	# Display the plot
	plt.show()

# def Plot2Cls(Cns, Cns2, angles, name, name2, name_x=r'\$\\alpha [deg]\$', x2s='G', title_comment='', invertedaxis=False, 
#             point_index=None, annotation_text=None, point_index2=None, annotation_text2=None, point_index3=None, annotation_text3=None):
#     if x2s[0] == 'G':
#         x2s = angles

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(angles, Cns, label=name, color='blue')
#     ax.plot(x2s, Cns2, label=name2, color='red')
#     ax.scatter(angles, Cns, color='blue')
#     ax.scatter(x2s, Cns2, color='red')

#     if name_x != r'\$\\alpha [deg]\$':
#         bounds = get_bounds(angles)
#         ax.set_xticks(np.arange(bounds[1], bounds[0], bounds[2]))
#         ax.set_xticks(np.arange(bounds[1], bounds[0], bounds[2] / 5), minor=True)
#     else:
#         ax.set_xticks(np.arange(-7.5, 17.5, 2.5))
#         ax.set_xticks(np.arange(-7.5, 17.5, 0.5), minor=True)

#     bounds = get_bounds(np.append(Cns, Cns2, axis=None))
#     ax.set_yticks(np.arange(bounds[1], bounds[0], bounds[2]))
#     ax.set_yticks(np.arange(bounds[1], bounds[0], bounds[2] / 5), minor=True)
#     # And a corresponding grid
#     ax.grid(which='both')

#     # Or if you want different settings for the grids:
#     ax.grid(which='minor', alpha=0.2)
#     ax.grid(which='major', alpha=0.5)
#     # Add labels and title
#     plt.xlabel(name_x)
#     plt.ylabel(name + ', ' + name2)
#     title = title_comment
#     plt.title(title)

#     # Add arrow and annotation if point_index and annotation_text are provided
#     def add_annotation(index, text):
#         if index is not None and text is not None:
#             x_point = angles[index] if index < len(angles) else None
#             y_point = Cns[index] if index < len(Cns) else None
#             if x_point is not None and y_point is not None:
#                 ax.annotate(
#                     text,
#                     xy=(x_point, y_point),
#                     xytext=(x_point + 0.05, y_point -.15 - (20-index)*.025),
#                     arrowprops=dict(facecolor='black', arrowstyle='->'),
#                     fontsize=10,
#                 )

#     add_annotation(point_index, annotation_text)
#     add_annotation(point_index2, annotation_text2)
#     add_annotation(point_index3, annotation_text3)

#     if invertedaxis:
#         plt.gca().invert_yaxis()
#     # Show legend
#     plt.legend()

#     # Display the plot
#     plt.show()
def Plot2Cls(Cns, Cns2, angles, name, name2, name_x=r'$\alpha [deg]$', x2s='G', title_comment='', invertedaxis=False, 
            point_index=None, annotation_text=None, point_index2=None, annotation_text2=None, point_index3=None, annotation_text3=None, scatter = False):
    if x2s[0] == 'G':
        x2s = angles
      
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if scatter:
        ax.scatter(angles, Cns, color='blue', label=name)
        ax.plot(x2s, Cns2, label=name2, color='red')
    else:
        ax.plot(angles, Cns, label=name, color='blue')
        ax.plot(x2s, Cns2, label=name2, color='red')
        ax.scatter(angles, Cns, color='blue')
        ax.scatter(x2s, Cns2, color='red')

    if name_x != r'$\alpha [deg]$':
        bounds = get_bounds(angles)
        ax.set_xticks(np.arange(bounds[1], bounds[0], bounds[2]))
        ax.set_xticks(np.arange(bounds[1], bounds[0], bounds[2] / 5), minor=True)
    else:
        ax.set_xticks(np.arange(-7.5, 17.5, 2.5))
        ax.set_xticks(np.arange(-7.5, 17.5, 0.5), minor=True)

    bounds = get_bounds(np.append(Cns, Cns2, axis=None))
    ax.set_yticks(np.arange(bounds[1], bounds[0], bounds[2]))
    ax.set_yticks(np.arange(bounds[1], bounds[0], bounds[2] / 5), minor=True)
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    # Add labels and title
    plt.xlabel(name_x)
    plt.ylabel(name + ', ' + name2)
    title = title_comment
    plt.title(title)

    # Add arrow and annotation if point_index and annotation_text are provided
    def add_annotation(index, text):
        if index is not None and text is not None:
            x_point = angles[index] if index < len(angles) else None
            y_point = Cns[index] if index < len(Cns) else None
            if x_point is not None and y_point is not None:
                ax.annotate(
                    text,
                    xy=(x_point, y_point),
                    xytext=(x_point + 0.008, y_point -.65),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10,
                )

    add_annotation(point_index, annotation_text)
    add_annotation(point_index2, annotation_text2)
    add_annotation(point_index3, annotation_text3)

    if invertedaxis:
        plt.gca().invert_yaxis()
    # Show legend
    plt.legend()

    # Display the plot
    plt.show()


def PlotCms(Cms, angles):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(angles, Cms, label='$C_m$', color='blue')  # First graph (blue)
	ax.scatter(angles, Cms)
	# major_ticks = np.arange(0, 101, 20)
	# minor_ticks = np.arange(0, 101, 5)
	ax.set_xticks(np.arange(-7.5, 17.5, 2.5))
	ax.set_xticks(np.arange(-7.5, 17.5, 0.5), minor=True)
	ax.set_yticks(np.arange(-0.5, 0.2, 0.1))
	ax.set_yticks(np.arange(-0.5, 0.2, 0.025), minor=True)
	# And a corresponding grid
	ax.grid(which='both')

	# Or if you want different settings for the grids:
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	# Add labels and title
	plt.xlabel(r'$\alpha$')
	plt.ylabel('$C_l$')
	plt.title(r'$C_m$ over $\alpha$')
	# plt.grid()
	# Show legend
	plt.legend()

	# Display the plot
	plt.show()


def PlotArbitrary(Thing, angles):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(angles, Thing, label='Thing', color='blue')  # First graph (blue)
	ax.scatter(angles, Thing)
	# major_ticks = np.arange(0, 101, 20)
	# minor_ticks = np.arange(0, 101, 5)

	plt.show()


def PlotWakeRake(velocities, y_values, alpha):
	# Convert positions to millimeters
	y_values_mm = np.array(y_values) * 1000  # Assuming y_values are in meters; convert to mm

	# Create the figure and axis
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	
	# Plot the data
	ax.plot(y_values_mm, velocities, label='$V_{wake}$', color='blue')  # Line plot (blue)
	ax.scatter(y_values_mm, velocities)  # Scatter points
	
	# Set appropriate ticks for better readability
	ax.set_xticks(np.linspace(min(y_values_mm), max(y_values_mm), 8))  # Fewer x-axis ticks
	ax.set_yticks(np.linspace(min(velocities), max(velocities), 8))  # Fewer y-axis ticks

	# Adjust tick label formatting to avoid overlap
	ax.tick_params(axis='x', labelrotation=45)
	
	# Add grid with customization
	ax.grid(which='both', alpha=0.5)
	ax.grid(which='minor', alpha=0.2)

	# Add labels and title
	plt.xlabel(r'$y$ (Position across wake) [mm]')
	plt.ylabel(r'$V_{wake}$ (Velocity)')
	plt.title(rf'Wake Velocity Profile at $\alpha = {alpha}^\circ$')
	
	# Show legend
	plt.legend()
	
	# Adjust layout for better spacing
	plt.tight_layout()
	
	# Display the plot
	plt.show()
