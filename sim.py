import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import common
import math

class Grid:
        '''
        Grid storing pressure at the center of each cell and velocity at the cell faces.

        The x-velocity is stored at the x-min face, and the y-velocity at the y-min face.

        The pressure field is (width, height), the u field is (width+1, height), and the v field is
        (width, height+1)
        '''
        def __init__(self, width, height):
                self.width = width
                self.height = height
                self.pressure = np.zeros((height, width))
                self.u = np.zeros((height, width+1))
                self.v = np.zeros((height+1, width))

                # Generate positions for velocity fields, used for plotting them at grid
                # edges rather than centers
                self.u_meshes = np.meshgrid(np.arange(0, self.width+1, 1), np.arange(0.5, self.height + 0.5, 1))
                self.v_meshes = np.meshgrid(np.arange(0.5, self.width+0.5, 1), np.arange(0, self.height + 1, 1))

        def plot(self):
                plt.title('Pressure and velocity')

                # Plot grid lines
                axes = plt.gca()
                x_ticks = np.arange(0, self.width+1, 1)
                y_ticks = np.arange(0, self.height+1, 1)
                axes.set_xticks(x_ticks)
                axes.set_yticks(y_ticks)
                plt.grid()

                # Plot pressure
                common.plot_scalar_field(self.pressure, self.width, self.height)

                # Plot velociy fields
                plt.quiver(self.u_meshes[0], self.u_meshes[1], self.u, np.zeros_like(self.u), angles='xy', scale = 200)
                plt.quiver(self.v_meshes[0], self.v_meshes[1], np.zeros_like(self.v), self.v, angles='xy', scale = 200)

        def interpolate(self, x, y, idx):
                field = (self.u, self.v)[idx]

                # U is offset vertically by 0.5
                if idx == 0: y -= 0.5
                # V is offset horizontally by 0.5

                elif idx == 1: x -= 0.5
                # Lowest x and y
                lx, ly = math.floor(x), math.floor(y)

                if lx < 0 or ly < 0 or lx+1 == self.width or ly+1 == self.height:
                        raise ValueError(f'x and y out of bounds: ({x}, {y})')

                bottom_left = field[ly, lx]
                bottom_right = field[ly, lx+1]
                top_left = field[ly+1, lx]
                top_right = field[ly+1, lx+1]

                dist_left = x - lx
                dist_right = lx+1 - x
                dist_bottom = y - ly
                dist_top = ly+1 - y

                return bottom_left * (1 - dist_left) * (1 - dist_bottom) \
                        + bottom_right * (1 - dist_right) * (1 - dist_bottom) \
                        + top_left * (1 - dist_left) * (1 - dist_top) \
                        + top_right * (1 - dist_right) * (1 - dist_top)

        def convect(self, dt):
                for y in range(1, self.height-1):
                        pass

def calc_time_step(u, v):
        '''
        Calculates the time step according to the CFL condition: Fluid should not flow more than
        one grid spacing in each step.
        '''
        grid_spacing = 1
        max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)))
        return grid_spacing / max_vel

def click_callback(grid, event):
        print(event.xdata, event.ydata, grid.interpolate(event.xdata, event.ydata, 1))

plt.connect('button_press_event', lambda event: click_callback(grid, event))

grid = Grid(10, 10)
grid.u = np.full(grid.u.shape, 10)
grid.v = np.full(grid.v.shape, 10)
grid.v[5,5] = 0
print(grid.u.shape, grid.u_meshes[0].shape, grid.u_meshes[1].shape)
grid.plot()

plt.show()
