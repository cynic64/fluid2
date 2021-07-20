import numpy as np
import matplotlib.pyplot as plt
import common

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
                self.pressure = np.zeros((width, height))
                self.u = np.zeros((width+1, height))
                self.v = np.zeros((width, height+1))

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

grid = Grid(10, 10)
grid.u = np.sin(grid.u_meshes[0] ** 2 + grid.u_meshes[1]) * 10
grid.v = np.cos(grid.v_meshes[0] ** 2 + grid.v_meshes[1]) * 10
grid.plot()

plt.show()
