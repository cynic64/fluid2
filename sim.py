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

                # Also generate positions for the averaged velocity fields, where velocities are at grid cell centers
                self.avg_meshes = np.meshgrid(np.arange(0.5, self.width+0.5, 1), np.arange(0.5, self.height+0.5, 1))

        def plot(self):
                # Subplot 1: individual velocities
                plt.subplot(1, 2, 1)
                plt.title('Pressure and velocity')
                self.plot_grid_lines()

                # Plot pressure
                common.plot_scalar_field(self.pressure, self.width, self.height)

                # Plot velociy fields
                plt.quiver(self.u_meshes[0], self.u_meshes[1], self.u, np.zeros_like(self.u), angles='xy', scale = 200)
                plt.quiver(self.v_meshes[0], self.v_meshes[1], np.zeros_like(self.v), self.v, angles='xy', scale = 200)

                # Subplot 2: average velocity
                plt.subplot(1, 2, 2)
                plt.title('Average velocity')
                self.plot_grid_lines()

                avg_u, avg_v = self.average_velocities()
                plt.quiver(self.avg_meshes[0], self.avg_meshes[1], avg_u, avg_v, angles='xy', scale = 200)

        def plot_grid_lines(self):
                axes = plt.gca()
                x_ticks = np.arange(0, self.width+1, 1)
                y_ticks = np.arange(0, self.height+1, 1)
                axes.set_xticks(x_ticks)
                axes.set_yticks(y_ticks)
                plt.grid()

        def interpolate(self, x, y, idx):
                field = (self.u, self.v)[idx]

                # U is offset vertically by 0.5
                if idx == 0: y -= 0.5
                # V is offset horizontally by 0.5

                elif idx == 1: x -= 0.5
                # Lowest x and y
                lx, ly = math.floor(x), math.floor(y)

                # The bounds depend on which field is being interpolated: U and V each have an extra cell in each direction (horizontal/vertical)
                max_width, max_height = self.width, self.height
                if idx == 0: max_width += 1
                elif idx == 1: max_height += 1

                if lx < 0 or ly < 0 or lx+1 == max_width or ly+1 == max_height:
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

        def apply_convection(self, dt):
                new_u = self.u.copy()
                new_v = self.v.copy()

                # Only convect inside the boundary. U has an extra cell horizontally
                for y in range(1, self.height-1):
                        for x in range(1, self.width):
                                # Get new velocity for u, offset by 0.5 vertically
                                new_u[y,x] = self.rk2_trace(x, y + 0.5, dt, 0)

                # V has an extra cell vertically
                for y in range(1, self.height):
                        for x in range(1, self.width-1):
                                # Get new velocity at v, offset by 0.5 horizontally
                                new_v[y,x] = self.rk2_trace(x + 0.5, y, dt, 1)

                self.u = new_u
                self.v = new_v

        def rk2_trace(self, x, y, dt, idx):
                # Order-two Runge-Kutta interpolation: take half an Euler step and use the velocity there as an average over the whole step
                half_x = x - self.interpolate(x, y, 0) * 0.5 * dt
                half_y = y - self.interpolate(x, y, 1) * 0.5 * dt
                half_vel = (self.interpolate(half_x, half_y, 0), self.interpolate(half_x, half_y, 1))

                final_x = x - half_vel[0] * dt
                final_y = y - half_vel[1] * dt
                return self.interpolate(final_x, final_y, idx)

        def average_velocities(self):
                u_avg = (self.u[:,:-1] + self.u[:,1:]) / 2
                v_avg = (self.v[:-1,:] + self.v[1:,:]) / 2
                return (u_avg, v_avg)

        def apply_viscosity(self, dt, nu):
                # Laplacians for U (dx and dy are 1, so there is no need to divide)
                u_diff2_x = self.u[1:-1,2:] - 2*self.u[1:-1,1:-1] + self.u[1:-1,:-2]
                u_diff2_y = self.u[2:,1:-1] - 2*self.u[1:-1,1:-1] + self.u[:-2,1:-1]
                self.u[1:-1,1:-1] += nu * dt * (u_diff2_x + u_diff2_y)

                # Laplacians for V
                v_diff2_x = self.v[1:-1,2:] - 2*self.v[1:-1,1:-1] + self.v[1:-1,:-2]
                v_diff2_y = self.v[2:,1:-1] - 2*self.v[1:-1,1:-1] + self.v[:-2,1:-1]
                self.v[1:-1,1:-1] += nu * dt * (v_diff2_x + v_diff2_y)

def calc_time_step(u, v):
        '''
        Calculates the time step according to the CFL condition: Fluid should not flow more than
        one grid spacing in each step.
        '''
        grid_spacing = 1
        max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)))
        return grid_spacing / max_vel

def click_callback(grid, event):
        plt.gcf().clear()
        #grid.apply_convection(0.05)
        grid.apply_viscosity(0.05, 1)
        grid.plot()
        plt.draw()

        #print(event.xdata, event.ydata, grid.interpolate(event.xdata, event.ydata, 0))

plt.connect('button_press_event', lambda event: click_callback(grid, event))

grid = Grid(10, 10)
grid.u = np.full(grid.u.shape, 0, dtype=float)
grid.v = np.full(grid.v.shape, 0, dtype=float)
grid.v[:,4] = 50
grid.plot()

plt.show()
