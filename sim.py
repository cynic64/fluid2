import numpy as np
import scipy.sparse.linalg
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
                self.u = np.zeros((height+2, width+1))
                self.v = np.zeros((height+1, width+2))

                # Generate positions for velocity fields, used for plotting them at grid
                # edges rather than centers
                self.u_meshes = np.meshgrid(np.arange(0, self.width+1, 1), np.arange(-0.5, self.height + 1.5, 1))
                self.v_meshes = np.meshgrid(np.arange(-0.5, self.width+1.5, 1), np.arange(0, self.height + 1, 1))

                # Also generate positions for the averaged velocity fields, where velocities are at grid cell centers
                self.avg_meshes = np.meshgrid(np.arange(0.5, self.width+0.5, 1), np.arange(0.5, self.height+0.5, 1))

                # Generate pressure coefficient matrix
                self.pressure_coef = np.zeros((width*height, width*height))
                for y in range(height):
                        for x in range(width):
                                idx = y*width + x

                                # In the row of this index, set the cell itself to the negative
                                # number of neighbors and the corresponding neighbors to 1
                                if x >= 1:
                                        # Left neighbor exists
                                        self.pressure_coef[idx,idx] -= 1
                                        self.pressure_coef[idx-1,idx] = 1
                                if x < width - 1:
                                        # Right neighbor exists
                                        self.pressure_coef[idx,idx] -= 1
                                        self.pressure_coef[idx+1,idx] = 1
                                if y >= 1:
                                        # Lower neighbor exists
                                        self.pressure_coef[idx,idx] -= 1
                                        self.pressure_coef[idx-width,idx] = 1
                                if y < height - 1:
                                        # Upper neighbor exists
                                        self.pressure_coef[idx,idx] -= 1
                                        self.pressure_coef[idx+width,idx] = 1

                # Directly set the pressure of one cell, because all that matters is the pressure
                # gradient
                self.pressure_coef[0,0] = 1

                print(self.pressure_coef)

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

        def apply_bcs(self):
                # Left
                self.u[:,0] = -self.u[:,1]
                # Right
                self.u[:,-1] = -self.u[:,-2]
                # Bottom
                self.u[0:1,:] = 0
                # Top
                self.u[-2:-1,:] = 10

                # Left
                self.v[:,0:1] = 0
                # Right
                self.v[:,-2:1] = 0
                # Bottom
                self.v[0,:] = -self.v[1,:]
                # Top
                self.v[-1,:] = -self.v[-2,:]

        def apply_convection(self, dt):
                dx, dy = 1, 1
                v_at_u = (self.v[1:,1:-2] + self.v[1:,2:-1] + self.v[:-1,1:-2] + self.v[:-1,2:-1]) / 4
                u_at_v = (self.u[1:-2,1:] + self.u[2:-1,1:] + self.u[1:-2,:-1] + self.u[2:-1,:-1]) / 4

                # Predict
                predicted_u = self.u.copy()
                u_diff_x_forward = (self.u[1:-1,2:] - self.u[1:-1,1:-1]) / dx
                u_diff_y_forward = (self.u[2:,1:-1] - self.u[1:-1,1:-1]) / dy
                predicted_u[1:-1,1:-1] += -dt * (self.u[1:-1,1:-1] * u_diff_x_forward + v_at_u * u_diff_y_forward)

                predicted_v = self.v.copy()
                v_diff_x_forward = (self.v[1:-1,2:] - self.v[1:-1,1:-1]) / dx
                v_diff_y_forward = (self.v[2:,1:-1] - self.v[1:-1,1:-1]) / dy
                predicted_v[1:-1,1:-1] += -dt * (u_at_v * v_diff_x_forward + self.v[1:-1,1:-1] * v_diff_y_forward)

                # Correct
                u_corrected = (self.u + predicted_u) / 2
                u_pred_diff_x_backward = (predicted_u[1:-1,1:-1] - predicted_u[1:-1,:-2]) / dx
                u_pred_diff_y_backward = (predicted_u[1:-1,1:-1] - predicted_u[:-2,1:-1]) / dy
                u_corrected[1:-1,1:-1] += -0.5 * dt * (self.u[1:-1,1:-1] * u_pred_diff_x_backward + v_at_u * u_pred_diff_y_backward)

                v_corrected = (self.v + predicted_v) / 2
                v_pred_diff_x_backward = (predicted_v[1:-1,1:-1] - predicted_v[1:-1,:-2]) / dx
                v_pred_diff_y_backward = (predicted_v[1:-1,1:-1] - predicted_v[:-2,1:-1]) / dy
                v_corrected[1:-1,1:-1] += -0.5 * dt * (u_at_v * v_pred_diff_x_backward + self.v[1:-1,1:-1] * v_pred_diff_y_backward)

                self.u[1:-1,1:-1] = u_corrected[1:-1,1:-1]
                self.v[1:-1,1:-1] = v_corrected[1:-1,1:-1]

        def average_velocities(self):
                u_avg = (self.u[1:-1,:-1] + self.u[1:-1,1:]) / 2
                v_avg = (self.v[:-1,1:-1] + self.v[1:,1:-1]) / 2
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

        def apply_pressure(self, dt, rho):
                # Solve for pressure
                u_diff_x = self.u[1:-1,1:] - self.u[1:-1,:-1]
                v_diff_y = self.v[1:,1:-1] - self.v[:-1,1:-1]
                divergence = u_diff_x + v_diff_y

                pressure_rhs = rho / dt * divergence
                pressure_rhs_flat = np.reshape(pressure_rhs, (self.width*self.height))

                pressure_flat = scipy.sparse.linalg.spsolve(self.pressure_coef, pressure_rhs_flat)
                self.pressure = np.reshape(pressure_flat, (self.width, self.height))

                # Apply pressure
                p_diff_x = self.pressure[:,1:] - self.pressure[:,:-1]
                p_diff_y = self.pressure[1:,:] - self.pressure[:-1,:]

                self.u[1:-1,1:-1] += -(dt / rho) * p_diff_x
                self.v[1:-1,1:-1] += -(dt / rho) * p_diff_y

def calc_time_step(u, v):
        '''
        Calculates the time step according to the CFL condition: Fluid should not flow more than
        one grid spacing in each step.
        '''
        grid_spacing = 1
        max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)))
        return grid_spacing / max_vel

def click_callback(grid, event):
        dt, nu, rho = 0.05, 0.05, 0.3
        plt.gcf().clear()

        for i in range(100):
                grid.apply_bcs()
                grid.apply_convection(dt)
                grid.apply_viscosity(dt, nu)
                grid.apply_pressure(dt, rho)

        grid.plot()
        plt.draw()

        #print(event.xdata, event.ydata, grid.interpolate(event.xdata, event.ydata, 0))

plt.connect('button_press_event', lambda event: click_callback(grid, event))

grid = Grid(60, 60)
grid.plot()

ct = 0
dt, nu, rho = 0.05, 0.1, 1.0
while 1:
        plt.clf()
        grid.plot()

        plt.savefig(f'img/{ct:04}.png', dpi=600)
        print(ct)

        for i in range(100):
                grid.apply_bcs()
                grid.apply_convection(dt)
                grid.apply_viscosity(dt, nu)
                grid.apply_pressure(dt, rho)
                print(f'\t{i}')

        ct += 1
