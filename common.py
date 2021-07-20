import numpy as np
import matplotlib.pyplot as plt

def plot_scalar_field(u, width, height):
        range = max(abs(np.min(u)), np.max(u))
        plt.imshow(u, cmap=plt.cm.coolwarm, origin='lower', vmin = -range, vmax=range, extent=(-1, width + 1, -1, height + 1))
        plt.colorbar()
