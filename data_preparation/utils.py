import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show


def visualize_ignition_raster(raster: np.array, cause: int, season: int):
    fig, ax = plt.subplots(figsize=(8, 6))

    show(raster, ax=ax, cmap="gray")

    img = ax.get_images()[0]
    cbar = plt.colorbar(img, ax=ax, cmap='gray')
    cbar.set_label("ignition value")

    ax.set_title(f"Ignition Grid for season {season} : Cause: {cause}")
    plt.show()