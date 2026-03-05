"""
Visualiser — reads the binary grid.dat file and renders a grayscale image
of the ray-traced sphere.

Usage:
    python visualizer.py [prefix]

The optional prefix is prepended to the output filename, e.g.
    python visualizer.py v100_1000_dp_
produces  ./images/v100_1000_dp_image.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import re
import pandas as pd

def load_dat(filename):
    with open(filename, "rb") as f:
        F, N, R, C = np.fromfile(f, dtype=np.int32, count=4)
        if F == 0:
            grid = np.fromfile(f, dtype=np.float64, count=(R * C)).reshape((R, C))
        else: 
            grid = np.fromfile(f, dtype=np.float32, count=(R * C)).reshape((R, C))

    return grid, N, R, C

def parse_files(location):
    match_string = r'grid.dat'
    files = []
    for f in os.listdir(location):
        match = re.match(match_string, f)
        if not match:
            continue
        files.append(f"{location}/{f}")

    df = pd.DataFrame({
        "files": files,
    })

    return df

def generate_image(file_path, prefix=""):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)

    grid, N, R, C = load_dat(file_path)

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="equal",
        cmap="gray"
    )

    ax.set_axis_off()

    output_name = f"./images/{prefix}image.png"
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    files = parse_files("./data")
    prefix = ""
    if len(sys.argv) == 2:
        prefix = sys.argv[1]

    for f in files['files']:
        print(f"Generating Image...")
        generate_image(f, prefix)
