import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Read CSV ---
df = pd.read_csv("../../../setup/data/blocks.csv", sep=',')
df = df[['Block', 'X', 'Y', 'Z', 'Dir']]

df['X'] = df['X'].astype(int)
df['Y'] = df['Y'].astype(int)
df['Z'] = df['Z'].astype(int)

# --- Track parameters ---
track_width = 32  # width of each block

# --- Helper: get rectangle corners for a block ---
def get_block_corners(x, z, width=32):
    """
    Returns 4 corners of a block (square) based on center x,z
    """
    w = width / 2
    corners = np.array([
        [x-w, z-w],
        [x+w, z-w],
        [x+w, z+w],
        [x-w, z+w]
    ])
    return corners

# --- Build track polygons ---
track_polygons = []
for idx, row in df.iterrows():
    corners = get_block_corners(row['X'], row['Z'], width=track_width)
    track_polygons.append(corners)

# --- Plot track ---
plt.figure(figsize=(10,10))
for poly in track_polygons:
    plt.fill(poly[:,0], poly[:,1], color='lightgray', edgecolor='black')

# Plot centerline
plt.plot(df['X'], df['Z'], 'r-o', label='Centerline')

plt.title("Trackmania Track Shape (Straights and 90° curves)")
plt.xlabel("X")
plt.ylabel("Z")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
