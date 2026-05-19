import numpy as np
from simulation import (
    LATTICE_Q, LATTICE_CX, LATTICE_CY, LATTICE_BB, 
    idx, calc_permutation, wall_permutation, 
    get_indexes_from_image
)

# Get walls and dimensions
walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")

# Calculate permutation
P = calc_permutation(size_x, size_y)
w_p = wall_permutation(P, walls, size_x, size_y)

# Check corner node (0,0)
corner = (0, 0)
is_corner_wall = corner in [(i,j) for i,j in walls]
print(f"Is (0,0) a wall? {is_corner_wall}")
print()

# Print the permutation indices for corner node and directions
print("=== CORNER NODE (0,0) ANALYSIS ===")
for q in range(LATTICE_Q):
    idx_00_q = idx(0, 0, q, size_y)
    # Where does this direction stream from in the original permutation?
    source_orig = P[idx_00_q]
    source_mod = w_p[idx_00_q]
    
    x = np.mod(0 + LATTICE_CX[q], size_x)
    y = np.mod(0 + LATTICE_CY[q], size_y)
    bb_q = LATTICE_BB[q]
    idx_neighbor_bb = idx(0, 0, bb_q, size_y)
    
    print(f"Direction {q} (cx={LATTICE_CX[q]:2d}, cy={LATTICE_CY[q]:2d}):")
    print(f"  Target neighbor after stream: ({x}, {y})")
    print(f"  P[{idx_00_q}] = {source_orig} vs w_p[{idx_00_q}] = {source_mod} (changed: {source_orig != source_mod})")
    print(f"  Bounce-back direction {bb_q}")
    print()

# Now check what got swapped at (0,0)
print("\n=== SWAP OPERATIONS AT (0,0) ===")
for q in range(LATTICE_Q):
    x = np.mod(0 + LATTICE_CX[q], size_x)
    y = np.mod(0 + LATTICE_CY[q], size_y)
    bb_q = LATTICE_BB[q]
    
    idx_target = idx(x, y, q, size_y)
    idx_source = idx(0, 0, bb_q, size_y)
    
    print(f"Direction {q}: Swap P[{idx_target}] <-> P[{idx_source}]")
    print(f"  At position ({x},{y}) direction {q} <-> At position (0,0) direction {bb_q}")
    
    # Check if there's a conflict (same index appearing in multiple swaps)
    if idx_target == idx_source:
        print(f"  ⚠️  SELF-SWAP: These are the same index!")
    print()
