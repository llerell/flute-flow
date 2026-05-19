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

# Check corner node (0,0) - specifically direction 0
idx_00_q0 = idx(0, 0, 0, size_y)

print("Direction 0 at corner (0,0):")
print(f"  P[{idx_00_q0}] = {P[idx_00_q0]}")
print(f"  w_p[{idx_00_q0}] = {w_p[idx_00_q0]}")
print(f"  Was swapped? {P[idx_00_q0] != w_p[idx_00_q0]}")
print()

if P[idx_00_q0] == w_p[idx_00_q0]:
    print("✓ FIXED: Direction 0 is NO LONGER swapped in wall_permutation!")
else:
    print("✗ PROBLEM: Direction 0 is still being swapped!")
