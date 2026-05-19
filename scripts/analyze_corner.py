import numpy as np
from simulation import (
    LATTICE_Q, LATTICE_CX, LATTICE_CY, LATTICE_BB, 
    idx, calc_permutation, wall_permutation, 
    get_indexes_from_image
)

# Get walls and dimensions
walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")

# Convert to set for faster lookup
wall_set = set((i,j) for i,j in walls)

print(f"Domain size: {size_x} x {size_y}")
print(f"Total walls: {len(walls)}")
print()

# Check corner (0,0) and its neighbors
corner = (0, 0)
print(f"Is (0,0) a wall? {corner in wall_set}")
print("\nNeighbors of (0,0) in each direction:")

for q in range(LATTICE_Q):
    x = np.mod(0 + LATTICE_CX[q], size_x)
    y = np.mod(0 + LATTICE_CY[q], size_y)
    bb_q = LATTICE_BB[q]
    
    is_wall = (x, y) in wall_set
    wall_marker = "WALL" if is_wall else "FLUID"
    
    print(f"  Dir {q} (cx={LATTICE_CX[q]:2d}, cy={LATTICE_CY[q]:2d}): pos ({x:3d}, {y:3d}) -> {wall_marker}")

print("\n" + "="*60)
print("CRITICAL ISSUE IDENTIFIED:")
print("="*60)
print()

# Check the specific problem with direction 0
print("Direction 0 (rest): cx=0, cy=0")
print("  - It doesn't stream anywhere (stays at origin)")
print("  - Bounce-back direction is also 0")
print("  - This creates a SELF-SWAP in the permutation!")
print()
print("For wall bounce-back, direction 0 should NOT be swapped.")
print("The rest component should bounce back implicitly,")
print("not through a permutation swap.")
