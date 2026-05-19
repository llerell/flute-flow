import numpy as np
from simulation import (
    LATTICE_Q, LATTICE_CX, LATTICE_CY, LATTICE_BB, 
    idx, calc_permutation, wall_permutation, 
    get_indexes_from_image
)

# Get walls and dimensions
walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")
P = calc_permutation(size_x, size_y)

# Track all swaps in the order they would be applied
swaps = []
for (i,j) in walls:
     for q in range(LATTICE_Q):
            x = np.mod(i + LATTICE_CX[q], size_x)
            y = np.mod(j + LATTICE_CY[q], size_y)
            bb_q = LATTICE_BB[q]
            
            idx_1 = idx(x, y, q, size_y)
            idx_2 = idx(i, j, bb_q, size_y)
            swaps.append((idx_1, idx_2, (i,j), q, (x,y)))

print(f"Total swaps scheduled: {len(swaps)}")
print()

# Look for conflicts at corner (0,0)
print("=" * 70)
print("SWAPS AFFECTING CORNER (0,0):")
print("=" * 70)

corner_swaps = [(idx1, idx2, wall, q, target) for idx1, idx2, wall, q, target in swaps if wall == (0,0)]

for idx1, idx2, wall, q, target in corner_swaps:
    print(f"Swap P[{idx1:7d}] <-> P[{idx2:7d}]  (wall {wall} dir {q} -> {target})")

print()
print("=" * 70)
print("CONFLICT CHECK: Do any indices appear in multiple swaps?")
print("=" * 70)

corner_swap_indices = set()
for idx1, idx2, _, _, _ in corner_swaps:
    corner_swap_indices.add(idx1)
    corner_swap_indices.add(idx2)

print(f"\nIndices involved in corner swaps: {sorted(corner_swap_indices)[:20]}...")
print()

# Check if multiple walls are trying to swap the same indices
print("=" * 70)
print("DETAILED SWAP ORDER & CONFLICTS:")
print("=" * 70)

# Apply swaps manually to see what happens
P_debug = np.copy(P)
wall_count = {}

for i, (idx_1, idx_2, wall, q, target) in enumerate(swaps):
    wall_key = wall
    if wall_key not in wall_count:
        wall_count[wall_key] = 0
    wall_count[wall_key] += 1
    
    # Print each swap involving corner
    if wall == (0,0):
        val_before_1 = P_debug[idx_1]
        val_before_2 = P_debug[idx_2]
        
        # Check if this swap undoes a previous swap
        print(f"\nSwap #{i}: {wall} dir {q}")
        print(f"  P[{idx_1}] (currently {val_before_1}) <-> P[{idx_2}] (currently {val_before_2})")
        
        # Apply
        P_debug[idx_1], P_debug[idx_2] = P_debug[idx_2], P_debug[idx_1]
        
        print(f"  After: P[{idx_1}] = {P_debug[idx_1]}, P[{idx_2}] = {P_debug[idx_2]}")

print(f"\n\nFinal P[0] after all swaps: {P_debug[0]} (original: {P[0]})")
print(f"Changed? {P_debug[0] != P[0]}")
