import numpy as np
from simulation import (
    LATTICE_Q, LATTICE_CX, LATTICE_CY, LATTICE_BB, 
    idx, get_indexes_from_image
)

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")

# Build the swaps like wall_permutation does
swaps_by_wall = {}
for wall_idx, (i,j) in enumerate(walls):
    swaps_by_wall[(i,j)] = []
    for q in range(LATTICE_Q):
        x = np.mod(i + LATTICE_CX[q], size_x)
        y = np.mod(j + LATTICE_CY[q], size_y)
        
        idx_1 = idx(x, y, q, size_y)
        idx_2 = idx(i, j, LATTICE_BB[q], size_y)
        
        swaps_by_wall[(i,j)].append((idx_1, idx_2))

# Check what swaps neighbor walls try to make that could interact
corner = (0, 0)
neighbors = [(i,j) for i,j in [(1,0), (0,1), (199,0), (0,499), (1,1), (199,1), (199,499), (1,499)] if (i,j) in swaps_by_wall]

print("Corner (0,0) swaps:")
for idx1, idx2 in swaps_by_wall[(0,0)]:
    print(f"  P[{idx1:7d}] <-> P[{idx2:7d}]")

print("\n" + "="*70)
print("NEIGHBOR SWAPS THAT INTERACT WITH CORNER'S INDICES:")
print("="*70)

corner_indices = set()
for idx1, idx2 in swaps_by_wall[(0,0)]:
    corner_indices.add(idx1)
    corner_indices.add(idx2)

print(f"\nCorner's indices: {sorted(corner_indices)}")
print()

for neighbor in neighbors:
    i, j = neighbor
    print(f"Wall {neighbor}:")
    has_conflict = False
    for idx1, idx2 in swaps_by_wall[(i,j)]:
        if idx1 in corner_indices or idx2 in corner_indices:
            print(f"  ⚠️  P[{idx1:7d}] <-> P[{idx2:7d}]  INTERACTS with corner!")
            has_conflict = True
    if not has_conflict:
        print(f"  (no conflicts with corner)")
