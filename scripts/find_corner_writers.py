from simulation import (
    LATTICE_Q, idx, calc_permutation, wall_permutation, get_indexes_from_image
)
import numpy as np

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image('assets/image.png')
wall_set = set((int(i), int(j)) for i,j in walls)
bc_left_set = set((int(i), int(j)) for i,j in bc_left)

P = calc_permutation(size_x, size_y)
w_p = wall_permutation(P, walls, size_x, size_y)

print('CORNER INDICES:')
corner_indices = set()
for q in range(LATTICE_Q):
    corner_indices.add(idx(0, 0, q, size_y))

corner_indices = sorted(list(corner_indices))
print(f'Corner uses indices: {corner_indices}')
print()

print("="*70)
print('WHAT CAN WRITE TO CORNER INDICES (using P)?')
print("="*70)
print()

# Find all positions that have a P[idx] pointing to corner
positions_writing_P = {}
for idx_val in corner_indices:
    sources = []
    for source_idx, target_idx in enumerate(P):
        if target_idx == idx_val:
            q = source_idx % LATTICE_Q
            xy = source_idx // LATTICE_Q
            j = xy % size_y
            i = xy // size_y
            is_bc = (i, j) in bc_left_set
            sources.append((i, j, q, is_bc))
    
    if sources:
        positions_writing_P[idx_val] = sources

for corner_idx in sorted(positions_writing_P.keys()):
    q_dst = corner_idx % LATTICE_Q
    print(f'Corner dir {q_dst}:')
    for i, j, q, is_bc in positions_writing_P[corner_idx]:
        bc_mark = " ⚠️  BC NODE!" if is_bc else ""
        print(f'  ← ({i:3d}, {j:3d}) dir {q}{bc_mark}')

print()
print("="*70)
print('SAME ANALYSIS WITH WALL PERMUTATION (w_p):')
print("="*70)
print()

positions_writing_wp = {}
for idx_val in corner_indices:
    sources = []
    for source_idx, target_idx in enumerate(w_p):
        if target_idx == idx_val:
            q = source_idx % LATTICE_Q
            xy = source_idx // LATTICE_Q
            j = xy % size_y
            i = xy // size_y
            is_bc = (i, j) in bc_left_set
            sources.append((i, j, q, is_bc))
    
    if sources:
        positions_writing_wp[idx_val] = sources

for corner_idx in sorted(positions_writing_wp.keys()):
    q_dst = corner_idx % LATTICE_Q
    print(f'Corner dir {q_dst}:')
    for i, j, q, is_bc in positions_writing_wp[corner_idx]:
        bc_mark = " ⚠️  BC NODE!" if is_bc else ""
        print(f'  ← ({i:3d}, {j:3d}) dir {q}{bc_mark}')
