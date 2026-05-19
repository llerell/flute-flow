import numpy as np
from simulation import (
    LATTICE_Q, LATTICE_CX, LATTICE_CY, LATTICE_BB, 
    idx, calc_permutation, wall_permutation, 
    get_indexes_from_image
)

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")
wall_set = set((i,j) for i,j in walls)
bc_left_set = set((i,j) for i,j in bc_left)

P = calc_permutation(size_x, size_y)
w_p = wall_permutation(P, walls, size_x, size_y)

# AFTER STREAM: Where does corner (0,0) get its distributions from?
corner = (0, 0)

print("="*70)
print("CORNER (0,0) - SOURCE OF EACH DISTRIBUTION AFTER STREAM")
print("="*70)
print()

for q in range(LATTICE_Q):
    idx_corner_q = idx(0, 0, q, size_y)
    
    # After stream with permutation w_p, corner dir q gets value from w_p[idx_corner_q]
    source_linear = w_p[idx_corner_q]
    
    # Decompose: which (i,j,q_src) is this?
    q_src = source_linear % LATTICE_Q
    xy_src = source_linear // LATTICE_Q
    j_src = xy_src % size_y
    i_src = xy_src // size_y
    
    is_wall = (i_src, j_src) in wall_set
    is_bc_left = (i_src, j_src) in bc_left_set
    
    status = []
    if is_wall:
        status.append("WALL")
    if is_bc_left:
        status.append("BC_LEFT")
    if not status:
        status.append("FLUID")
    
    marker = ""
    if is_bc_left:
        marker = " ⚠️  BC NODE!"
    
    print(f"Corner dir {q}: w_p[{idx_corner_q:7d}] = {source_linear:7d}")
    print(f"  ← ({i_src:3d}, {j_src:3d}) dir {q_src:d}  [{', '.join(status)}]{marker}")

print()
print("="*70)
print("BC NODES NEAR CORNER:")
print("="*70)

# Show BC nodes within distance 3 of corner
near_bc = [(int(bc[0]), int(bc[1])) for bc in bc_left if abs(bc[0] - 0) <= 3 and abs(bc[1] - 1) <= 3]
print(f"BC nodes near corner: {sorted(near_bc)[:15]}")

print()
print("="*70)
print("TRACING BC NODE (2,1) -> CORNER (0,0):")
print("="*70)
print()

bc_node = (2, 1)
print(f"BC node {bc_node}:")

# What values from (2,1) end up at corner?
for q in range(LATTICE_Q):
    idx_bc = idx(2, 1, q, size_y)
    
    # Find which destinations this (2,1,q) feeds into
    # by checking where w_p[x] == idx_bc
    destinations = []
    for dest_idx in range(len(w_p)):
        if w_p[dest_idx] == idx_bc:
            q_dst = dest_idx % LATTICE_Q
            xy_dst = dest_idx // LATTICE_Q
            j_dst = xy_dst % size_y
            i_dst = xy_dst // size_y
            destinations.append((i_dst, j_dst, q_dst, dest_idx))
    
    if destinations:
        print(f"\n  Dir {q} (from w_p[{idx_bc}]={w_p[idx_bc]}) streams to:")
        for i_dst, j_dst, q_dst, dest_idx in destinations:
            corner_marker = " ← CORNER!" if (i_dst, j_dst) == (0, 0) else ""
            print(f"    ({i_dst}, {j_dst}) dir {q_dst} (w_p index {dest_idx}){corner_marker}")
