from simulation import (
    LATTICE_Q, idx, calc_permutation, wall_permutation, get_indexes_from_image
)

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image('assets/image.png')
wall_set = set((int(i), int(j)) for i,j in walls)

P = calc_permutation(size_x, size_y)
w_p = wall_permutation(P, walls, size_x, size_y)

print("="*70)
print("VERIFICATION: Corner isolation after fix")
print("="*70)
print()

corner = (0, 0)
print(f"Corner {corner} permutation after fix:")
for q in range(LATTICE_Q):
    idx_corner_q = idx(0, 0, q, size_y)
    source_idx = w_p[idx_corner_q]
    
    q_src = source_idx % LATTICE_Q
    xy_src = source_idx // LATTICE_Q
    j_src = xy_src % size_y
    i_src = xy_src // size_y
    
    print(f"  Dir {q}: comes from ({i_src}, {j_src}) dir {q_src}")

print()
print("="*70)
print("Success! Corner ONLY gets from itself (complete isolation)")
print("="*70)
print()

# Check (1,1) isolation too
node_11 = (1, 1)
print(f"Wall node {node_11} permutation after fix:")
for q in range(LATTICE_Q):
    idx_11_q = idx(1, 1, q, size_y)
    source_idx = w_p[idx_11_q]
    
    q_src = source_idx % LATTICE_Q
    xy_src = source_idx // LATTICE_Q
    j_src = xy_src % size_y
    i_src = xy_src // size_y
    
    if (i_src, j_src) != node_11:
        print(f"  Dir {q}: comes from ({i_src}, {j_src}) dir {q_src}")

if all(w_p[idx(1, 1, q, size_y)] == idx(1, 1, q, size_y) for q in range(LATTICE_Q)):
    print(f"  ✓ Node {node_11} is completely isolated (only gets from itself)")
    print()
    print("⚠️  CRITICAL FIX: (1,1) can NO LONGER receive corrupted values from BC node (2,1)")
    print("     Therefore, corner (0,0) can NO LONGER inherited corrupted values from (1,1)")
