import numpy as np
from simulation import (
    LATTICE_Q, LATTICE_CX, LATTICE_CY, LATTICE_BB, 
    idx, calc_permutation, wall_permutation, 
    get_indexes_from_image
)

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")
wall_set = set((int(i), int(j)) for i, j in walls)
bc_left_set = set((int(i), int(j)) for i, j in bc_left)

P = calc_permutation(size_x, size_y)
w_p = wall_permutation(P, walls, size_x, size_y)

# BC vel indices that get MODIFIED by the BC kernel
BC_VEL_LEFT = [0, 2, 4, 3, 6, 7, 1, 5, 8]
WRITES_TO = [6, 7, 8]  # Indices written by BC kernel (from flute.cl)

print("="*70)
print("HYPOTHESIS CHECK: Can BC-modified data reach corner?")
print("="*70)
print()

# Step 1: After stream, corner gets data from:
corner = (0, 0)
sources_to_corner = {}
for q in range(LATTICE_Q):
    idx_corner_q = idx(0, 0, q, size_y)
    source_idx = w_p[idx_corner_q]
    
    q_src = source_idx % LATTICE_Q
    xy_src = source_idx // LATTICE_Q
    j_src = xy_src % size_y
    i_src = xy_src // size_y
    
    sources_to_corner[q] = ((i_src, j_src), q_src)

print("Corner sources after stream:")
for q, (pos, q_src) in sources_to_corner.items():
    print(f"  Dir {q} ← {pos} dir {q_src}")

print()
print("="*70)
print("CRITICAL QUESTION: Can corner indices be WRITTEN TO by any position?")
print("="*70)
print()

# Reverse: for each direction at corner, what positions CAN write to it?
print("For each corner direction, what positions have it as a destination?")
print()

for corner_q in range(LATTICE_Q):
    idx_corner_q = idx(0, 0, corner_q, size_y)
    
    # Find all positions (i,j) and directions q_src such that
    # w_p[ idx(i, j, q_src, size_y) ] == idx_corner_q
    
    # Actually, this is backwards. w_p[ idx_corner_q ] tells us where corner gets from.
    # The reverse would be: what positions write TO corner?
    
    # Let me check: for each BC node, what does it stream into?
    pass

print()
print("="*70) 
print("FORWARD TRACE: BC NODE (2,1) → WHAT HAPPENS TO ITS DATA?")
print("="*70)
print()

bc_node = (2, 1)

# BC kernel modifies directions BC_VEL_LEFT[6,7,8] = [1, 5, 8]
bc_writes = [BC_VEL_LEFT[6], BC_VEL_LEFT[7], BC_VEL_LEFT[8]]
print(f"BC kernel at (2,1) modifies directions: {bc_writes}")
print()

# After BC modification, these values at (2,1) will stream onward
# Following the permutation from the NEXT timestep
# But wait - the BC kernel modifies M_g
# Then collision reads from M_g

# The key: does anything from (2,1) reach the corner's directions?

# In the next stream step, data from (2,1) would stream based on normal direction vectors
# dir 1: (2,1) → (3,1)
# dir 5: (2,1) → (3,2)
# dir 8: (2,1) → (3,0)

for orig_q in bc_writes:
    x_str = np.mod(2 + LATTICE_CX[orig_q], size_x)
    y_str = np.mod(1 + LATTICE_CY[orig_q], size_y)
    
    target = (x_str, y_str)
    
    if target == corner:
        print(f"⚠️  BC dir {orig_q} at (2,1) streams to CORNER {corner}")
    else:
        print(f"    BC dir {orig_q} at (2,1) streams to {target}")

print()
print("=" * 70)
print("THE REAL ISSUE: Streaming happens in ONE direction")
print("=" * 70)
print()

print("Corner gets from itself (via permutation bounce-back)")
print("But bc kernel modifies (2,1), not (0,0)")
print()
print("UNLESS: The permutation P routes a neighbor (like 1,1) to corner?")
print("Then BC modifies (1,1) → corner changes")
print()

# Check ALL positions that can stream to corner
print("Positions that can naturally stream to corner (without permutation):")
for q in range(LATTICE_Q):
    source_i = np.mod(0 - LATTICE_CX[q], size_x)  # Work backwards
    source_j = np.mod(0 - LATTICE_CY[q], size_y)
    
    if (source_i, source_j) == (0, 0):
        continue  # Skip self
    
    print(f"  Dir {q}: naturally from ({source_i}, {source_j})")
    
    if (source_i, source_j) in bc_left_set:
        print(f"         ⚠️  That's a BC node!")
