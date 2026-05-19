import numpy as np
from simulation import (
    LATTICE_Q, idx, get_indexes_from_image
)

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")

# Convert to int tuples for easier checking
bc_left_set = set((int(i), int(j)) for i, j in bc_left)
bc_right_set = set((int(i), int(j)) for i, j in bc_right)

corner = (0, 0)

print(f"Is corner (0,0) a BC_LEFT node? {corner in bc_left_set}")
print(f"Is corner (0,0) a BC_RIGHT node? {corner in bc_right_set}")
print()

# Check what indices the BC_LEFT kernel modifies
print("="*70)
print("BC_LEFT KERNEL MODIFIES INDICES FOR DIRECTIONS: 6, 7, 8")
print("="*70)
print()

# The velocity_bc_left kernel does: 
# N_out[indexes[6]] = ...
# N_out[indexes[7]] = ...
# N_out[indexes[8]] = ...

# where indexes[q] = xyq(xy, bc_vel_left[q])
# and bc_vel_left[] = {0, 2, 4, 3, 6, 7, 1, 5, 8}

BC_VEL_LEFT = [0, 2, 4, 3, 6, 7, 1, 5, 8]

print("For a BC_LEFT node at (x,y), it modifies:")
print(f"  N_out[xy, {BC_VEL_LEFT[6]}] = indexes[6]")
print(f"  N_out[xy, {BC_VEL_LEFT[7]}] = indexes[7]")
print(f"  N_out[xy, {BC_VEL_LEFT[8]}] = indexes[8]")
print()

# Now check: do any BC_LEFT nodes have streaming destinations that hit corner?
print("="*70)
print("BC_LEFT NODES WHOSE OUTPUT WRITES TO CORNER")
print("="*70)
print()

LATTICE_CX = np.array([0, 1, 0, -1,  0, 1, -1, -1, 1])
LATTICE_CY = np.array([0, 0, 1,  0, -1, 1,  1, -1,-1])

corner_hit = False

for bc_idx, (i, j) in enumerate(bc_left):
    i, j = int(i), int(j)
    
    # BC kernel writes to indices[6], indices[7], indices[8]
    # where indices[q] = xyq(xy, bc_vel_left[q])
    
    for write_q in [6, 7, 8]:  # What BC kernel writes
        indices_q = BC_VEL_LEFT[write_q]  # Which direction in the permutation
        
        # This write goes to (i,j) direction indices_q
        # But wait - the BC kernel is writing to N_out, not directly to positions
        # It's modifying indices[6], indices[7], indices[8]
        # where indices[q] = xyq(xy, bc_vel_left[q])
        
        # So it's writing to:
        # N_out[0 + LATTICE_Q * (j + size_y * i), bc_vel_left[write_q]]
        bc_write_dir = BC_VEL_LEFT[write_q]
        
        # Is this at the corner?
        if i == 0 and j == 0:
            print(f"⚠️  BC node ({i},{j}) writes to corner dir {bc_write_dir}")
            corner_hit = True

if not corner_hit:
    print("✓ No BC_LEFT node writes directly to corner")
    
print()
print("="*70)
print("BUT: BC nodes can INDIRECTLY affect corner through streaming!")
print("="*70)
print()

# The issue might be: 
# 1. BC_LEFT modifies values at nodes (2,1), (3,1), etc.
# 2. Those modified values stream outward
# 3. If the permutation routes any of those to corner, corner changes

# Check: after BC kernel modifies (2,1), can any of those changes reach corner?
bc_near = (2, 1)
print(f"Example: BC node {bc_near}")
print(f"BC kernel modifies its directions: {[BC_VEL_LEFT[q] for q in [6,7,8]]}")
print()

# Those directions then participate in streaming...
# The issue is the permutation + streaming chain.
