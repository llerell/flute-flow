import numpy as np
from simulation import get_indexes_from_image

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image("assets/image.png")

wall_set = set((i,j) for i,j in walls)
bc_left_set = set((i,j) for i,j in bc_left)
bc_right_set = set((i,j) for i,j in bc_right)

corner = (0, 0)

print(f"Position (0,0):")
print(f"  Is wall? {corner in wall_set}")
print(f"  Is bc_left? {corner in bc_left_set}")
print(f"  Is bc_right? {corner in bc_right_set}")
print()

# Check neighbors
neighbors = [
    (1, 0, "right"), (0, 1, "up"), (-1, 0, "left (wraps)"), (0, -1, "down (wraps)"),
    (1, 1, "up-right"), (-1, 1, "up-left (wraps)"), (-1, -1, "down-left (wraps)"), (1, -1, "down-right (wraps)")
]

print("Neighbors of (0,0):")
for dx, dy, name in neighbors:
    x = np.mod(0 + dx, size_x)
    y = np.mod(0 + dy, size_y)
    is_wall = (x, y) in wall_set
    is_left = (x, y) in bc_left_set
    is_right = (x, y) in bc_right_set
    
    status = []
    if is_wall:
        status.append("WALL")
    if is_left:
        status.append("BC_LEFT")
    if is_right:
        status.append("BC_RIGHT")
    if not status:
        status.append("FLUID")
    
    print(f"  ({x:3d}, {y:3d}) {name:20s} -> {', '.join(status)}")

print()
print("bc_left nodes near (0,0):")
for i, j in bc_left[:20]:
    if abs(i) < 10 and abs(j) < 10:
        print(f"  ({i}, {j})")
