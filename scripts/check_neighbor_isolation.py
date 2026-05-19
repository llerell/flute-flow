from simulation import get_indexes_from_image, LATTICE_CX, LATTICE_CY
import numpy as np

walls, size_x, size_y, bc_left, bc_right = get_indexes_from_image('assets/image.png')
wall_set = set((int(i), int(j)) for i,j in walls)

check_neighbors = [(1,0), (0,1), (199,0), (0,499), (1,1), (199,1), (199,499), (1,499)]

print('Are the corner source neighbors ALSO only surrounded by walls?')
print()

for node in check_neighbors:
    i, j = node
    is_wall = node in wall_set
    print(f'{node}: wall={is_wall}')
    if is_wall:
        fluid_neighbors = []
        for q in range(9):
            x = np.mod(i + LATTICE_CX[q], size_x)
            y = np.mod(j + LATTICE_CY[q], size_y)
            if (x, y) not in wall_set:
                fluid_neighbors.append((x, y))
        
        if fluid_neighbors:
            print(f'  Fluid neighbors: {fluid_neighbors}')
        else:
            print(f'  ⚠️  ALL NEIGHBORS ARE WALLS')
