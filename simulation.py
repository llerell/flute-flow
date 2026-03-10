import numpy as np
import os
from evtk import hl as vtkhl
import imageio.v2 as imageio

#=========================================

# 6   2   5
#  \  |  /
# 3---0---1  
#  /  |  \
# 7   4   8

#=========================================


LATTICE_D = 2
LATTICE_Q = 9

# 9 directions of D2Q9 lattice
LATTICE_C = np.zeros([LATTICE_Q, LATTICE_D], dtype=np.int32)
LATTICE_C[0,:] = np.array([0,0])
LATTICE_C[1,:] = np.array([1,0])
LATTICE_C[2,:] = np.array([0,1])
LATTICE_C[3,:] = np.array([-1,0])
LATTICE_C[4,:] = np.array([0,-1])
LATTICE_C[5,:] = np.array([1,1])
LATTICE_C[6,:] = np.array([-1, 1])
LATTICE_C[7,:] = np.array([-1, -1])
LATTICE_C[8,:] = np.array([1, -1])

# weights of the D2Q9 lattice
LATTICE_W = np.zeros(LATTICE_Q)
LATTICE_W[0] = 4./9.
LATTICE_W[1:5] = 1./9.
LATTICE_W[5:] = 1./36.

# x and y components of the lattice velocities
LATTICE_CX = LATTICE_C[:,0]
LATTICE_CY = LATTICE_C[:,1]

# bounce-back

LATTICE_BB = np.ones(LATTICE_Q, dtype=np.int32)
LATTICE_BB[0] = 0 
LATTICE_BB[1] = 3
LATTICE_BB[2] = 4 
LATTICE_BB[3] = 1 
LATTICE_BB[4] = 2 
LATTICE_BB[5] = 7 
LATTICE_BB[6] = 8 
LATTICE_BB[7] = 5
LATTICE_BB[8] = 6 
# inverse of the square of the speed of sound in the lattice
LATTICE_INVCS2 = 3.

# dimensions of the simulation grid

NU = 0.01
TAU = NU*LATTICE_INVCS2 + 1./2.



cpt = iter(range(1000000)) #image counter

def get_walls_from_image(path):
    img = imageio.imread(path)
    size_x, size_y = img.shape[:2]
    walls = np.argwhere(np.sum(img, axis=2)<20)
    return walls, size_x, size_y

  
WALLS, SIZE_X, SIZE_Y = get_walls_from_image("assets/image.png")
points = np.zeros([SIZE_X, SIZE_Y, LATTICE_Q, LATTICE_D])


def save_to_vtk(N, name):
    if not os.path.exists("images"):
        os.makedirs("images")

    rho, u, v = flow_properties(N)
    u = np.reshape(u , (SIZE_X, SIZE_Y, 1), order='C')
    v = np.reshape(v , (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
    pointData={"p": rho - 1., "u": u, "v": v}) 
    

   
def equilibrium_distribution(N: np.array) -> np.array:
    rho, u, v = flow_properties(N)
    return equilibrium_from_moments(rho, u, v)


def equilibrium_from_moments(rho: np.array, u: np.array, v: np.array) -> np.array:
    def p(a,b):
        return np.tensordot(a,b, axes=0)
    
    vc = LATTICE_INVCS2 * (  p(u, LATTICE_CX) 
                           + p(v, LATTICE_CY))
    Neq = p(rho, LATTICE_W) * (  vc 
                               + vc*vc/2
                               - p(u*u + v*v, np.ones(LATTICE_Q)) * LATTICE_INVCS2/2.
                               + 1)
    return Neq


def flow_properties(N: np.array) -> tuple[np.array, np.array, np.array]:
    rho = np.sum(N, axis=2)
    u = np.sum(N * LATTICE_CX, axis=2) / rho
    v = np.sum(N * LATTICE_CY, axis=2) / rho
    return rho, u, v


def collide(N):
    Nm = N - (N - equilibrium_distribution(N))/TAU
    return Nm


def idx(i,j,q):
    return q + LATTICE_Q * (j + SIZE_Y * i)


def calc_permutation():
    P = np.zeros(SIZE_X * SIZE_Y * LATTICE_Q, dtype=np.int32)
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                P[idx(x,y,q)] = idx(i,j,q)
    return P


def stream(Nm, P):
    N = np.reshape(
            np.reshape(Nm, (SIZE_X * SIZE_Y * LATTICE_Q))[P],
            (SIZE_X, SIZE_Y, LATTICE_Q)
        )
    return N


def wall(N, i, j):
    # [i,j] is a wall node
    for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                N[x, y, q] = N[i, j, LATTICE_BB[q]]
    return N


def wall_permutation(Pm, walls):
    w_p = np.copy(Pm)
    for (i,j) in walls:
         for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                w_p[idx(x, y, q)], w_p[idx(i, j, LATTICE_BB[q])] = Pm[idx(i, j, LATTICE_BB[q])], Pm[idx(x, y, q)] 
    return w_p


def bounce_back(Nm, w_p):
    N = np.reshape(
            np.reshape(Nm, (SIZE_X * SIZE_Y * LATTICE_Q))[w_p],
            (SIZE_X, SIZE_Y, LATTICE_Q)
        )
    return N


def main():
    rho = np.ones((SIZE_X, SIZE_Y))
    u = 0. * np.ones((SIZE_X, SIZE_Y))
    v = 0. * np.ones((SIZE_X, SIZE_Y))
    rho[20:25, 20:25] = 1.015
    N = equilibrium_from_moments(rho, u, v)
    P = calc_permutation()
    w_p = wall_permutation(P, WALLS)

    for t in range(500):
        N = stream(N, w_p)
        N = collide(N)
        save_to_vtk(N, "test")

        if np.isnan(N).any():
            print(f"Instability detected at step {t}!")
            break


if __name__ == "__main__":
    main()
