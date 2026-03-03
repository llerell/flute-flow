import numpy as np
import os
from evtk import hl as vtkhl

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

# inverse of the square of the speed of sound in the lattice
LATTICE_INVCS2 = 3.

# dimensions of the simulation grid
SIZE_X = 300
SIZE_Y = 400

NU = 0.01
TAU = NU*LATTICE_INVCS2 + 1./2.
points = np.zeros([SIZE_X, SIZE_Y, LATTICE_Q, LATTICE_D])

cpt = iter(range(1000000)) #image counter


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
    P=np.zeros(SIZE_X*SIZE_Y*LATTICE_Q, dtype=np.int32)
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                
                P[idx(x,y,q)] = idx(i,j,q)
    return P

def stream(Nm, P):
    N = np.reshape(np.reshape(Nm, (SIZE_X * SIZE_Y * LATTICE_Q))[P],
                    (SIZE_X, SIZE_Y, LATTICE_Q))
    return N


def main():
    rho = np.ones((SIZE_X, SIZE_Y))
    u = 0. * np.ones((SIZE_X, SIZE_Y))
    v = 0. * np.ones((SIZE_X, SIZE_Y))
    rho[15:17, 15:17] = 1.01
    N = equilibrium_from_moments(rho, u, v)
    P = calc_permutation()

    for t in range(100):
        N = stream(N, P)
        N = collide(N)
        save_to_vtk(N, "test")

        if np.isnan(N).any():
            print(f"Instability detected at step {t}!")
            break


if __name__ == "__main__":
    main()
